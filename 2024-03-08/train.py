import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.distributed as dist

from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from data import CIFAR10Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../../datasets/CIFAR-10")
parser.add_argument("--distribution", type=bool, default=True)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--epoch", type=int, default=20)

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label ='Train_Loss', marker ='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label ='Validation_Loss', marker ='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    title = "ResNet50 with pretrained"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f'./result/{title}.png')

def train(args):
    os.makedirs("result", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = CIFAR10Dataset(args.data_path, True), \
                                 CIFAR10Dataset(args.data_path, False)
    
    if args.distribution:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None

    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                           shuffle=(not args.distribution), num_workers=4, pin_memory=True), \
                               DataLoader(val_dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=4, pin_memory=True)
    
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    if args.distribution:
        net = DistributedDataParallel(net)
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []

    for epoch in range(args.epoch):
        if args.distribution:
            train_sampler.set_epoch(epoch)

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        net.train()
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        net.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)

        if rank == 0 or not args.distribution:
            print(f"Epoch [{epoch+1}/{args.epoch}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    if rank == 0 or not args.distribution:
        plot_loss(train_losses, val_losses)
    
    if args.distribution:
        dist.destroy_process_group()

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
