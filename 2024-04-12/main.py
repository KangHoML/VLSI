import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

from data import build_dataset
from net import ConvNeXt_T

def parse_cfgs(input_str):
    if input_str is None:
        return None
    
    items = input_str.split(',')
    result = []
    for item in items:
        numbers = tuple(map(int, item.strip().split()))
        if len(numbers) == 2:
            result.append(numbers)
        else:
            raise argparse.ArgumentTypeError()
    return result

parser = argparse.ArgumentParser()
# -- hyperparameter about data
parser.add_argument("--data_path", type=str, default="../../datasets/CIFAR100/")

# -- hyperparameter about ddp &amp
parser.add_argument("--is_ddp", type=bool, default=False)
parser.add_argument("--is_amp", type=bool, default=False)

# -- hyperparameter about model
parser.add_argument("--pretrained", type=bool, default=False)
parser.add_argument("--patch_size", type=int, default=4)
parser.add_argument("--cfgs", type=parse_cfgs, default=None)

# -- hyperparameter about train
parser.add_argument("--optimizer", type=str, default='SGD')
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--lr_scheduler", type=str, default=None)
parser.add_argument("--step_size", type=int, default=1)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)

# -- hyperparamter for saving model
parser.add_argument("--save", type=str, default="convnext")

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label ='Train_Loss', marker ='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label ='Validation_Loss', marker ='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    title = f"{args.save}"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f'./result/{title}.png')

# optimizer type 설정
def get_optimizer():
    if args.optimizer == 'SGD':
        return SGD
    elif args.optimizer == 'Adam':
        return Adam
    elif args.optimizer == 'AdamW':
        return AdamW
    else:
        raise ValueError(args.optimizer)

# scheduler type 설정
def get_scheduler():
    if args.lr_scheduler == "Step":
        return StepLR
    elif args.lr_scheduler == "Cosine":
        return CosineAnnealingLR
    elif args.lr_scheduler == "Cycle":
        return OneCycleLR
    else:
        raise ValueError(args.lr_scheduler)

# train code
def train(net, train_loader, criterion, optimizer, scheduler, scaler, device):
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    net.train()

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        train_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step()
    
    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total

    return train_loss, train_accuracy

# evaluate code
def eval(net, val_loader, criterion, device):
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    net.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            if criterion is not None:
                val_loss += loss.item()
    
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    return val_loss, val_accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NSMCDataset
    train_dataset, val_dataset = build_dataset(root=args.data_path, train=True), build_dataset(root=args.data_path, train=False)
    
    # set the ddp
    if args.is_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        rank = 0
    
    # define the dataloader
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                                          shuffle=(not args.is_ddp), num_workers=4, pin_memory=True), \
                               DataLoader(val_dataset, batch_size=args.batch_size, 
                                          shuffle=False, num_workers=4, pin_memory=True)

    # define the model instance
    net = ConvNeXt_T(patch_size=args.patch_size, cfgs=args.cfgs, pretrained=args.pretrained).to(device)
    if args.is_ddp:
        net = DistributedDataParallel(net)
    
    # define the CrossEntropyLoss
    criterion = CrossEntropyLoss()
    
    # define the optimizer
    optimizer_type = get_optimizer()
    optimizer = optimizer_type(net.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)

    # define the schedular
    scheduler = None
    if args.lr_scheduler is not None:
        scheduler_type = get_scheduler()
        try:
            scheduler = scheduler_type(optimizer, step_size=args.step_size, gamma=args.gamma)
        except:
            scheduler = scheduler_type(optimizer, T_max=args.step_size)

    # define the scaler
    if args.is_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    train_losses = []
    val_losses = []
    best_loss = float('inf')

    for epoch in range(args.epoch):
        if args.is_ddp:
            train_sampler.set_epoch(epoch)

        train_loss, train_accuracy = train(net, train_loader, criterion, optimizer, scheduler, scaler, device)
        train_losses.append(train_loss)

        val_loss, val_accuracy = eval(net, val_loader, criterion, device)
        val_losses.append(val_loss)

        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epoch}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            if val_loss < best_loss:
                best_loss = val_loss
                if args.is_ddp:
                    weights = net.module.state_dict()
                else:
                    weights = net.state_dict()
                torch.save(weights, f'./result/{args.save}.pth')

    if rank == 0:
        plot_loss(train_losses, val_losses)

    if args.is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    os.makedirs("result", exist_ok=True)
    main()
