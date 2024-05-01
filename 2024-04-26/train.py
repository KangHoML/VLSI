import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from data import build_dataset
from net import AutoEncoder

parser = argparse.ArgumentParser()
# -- hyperparameter about data
parser.add_argument("--data_path", type=str, default="../../datasets/OxfordPet/")

# -- hyperparameter about ddp &amp
parser.add_argument("--is_ddp", type=bool, default=False)
parser.add_argument("--is_amp", type=bool, default=False)

# -- hyperparameter about train
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)

# -- hyperparamter for saving model
parser.add_argument("--save", type=str, default="ckpt1")

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
    plt.close()

# train code
def train(net, train_loader, criterion, optimizer, scaler, device):
    train_loss = 0.0
    
    net.train()
    for inputs, _ in tqdm(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                _, outputs = net(inputs)
                loss = criterion(outputs, inputs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, outputs = net(inputs)
            loss = criterion(outputs, inputs)
            
            loss.backward()
            optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)

    return train_loss

# evaluate code
def eval(net, val_loader, criterion, device):
    val_loss = 0.0
    
    net.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader):
            inputs = inputs.to(device)

            _, outputs = net(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item()
        
    val_loss /= len(val_loader)

    return val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Oxford Dataset
    train_dataset, val_dataset = build_dataset(root=args.data_path)
    
    # set the ddpl
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
    net = AutoEncoder().to(device)
    if args.is_ddp:
        net = DistributedDataParallel(net)
    
    # define the CrossEntropyLoss
    criterion = CrossEntropyLoss()
    
    # define the optimizer
    optimizer = Adam(net.parameters(), lr=args.learning_rate)

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

        train_loss = train(net, train_loader, criterion, optimizer, scaler, device)
        train_losses.append(train_loss)

        val_loss = eval(net, val_loader, criterion, device)
        val_losses.append(val_loss)

        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epoch}]")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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