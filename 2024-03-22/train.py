import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

from data import IMDBDataset
from net import SentenceClassifier

parser = argparse.ArgumentParser()
# -- hyperparameter about data
parser.add_argument("--data_path", type=str, default="../../datasets/")
parser.add_argument("--threshold", type=int, default=3)
parser.add_argument("--max_len", type=int, default=500)

# -- hyperparameter about ddp &amp
parser.add_argument("--is_ddp", type=bool, default=False)
parser.add_argument("--is_amp", type=bool, default=False)

# -- hyperparameter about model
parser.add_argument("--model", type=str, default='rnn')
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--bidirectional", type=bool, default=False)

# -- hyperparameter about train
parser.add_argument("--optimizer", type=str, default='Adam')
parser.add_argument("--lr_scheduler", type=str, default='Step')
parser.add_argument("--step_size", type=float, default=1.0)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--epoch", type=int, default=10)

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label ='Train_Loss', marker ='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label ='Validation_Loss', marker ='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    title = f"{args.model}"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f'./result/{title}.png')

def get_optimizer():
    if args.optimizer == 'SGD':
        return SGD
    elif args.optimizer == 'Adam':
        return Adam
    else:
        raise ValueError(args.optimizer)

def get_scheduler():
    if args.lr_scheduler == "Step":
        return StepLR
    elif args.lr_scheduler == "Cosine":
        return CosineAnnealingLR
    else:
        raise ValueError(args.lr_scheduler)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # IMDBDataset
    train_dataset, val_dataset = IMDBDataset(args.data_path, threshold=args.threshold, max_len=args.max_len, mode='train'), \
                                 IMDBDataset(args.data_path, threshold=args.threshold, max_len=args.max_len, mode='val')
    vocab_size = len(train_dataset.vocab)
    
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
    net = SentenceClassifier(vocab_size, hidden_size=64, embed_size=128, n_layers=args.n_layers, 
                             dropout=args.dropout, bidirectional=args.bidirectional, model_type=args.model).to(device)
    if args.is_ddp:
        net = DistributedDataParallel(net, find_unused_parameters=True)
    
    # define the CrossEntropyLoss
    criterion = CrossEntropyLoss()
    
    # define the optimizer
    optimizer_type = get_optimizer()
    optimizer = optimizer_type(net.parameters(), lr=args.learning_rate)

    # define the schedular
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
        
        scheduler.step()
        
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
                torch.save(weights, f'./result/{args.model}.pth')

    if rank == 0:
        plot_loss(train_losses, val_losses)

    if args.is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    os.makedirs("result", exist_ok=True)
    main()


