import sys
import argparse
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from data import MNISTDataest
from net import MNISTNetwork

BATCH_SIZE = 32
EPOCH = 10

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./datasets/MNIST/")
parser.add_argument("--act_func", type=str, default="ReLU",
                    choices=["Tanh", "ReLU", "ELU", "SiLU"])
parser.add_argument("--learning_rate", type=int, default=1e-2,
                    choices=[1e-2, 1e-3, 1e-4])

def plot_loss(train_losses, val_losses, act_func, lr):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label ='Train_Loss', marker ='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label ='Validation_Loss', marker ='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    title = f"{act_func} Activation Function & lr = {lr}"
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig(f'./result/{title}.png')


def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device : {device}")

    train_dataset, val_dataset = MNISTDataest(args.data_path, True), \
                                 MNISTDataest(args.data_path, False)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4), \
                               DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    net = MNISTNetwork(args.act_func).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.005)

    train_losses = []
    val_losses = []
    for epoch in range(EPOCH):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        net.train()

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct == (predicted == labels).sum().item()

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
                val_correct == (predicted == labels).sum().item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCH}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    plot_loss(train_losses, val_losses, act_func=args.act_func, lr=args.learning_rate)
    
if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
    