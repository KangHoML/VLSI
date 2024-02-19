import os
import torch
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm

from data import CIFAR10Dataset
from vggnet import VGG
from resnet import ResNet
from customnet import CustomNet

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./datasets/CIFAR-10")
parser.add_argument("--model", type=str, default="VGG16")
parser.add_argument("--bn_flag", type=bool, default=True)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--epoch", type=int, default=20)

def get_model(model, bn_flag, kernel_size):
    if "VGG" in model:
        return VGG(cfg=model, bn=bn_flag)
    elif "ResNet" in model:
        return ResNet(cfg=model)
    elif "CustomNet" in model:
        return CustomNet(kernel_size)
    else:
        raise NotImplementedError(model)

def plot_loss(train_losses, val_losses, model, bn_flag, kernel_size):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label ='Train_Loss', marker ='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label ='Validation_Loss', marker ='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if "VGG" in model:
        title = f"{model} Architecture with bn_flag={bn_flag}"
    elif "ResNet" in model:
        title = f"{model} Architecture"
    else:
        title = f"{model} Architecture with kernel_size = {kernel_size}"
    plt.title(title)
    plt.legend()
    plt.grid()
    
    if "CustomNet" in model:
        plt.savefig(f'./result/{model}_{kernel_size}.png')
    else:
        plt.savefig(f'./result/{model}.png')

def train(args):
    os.makedirs("result", exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device : {device}")

    train_dataset, val_dataset = CIFAR10Dataset(args.data_path, True), \
                                 CIFAR10Dataset(args.data_path, False)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4), \
                               DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    net = get_model(args.model, args.bn_flag, args.kernel_size).to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001)

    train_losses = []
    val_losses = []

    for epoch in range(args.epoch):
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

        print(f"Epoch [{epoch+1}/{args.epoch}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    plot_loss(train_losses, val_losses, model=args.model, bn_flag=args.bn_flag, kernel_size=args.kernel_size)

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)




    
    
