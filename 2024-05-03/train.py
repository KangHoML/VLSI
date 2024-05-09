import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
import torchmetrics

from net import CNNAutoEncoder, TransformerAutoEncoder

IMG_SIZE = 64

parser = argparse.ArgumentParser()
# -- hyperparameter about data
parser.add_argument("--data_path", type=str, default="data")

# -- hyperparameter about ddp &amp
parser.add_argument("--is_ddp", type=bool, default=False)
parser.add_argument("--is_amp", type=bool, default=False)

# -- hyperparameter about train
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--mode", type=str, default="CNN")

# -- hyperparamter for saving model
parser.add_argument("--save", type=str, default="ckpt1")

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label ='Train_Loss', marker ='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label ='Validation_Loss', marker ='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    title = f"{args.save}_loss"
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
    psnr = 0.0
    ssim = 0.0
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    net.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader):
            inputs = inputs.to(device)

            with torch.cuda.amp.autocast():
                _, outputs = net(inputs)
                loss = criterion(outputs, inputs)

            val_loss += loss.item()
            psnr += psnr_metric(outputs, inputs).item()
            ssim += ssim_metric(outputs, inputs).item()
        
    val_loss /= len(val_loader)
    psnr /= len(val_loader)
    ssim /= len(val_loader)
    print(f"Test Loss: {val_loss:.4f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    return val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Oxford Dataset
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8),
        #v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
    ])

    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8),
        #v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
    ])

    # train_datasets
    train_dataset = CIFAR10(root = args.data_path, download=True, train = True, transform=train_transforms)
    val_dataset  = CIFAR10(root = args.data_path, download=True, train = False, transform=val_transforms)
    
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
    if args.mode == "CNN":
        net = CNNAutoEncoder().to(device)
    elif args.mode == "Transformer":
        net = TransformerAutoEncoder().to(device)

    if args.is_ddp:
        net = DistributedDataParallel(net)
    
    # define the loss function
    criterion = MSELoss()
    
    # define the optimizer
    optimizer = Adam(net.parameters(), lr=args.learning_rate)

    # define the scaler
    if args.is_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None


    view_data = next(iter(train_loader))[0][:5] # 첫번째 배치에서 이미지를 가져옵니다.
    view_data = view_data.to(device)

    train_losses = []
    val_losses = []
    best_loss = float('inf')
    os.makedirs(f"result/epoch_{args.mode}", exist_ok=True)
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

        test_x = view_data
        _, decoded_data = net(test_x)

        # Plot and save images
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
        for idx, ax in enumerate(axes.flat):
            ax.axis('off')
            if idx < 5:
                original_image = view_data[idx].detach().cpu().permute(1, 2, 0).numpy()
                print(f"Original Image - Min: {original_image.min()}, Max: {original_image.max()}")
                #original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())  # Normalize pixel values to [0, 1]
                original_image = np.clip(original_image, 0, 1)
                ax.imshow(original_image)
                ax.set_title('Original')
            else:
                decoded_image = decoded_data[idx-5].detach().cpu().permute(1, 2, 0).numpy()
                print(f"Decoded Image - Min: {decoded_image.min()}, Max: {decoded_image.max()}")
                #decoded_image = (decoded_image - decoded_image.min()) / (decoded_image.max() - decoded_image.min())  # Normalize pixel values to [0, 1]
                decoded_image = np.clip(decoded_image, 0, 1)
                ax.imshow(decoded_image)
                ax.set_title('Decoded')
        plt.tight_layout()
        plt.savefig(f'./result/epoch_{args.mode}/epoch_{epoch}_images.png')
        plt.close(fig)
    
    if rank == 0:
        plot_loss(train_losses, val_losses)

    if args.is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    os.makedirs("result", exist_ok=True)
    main()


