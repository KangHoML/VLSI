import os
import torchvision.transforms as T

from torchvision import datasets

def build_dataset(root, train=True):
    if train:
        dir_root = os.path.join(root, "train")
        transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(32, padding=4),
                T.ColorJitter(brightness=0.2, contrast =0.2, saturation =0.2, hue =0.1),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4882, 0.4465], std =[0.2023, 0.1994, 0.2010])
            ])
    else:
        dir_root = os.path.join(root, "val")
        transform = T.Compose([
                T.CenterCrop(32),
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4882, 0.4465], std =[0.2023, 0.1994, 0.2010])
            ])
        
    os.makedirs(dir_root, exist_ok=True)
    dataset = datasets.CIFAR100(root=dir_root, train=train, transform=transform, download=True)
    return dataset

if __name__ == '__main__':
    root = "../../datasets/CIFAR100/"
    train_dataset, val_dataset = build_dataset(root, train=True), build_dataset(root, train=False)
    img, label = train_dataset[0]
    print(f"img shape: {img.shape}, label: {label}")

