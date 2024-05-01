import os
import torchvision.transforms as T

from torchvision import datasets
from torch.utils.data import random_split

# train과 validation 데이터셋을 ratio만큼 분할 (8:2)
def split_dataset(dataset, ratio=0.2):
    data_size = len(dataset)
    val_size = int(data_size * ratio)
    train_size = data_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

# 데이터셋 구축
def build_dataset(root):
    transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
            ])
        
    os.makedirs(root, exist_ok=True)
    dataset = datasets.OxfordIIITPet(root=root, split="trainval", transform=transform, download=True)
    train_dataset, val_dataset = split_dataset(dataset)

    return train_dataset, val_dataset

if __name__ == '__main__':
    root = "../../datasets/OxfordPet/"
    train_dataset, val_dataset = build_dataset(root)
    img, label = train_dataset[0]
    print(f"img shape: {img.shape}, label: {label}")