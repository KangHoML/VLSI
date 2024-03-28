import os
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset

class MNISTDataest(Dataset):
    def __init__(self, root, train):
        super().__init__()
        
        if train:
            self.root = os.path.join(root, "train")
            self.transform = T.Compose([
                T.RandomCrop(28),
                T.RandomRotation(15),
                T.ToTensor()
            ])
        else:
            self.root = os.path.join(root, "val")
            self.transform = T.Compose([
                T.CenterCrop(28),
                T.ToTensor()
            ])

        data_list = []
        for i in range(10):
            dir_root = os.path.join(self.root, str(i))
            for img in os.listdir(dir_root):
                img_path = os.path.join(dir_root, img)
                data_list.append((i, img_path))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        label, img_path = self.data_list[idx]

        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        
        return img, label

if __name__ == "__main__":
    data_path = "../../workspace/datasets/MNIST/"
    example_dataset = MNISTDataest(data_path, train=False)
    img, target = example_dataset[10]
    print(f"img shape : {img.shape}")