import os
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    name2idx = {
        'airplane' : 0, 'automobile' : 1, 'bird' : 2,
        'cat' : 3, 'deer' : 4, 'dog' : 5, 'frog' : 6,
        'horse' : 7, 'ship' : 8, 'truck' : 9
    }

    def __init__(self, root, train):
        super().__init__()
        
        if train:
            self.root = os.path.join(root, "train")
            self.transform = T.Compose([ 
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
            ])

        else:
            self.root = os.path.join(root, "val")
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
            ])
        
        data_list = []
        for class_name in os.listdir(self.root):
            dir_root = os.path.join(self.root, class_name)
            for img in os.listdir(dir_root):
                img_path = os.path.join(dir_root, img)
                data_list.append((self.name2idx[class_name], img_path))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        label, img_path = self.data_list[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label
    
if __name__ == "__main__":
    data_path = "../../datasets/CIFAR-10/"
    example_dataset = CIFAR10Dataset(data_path, train=False)
    img, target = example_dataset[10]
    print(f"img shape : {img.shape}")