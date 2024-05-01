import torch
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data import build_dataset
from net import AutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../../datasets/OxfordPet/")
parser.add_argument("--sample", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--load", type=str, default="ckpt1.pth")

def plot_img(ori, pre):
    fig, axes = plt.subplot(nrows=2, ncols=args.sample, figsize=(args.sample * 3, 6))

    for i in range(args.samples):
        ax = axes[0, i]

        ori_img = ori[i].detach().cpu().numpy().transpose(1, 2, 0)
        ax.imshow(ori_img)
        ax.axis('off')

        ax = ax[1, i]

        pre_img = pre[i].detach().cpu().numpy().transpose(1, 2, 0)
        ax.imshow(pre_img)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'./result/result_img.png')
    plt.close()


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_dataset = build_dataset(root=args.data_path)
    data_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                          shuffle=False, num_workers=4, pin_memory=True)
    
    net = AutoEncoder()
    weight_path = f'./pth/{args.load}.pth'
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)

    net.eval()
    with torch.no_grad():
        inputs, _ = next(iter(data_loader))
        _, outputs = net(inputs.to(device))
        plot_img(inputs[:args.sample], outputs[:args.sample])

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    test()