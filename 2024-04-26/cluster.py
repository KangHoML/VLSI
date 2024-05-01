import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, MeanShift

from data import build_dataset
from net import AutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../../datasets/OxfordPet/")
parser.add_argument("--cluster", type=str, default='KMeans')
parser.add_argument("--load", type=str, default="resnet_ae")

def get_cluster():
    if args.cluster == "KMeans":
        return KMeans(n_clusters=37)
    elif args.cluster == "MeanShift":
        return MeanShift()
    elif args.cluster == "DBSCAN":
        return DBSCAN()

def plot_cluster(embedded, labels):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='viridis', s=5, alpha=0.5)
    plt.colorbar(scatter)
    plt.title(f'Clustering by {args.cluster}')
    plt.savefig(f'{args.load}_cl.png')

def cluster():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset, val_dataset = build_dataset(root=args.data_path)
    dataset = train_dataset + val_dataset
    data_loader = DataLoader(dataset, batch_size=16, 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    net = AutoEncoder()
    net.load_state_dict(torch.load(f'./result/{args.load}.pth'))
    net.to(device)

    # feature 추출
    net.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader):
            inputs = inputs.to(device)
            latent, _ = net(inputs)
            features.append(latent.view(latent.size(0), -1).detach().cpu().numpy())
    features = np.concatenate(features, axis=0)
    
    # 군집화
    cluster_net = get_cluster()
    labels = cluster_net.fit_predict(features)

    # 시각화
    embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)
    plot_cluster(embedded, labels)


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    cluster()