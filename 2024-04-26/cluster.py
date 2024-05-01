import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering

from data import build_dataset
from net import AutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../../datasets/OxfordPet/")
parser.add_argument("--cluster", type=str, default='KMeans')
parser.add_argument("--load", type=str, default="resnet_ae")

def get_cluster():
    if args.cluster == "KMeans":
        return KMeans(n_clusters=37)
    elif args.cluster == "Hierarchical":
        return AgglomerativeClustering(n_clusters=37)

def plot_cluster(embedded, labels):
    plt.figure(figsize=(12, 10))
    
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='viridis', s=5, alpha=0.5)
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('viridis')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(label / len(unique_labels)), markersize=10) for label in unique_labels]
    legend_labels = [f'Cluster {label}' for label in unique_labels]
    
    plt.legend(handles, legend_labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(f'Clustering by {args.cluster}')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f'./result/{args.load}_{args.cluster}_cl.png', bbox_inches='tight')

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