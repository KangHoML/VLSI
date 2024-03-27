import os
import torch
import argparse
from torch.utils.data import DataLoader

from data import IMDBDataset
from net import SentenceClassifier  # 이미 정의된 모델 클래스 사용
from train import collate_fn

parser = argparse.ArgumentParser()
# -- path setting
parser.add_argument("--data_path", type=str, default="../../datasets/IMDB/")
parser.add_argument("--model_path", type=str, default="./results/gru.pth")

# -- model setting
parser.add_argument("--model", type=str, default='gru')
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--bidirectional", type=bool, default=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = IMDBDataset(root=args.data_path, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(test_dataset.vocab)
    net = SentenceClassifier(vocab_size, hidden_size=args.hidden_size, embed_size=args.embed_size, n_layers=args.n_layers, 
                             dropout=args.dropout, bidirectional=args.bidirectional, model_type=args.model).to(device)
    net.load_state_dict(torch.load(args.model_path))
    net = net.to(device)

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    main()