import os
import torch
import argparse

from collections import OrderedDict
from torch.utils.data import DataLoader

from data import IMDBDataset
from net import SentenceClassifier  # 이미 정의된 모델 클래스 사용
from train import collate_fn, tokenizer_type

parser = argparse.ArgumentParser()
# -- path setting
parser.add_argument("--data_path", type=str, default="../../datasets/IMDB/")
parser.add_argument("--weight_path", type=str, default="../../pth/IMDB/")
parser.add_argument("--pth_name", type=str, default="lstm_ddp")

# -- data setting
parser.add_argument("--tokenizer", type=str, default='torchtext')
parser.add_argument("--vocab_size", type=int, default=40000)

# -- model setting
parser.add_argument("--model", type=str, default='gru')
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--embed_size", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--bidirectional", type=bool, default=False)

# -- batch_size
parser.add_argument("--batch_size", type=int, default=32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = IMDBDataset(root=args.data_path, train=False, tokenizer=tokenizer_type(), vocab_size=args.vocab_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    weight_path = os.path.join(args.weight_path, args.pth_name + '.pth')
    state_dict = torch.load(weight_path, map_location=device)
    vocab_size = state_dict['embedding.weight'].size()[0]
    net = SentenceClassifier(vocab_size, hidden_size=args.hidden_size, embed_size=args.embed_size, n_layers=args.n_layers, 
                             dropout=args.dropout, bidirectional=args.bidirectional, model_type=args.model).to(device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.'
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
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
    print(f'{args.pth_name} Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    main()