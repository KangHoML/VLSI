import os
import torch
import argparse

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

from data import NSMCDataset
from net import SentenceClassifier  # 이미 정의된 모델 클래스 사용
from train import tokenizer_type

parser = argparse.ArgumentParser()
# -- path setting
parser.add_argument("--weight_path", type=str, default="./result/")
parser.add_argument("--pth_name", type=str, default="okt")

# -- data setting
parser.add_argument("--tokenizer", type=str, default='okt')
parser.add_argument("--vocab_size", type=int, default=5000)
parser.add_argument("--max_len", type=int, default=32)

# -- batch_size
parser.add_argument("--batch_size", type=int, default=16)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = NSMCDataset(tokenizer=tokenizer_type(args.tokenizer), n_vocab=args.vocab_size, max_len=args.max_len, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    weight_path = os.path.join(args.weight_path, args.pth_name + '.pth')
    state_dict = torch.load(weight_path, map_location=device)
    vocab_size = state_dict['embedding.weight'].size()[0]
    net = SentenceClassifier(vocab_size, hidden_size=128, embed_size=100, n_layers=2).to(device)

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
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = net(inputs)
            predicted = torch.sigmoid(outputs) > .5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'{args.pth_name} Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    main()