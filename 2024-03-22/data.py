import os
import torch
import pandas as pd

from torch.utils.data import Dataset, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class IMDBDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        
        # setting
        self.root = os.path.join(root, "train.csv")
        self.data = pd.read_csv(self.root) # data load
        self.tokenizer = get_tokenizer('basic_english') # tokenizer 정의
        
        # text 데이터
        raw_text = self.data['text']
        self.vocab = self._build_vocab(self.data['text']) # vocab 생성

        # text 데이터 전처리
        text_data = []
        for sent in raw_text:
            tokenized_sent = self.tokenizer(sent) # 문장을 tokenize
            encoded_sent = self.vocab(tokenized_sent) # vocab을 통해 encoding된 리스트 반환
            text_data.append(encoded_sent)
        self.text_data = text_data
        
        # label 데이터
        label_data = self.data['sentiment'].replace(['pos', 'neg'], [1, 0])
        self.label_data = label_data.to_list()
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.text_data[idx], dtype=torch.long)
        label = self.label_data[idx]

        return text, label

    # tokenizer를 활용하여 text tokenize
    def _yield_tokens(self, raw_text):
        for sent in raw_text:
            yield self.tokenizer(sent)

    # tokenized된 text를 통해 vocab 생성
    def _build_vocab(self, raw_text):
        vocab = build_vocab_from_iterator(self._yield_tokens(raw_text), specials=["<pad>", "<unk>"])
        vocab.set_default_index(vocab["<pad>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    # train과 validation 데이터셋을 ratio만큼 분할 (8:2)
    def split_dataset(self, ratio=0.2):
        data_size = len(self)
        val_size = int(data_size * ratio)
        train_size = data_size - val_size

        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        return train_dataset, val_dataset

if __name__ == '__main__':
    data_path = '../../datasets/IMDB/'
    dataset = IMDBDataset(data_path)
    train_dataset, _ = dataset.split_dataset(ratio=0.2)
    
    text, label = train_dataset[0]
    print(f"text shape: {text.shape}")
    