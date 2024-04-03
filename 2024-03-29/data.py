import os
import pickle
import torch
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, random_split
from konlpy.tag import Okt
from collections import Counter

class NSMCDataset(Dataset):
    def __init__(self, tokenizer, n_vocab, max_len, train=True):
        super().__init__()

        # tokenizer 정의
        self.tokenizer = tokenizer
        raw_data = load_dataset("nsmc")

        if train:
            raw_data = raw_data["train"]
            tokenized_file = f'./pkl/train_tokenized_{tokenizer.__class__.__name__}.pk1'
        else:
            raw_data = raw_data["test"]
            tokenized_file = f'./pkl/test_tokenized_{tokenizer.__class__.__name__}.pk1'

        # 기존 데이터
        raw_text_data = raw_data["document"]
        
        # tokenize 하기
        if os.path.exists(tokenized_file):
            with open(tokenized_file, 'rb') as f:
                tokenized_text_data = pickle.load(f)
        else:
            raw_text_data = raw_data["document"]   
            tokenized_text_data = self._tokenize(raw_text_data)
            with open(tokenized_file, 'wb') as f:
                pickle.dump(tokenized_text_data, f)

        # vocab 만들기
        self.vocab = self._build_vocab(tokenized_text_data, n_vocab=n_vocab, special_tokens=["<pad>", "<unk>"])

        # 인코딩
        token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        encoded_texts = [
            [token_to_id.get(token, token_to_id["<unk>"]) for token in text] for text in tokenized_text_data
        ]

        # 패딩
        padded_texts = self._pad_sequence(encoded_texts, max_len, token_to_id["<pad>"])

        # 데이터
        self.text_data = padded_texts
        self.label_data = raw_data["label"]

    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.text_data[idx], dtype=torch.long)
        label = torch.tensor(self.label_data[idx], dtype=torch.float32)

        return text, label

    def _tokenize(self, texts):
        tokenized_texts = [self.tokenizer.morphs(text) for text in tqdm(texts)]
        return tokenized_texts
    
    def _build_vocab(self, tokenized_texts, n_vocab, special_tokens):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        vocab = special_tokens
        
        for token, _ in counter.most_common(n_vocab):
            vocab.append(token)
        
        return vocab

    def _pad_sequence(self, encoded_texts, max_len, pad_value):
        result = list()

        for seq in encoded_texts:
            seq = seq[:max_len]
            pad_length = max_len - len(seq)
            padded_seq = seq + [pad_value] * pad_length
            result.append(padded_seq)
        
        return np.asarray(result)
        
        
    # train과 validation 데이터셋을 ratio만큼 분할 (9:1)
    def split_dataset(self, ratio=0.1):
        data_size = len(self)
        val_size = int(data_size * ratio)
        train_size = data_size - val_size

        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        return train_dataset, val_dataset
    
if __name__ == "__main__":
    tokenizer = Okt()

    dataset = NSMCDataset(tokenizer=tokenizer, n_vocab=5000, max_len=32, train=True)

    train_dataset, val_dataset = dataset.split_dataset()
    text, label = train_dataset[0]
    
    print(f"text: {text.shape}, label: {label.shape}")
    print(f"    text: {text}")
    print(f"    label: {label}")
    