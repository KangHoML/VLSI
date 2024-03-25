import os
import nltk
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    def __init__(self, root, threshold=3, max_len=500, mode='train'):
        self.root = os.path.join(root, "IMDBDataset.csv")
        self.mode = mode
        self.df = self._load_data()
        
        text_data, label_data = self._preprocess_data()
        text_data = self._tokenize(text_data)
        
        self.vocab = self._build_vocab(threshold)
        self.index_to_vocab = {v: k for k, v in self.vocab.items()}
        
        text_data = self._text_to_seq(text_data, self.vocab)
        self.text_data = self._pad_sequence(text_data, max_len)
        self.label_data = label_data.to_list()

    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.text_data[idx], dtype=torch.long)
        label = self.label_data[idx]
        return text, label

    def _load_data(self):
        nltk.download('punkt')
        return pd.read_csv(self.root)
    
    def _preprocess_data(self):
        self.df['sentiment'] = self.df['sentiment'].replace(['positive', 'negative'], [1, 0])
        texts, labels = self.df['review'], self.df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.5, 
                                                    random_state=0, stratify=labels)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                        random_state=0, stratify=y_train)
        self.train_texts = X_train

        if self.mode == 'train':
            return X_train, y_train
        elif self.mode == 'val':
            return X_val, y_val
        elif self.mode == 'test':
            return X_test, y_test
        else:
            raise NotImplementedError(self.mode)
        
    def _tokenize(self, sents):
        tokenized_sents = []
        for sent in tqdm(sents):
            tokenized_sent = word_tokenize(sent)
            tokenized_sent = [word.lower() for word in tokenized_sent]
            tokenized_sents.append(tokenized_sent)
        
        return tokenized_sents

    def _build_vocab(self, threshold):
        word_list = []
        for sent in self.train_texts:
            for word in sent:
                word_list.append(word)
        word_count = Counter(word_list)
        vocab = sorted(word_count, key=word_count.get, reverse=True)
        
        rare_cnt = 0
        for _, value in word_count.items():
            if (value < threshold):
                rare_cnt += 1
        vocab_size = len(word_count) - rare_cnt
        vocab = vocab[:vocab_size]

        word_to_idx = {}
        word_to_idx['<PAD>'] = 0 # padding token
        word_to_idx['<UNK>'] = 1 # unknown token

        for idx, word in enumerate(vocab):
            word_to_idx[word] = idx + 2
        
        return word_to_idx

    def _text_to_seq(self, text, word_to_idx):
        encoded = []
        for sent in text:
            seq = []
            for word in sent:
                try:
                    seq.append(word_to_idx[word])
                except KeyError:
                    seq.append(word_to_idx['<UNK>'])
            
            encoded.append(seq)
        
        return encoded

    def _pad_sequence(self, text, max_len):
        features = np.zeros((len(text), max_len), dtype=int)
        for i, sent in enumerate(text):
            if len(sent) != 0:
                features[i, :len(sent)] = np.array(sent)[:max_len]
        return features

if __name__ == '__main__':
    data_path = '../../datasets/'
    train_dataset = IMDBDataset(data_path, mode='train')
    text, label = train_dataset[10]
    print(f"text shape: {text.shape}")
    