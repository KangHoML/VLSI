import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import re
import unicodedata
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, root, num_sample, max_len=None):
        super().__init__()
        self.max_len = max_len

        # load the text data
        encoder_input, decoder_input, decoder_target = [], [], []
        with open(root, "r") as lines:
            for i, line in enumerate(lines):
                if i >= num_sample:
                    break
                src_line, trg_line, _ = line.strip().split("\t")

                src_line = self._preprocess_sent(src_line)
                trg_line = self._preprocess_sent(trg_line)

                trg_in = "<sos> " + trg_line
                trg_out = trg_line + " <eos>"

                encoder_input.append(src_line.split())
                decoder_input.append(trg_in.split()) # teacher forcing
                decoder_target.append(trg_out.split())

        # vocab
        self.src_vocab = self._build_vocab(encoder_input)
        self.trg_vocab = self._build_vocab(decoder_input + decoder_target)
        self.index_to_src = {v: k for k, v in self.src_vocab.items()}
        self.index_to_trg = {v: k for k, v in self.trg_vocab.items()}

        # encoded text to seq
        encoder_input = self._text_to_seq(encoder_input, self.src_vocab)
        decoder_input = self._text_to_seq(decoder_input, self.trg_vocab)
        decoder_target = self._text_to_seq(decoder_target, self.trg_vocab)

        # padding
        self.encoder_input = self._pad_sequence(encoder_input, self.max_len)
        self.decoder_input = self._pad_sequence(decoder_input, self.max_len)
        self.decoder_target = self._pad_sequence(decoder_target, self.max_len)
    
    def __len__(self):
        return len(self.encoder_input)
    
    def __getitem__(self, idx):
        return {
            "encoder_input": torch.tensor(self.encoder_input[idx], dtype=torch.long),
            "decoder_input": torch.tensor(self.decoder_input[idx], dtype=torch.long),
            "decoder_target": torch.tensor(self.decoder_target[idx], dtype=torch.long)
        }
        
    def _preprocess_sent(self, sent):
        # remove the accent in French
        sent = ''.join(c for c in unicodedata.normalize('NFD', sent.lower()) 
                        if unicodedata.category(c) != 'Mn')
        
        # add space between word and punctuation
        sent = re.sub(r"([?.!,¿])", r" \1", sent)

        # convert characters to space except (a-z, A-Z, ".", "?", "!", ",")
        sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

        # compress the space more than 2
        sent = re.sub(r"\s+", " ", sent)

        return sent

    def _build_vocab(self, text):
        word_list = []

        for sent in text:
            for word in sent:
                word_list.append(word)

        # sort vocab by frequency (high -> low)
        word_count = Counter(word_list)
        vocab = sorted(word_count, key=word_count.get, reverse=True)

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
        if max_len is None:
            max_len = max([len(sent) for sent in text])
        
        features = np.zeros((len(text), max_len), dtype=int)
        for i, sent in enumerate(text):
            if len(sent) != 0:
                features[i, :len(sent)] = np.array(sent)[:max_len]
        return features

def split_dataset(dataset, split_ratio=0.9):
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

if __name__ == "__main__":
    dataset = TextDataset(root='../../datasets/fra_eng.txt', num_sample=33000)
    train_dataset, val_dataset = split_dataset(dataset)
    
    encoder_input, decoder_input, decoder_target = train_dataset[1]["encoder_input"], \
                                                   train_dataset[1]["decoder_input"], \
                                                   train_dataset[1]["decoder_target"]
    
    print(f"(Vocab Size) Src: {len(dataset.src_vocab)}, Trg: {len(dataset.trg_vocab)}")
    print(f"(Dataset Size) Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"(Input Shape) enc_in: {encoder_input.shape}, dec_in: {decoder_input.shape}, dec_out: {decoder_target.shape}")