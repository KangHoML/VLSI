import os
import torch
import pandas as pd
import urllib.request

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def load_data(path):
    url = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path)

class ChatDataset(Dataset):
    def __init__(self, root, tokenizer):
        super().__init__()
        self.data = load_data(root)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        q = self.data.Q.iloc[idx]
        a = self.data.A.iloc[idx]

        bos_token = self.tokenizer.bos_token_id
        eos_token = self.tokenizer.eos_token_id

        sent = self.tokenizer.encode('' + q + '' + a, add_special_tokens=False)
        sent_tensor = torch.tensor([bos_token] + sent + [eos_token], dtype=torch.long)
        
        return sent_tensor


if __name__ == "__main__":
    data_path = "/home/ho/workspace/dataset/ChatBotData.csv"
    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='', eos_token='', pad_token='')
    
    example_dataset = ChatDataset(data_path, tokenizer)
    print(example_dataset[10].shape)