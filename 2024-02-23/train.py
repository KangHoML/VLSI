import torch
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam

from data import ChatDataset
from net import Transformer

def create_collate_fn(tokenizer):
    def collate_fn(batch):
        return pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    return collate_fn

def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    # Load Dataset
    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='', eos_token='', pad_token='')
    dataset = ChatDataset(args.path, tokenizer)
    data_lodaer = DataLoader(dataset, batch_size=args.batch_size, 
                             collate_fn=create_collate_fn(tokenizer))
    
    # 
    
    

