import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, filter_size=2048, 
                 num_layer=6, prob_drop=0.1, num_heads=8):
        super().__init__()

        # embed the vocab
        self.embed = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.embed, mean=0, std=hidden_size**-0.5)
        
        # 
    raise NotImplemented