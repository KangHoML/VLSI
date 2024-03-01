import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=embed_size)

    def forward(self, input, hidden_state, cell_state):
        x = self.embed(input)
        x = x.view((1, 1, -1)) # (1, 1, embed_size)

        x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))

        return hidden_state, cell_state

class Attention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

    # key = value : All hidden states of Encoder
    # query : (t-1) time-step's hidden state of Decoder
    def forward(self, value, query):
        attn_score = torch.matmul(value, query.view((1, self.embed_size, 1)))
        attn_distribution = F.softmax(attn_score, dim=0)
        attn_value = torch.sum(value * attn_distribution, dim=0).view(1, 1, self.embed_size)
        concat_vector = torch.cat((attn_value, query), dim=2)
        
        return concat_vector

        
        