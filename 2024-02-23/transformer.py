import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# input : (seq_len, emb_size)
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attn_size = hidden_size // num_heads

        self.query_embed= nn.Linear(hidden_size, hidden_size)
        self.key_embed= nn.Linear(hidden_size, hidden_size)
        self.value_embed= nn.Linear(hidden_size, hidden_size)
        self.output_embed= nn.Linear(hidden_size, hidden_size)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        d_k = key.size()[-1]
        key_T = torch.transpose(key, 3, 2)

        attn_score = torch.matmul(query, key_T) / math.sqrt(d_k)
        
        if mask is not None:
            attn_score = attn_score.masked_fill(mask==0, float("-1e20"))

        attn_weight = F.softmax(attn_score, -1)
        attn_value = torch.matmul(attn_weight, value)

        return attn_value
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size()[0]

        # (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, seq_len, attn_size)
        query = self.query_embed(query).view(batch_size, -1, self.num_heads, self.attn_size).transpose(1, 2)
        key = self.key_embed(key).view(batch_size, -1, self.num_heads, self.attn_size).transpose(1, 2)
        value = self.value_embed(value).view(batch_size, -1, self.num_heads, self.attn_size).transpose(1, 2)

        attn_value = self.scaled_dot_product_attention(query, key, value, mask)
        attn_value = torch.transpose(attn_value, 1, 2).contiguous().view((batch_size, -1, self.hidden_size))

        return attn_value

class FeedForword(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)

        return x

'''
add th
'''
def positional_encoding(hidden_size, max_seq_len, device):
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(max_seq_len, hidden_size)
        pe.requires_grad = False
        div_term = torch.pow(torch.ones(hidden_size//2).fill_(10000),
                             torch.arange(0, hidden_size, 2) / torch.tensor(hidden_size, dtype=torch.float32))
        
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)

        return pe.to(device)

'''
Input : embedded & positional encoded input sequence of encoder or (t-1) EncoderLayer's output (batch_size, seq_len, hidden_size)
Output : Output of present EncoderLayer's output (batch_size, seq_len, hidden_size)
'''
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.attention = MultiHeadAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffnn = FeedForword(hidden_size)

    def forward(self, src, mask):
        attn_value = self.attention(src, src, src, mask)
        res_out = src + self.norm(attn_value)
        
        ffnn_output = self.ffnn(res_out)
        output = res_out + self.norm(ffnn_output)

        return output

'''
Input(encoder_input) : input sequence of encoder & encoding with source_vocab (batch_size, seq_len)
Output(encoder_output) : context_vector of input sequence (batch_size, seq_len, hidden_size)
'''
class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, src_vocab_size, max_seq_len, device):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, hidden_size)
        self.positional_encoding = positional_encoding(hidden_size, max_seq_len, device)
        self.encoder = nn.ModuleList([EncoderLayer(hidden_size) for _ in range(num_layers)])

        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, src, mask):
        src_len = src.size(1) # sequence length of input sequence
        src_embed = self.src_embed(src) # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        src_embed += self.positional_encoding[:, :src_len, :]
        encoder_output = src_embed
        for encoder in self.encoder:
            encoder_output = encoder(encoder_output, mask)
        
        return encoder_output

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffnn = FeedForword(hidden_size)
    
    def forward(self, trg, encoder_output, pad_mask, look_ahead_mask):
        look_ahead_attn = self.attention(trg, trg, trg, look_ahead_mask)
        res_out = trg + self.norm(look_ahead_attn)
        
        attn = self.attention(res_out, encoder_output, encoder_output, pad_mask)
        res_out = res_out + self.norm(attn)

        ffnn_out = self.ffnn(res_out)
        output = res_out + self.norm(ffnn_out)

        return output
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, trg_vocab_size, max_seq_len, device):
        super().__init__()
        self.decoder = nn.ModuleList([DecoderLayer(hidden_size) for _ in range(num_layers)])
        self.positional_encoding = positional_encoding(hidden_size, max_seq_len, device)
        self.trg_embed = nn.Embedding(trg_vocab_size, hidden_size)

        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, trg, encoder_output, pad_mask, look_ahead_mask):
        trg_len = trg.size(1)
        trg_embed = self.trg_embed(trg)
        trg_embed += self.positional_encoding[:, :trg_len, :]
        decoder_output = trg_embed
        for decoder in self.decoder:
            decoder_output = decoder(decoder_output, encoder_output, pad_mask, look_ahead_mask)

        return decoder_output
    
class Transformer(nn.Module):
    def __init__(self, hidden_size, num_layers, src_vocab_size, trg_vocab_size,
                 src_pad_idx, trg_pad_idx, max_seq_len, device):
        super().__init__()

        self.hidden_size = hidden_size
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.max_seq_len = max_seq_len
        self.device = device

        self.encoder = Encoder(hidden_size, num_layers, src_vocab_size, max_seq_len, device)
        self.decoder= Decoder(hidden_size, num_layers, trg_vocab_size, max_seq_len, device)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_size, trg_vocab_size),
            nn.LogSoftmax(dim=-1)
        )
    
    def create_pad_mask(self, query, key):
        len_query, len_key = query.size(1), key.size(1)

        key = key.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        key = key.repeat(1, 1, len_query, 1)

        query = query.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        query = query.repeat(1, 1, 1, len_key)

        mask = key & query
        return mask
    
    def create_look_ahead_mask(self, trg):
        n, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(n, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.create_pad_mask(src, src) # mask the <pad> token in encoder_input sequence
        encoder_output = self.encoder(src, src_mask)

        trg_pad_mask = self.create_pad_mask(trg, src)
        trg_look_ahead_mask = self.create_look_ahead_mask(trg)

        decoder_output = self.decoder(trg, encoder_output, trg_pad_mask, trg_look_ahead_mask)

        output = self.output(decoder_output)

        return output

if __name__ == "__main__":
    device = torch.device("cuda")
    src_vocab_size, trg_vocab_size = 4488, 7884

    net = Transformer(512, 6, src_vocab_size, trg_vocab_size,
                      src_pad_idx=1, trg_pad_idx=1, max_seq_len=16, device=device).to(device)
    
    src = torch.randint(low=0, high=src_vocab_size, size=(16, 16), dtype=torch.long).to(device)
    trg = torch.randint(low=0, high=trg_vocab_size, size=(16, 16), dtype=torch.long).to(device)

    output = net(src, trg)
    print(output.shape)