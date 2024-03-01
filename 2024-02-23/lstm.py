import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, src_vocab_size, embed_size, hidden_size, prob_drop):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Embedding(src_vocab_size, embed_size, padding_idx=0),
            nn.Dropout(prob_drop)
        )
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        
        return output, hidden, cell
    
class DecoderLSTM(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, hidden_size, prob_drop):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Embedding(trg_vocab_size, embed_size, padding_idx=0),
            nn.Dropout(prob_drop)
        )
        self.softmax = nn.Softmax(dim=2)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, trg_vocab_size)
        
    def attention(self, encoder_output, hidden):
        # torch.bmm = batch matrix multiplication
        attn_score = torch.bmm(encoder_output, hidden.transpose(0, 1).transpose(1, 2))
        attn_weight = self.softmax(attn_score)
        attn_value = torch.bmm(attn_weight.transpose(1, 2), encoder_output)

        return attn_value

    def forward(self, x, encoder_output, hidden, cell):
        x = self.embedding(x)
        context = self.attention(encoder_output, hidden)

        seq_len = x.shape[1]
        context = context.repeat(1, seq_len, 1)
        
        x = torch.cat((x, context), dim=2)

        output, (hidden, cell) = self.lstm(x, (hidden, cell))

        output = self.fc(output)

        return output, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, prob_drop=0.1):
        super().__init__()
        self.encoder = EncoderLSTM(src_vocab_size, embed_size, hidden_size, prob_drop)
        self.decoder = DecoderLSTM(trg_vocab_size, embed_size, hidden_size, prob_drop)

    def forward(self, src, trg):
        encoder_output, hidden, cell = self.encoder(src)
        output, _, _ = self.decoder(trg, encoder_output, hidden, cell)

        return output
    
if __name__ == "__main__":
    src_vocab_size, trg_vocab_size = 4488, 7884

    net = Seq2Seq(src_vocab_size, trg_vocab_size, embed_size=256, hidden_size=256)
    src = torch.randint(low=0, high=src_vocab_size, size=(16, 7), dtype=torch.long)
    trg = torch.randint(low=0, high=trg_vocab_size, size=(16, 16), dtype=torch.long)

    random_output = net(src, trg)
    print(random_output.shape)
