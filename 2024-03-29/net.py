import torch
import torch.nn as nn
 
'''
Input:
    input sequence of text (batch_size, seq_len)
Output:
    binary class(0 or 1) of sentence (batch_size, 2)
'''
class SentenceClassifier(nn.Module):
    def __init__(self, n_vocab, hidden_size, embed_size, n_layers,
                 dropout=0.5, bidirectional=True, model_type='lstm'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.model_type = model_type
        
        self.embedding = nn.Embedding(n_vocab, embed_size, padding_idx=0)

        if model_type == "rnn":
            self.context = nn.RNN(embed_size, hidden_size, num_layers=n_layers, 
                                  bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif model_type == "lstm":
            self.context = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, 
                                   bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif model_type == "gru":
            self.context = nn.GRU(embed_size, hidden_size, num_layers=n_layers, 
                                  bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        if bidirectional:
            self.classifier = nn.Linear(hidden_size * 2, 1)
        else:
            self.classifier = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)

        output, _ = self.context(embedded)
        output = output[:, -1, :] # last time_step of output
        
        output = self.dropout(output)
        logits = self.classifier(output)
        
        return logits
    
if __name__ == "__main__":
    src_vocab_size = 5002
    
    net = SentenceClassifier(src_vocab_size, hidden_size=64, embed_size=128, n_layers=2)
    random_input = torch.randint(low=0, high=src_vocab_size, size=(16, 32), dtype=torch.long)

    random_output = net(random_input)
    print(f'Output Shape: {random_output.shape}')
