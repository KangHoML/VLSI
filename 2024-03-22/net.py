import torch
import torch.nn as nn
 
'''
Input:
    input sequence of text (batch_size, seq_len)
Output:
    binary class(0 or 1) of sentence (batch_size, 2)
'''
class SentenceClassifier(nn.Module):
    def __init__(self, n_vocab, hidden_size, embed_size, n_layers=1,
                 dropout=0, bidirectional=False, model_type='lstm'):
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
            self.classifier = nn.Linear(hidden_size * 2, 2)
        else:
            self.classifier = nn.Linear(hidden_size, 2)
        
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def forward(self, x):
        embedded = self.embedding(x)
        h_0 = self._init_state(batch_size=embedded.size(0))

        output, _ = self.context(embedded, h_0)
        output = output[:, -1, :] # last time_step of output
        
        output = self.dropout(output)
        logits = self.classifier(output)
        
        return logits
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        num_directions = 2 if self.bidirectional else 1

        h_0 = weight.new(self.n_layers * num_directions, batch_size, self.hidden_size).zero_()
        c_0 = weight.new(self.n_layers * num_directions, batch_size, self.hidden_size).zero_()

        if self.model_type == 'lstm':
            return (h_0, c_0)
        else:
            return h_0
    
    # init weights    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

if __name__ == "__main__":
    src_vocab_size = 40710
    
    net = SentenceClassifier(src_vocab_size, hidden_size=64, embed_size=128, n_layers=2, model_type='rnn')
    random_input = torch.randint(low=0, high=src_vocab_size, size=(64, 500), dtype=torch.long)

    random_output = net(random_input)
    print(f'Output Shape: {random_output.shape}')
