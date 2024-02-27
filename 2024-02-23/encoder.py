import torch
import torch.nn as nn

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

'''
Multi-Head Attention Block
    Inputs
        - query, key, value : (batch_size, length, hidden_size)
        - mask : prevent attention to certain positions
    Output
        - scaled_attention_value matrix : (batch_size, length, hidden_size)

'''
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, prob_drop, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.attention_size = attention_size = hidden_size // num_heads

        # multi attention heads
        self.linear_q = nn.Linear(hidden_size, num_heads * attention_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, num_heads * attention_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, num_heads * attention_size, bias=False)

        # initialize weight
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        # output layer
        self.dropout = nn.Dropout(prob_drop)
        self.output = nn.Linear(num_heads * attention_size, hidden_size, bias=False)
        initialize_weight(self.output)

    # split q, k, v by num_heads
    # output shape = (batch_size, num_heads, length, attention_size)
    def split_heads(self, inputs, batch_size):
        inputs = inputs.reshape(batch_size, -1, self.num_heads, self.attention_size)
        return inputs.transpose(1, 2)
    
    # calculate attention value
    # attention = softmax((QK^T) / sqrt(attention_size))V
    # output shape = (batch_size, num_heads, length, attention_size)
    def scaled_dot_product_attention(self, q, k, v, mask):
        attention_score = torch.matmul(q, k.transpose(2, 3))
        attention_score /= (self.attention_size ** -0.5)
        
        # masking: outputs of softmax are 0
        if mask is not None:
            attention_score += (mask * -1e9)

        attention_distribution = torch.softmax(attention_score, dim=3)
        attention_value = self.dropout(attention_distribution)
        attention_value = attention_value.matmul(v)
        
        return attention_value
    
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention_value = self.scaled_dot_product_attention(q, k, v, mask)
        attention_value = attention_value.permute(0, 2, 1, 3).contiguous()
        concat_attention = attention_value.view(batch_size, -1, self.num_heads * self.attention_size)

        output = self.output(concat_attention)
        
        return output

'''
Point-wise Feed Forward Neural Network
    Input
        - output of Attention Block : (batch_size, length, hidden_size)
    Output
        - output of FFNN : (batch_size, length, hidden_size)
'''
class FeedForward(nn.Module):
    def __init__(self, hidden_size, filter_size, prob_drop):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, filter_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(prob_drop)
        self.fc2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.fc1)
        initialize_weight(self.fc2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, prob_drop):
        super().__init__()

        self.self_attention = MultiHeadAttention(hidden_size, prob_drop)
        self.ffnn = FeedForward(hidden_size, filter_size, prob_drop)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(prob_drop)

    def forward(self, x, mask):
        self_attention = self.self_attention(x, x, x, mask)
        self_attention = self.dropout(self_attention)
        x = x + self_attention
        x = self.layer_norm(x)

        ffnn = self.ffnn(x)
        ffnn = self.dropout(ffnn)
        x = x + ffnn
        x = self.layer_norm(x)

        return x

class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, prob_drop, num_layer):
        super().__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, prob_drop)
                    for _ in range(num_layer)]
        self.layers = nn.ModuleList(encoders)

    def forward(self, input, mask):
        context = input
        for layer in self.layers:
            context = layer(context, mask)

        return context

if __name__ == "__main__":
    # (batch_size, length, hidden_size)
    input_tensor = torch.randn(64, 50, 512)
    encoder = Encoder(hidden_size=512, filter_size=2048, prob_drop=0.1, num_layer=6)

    context = encoder(input_tensor, mask=None)
    print(f"Context shape: {context.shape}")