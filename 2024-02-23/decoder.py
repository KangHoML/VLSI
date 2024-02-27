import torch
import torch.nn as nn

from encoder import MultiHeadAttention, FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, prob_drop, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_size, prob_drop, num_heads)
        self.ffnn = FeedForward(hidden_size, filter_size, prob_drop)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(prob_drop)

    def forward(self, x, encoded_output, look_ahead_mask, pad_mask):
        self_attention = self.attention(x, x, x, look_ahead_mask)
        self_attention = self.dropout(self_attention)
        x = x + self_attention
        x = self.layer_norm(x)

        attention = self.attention(x, encoded_output, encoded_output, pad_mask)
        attention = self.dropout(attention)
        x = x + attention
        x = self.layer_norm(x)

        ffnn = self.ffnn(x)
        ffnn = self.dropout(ffnn)
        x = x + ffnn
        x = self.layer_norm(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, prob_drop, num_layer, num_heads):
        super().__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, prob_drop, num_heads)
                    for _ in range(num_layer)]
        self.layers = nn.ModuleList(decoders)

    def forward(self, target, encoded_output, look_ahead_mask, pad_mask):
        output = target
        for layer in self.layers:
            output = layer(output, encoded_output, look_ahead_mask, pad_mask)
        
        return output

if __name__ == "__main__":
    # (batch_size, length, hidden_size)
    target_tensor = torch.randn(64, 50, 512)
    encoded_tensor = torch.randn(64, 50, 512)
    look_ahead_mask = torch.triu(torch.ones((50, 50)) * float('-inf'), diagonal=1)

    decoder = Decoder(hidden_size=512, filter_size=2048, prob_drop=0.1, num_layer=6, num_heads=8)

    context = decoder(target_tensor, encoded_tensor, look_ahead_mask, pad_mask=None)
    print(f"Context shape: {context.shape}")
