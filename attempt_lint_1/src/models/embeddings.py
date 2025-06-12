import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    """
    Generates token embeddings and adds positional encodings.
    """
    def __init__(self, vocab_size, d_model, max_len=5000, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # Positional Encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of token IDs (batch_size, seq_len)
        Returns:
            torch.Tensor: Embedded tensor with positional encoding (batch_size, seq_len, d_model)
        """
        # Token embedding scaled by sqrt(d_model) as per paper
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        # Ensure positional encoding is on the same device as x
        x = x + self.pe[:, :x.size(1)].to(x.device)
        
        return self.dropout(x)