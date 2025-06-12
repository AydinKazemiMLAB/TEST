import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Computes Scaled Dot-Product Attention.
    Args:
        query (torch.Tensor): Query tensor (..., seq_len_q, d_k)
        key (torch.Tensor): Key tensor (..., seq_len_k, d_k)
        value (torch.Tensor): Value tensor (..., seq_len_v, d_v)
        mask (torch.Tensor, optional): Mask tensor (..., seq_len_q, seq_len_k).
                                       Positions with True are masked (set to -inf).
        dropout (torch.nn.Dropout, optional): Dropout layer.
    Returns:
        torch.Tensor: Output tensor (..., seq_len_q, d_v)
        torch.Tensor: Attention weights (..., seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Ensure mask is broadcastable to scores
        # For padding mask: (batch_size, 1, 1, seq_len_k)
        # For look-ahead mask: (1, 1, seq_len_q, seq_len_k)
        # Combined mask: (batch_size, 1, seq_len_q, seq_len_k)
        scores = scores.masked_fill(mask == 0, -1e9) # Use -1e9 for numerical stability

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.
    """
    def __init__(self, h, d_model, dropout_rate=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model // h # Dimension of keys/queries per head
        self.d_v = d_model // h # Dimension of values per head
        self.h = h # Number of heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        """
        Performs multi-head attention.
        Args:
            query (torch.Tensor): Query tensor (batch_size, seq_len_q, d_model)
            key (torch.Tensor): Key tensor (batch_size, seq_len_k, d_model)
            value (torch.Tensor): Value tensor (batch_size, seq_len_v, d_model)
            mask (torch.Tensor, optional): Mask tensor (batch_size, 1, seq_len_q, seq_len_k).
                                           Positions with True are masked (set to -inf).
        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len_q, d_model)
            torch.Tensor: Attention weights (batch_size, h, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1) Apply linear projections and split into heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k/d_v) -> (batch_size, h, seq_len, d_k/d_v)
        query = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.h, self.d_v).transpose(1, 2)

        # 2) Apply scaled dot-product attention
        x, attn_weights = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concatenate heads and apply final linear layer
        # (batch_size, h, seq_len_q, d_v) -> (batch_size, seq_len_q, h * d_v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_v)
        
        return self.linear_out(x), attn_weights