import torch.nn as nn
from models.attention import MultiHeadAttention
from models.feed_forward import PositionwiseFeedForward

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.
    Note for simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.
    Consists of Multi-Head Self-Attention and Position-wise Feed-Forward.
    """
    def __init__(self, d_model, h, d_ff, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout_rate) for _ in range(2)
        ])
        self.d_model = d_model

    def forward(self, x, src_mask):
        """
        Args:
            x (torch.Tensor): Input tensor from previous layer (batch_size, seq_len, d_model)
            src_mask (torch.Tensor): Source padding mask (batch_size, 1, 1, seq_len)
        Returns:
            torch.Tensor: Output tensor of this encoder layer (batch_size, seq_len, d_model)
        """
        # Self-attention sub-layer
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x, src_mask)[0])
        
        # Feed-forward sub-layer
        x = self.sublayer_connections[1](x, self.feed_forward)
        return x


class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer Decoder.
    Consists of Masked Multi-Head Self-Attention, Encoder-Decoder Attention,
    and Position-wise Feed-Forward.
    """
    def __init__(self, d_model, h, d_ff, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout_rate)
        self.encoder_decoder_attn = MultiHeadAttention(h, d_model, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout_rate) for _ in range(3)
        ])
        self.d_model = d_model

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Args:
            x (torch.Tensor): Input tensor from previous decoder layer (batch_size, tgt_seq_len, d_model)
            enc_output (torch.Tensor): Output from the encoder stack (batch_size, src_seq_len, d_model)
            src_mask (torch.Tensor): Source padding mask (batch_size, 1, 1, src_seq_len)
            tgt_mask (torch.Tensor): Target combined padding and look-ahead mask (batch_size, 1, tgt_seq_len, tgt_seq_len)
        Returns:
            torch.Tensor: Output tensor of this decoder layer (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention sub-layer
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        
        # Encoder-decoder attention sub-layer
        # Query from decoder, Key/Value from encoder output
        x = self.sublayer_connections[1](x, lambda x: self.encoder_decoder_attn(x, enc_output, enc_output, src_mask)[0])
        
        # Feed-forward sub-layer
        x = self.sublayer_connections[2](x, self.feed_forward)
        return x