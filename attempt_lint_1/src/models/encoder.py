import torch.nn as nn
from models.transformer_layers import EncoderLayer

class Encoder(nn.Module):
    """
    Stacks N identical EncoderLayer's.
    """
    def __init__(self, num_layers, d_model, h, d_ff, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout_rate) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        """
        Processes source input through the encoder stack.
        Args:
            src (torch.Tensor): Embedded source input (batch_size, src_seq_len, d_model)
            src_mask (torch.Tensor): Source padding mask (batch_size, 1, 1, src_seq_len)
        Returns:
            torch.Tensor: Output of the encoder stack (batch_size, src_seq_len, d_model)
        """
        x = src
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)