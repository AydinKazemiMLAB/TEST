import torch.nn as nn
from models.transformer_layers import DecoderLayer

class Decoder(nn.Module):
    """
    Stacks N identical DecoderLayer's, handling masked attention and encoder-decoder attention.
    """
    def __init__(self, num_layers, d_model, h, d_ff, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout_rate) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        """
        Processes target input and encoder output through the decoder stack.
        Args:
            tgt (torch.Tensor): Embedded target input (batch_size, tgt_seq_len, d_model)
            enc_output (torch.Tensor): Output from the encoder stack (batch_size, src_seq_len, d_model)
            src_mask (torch.Tensor): Source padding mask (batch_size, 1, 1, src_seq_len)
            tgt_mask (torch.Tensor): Target combined padding and look-ahead mask (batch_size, 1, tgt_seq_len, tgt_seq_len)
        Returns:
            torch.Tensor: Output of the decoder stack (batch_size, tgt_seq_len, d_model)
        """
        x = tgt
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)