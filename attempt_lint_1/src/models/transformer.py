import torch
import torch.nn as nn
from models.embeddings import Embeddings
from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(nn.Module):
    """
    Assembles the complete Transformer model: embeddings, encoder, decoder, and final linear + softmax.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, N, h, d_ff, dropout_rate, max_seq_len):
        super().__init__()
        self.src_embeddings = Embeddings(src_vocab_size, d_model, max_len, dropout_rate)
        self.tgt_embeddings = Embeddings(tgt_vocab_size, d_model, max_len, dropout_rate)
        
        self.encoder = Encoder(N, d_model, h, d_ff, dropout_rate)
        self.decoder = Decoder(N, d_model, h, d_ff, dropout_rate)
        
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # Share weights between target embeddings and output linear layer
        # as per the paper (Section 3.4)
        self.output_linear.weight = self.tgt_embeddings.token_embedding.weight

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Performs a full forward pass from source to target logits.
        Args:
            src (torch.Tensor): Source input token IDs (batch_size, src_seq_len)
            tgt (torch.Tensor): Target input token IDs (batch_size, tgt_seq_len)
            src_mask (torch.Tensor): Source padding mask (batch_size, 1, 1, src_seq_len)
            tgt_mask (torch.Tensor): Target combined padding and look-ahead mask (batch_size, 1, tgt_seq_len, tgt_seq_len)
        Returns:
            torch.Tensor: Logits for the next token probabilities (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Embed source and target inputs
        src_embedded = self.src_embeddings(src)
        tgt_embedded = self.tgt_embeddings(tgt)
        
        # Encoder forward pass
        enc_output = self.encoder(src_embedded, src_mask)
        
        # Decoder forward pass
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
        
        # Final linear layer to get logits over target vocabulary
        logits = self.output_linear(dec_output)
        
        return logits