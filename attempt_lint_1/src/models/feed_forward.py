import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise fully connected feed-forward network.
    """
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, d_model)
        """
        return self.w_2(self.dropout(self.relu(self.w_1(x))))