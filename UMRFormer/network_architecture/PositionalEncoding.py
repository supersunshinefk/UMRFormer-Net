import torch
import torch.nn as nn

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length, bool_x=True):
        super(LearnedPositionalEncoding, self).__init__()

        # self.position_embeddings = nn.Parameter(torch.zeros(1, 4096, 512)) #8x
        if bool_x is True:
            self.position_embeddings = nn.Parameter(torch.zeros(1, 256, embedding_dim)) #8x
        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, 2048, embedding_dim))

    def forward(self, x, position_ids=None):
        B,H_W_L,C = x.shape
        self.position_embeddings = nn.Parameter(torch.zeros(1, H_W_L, C).cuda())
        position_embeddings = self.position_embeddings
        return x + position_embeddings


