import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        """
        :return: (Batch,Time,Channel) corresponds to batch size, block size, vocab size
        """
        return self.token_embedding_table(idx)
