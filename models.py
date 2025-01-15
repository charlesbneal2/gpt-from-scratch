import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Head(nn.Module):
    def __init__(self, head_size, block_size, n_embed):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(-2, -1) * C ** -0.5).masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ self.value(x)


class BigramModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed=32):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embed, block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        """
        :return: logits, loss
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        logits = self.lm_head(self.sa_head(tok_emb + pos_emb))

        b, t, c = logits.shape
        return logits, None if targets is None else F.cross_entropy(logits.view(b * t, c), targets.view(b * t))

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
