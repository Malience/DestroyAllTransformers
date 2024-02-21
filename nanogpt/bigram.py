import torch
import torch.nn as nn
from torch.nn import functional as F

from lang_model import LanguageModel

class SAHead(nn.Module):
    ''' Self-Attention Head '''

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)     # (B,T,C) # Shouldn't these be (B,T,head_size)?
        q = self.query(x)   # (B,T,C) # Shouldn't these be (B,T,head_size)?

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) => (B,T,T) # Shouldn't this be (B,T,head_size) @ (B,T,head_size) => (B,T,T)?
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        v = self.value(x) # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) => (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([SAHead(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

    
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size=block_size, dropout=dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(LanguageModel):
    def __init__(self, vocab_size, block_size, n_embd=32):
        super().__init__()

        n_layer = 6
        n_head = 6
        dropout = 0.2

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f =nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape


        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = tok_emb + pos_embd # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss