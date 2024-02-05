import torch
import torch.nn as nn
from torch.nn import functional as F

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx