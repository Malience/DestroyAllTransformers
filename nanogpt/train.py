import torch

from tokenizer import TokenizerType, Tokenizer
from bigram import BigramLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
torch.manual_seed(1337)

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embd = 384
# -----------

# input
text_file = 'input/TinyShakespeare'

with open(text_file + '.txt', 'r', encoding='utf-8') as f:
    text = f.read()


tokenizer = Tokenizer(TokenizerType.CHAR)

tokenized_text = tokenizer.encode(text)
print(tokenizer.vocab_size)

# dataset
data = torch.tensor(tokenized_text, dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split='train'):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
# ---------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch()

model = BigramLanguageModel(tokenizer.vocab_size, block_size, n_embd)
out, loss = model(xb, yb)
print(out.shape)
print(loss)


# training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

print(tokenizer.decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=500, block_size=block_size)[0].tolist()))

