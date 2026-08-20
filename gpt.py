"""
gpt.py -- a character-level GPT, built from scratch.

Written while following Andrej Karpathy's "Let's build GPT" lecture, then
reorganized from the original Jupyter notebook into a standalone script.
Trains on plain text (default: arXiv hep-ph abstracts in data/hepph.txt,
fetched with fetch_arxiv.py) and writes a generated sample when done.

Usage:
    python gpt.py

Everything is configured in the hyperparameter block below, in the spirit of
Karpathy's nanoGPT: edit the numbers, run the file.

  Wei Xue, August 2026.
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------
data_path    = 'data/hepph.txt'
out_sample   = 'data/generated_text.txt'

n_embd       = 48       # embedding dimension
nhead        = 4        # attention heads per layer
dhead        = 12       # dimension per head  (nhead * dhead = n_embd)
Nx           = 2        # number of transformer blocks
block_size   = 32       # context length
batch_size   = 64
lr           = 1e-3
weight_decay = 0.01
runNum       = 5000     # training iterations
dropout_rate = 0.2
eps          = 1e-5     # layernorm epsilon
train_frac   = 0.8
sample_len   = 5000     # characters to generate at the end
seed         = 1337

# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')        # Apple Silicon GPU
else:
    device = torch.device('cpu')
print(f'device: {device}')

torch.manual_seed(seed)

# ---------------------------------------------------------------------------
# data: read text, build the character vocabulary, encode
# ---------------------------------------------------------------------------
with open(data_path) as f:
    content = f.read()

chalist = sorted(set(content))
vol_size = len(chalist)
stoi = {st: ii for ii, st in enumerate(chalist)}
itos = {ii: st for ii, st in enumerate(chalist)}


def encode(text):
    return torch.tensor([stoi[ss] for ss in text], dtype=torch.long)


datalen = int(train_frac * len(content))
# encode once and keep the tensors on the device, so the training loop
# does no host-to-device copies
train_data = encode(content[:datalen]).to(device)
val_data = encode(content[datalen:]).to(device)
print(f'data: {len(content):,} chars, vocab {vol_size}, '
      f'train {len(train_data):,} / val {len(val_data):,}')


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class HeadAttention(torch.nn.Module):
    """A single self-attention head, written out explicitly."""

    def __init__(self, n_embd, dhead):
        super().__init__()
        self.qW = torch.nn.Linear(n_embd, dhead, bias=False)
        self.kW = torch.nn.Linear(n_embd, dhead, bias=False)
        self.vW = torch.nn.Linear(n_embd, dhead, bias=False)
        self.dhead = dhead
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, xin):                            # xin: (B, T, C)
        Q = self.qW(xin)                               # (B, T, H)
        K = self.kW(xin)                               # (B, T, H)
        V = self.vW(xin)                               # (B, T, H)

        # scaled dot-product scores; the 1/sqrt(dhead) keeps the variance of
        # the logits O(1) so the softmax does not saturate at initialization
        qk = Q @ K.transpose(-2, -1) * self.dhead ** -0.5      # (B, T, T)

        # causal mask: position t may only attend to positions <= t
        T = xin.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=xin.device).triu(diagonal=1)
        qk = qk.masked_fill(mask, float('-inf'))

        h = F.softmax(qk, dim=-1)
        h = self.dropout(h)
        return h @ V                                   # (B, T, H)


class MultiHeadAttention(torch.nn.Module):
    """nhead independent attention heads, concatenated and projected.
    Pre-LayerNorm and a residual connection around the whole block."""

    def __init__(self, n_embd, dhead, nhead):
        super().__init__()
        self.ln = torch.nn.LayerNorm(n_embd, eps=eps)
        self.heads = torch.nn.ModuleList(
            [HeadAttention(n_embd, dhead) for _ in range(nhead)])
        self.proj = torch.nn.Linear(nhead * dhead, n_embd)

    def forward(self, xin):
        x = self.ln(xin)
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, nhead*dhead)
        out = self.proj(out)
        return out + xin                               # residual


class FeedForward(torch.nn.Module):
    """Position-wise MLP with a 4x hidden expansion, pre-LN, residual."""

    def __init__(self, n_embd):
        super().__init__()
        self.ln = torch.nn.LayerNorm(n_embd, eps=eps)
        self.linear1 = torch.nn.Linear(n_embd, 4 * n_embd)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(4 * n_embd, n_embd)

    def forward(self, xin):
        x = self.ln(xin)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x + xin                                 # residual


class gptModel(torch.nn.Module):
    def __init__(self, vol_size, n_embd, block_size, dhead, nhead, Nx):
        super().__init__()
        self.Cmap = torch.nn.Embedding(vol_size, n_embd)   # token embedding
        self.pMap = torch.nn.Embedding(block_size, n_embd) # learned position embedding
        self.block_size = block_size

        self.block = torch.nn.Sequential(
            *[layer for _ in range(Nx)
              for layer in [MultiHeadAttention(n_embd, dhead, nhead),
                            FeedForward(n_embd)]],
            torch.nn.LayerNorm(n_embd, eps=eps),
            torch.nn.Linear(n_embd, vol_size))         # language-model head

    def forward(self, xin):                            # xin: (B, T)
        B, T = xin.shape
        assert T <= self.block_size

        x1 = self.Cmap(xin)                            # (B, T, C)
        positions = torch.arange(T, dtype=torch.long, device=xin.device)
        x2 = self.pMap(positions)                      # (T, C), broadcast over B
        x = x1 + x2
        for bb in self.block:
            x = bb(x)
        return x                                       # logits (B, T, vol_size)

    @torch.no_grad()
    def eval_loss(self, datai):
        """Average loss over datai, in non-overlapping block_size windows."""
        self.eval()
        da1 = datai.unfold(dimension=0, size=self.block_size + 1, step=self.block_size)
        lossT, num_b = 0.0, 0
        for ii in range(0, len(da1), batch_size):
            batch = da1[ii: ii + batch_size]
            xid = batch[:, :-1]
            yid = batch[:, 1:]
            logits = self.forward(xid)
            loss = F.cross_entropy(logits.reshape(-1, vol_size), yid.reshape(-1))
            lossT += loss.item()
            num_b += 1
        self.train()
        return lossT / num_b

    @torch.no_grad()
    def generate(self, startword, wordNum=10):
        """Sample wordNum tokens autoregressively, conditioning on at most
        the last block_size tokens."""
        self.eval()
        for _ in range(wordNum):
            xid = startword[-self.block_size:].view(1, -1)
            logits = self.forward(xid)
            next_logits = logits[0, -1, :]
            probabilities = F.softmax(next_logits, dim=-1)
            # sample from the distribution rather than taking the argmax
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            startword = torch.cat([startword, next_token_id])
        self.train()
        return startword


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
model = gptModel(vol_size=vol_size, n_embd=n_embd, block_size=block_size,
                 dhead=dhead, nhead=nhead, Nx=Nx).to(device)
print(f'model: {sum(p.numel() for p in model.parameters()):,} parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

positions = torch.arange(block_size, device=device)
lossi = []

for ii in range(runNum):
    # choose a random batch of block_size windows
    lenmax = len(train_data) - block_size
    bstart = torch.randint(lenmax, (batch_size,), device=device)
    index = bstart[:, None] + positions[None, :]       # (B, T)
    xid = train_data[index]                            # (B, T)
    yid = train_data[index + 1]                        # next-character targets

    optimizer.zero_grad(set_to_none=True)
    logits = model(xid)                                # (B, T, vol_size)
    loss = F.cross_entropy(logits.reshape(-1, vol_size), yid.reshape(-1))
    loss.backward()
    optimizer.step()

    lossi.append(loss.item())
    if ii % 500 == 0 or ii == runNum - 1:
        print(f'step {ii}, loss = {loss.item():.4f}')

# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
out1 = model.eval_loss(train_data)
out2 = model.eval_loss(val_data)
print(f'training data, loss = {out1:.4f};  validation data, loss = {out2:.4f}')

# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
startword = torch.zeros(1, dtype=torch.long, device=device)
ids = model.generate(startword, sample_len)
text = ''.join(itos[ii.item()] for ii in ids.cpu())
print(text[:1000])

with open(out_sample, 'w', encoding='utf-8') as f:
    f.write(text)
print(f'Saved to {out_sample}')
