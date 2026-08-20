# nanoGPT on hep-ph

A character-level GPT written from scratch in PyTorch, after watching Andrej
Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) —
then **reimplemented from memory**, not typed along with the video. The
architecture matches; the code is my own (a couple of bugs were debugged with
LLM assistance). Divergences from the reference implementation are deliberate
or instructive, and the commit history is the honest record.

Instead of Shakespeare, it trains on something closer to home: arXiv **hep-ph
abstracts**.

Everything lives in one file. Multi-head causal self-attention, pre-LayerNorm
transformer blocks with residual connections, learned position embeddings, and
an AdamW training loop — no `nn.Transformer`, no attention shortcuts.

## Run it

```bash
# 1. build the corpus (~1.1 MB of recent hep-ph abstracts, stdlib only)
python fetch_arxiv.py --out data/hepph.txt

# 2. train and generate
python gpt.py
```

Hyperparameters are a block at the top of `gpt.py` — edit the numbers, run the
file. Runs on CUDA, Apple Silicon (MPS), or CPU, whichever it finds.

## What comes out — and what scale buys

Two configurations, same code, same data, 5000 steps each. (Samples are shown
as code blocks on purpose: GitHub renders `$...$` as real LaTeX, and the small
model's LaTeX does not survive that.)

**Small** (`n_embd=48`, 2 blocks, `block_size=32` — ~67k parameters):

```text
We by decayns. Hown becom uncaly the QCDNCL emitate, and centroles ship
bonated eynisficiles for the rotak' lecusively lighting vanrop to malough
[...] only $\phrande^$ \sim and frameterpondificing indersum $\Ombreq
T^{D}^{-}^2$
```

It has learned the *shape* of a physics abstract — LaTeX-ish math, plausible
jargon morphology — but it cannot spell `\Omega`, and with a 32-character
window it literally cannot see an opening `$` when it emits the closing one.

**Scaled up** (`n_embd=192`, `nhead=6`, 4 blocks, `block_size=128` — ~1.8M
parameters, sample in `data/generated_text2.txt`):

```text
Than strategies the SNR separation is through origin $S_h^{1/2} \sim m_i/(N)$,
$Z_{cs}$, and $U(1)'$, $Z_X$ is mixpressed. Specificantly, the classes of
electrons are presented in quasi-PDF associated with the CKM selected sectors
remain only a photon locality of interactions studied by a tanalytic scaling
systemption [...] the first likelihood-based terms have discrete hole models,
consistent effective theory
```

Same architecture, ~27x the parameters and 4x the context window, and the
failures move up a level: the LaTeX is now mostly *valid* ($S_h^{1/2}$,
$Z_{cs}$, $U(1)'$ — spelled, subscripted, and closed), the vocabulary is real
(CKM, quasi-PDF, LHCb, Hermitian), and what's left is that the sentences mean
nothing. Spelling and syntax were capacity problems; meaning is not, and no
amount of `n_embd` at this corpus size will buy it. That boundary — which
failures scale fixes and which it doesn't — is the most instructive thing this
repo produces. A real tokenizer (BPE) is the next lever: it turns `\alpha_s`
from eight memorized characters into one unit, and spends the freed capacity
above the character level.

## Files

| File | What it is |
|---|---|
| `gpt.py` | model + training + generation, single file |
| `fetch_arxiv.py` | builds the corpus from the arXiv API; maps Unicode math to a compact ~90-character vocabulary |
| `data/hepph.txt` | the training corpus (rebuild with `fetch_arxiv.py`) |

MIT-licensed; abstracts fetched via the [arXiv API](https://info.arxiv.org/help/api/index.html).
