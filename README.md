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

## What comes out

After 5000 steps of the small default config (~67k parameters), the model
writes text like this:

```text
We by decayns. Hown becom uncaly the QCDNCL emitate, and centroles ship
bonated eynisficiles for the rotak' lecusively lighting vanrop to malough
[...] only $\phrande^$ \sim and frameterpondificing indersum $\Ombreq
T^{D}^{-}^2$
```

(Shown as a code block on purpose: GitHub renders `$...$` as real LaTeX, and
this model's LaTeX is not yet valid enough to survive that.)

It has learned the *shape* of a physics abstract — LaTeX-ish math, "We show
that...", plausible jargon morphology — but not yet how to spell `\Omega` or
close a bracket. Both are expected at this scale: a character-level model has
to memorize every command letter-by-letter, and the default context window
(32 characters) is too short to see an opening `$` when it emits the closing
one. Scaling up `n_embd`, `Nx`, and `block_size` fixes both; so, eventually,
does a real tokenizer.

## Files

| File | What it is |
|---|---|
| `gpt.py` | model + training + generation, single file |
| `fetch_arxiv.py` | builds the corpus from the arXiv API; maps Unicode math to a compact ~90-character vocabulary |
| `data/hepph.txt` | the training corpus (rebuild with `fetch_arxiv.py`) |

MIT-licensed; abstracts fetched via the [arXiv API](https://info.arxiv.org/help/api/index.html).
