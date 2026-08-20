#!/usr/bin/env python3
"""
Build a character-level training corpus from arXiv hep-ph abstracts.

Produces a plain text file in the same spirit as tiny-shakespeare: one long
stream of text, abstracts separated by blank lines, with the character
vocabulary deliberately kept small so a char-level GPT can actually learn it.

Standard library only -- no pip install needed. Runs on macOS or lxplus.

    python3 fetch_arxiv.py                        # ~1.1 MB, like tiny-shakespeare
    python3 fetch_arxiv.py --target-chars 3000000 # bigger
    python3 fetch_arxiv.py --category hep-th      # a different corner of arXiv
    python3 fetch_arxiv.py --self-test            # check parsing/cleaning offline

Please leave --delay at 3 seconds or more: that is the interval arXiv asks API
clients to respect, and this script is polite by default.

  Wei Xue, August 2026.
"""

import argparse, html, random, re, sys, time, unicodedata
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET

API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------

# Characters that turn up constantly in physics abstracts and would otherwise
# each become their own rare token in a char-level vocabulary.
REPLACEMENTS = {
    '‘': "'", '’': "'", '“': '"', '”': '"',
    '–': '-', '—': '-', '−': '-', '‐': '-', '‑': '-',
    ' ': ' ', ' ': ' ', ' ': ' ', ' ': ' ', '​': '',
    '…': '...', '×': 'x', '·': '.', '′': "'",
    '→': ' -> ', '←': ' <- ', '⇒': ' => ', '↔': ' <-> ',
    '≤': ' <= ', '≥': ' >= ', '≈': ' ~ ', '≡': ' = ',
    '±': ' +/- ', '≃': ' ~ ', '∼': ' ~ ', '≪': ' << ',
    '≫': ' >> ', '≠': ' != ', '∞': ' infinity ',
    '†': '+', '∗': '*', '⋅': '.', 'º': ' deg ',
    '°': ' deg ', '″': '"', '˜': '~', '­': '',
    '∈': ' in ', '∉': ' notin ', '⊂': ' subset ', '∪': ' union ',
    '∩': ' cap ', '∀': ' forall ', '∃': ' exists ', '∇': ' nabla ',
    '∂': ' partial ', '√': ' sqrt ', '∑': ' sum ', '∏': ' prod ',
    '∫': ' int ', '⟨': '<', '⟩': '>', '〈': '<', '〉': '>',
    '‖': '||', '∼': ' ~ ', '≲': ' <~ ', '≳': ' >~ ',
    '⊗': ' x ', '⊕': ' + ', '∅': ' empty ', '∝': ' propto ',
    'ℏ': ' hbar ', 'ℓ': 'l', '⟶': ' -> ', '％': '%',
}

# Greek letters -> LaTeX names. hep-ph abstracts are full of these, and the
# LaTeX spelling is already present elsewhere in the same corpus, so mapping
# to it makes the text more self-consistent rather than less.
GREEK = {
    'α': r'\alpha ',  'β': r'\beta ',   'γ': r'\gamma ',
    'δ': r'\delta ',  'ε': r'\epsilon ', 'ζ': r'\zeta ',
    'η': r'\eta ',    'θ': r'\theta ',  'ι': r'\iota ',
    'κ': r'\kappa ',  'λ': r'\lambda ', 'μ': r'\mu ',
    'ν': r'\nu ',     'ξ': r'\xi ',     'π': r'\pi ',
    'ρ': r'\rho ',    'σ': r'\sigma ',  'τ': r'\tau ',
    'υ': r'\upsilon ', 'φ': r'\phi ',   'χ': r'\chi ',
    'ψ': r'\psi ',    'ω': r'\omega ',
    'Γ': r'\Gamma ',  'Δ': r'\Delta ',  'Θ': r'\Theta ',
    'Λ': r'\Lambda ', 'Ξ': r'\Xi ',     'Π': r'\Pi ',
    'Σ': r'\Sigma ',  'Φ': r'\Phi ',    'Ψ': r'\Psi ',
    'Ω': r'\Omega ',
}


def clean(text):
    """Normalise one abstract into a compact, low-vocabulary ASCII stream."""
    text = html.unescape(text)
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)
    for k, v in GREEK.items():
        text = text.replace(k, v)

    # strip accents (résumé -> resume) rather than keeping one-off code points
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # anything still outside ASCII becomes a space rather than a one-off token
    text = ''.join(c if ord(c) < 128 else ' ' for c in text)

    # our Greek mapping appends a space; undo it before a sub/superscript so
    # "\alpha _s" reads as "\alpha_s" the way it was written
    text = re.sub(r'\\([a-zA-Z]+) ([_^{/,.;:)\]}])', r'\\\1\2', text)

    # arXiv hard-wraps abstracts; join them back into paragraphs
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def restrict_vocab(text, min_count, verbose=True):
    """Drop characters that appear too rarely to ever be learned."""
    from collections import Counter
    counts = Counter(text)
    rare = {c for c, n in counts.items() if n < min_count}
    if rare and verbose:
        shown = ''.join(sorted(rare))[:120]
        print(f"[vocab] dropping {len(rare)} characters seen < {min_count} times: {shown!r}")
    if rare:
        text = ''.join(' ' if c in rare else c for c in text)
        text = re.sub(r'[ \t]+', ' ', text)
    return text


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def parse_atom(xml_bytes):
    """Return [(title, abstract), ...] from one arXiv API response."""
    root = ET.fromstring(xml_bytes)
    out = []
    for entry in root.findall(ATOM + 'entry'):
        summary = entry.find(ATOM + 'summary')
        title = entry.find(ATOM + 'title')
        if summary is None or summary.text is None:
            continue
        t = title.text if (title is not None and title.text) else ''
        out.append((t, summary.text))
    return out


def fetch_page(category, start, per_page, delay, retries=3):
    q = urllib.parse.urlencode({
        'search_query': f'cat:{category}',
        'start': start,
        'max_results': per_page,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
    })
    url = f"{API}?{q}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'char-gpt-corpus-builder/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return parse_atom(r.read())
        except Exception as e:
            wait = delay * (attempt + 2)
            print(f"[warn] {type(e).__name__}: {e} -- retrying in {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
    return []


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--category',     default='hep-ph')
    p.add_argument('--out',          default='hepph.txt')
    p.add_argument('--target-chars', type=int, default=1_100_000,
                   help='stop once the corpus reaches this size '
                        '(1.1M = tiny-shakespeare)')
    p.add_argument('--per-page',     type=int, default=100, help='<= 200')
    p.add_argument('--max-pages',    type=int, default=60)
    p.add_argument('--delay',        type=float, default=3.0,
                   help="seconds between API calls; arXiv asks for >= 3")
    p.add_argument('--min-chars',    type=int, default=300,
                   help='skip abstracts shorter than this')
    p.add_argument('--min-char-count', type=int, default=25,
                   help='drop characters rarer than this from the vocabulary')
    p.add_argument('--titles',       action='store_true',
                   help='prepend each title as its own line')
    p.add_argument('--no-shuffle',   action='store_true',
                   help='keep reverse-chronological order (default shuffles, so '
                        'the 80/20 train/val split is not a split in time)')
    p.add_argument('--seed',         type=int, default=1337)
    p.add_argument('--self-test',    action='store_true',
                   help='run the parser and cleaner on a built-in sample, no network')
    args = p.parse_args()

    if args.self_test:
        return self_test()

    print(f"[fetch] category {args.category}, target {args.target_chars:,} chars")
    seen, records, total = set(), [], 0

    for page in range(args.max_pages):
        start = page * args.per_page
        batch = fetch_page(args.category, start, args.per_page, args.delay)
        if not batch:
            print(f"[fetch] no results at start={start}; stopping")
            break

        kept = 0
        for title, summary in batch:
            a = clean(summary)
            if len(a) < args.min_chars:
                continue
            key = a[:120]
            if key in seen:
                continue
            seen.add(key)
            records.append((clean(title), a))
            total += len(a) + 2
            kept += 1

        print(f"[fetch] start={start:5d}  got {len(batch):3d}  kept {kept:3d}  "
              f"total {total:,} chars")
        if total >= args.target_chars:
            break
        time.sleep(args.delay)

    if not records:
        print("[error] nothing fetched. Check network access and try again.",
              file=sys.stderr)
        return 1

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(records)

    parts = []
    for title, abstract in records:
        parts.append(f"{title}\n{abstract}" if args.titles else abstract)
    text = "\n\n".join(parts) + "\n"

    text = restrict_vocab(text, args.min_char_count)

    with open(args.out, 'w') as f:
        f.write(text)

    report(text, len(records), args.out)
    return 0


def report(text, n_abstracts, path):
    from collections import Counter
    vocab = sorted(set(text))
    counts = Counter(text)
    print("-" * 66)
    print(f"[done] {path}")
    print(f"  abstracts     : {n_abstracts:,}")
    print(f"  characters    : {len(text):,}")
    print(f"  vocabulary    : {len(vocab)}   (tiny-shakespeare has 65)")
    print(f"  chars/abstract: {len(text)//max(n_abstracts,1):,}")
    print(f"  alphabet      : {''.join(vocab)!r}")
    tail = [f"{c!r}:{n}" for c, n in counts.most_common()[-8:]]
    print(f"  rarest        : {', '.join(tail)}")
    print("-" * 66)
    print("Train on it with:")
    print(f"  python gpt.py --data {path} --preset karpathy --amp --fast-attn")


SAMPLE_ATOM = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Probing the Higgs self-coupling</title>
    <summary>  We study the di-Higgs production cross section at
the LHC, including next-to-leading order corrections of order
&#945;_s^3. The trilinear coupling &#955; is constrained to
&#955;/&#955;_SM &#8712; [-1.5, 6.7] at 95% CL, and we find
&#963; = 32.7 &#177; 2.1 fb &#8212; a 30% improvement.
</summary>
  </entry>
  <entry>
    <title>Short one</title>
    <summary>Too short.</summary>
  </entry>
</feed>'''


def self_test():
    """Exercise the parse + clean path with no network."""
    entries = parse_atom(SAMPLE_ATOM.encode())
    assert len(entries) == 2, entries
    title, summary = entries[0]
    out = clean(summary)
    print("[self-test] cleaned abstract:")
    print("   ", out)
    checks = {
        'no newlines':        '\n' not in out,
        'greek -> latex':     r'\alpha' in out and r'\lambda' in out,
        'plusminus expanded': '+/-' in out,
        'emdash -> hyphen':   '—' not in out,
        'entities decoded':   '&#' not in out and '&amp;' not in out,
        'ascii only':         all(ord(c) < 128 for c in out),
        'no double spaces':   '  ' not in out,
    }
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    body = "\n\n".join(clean(s) for _, s in entries)
    body = restrict_vocab(body, min_count=2, verbose=True)
    print(f"[self-test] vocabulary after restriction: {len(set(body))} chars")
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
