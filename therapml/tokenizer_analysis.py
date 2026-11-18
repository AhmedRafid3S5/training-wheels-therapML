"""
BPE Tokenizer Scaling Analysis
================================
Trains BPETokenizers across vocab sizes V ∈ {256, 1000, 5000, 10000, 32000}
and measures, per vocab size:
  - Average tokens per sentence
  - Compression ratio (bytes per token)
  - Rare-word coverage (how many words the tokenizer can represent without fallback)

Plots compression ratio vs. vocab size.
"""

import sys
import time
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from therapml.tokenizer import BasicTokenizer

# ── Corpus ────────────────────────────────────────────────────────────────────
CORPUS_PATH = ROOT / "tests" / "fixtures" / "tinyshakespeare.txt"

TRAIN_CHARS   = 100_000     # characters used for BPE training
EVAL_CHARS    = 20_000     # held-out slice for metrics (must NOT overlap train)
EVAL_OFFSET   = 100_000    # start of held-out slice

VOCAB_SIZES   = [256, 1000, 5000, 10_000, 32_000]

# A handful of rare / archaic Shakespearean words to test coverage
RARE_WORDS = [
    "honorificabilitudinitatibus",   # Shakespeare's longest word
    "sesquipedalian",
    "defenestration",
    "incomprehensibilities",
    "supercalifragilisticexpialidocious",
    "antidisestablishmentarianism",
    "floccinaucinihilipilification",
    "pneumonoultramicroscopicsilicovolcanoconiosis",
    "quizzaciously",
    "unimaginatively",
]


def load_corpus():
    text = CORPUS_PATH.read_text(encoding="utf-8")
    return text[:TRAIN_CHARS], text[EVAL_OFFSET: EVAL_OFFSET + EVAL_CHARS]


def split_sentences(text: str) -> list[str]:
    """
    Rough sentence splitter: split on '.', '!', '?' followed by whitespace.
    Filters out empty strings.
    """
    import re
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def measure_rare_word_coverage(tokenizer: BasicTokenizer, words: list[str]) -> dict:
    """
    For each rare word, encode it and measure:
      - # tokens produced  (smaller = better merging)
      - whether it encodes/decodes losslessly
    Returns a summary dict.
    """
    results = {}
    for word in words:
        ids = tokenizer.encode(word)
        decoded = tokenizer.decode(ids)
        results[word] = {
            "tokens": len(ids),
            "lossless": decoded == word,
        }
    return results


def analyze_vocab(vocab_size: int, train_text: str, eval_text: str, eval_sentences: list[str]):
    print(f"\n  → Training V={vocab_size:>6,} ...", end="", flush=True)
    t0 = time.perf_counter()
    tok = BasicTokenizer()
    tok.train(train_text, vocab_size)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s", flush=True)

    # 1. Encode the entire eval slice to get overall stats
    ids = tok.encode(eval_text)
    n_bytes = len(eval_text.encode("utf-8"))
    n_tokens = len(ids)
    compression_ratio = n_bytes / n_tokens      # bytes per token (higher = better compression)

    # 2. Per-sentence average tokens
    per_sentence_tokens = [len(tok.encode(s)) for s in eval_sentences]
    avg_tokens_per_sentence = sum(per_sentence_tokens) / len(per_sentence_tokens) if per_sentence_tokens else 0

    # 3. Rare-word coverage
    rare = measure_rare_word_coverage(tok, RARE_WORDS)
    avg_rare_tokens = sum(r["tokens"] for r in rare.values()) / len(RARE_WORDS)
    lossless_pct = 100.0 * sum(r["lossless"] for r in rare.values()) / len(RARE_WORDS)

    return {
        "vocab_size": vocab_size,
        "train_time_s": elapsed,
        "compression_ratio": compression_ratio,
        "avg_tokens_per_sentence": avg_tokens_per_sentence,
        "avg_rare_word_tokens": avg_rare_tokens,
        "rare_lossless_pct": lossless_pct,
    }


def print_table(results: list[dict]):
    header = (
        f"\n{'Vocab':>8}  |  {'Train(s)':>8}  |  {'Bytes/Tok':>10}  |  "
        f"{'AvgTok/Sent':>12}  |  {'RareWordTok(avg)':>16}  |  {'Lossless%':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['vocab_size']:>8,}  |  {r['train_time_s']:>8.1f}  |  "
            f"{r['compression_ratio']:>10.3f}  |  {r['avg_tokens_per_sentence']:>12.1f}  |  "
            f"{r['avg_rare_word_tokens']:>16.2f}  |  {r['rare_lossless_pct']:>9.1f}%"
        )
    print(sep)


def plot_results(results: list[dict], save_path: str = "tokenizer_analysis.png"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    vocab_sizes     = [r["vocab_size"]             for r in results]
    comp_ratios     = [r["compression_ratio"]       for r in results]
    avg_sent_tokens = [r["avg_tokens_per_sentence"] for r in results]
    rare_tokens     = [r["avg_rare_word_tokens"]    for r in results]

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Plot 1: Compression Ratio vs Vocab Size ───────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(vocab_sizes, comp_ratios, marker='o', linewidth=2.5, color='#1f77b4')
    ax1.fill_between(vocab_sizes, comp_ratios, alpha=0.15, color='#1f77b4')
    for x, y in zip(vocab_sizes, comp_ratios):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha='center', fontsize=9, fontweight='bold')
    ax1.set_title("Compression Ratio vs. Vocab Size", fontweight='bold')
    ax1.set_xlabel("Vocabulary Size")
    ax1.set_ylabel("Bytes per Token  (↑ = better compression)")
    ax1.set_xscale("log")
    ax1.set_xticks(vocab_sizes)
    ax1.set_xticklabels([str(v) for v in vocab_sizes], rotation=20, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.55)

    # ── Plot 2: Avg Tokens per Sentence ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(range(len(vocab_sizes)), avg_sent_tokens,
                   color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.6)
    ax2.set_xticks(range(len(vocab_sizes)))
    ax2.set_xticklabels([str(v) for v in vocab_sizes], rotation=20, ha='right')
    ax2.bar_label(bars, labels=[f"{v:.1f}" for v in avg_sent_tokens],
                  padding=3, fontsize=9, fontweight='bold')
    ax2.set_title("Avg Tokens per Sentence vs. Vocab Size", fontweight='bold')
    ax2.set_xlabel("Vocabulary Size")
    ax2.set_ylabel("Average Tokens per Sentence  (↓ = shorter)")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.55)

    # ── Plot 3: Rare-Word Token Cost ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(vocab_sizes, rare_tokens, marker='s', linewidth=2.5, color='#2ca02c', linestyle='--')
    ax3.fill_between(vocab_sizes, rare_tokens, alpha=0.15, color='#2ca02c')
    for x, y in zip(vocab_sizes, rare_tokens):
        ax3.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha='center', fontsize=9, fontweight='bold')
    ax3.set_title("Rare Word Cost vs. Vocab Size", fontweight='bold')
    ax3.set_xlabel("Vocabulary Size")
    ax3.set_ylabel("Avg Tokens per Rare Word  (↓ = better merging)")
    ax3.set_xscale("log")
    ax3.set_xticks(vocab_sizes)
    ax3.set_xticklabels([str(v) for v in vocab_sizes], rotation=20, ha='right')
    ax3.grid(True, linestyle='--', alpha=0.55)

    fig.suptitle(
        "BPE Tokenizer Scaling Analysis  —  TinyShakespeare Corpus",
        fontsize=14, fontweight='bold', y=1.01
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved → {save_path}")
    plt.show()


def main():
    print("Loading corpus…")
    train_text, eval_text = load_corpus()
    eval_sentences = split_sentences(eval_text)
    print(f"  Train slice : {len(train_text):,} chars")
    print(f"  Eval slice  : {len(eval_text):,} chars  |  {len(eval_sentences):,} sentences")
    print(f"  Vocab sizes : {VOCAB_SIZES}")

    results = []
    for vs in VOCAB_SIZES:
        result = analyze_vocab(vs, train_text, eval_text, eval_sentences)
        results.append(result)

    print("\n\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print_table(results)

    # Rare-word breakdown for the largest vocab (most interesting)
    print("\n\nRare-word token breakdown at V=32,000:")
    tok_32k = BasicTokenizer()
    tok_32k.train(train_text, 32_000)
    rare_detail = measure_rare_word_coverage(tok_32k, RARE_WORDS)
    max_w = max(len(w) for w in RARE_WORDS)
    for word, info in rare_detail.items():
        status = "✓" if info["lossless"] else "✗ (lossy!)"
        print(f"  {word:<{max_w}}  →  {info['tokens']:>3} tokens  {status}")

    plot_results(results)


if __name__ == "__main__":
    main()
