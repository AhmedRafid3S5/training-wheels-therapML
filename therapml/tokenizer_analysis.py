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

# ── Special Tokens & Edge Cases Suite ─────────────────────────────────────────
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

EDGE_CASES = {
    "Accents": "café naïve résumé Ångström",
    "Combining Characters": "e\u0301 a\u0308 n\u0303",
    "Cyrillic Script": "Съешь же ещё этих мягких французских булок",
    "Greek Script": "Τάχιστη αλώπηξ βαφής ψημένη γη",
    "CJK Ideographs": "日本語の文章と中文句子와 한국어",
    "Arabic (RTL)": "النص العربي يكتب من اليمين",
    "Hindi (Devanagari)": "नमस्ते दुनिया",
    "Emoji & Modifiers": "😀😃🚀🔥 👍🏽 👋🏿",
    "Emoji ZWJ & Flags": "👩‍💻 👨‍👩‍👧‍👦 🇧🇩 🇺🇸",
    "Whitespace Strictness": "a  b   c\t\td\n\ne   padded   ",
    "Mixed Language & Emoji": "Hello世界! Bonjour мир 🌍 مرحبا",
    "Special Lookalikes": "<BOS <EOS> > < PAD> <UNK",
    "Control / Math Symbols": "null\x00bell\x07esc\x1b  ∀x∈ℝ: x² ≥ 0 ∑∫√∞",
}

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
    # Pass special tokens for any vocab size that can accommodate them (> 256 + 4)
    specials = SPECIAL_TOKENS if vocab_size > 256 + len(SPECIAL_TOKENS) else []
    tok.train(train_text, vocab_size, special_tokens=specials)
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

    metrics = {
        "vocab_size": vocab_size,
        "train_time_s": elapsed,
        "compression_ratio": compression_ratio,
        "avg_tokens_per_sentence": avg_tokens_per_sentence,
        "avg_rare_word_tokens": avg_rare_tokens,
        "rare_lossless_pct": lossless_pct,
    }
    return metrics, tok


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


def show_first_20_merged_tokens(tokenizers: dict[int, BasicTokenizer], n_tokens: int = 20):
    """
    Shows the first `n_tokens` learned merge tokens (starting at ID 256)
    for each trained vocabulary size.
    """
    print("\n\n" + "=" * 80)
    print(f"  FIRST {n_tokens} LEARNED MERGE TOKENS (IDs 256 .. {256 + n_tokens - 1})")
    print("=" * 80)

    for vocab_size, tok in tokenizers.items():
        print(f"\n--- Vocab Size: {vocab_size:,} ---")
        merged_ids = [tid for tid in range(256, 256 + n_tokens) if tid in tok.vocab]

        if not merged_ids:
            print("  (No merges learned — pure byte tokens 0..255)")
            continue

        for tid in merged_ids:
            raw_bytes = tok.vocab[tid]
            try:
                display_str = repr(raw_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                display_str = "<non-utf8 bytes>"

            print(f"  ID {tid:>3}:  {raw_bytes!r:<20}  ->  {display_str}")


def analyze_special_tokens_and_edge_cases(tok: BasicTokenizer):
    """
    Evaluates and verifies how the tokenizer handles:
      1. Special tokens (<PAD>, <BOS>, <EOS>, <UNK>) - atomic matching & isolated ID assignments.
      2. Unicode characters, multi-script typography & combining accents.
      3. Emojis, skin-tone modifiers, zero-width-joiner (ZWJ) sequences & flags.
      4. Whitespace strictness - ensuring whitespace is never stripped or normalized.
      5. Fault tolerance - ensuring partial/malformed UTF-8 decodes with U+FFFD without crashing.
      6. Guarantee that <UNK> is never emitted for arbitrary regular text.
    """
    print("\n\n" + "=" * 80)
    print("  SPECIAL TOKENS & EDGE CASES BATTERY")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Special Tokens Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 1. Special Tokens Verification ---")
    unk_id = tok.special_tokens.get("<UNK>")

    for s in SPECIAL_TOKENS:
        # Every special token must be assigned a unique ID in the vocab
        s_id = tok.special_tokens.get(s)
        encoded = tok.encode(s)
        is_atomic = (len(encoded) == 1 and encoded[0] == s_id)
        decoded = tok.decode(encoded)
        status = "✓ PASS" if (is_atomic and decoded == s) else "✗ FAIL"
        print(f"  {s:<8} → Token ID: {s_id!s:<6} | Encoded: {encoded!s:<8} | Roundtrip: {status}")

    # Test special tokens embedded in text
    sample_text = "<BOS>To be, or not to be.<EOS><PAD><PAD>"
    embedded_ids = tok.encode(sample_text)
    embedded_roundtrip = tok.decode(embedded_ids)
    print(f"\n  Embedded special tokens:")
    print(f"    Input     : {sample_text!r}")
    print(f"    Token IDs : {embedded_ids}")
    print(f"    Match     : {'✓ PASS' if embedded_roundtrip == sample_text else '✗ FAIL'}")

    # Special lookalikes should NOT be treated as special tokens
    lookalike = "<BOS <EOS > PAD>"
    lookalike_ids = tok.encode(lookalike)
    has_specials = any(tid in tok.special_tokens.values() for tid in lookalike_ids)
    print(f"\n  Special lookalikes (e.g. {lookalike!r}):")
    print(f"    Treated as regular text: {'✓ PASS' if not has_specials else '✗ FAIL'}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Edge Cases (Unicode, Emojis, Scripts, Whitespace)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 2. Edge Cases Roundtrip & Invariants ---")
    header = f"  {'Category':<24} | {'Tokens':>6} | {'No <UNK>':>8} | {'Lossless':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cat, text in EDGE_CASES.items():
        ids = tok.encode(text)
        decoded = tok.decode(ids)

        no_unk = (unk_id not in ids) if unk_id is not None else True
        lossless = (decoded == text)

        print(f"  {cat:<24} | {len(ids):>6} | {'✓' if no_unk else '✗':>8} | {'✓' if lossless else '✗':>8}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Whitespace Normalization Invariant Check
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 3. Whitespace Strictness (No Collapsing/Stripping) ---")
    ws_1 = tok.encode("to be")
    ws_2 = tok.encode("to  be")
    ws_3 = tok.encode("to\tbe")
    ws_distinct = (ws_1 != ws_2) and (ws_1 != ws_3) and (ws_2 != ws_3)
    print(f"  'to be' vs 'to  be' vs 'to\\tbe' distinct encodings: {'✓ PASS' if ws_distinct else '✗ FAIL'}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Fault Tolerance (Partial UTF-8 Bytes)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 4. Fault Tolerance & Partial UTF-8 Byte Handling ---")
    rocket_bytes = "🚀".encode("utf-8")  # 4 bytes: [240, 159, 154, 128]
    partial_id = [rocket_bytes[0]]       # Take only the first lead byte
    partial_decoded = tok.decode(partial_id)
    safe_replace = (partial_decoded == "\ufffd")
    print(f"  Single byte of 4-byte emoji decodes to U+FFFD (no crash): {'✓ PASS' if safe_replace else '✗ FAIL'}")


def main():
    print("Loading corpus…")
    train_text, eval_text = load_corpus()
    eval_sentences = split_sentences(eval_text)
    print(f"  Train slice : {len(train_text):,} chars")
    print(f"  Eval slice  : {len(eval_text):,} chars  |  {len(eval_sentences):,} sentences")
    print(f"  Vocab sizes : {VOCAB_SIZES}")

    results = []
    tokenizers = {}
    for vs in VOCAB_SIZES:
        result, tok = analyze_vocab(vs, train_text, eval_text, eval_sentences)
        results.append(result)
        tokenizers[vs] = tok

    print("\n\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print_table(results)

    # Rare-word breakdown for the largest vocab (most interesting)
    largest_vocab = max(VOCAB_SIZES)
    print(f"\n\nRare-word token breakdown at V={largest_vocab:,}:")
    tok_large = tokenizers[largest_vocab]
    rare_detail = measure_rare_word_coverage(tok_large, RARE_WORDS)
    max_w = max(len(w) for w in RARE_WORDS)
    for word, info in rare_detail.items():
        status = "✓" if info["lossless"] else "✗ (lossy!)"
        print(f"  {word:<{max_w}}  →  {info['tokens']:>3} tokens  {status}")

    plot_results(results)

    # Display the first 20 tokens after ID 255 for each vocab size
    show_first_20_merged_tokens(tokenizers)

    # Battery of special tokens & edge cases verification
    analyze_special_tokens_and_edge_cases(tok_large)


if __name__ == "__main__":
    main()
