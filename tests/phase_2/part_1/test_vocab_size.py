"""Phase 2 · Part 1 · Task 2 — Vocabulary Size Experiment

Tests validate the vocabulary-size / sequence-length trade-off:

* compression ratio (bytes/token) grows with vocabulary size,
* average tokens per sentence shrinks with vocabulary size,
* rare words remain representable (byte fallback) but cost more tokens than
  common words.

The experiment uses smaller vocabularies than the PDF (V ∈ {256, 1000, 5000,
10000, 32000}) to keep test time short on a 50 KB training slice. Run the full
sweep in your own experiment script.
"""

import itertools
import re
from collections import Counter

import pytest

from tests.adapters import run_bpe_decode, run_bpe_encode

VOCAB_SIZES = [256, 512, 1000, 2000]

RARE_WORDS = [
    "Supercalifragilisticexpialidocious",
    "antidisestablishmentarianism",
    "Pneumonoultramicroscopicsilicovolcanoconiosis",
    "Zyzzyva",
    "Mxyzptlk",
    "Llanfairpwllgwyngyll",
]


def _bytes_per_token(tokenizer, text: str) -> float:
    return len(text.encode("utf-8")) / len(run_bpe_encode(tokenizer, text))


@pytest.fixture(scope="module")
def tokenizers(train_bpe_cached):
    return {v: train_bpe_cached(v) for v in VOCAB_SIZES}


@pytest.fixture(scope="module")
def sentences(heldout_text) -> list[str]:
    return [line for line in heldout_text.splitlines() if line.strip()]


class TestVocabSizeExperiment:
    def test_vocab_256_has_ratio_one(self, tokenizers, heldout_text):
        assert _bytes_per_token(tokenizers[256], heldout_text) == pytest.approx(1.0)

    def test_compression_ratio_increases_with_vocab_size(self, tokenizers, heldout_text):
        ratios = [_bytes_per_token(tokenizers[v], heldout_text) for v in VOCAB_SIZES]
        print("\n  V      bytes/token")
        for v, r in zip(VOCAB_SIZES, ratios):
            print(f"  {v:<6} {r:.3f}")
        for i in range(len(VOCAB_SIZES) - 1):
            assert ratios[i + 1] > ratios[i], (
                f"Compression should improve from V={VOCAB_SIZES[i]} ({ratios[i]:.3f}) "
                f"to V={VOCAB_SIZES[i + 1]} ({ratios[i + 1]:.3f})"
            )

    def test_avg_tokens_per_sentence_decreases(self, tokenizers, sentences):
        avgs = [sum(len(run_bpe_encode(tokenizers[v], s)) for s in sentences) / len(sentences) for v in VOCAB_SIZES]
        print("\n  V      avg tokens/sentence")
        for v, a in zip(VOCAB_SIZES, avgs):
            print(f"  {v:<6} {a:.2f}")
        for a, b in itertools.pairwise(avgs):
            assert b < a, f"Average tokens per sentence should shrink as V grows: {avgs}"


class TestRareWords:
    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_rare_words_roundtrip(self, tokenizers, vocab_size):
        tok = tokenizers[vocab_size]
        for word in RARE_WORDS:
            ids = run_bpe_encode(tok, " " + word)
            assert len(ids) >= 1
            assert run_bpe_decode(tok, ids) == " " + word

    def test_common_words_compress_better_than_rare_words(self, tokenizers, bpe_train_text):
        tok = tokenizers[max(VOCAB_SIZES)]
        common = [w for w, _ in Counter(re.findall(r"[a-z]+", bpe_train_text)).most_common(50)]
        common_ratio = sum(_bytes_per_token(tok, " " + w) for w in common) / len(common)
        rare_ratio = sum(_bytes_per_token(tok, " " + w) for w in RARE_WORDS) / len(RARE_WORDS)
        print(f"\nbytes/token  common words: {common_ratio:.2f}   rare words: {rare_ratio:.2f}")
        assert common_ratio > rare_ratio
