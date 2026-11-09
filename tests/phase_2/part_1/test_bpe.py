"""Phase 2 · Part 1 · Task 1 — Byte Pair Encoding (BPE) from Scratch

Tests validate a byte-level BPE tokenizer (``train`` / ``encode`` / ``decode``)
trained on a TinyShakespeare subset:

* structural invariants of the learned vocabulary,
* the greedy "merge the most frequent pair" rule on a hand-checked corpus,
* the round-trip invariant ``decode(encode(text)) == text``,
* compression on held-out text,
* training / encoding latency.
"""

import time

import pytest

from tests.adapters import get_bpe_vocab, run_bpe_decode, run_bpe_encode, run_train_bpe

VOCAB_SIZE = 1000

ROUNDTRIP_STRINGS = [
    "",
    "Hello, world!",
    "First Citizen:\nBefore we proceed any further, hear me speak.",
    "KING HENRY: What say you, my lords?",
    "To be, or not to be, that is the question:",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\nand\n\n\nblank lines",
    "numbers 0123456789 and 3.14159",
    "punctuation!?.,;:'\"()[]{}<>-_/\\|@#$%^&*~`",
    "café naïve résumé façade",
    "Здравствуй, мир",
    "你好，世界",
    "こんにちは世界",
    "مرحبا بالعالم",
    "emoji 👋🌍🚀🔥",
    "family 👨\u200d👩\u200d👧\u200d👦 and flag 🇧🇩",
    "Mixed: Hello 世界 مرحبا мир 🌍!",
    "zero\u200bwidth and non\u00a0breaking",
    "e\u0301 (combining accent)",
    "supercalifragilisticexpialidocious",
]


@pytest.fixture(scope="module")
def tokenizer(train_bpe_cached):
    return train_bpe_cached(VOCAB_SIZE)


@pytest.fixture(scope="module")
def vocab(tokenizer) -> dict[int, bytes]:
    return get_bpe_vocab(tokenizer)


# ============================================================================
# Vocabulary structure
# ============================================================================


class TestBPEVocabulary:
    def test_vocab_size_is_exact(self, vocab):
        assert len(vocab) == VOCAB_SIZE

    def test_ids_are_contiguous(self, vocab):
        assert set(vocab.keys()) == set(range(VOCAB_SIZE))

    def test_first_256_ids_are_raw_bytes(self, vocab):
        for i in range(256):
            assert vocab[i] == bytes([i]), f"vocab[{i}] should be the single byte {bytes([i])!r}"

    def test_vocab_entries_are_unique(self, vocab):
        assert len(set(vocab.values())) == len(vocab)

    def test_merged_tokens_are_concatenations_of_earlier_tokens(self, vocab):
        """Every merged token (id ≥ 256) = vocab[a] + vocab[b] with a, b < id."""
        id_of: dict[bytes, int] = {}
        for token_id in sorted(vocab):
            token = vocab[token_id]
            if token_id >= 256:
                ok = any(token[:split] in id_of and token[split:] in id_of for split in range(1, len(token)))
                assert ok, f"Token {token_id} ({token!r}) is not a merge of two earlier tokens"
            id_of[token] = token_id

    def test_training_is_deterministic(self, bpe_train_text):
        corpus = bpe_train_text[:10_000]
        v1 = get_bpe_vocab(run_train_bpe(corpus, 300))
        v2 = get_bpe_vocab(run_train_bpe(corpus, 300))
        assert v1 == v2


# ============================================================================
# The merge rule
# ============================================================================


class TestBPEMergeRule:
    """Hand-checked corpus with no frequency ties (with or without GPT-2 style
    pre-tokenization):

        "abababab cdcd"
        merge 1: (a, b)   count 4  → 256 = b"ab"
        merge 2: (ab, ab) count 3  → 257 = b"abab"
        merge 3: (c, d)   count 2  → 258 = b"cd"
    """

    CORPUS = "abababab cdcd"

    @pytest.fixture(scope="class")
    def toy_tokenizer(self):
        return run_train_bpe(self.CORPUS, 259)

    def test_merges_follow_frequency_order(self, toy_tokenizer):
        vocab = get_bpe_vocab(toy_tokenizer)
        assert vocab[256] == b"ab"
        assert vocab[257] == b"abab"
        assert vocab[258] == b"cd"

    def test_encode_applies_learned_merges(self, toy_tokenizer):
        assert run_bpe_encode(toy_tokenizer, self.CORPUS) == [257, 257, ord(" "), 258, 258]

    def test_encode_unseen_pairs_stay_bytes(self, toy_tokenizer):
        assert run_bpe_encode(toy_tokenizer, "ab") == [256]
        assert run_bpe_encode(toy_tokenizer, "ba") == [ord("b"), ord("a")]
        assert run_bpe_encode(toy_tokenizer, "abc") == [256, ord("c")]

    def test_zero_merges_is_raw_utf8_bytes(self, bpe_train_text):
        tok = run_train_bpe(bpe_train_text[:5_000], 256)
        text = "Hello 世界 🚀"
        assert run_bpe_encode(tok, text) == list(text.encode("utf-8"))


# ============================================================================
# Round-trip invariant
# ============================================================================


class TestBPERoundtrip:
    @pytest.mark.parametrize("text", ROUNDTRIP_STRINGS, ids=lambda s: repr(s[:20]))
    def test_roundtrip(self, tokenizer, text):
        ids = run_bpe_encode(tokenizer, text)
        assert run_bpe_decode(tokenizer, ids) == text

    def test_roundtrip_heldout_shakespeare(self, tokenizer, heldout_text):
        ids = run_bpe_encode(tokenizer, heldout_text)
        assert run_bpe_decode(tokenizer, ids) == heldout_text

    def test_encode_empty_string(self, tokenizer):
        assert run_bpe_encode(tokenizer, "") == []
        assert run_bpe_decode(tokenizer, []) == ""

    def test_token_ids_are_valid_ints(self, tokenizer, heldout_text):
        ids = run_bpe_encode(tokenizer, heldout_text[:5_000])
        assert all(isinstance(i, int) for i in ids)
        assert all(0 <= i < VOCAB_SIZE for i in ids)


# ============================================================================
# Compression
# ============================================================================


class TestBPECompression:
    def test_compresses_heldout_text(self, tokenizer, heldout_text):
        n_bytes = len(heldout_text.encode("utf-8"))
        n_tokens = len(run_bpe_encode(tokenizer, heldout_text))
        ratio = n_bytes / n_tokens
        print(f"\nBPE V={VOCAB_SIZE}: {ratio:.2f} bytes/token on held-out text")
        assert ratio >= 1.8, f"Compression ratio {ratio:.2f} bytes/token is too low (expected ≥ 1.8)"


# ============================================================================
# Latency benchmarks
# ============================================================================


class TestBPEPerformance:
    """No reference library (tiktoken / sentencepiece) is allowed, so these use
    generous absolute budgets. A straightforward minbpe-style implementation
    passes comfortably. They exist to catch pathologically slow code (e.g.
    re-encoding the whole corpus from scratch for every merge)."""

    def test_training_latency(self, bpe_train_text):
        t0 = time.perf_counter()
        run_train_bpe(bpe_train_text, VOCAB_SIZE)
        elapsed = time.perf_counter() - t0
        print(f"\nBPE training: {len(bpe_train_text):,} chars → V={VOCAB_SIZE} in {elapsed:.2f}s")
        assert elapsed <= 30.0, f"BPE training took {elapsed:.1f}s (max 30s)"

    def test_encode_throughput(self, tokenizer, heldout_text):
        n_bytes = len(heldout_text.encode("utf-8"))
        t0 = time.perf_counter()
        run_bpe_encode(tokenizer, heldout_text)
        elapsed = time.perf_counter() - t0
        kb_per_s = n_bytes / 1024 / elapsed
        print(f"\nBPE encode: {kb_per_s:.1f} KB/s")
        assert kb_per_s >= 10.0, f"Encoding throughput {kb_per_s:.1f} KB/s is too slow (min 10 KB/s)"

    def test_decode_throughput(self, tokenizer, heldout_text):
        ids = run_bpe_encode(tokenizer, heldout_text)
        n_bytes = len(heldout_text.encode("utf-8"))
        t0 = time.perf_counter()
        for _ in range(10):
            run_bpe_decode(tokenizer, ids)
        elapsed = (time.perf_counter() - t0) / 10
        kb_per_s = n_bytes / 1024 / elapsed
        print(f"\nBPE decode: {kb_per_s:.1f} KB/s")
        assert kb_per_s >= 1000.0, f"Decoding throughput {kb_per_s:.1f} KB/s is too slow (min 1000 KB/s)"
