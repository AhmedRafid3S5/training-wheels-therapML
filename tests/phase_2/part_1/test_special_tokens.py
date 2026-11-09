"""Phase 2 · Part 1 · Task 3 — Special Tokens & Edge Cases

Tests validate:

* ``<PAD>``, ``<BOS>``, ``<EOS>``, ``<UNK>`` are single, atomic tokens that are
  never split, never merged, and never learned as part of a merge,
* lossless handling of Unicode, emoji, whitespace and mixed-language text,
* ``<UNK>`` is never emitted (a byte-level BPE can represent every string),
* decoding invalid UTF-8 (e.g. half an emoji) does not crash.
"""

import pytest

from tests.adapters import get_bpe_vocab, run_bpe_decode, run_bpe_encode
from tests.phase_2.common import SPECIAL_TOKENS

VOCAB_SIZE = 600

EDGE_CASES = {
    "accents": "café naïve résumé Ångström",
    "combining": "e\u0301 a\u0308 n\u0303",
    "cyrillic": "Съешь же ещё этих мягких французских булок",
    "greek": "Τάχιστη αλώπηξ βαφής ψημένη γη",
    "cjk": "日本語の文章と中文句子와 한국어",
    "arabic_rtl": "النص العربي ي\u064fكتب من اليمين",
    "hindi": "नमस\u094dत\u0947 द\u0941निया",
    "emoji": "😀😃😄😁 🚀🔥💯",
    "emoji_zwj": "👩\u200d💻 👨\u200d👩\u200d👧\u200d👦 🏳\ufe0f\u200d🌈",
    "emoji_skin_tone": "👍🏽 👋🏿",
    "flags": "🇧🇩 🇯🇵 🇺🇸",
    "mixed_language": "Hello世界! Bonjour мир 🌍 مرحبا",
    "multiple_spaces": "a  b   c    d",
    "tabs_newlines": "a\tb\n\nc\r\nd",
    "leading_trailing_ws": "   padded   ",
    "only_whitespace": " \t\n \u3000",
    "unicode_spaces": "nbsp\u00a0em\u2003thin\u2009ideographic\u3000",
    "zero_width": "zero\u200bwidth\u200cnon\u200djoiner\ufeff",
    "control_chars": "null\x00bell\x07esc\x1b",
    "math_symbols": "∀x∈ℝ: x² ≥ 0 ∑∫√∞",
    "special_lookalike": "<BOS <EOS> > < PAD> <UNK",
}


@pytest.fixture(scope="module")
def corpus_with_specials(bpe_train_text) -> str:
    """Training text with special tokens sprinkled in so that a naive trainer
    would happily learn merges like ``b"<EOS>"``."""
    chunks = [bpe_train_text[i : i + 500] for i in range(0, len(bpe_train_text), 500)]
    return "".join(f"<BOS>{c}<EOS><PAD><PAD>" for c in chunks)


@pytest.fixture(scope="module")
def tokenizer(train_bpe_cached, corpus_with_specials):
    return train_bpe_cached(VOCAB_SIZE, tuple(SPECIAL_TOKENS), corpus_with_specials)


@pytest.fixture(scope="module")
def vocab(tokenizer):
    return get_bpe_vocab(tokenizer)


@pytest.fixture(scope="module")
def special_ids(vocab) -> dict[str, int]:
    inverse = {b: i for i, b in vocab.items()}
    return {s: inverse[s.encode("utf-8")] for s in SPECIAL_TOKENS if s.encode("utf-8") in inverse}


# ============================================================================
# Special tokens
# ============================================================================


class TestSpecialTokens:
    def test_vocab_size_includes_special_tokens(self, vocab):
        assert len(vocab) == VOCAB_SIZE

    def test_each_special_token_in_vocab_once(self, vocab):
        values = list(vocab.values())
        for s in SPECIAL_TOKENS:
            assert values.count(s.encode("utf-8")) == 1, f"{s} must appear exactly once in the vocab"

    def test_special_token_ids_come_after_merges(self, special_ids):
        n = len(SPECIAL_TOKENS)
        assert sorted(special_ids.values()) == list(range(VOCAB_SIZE - n, VOCAB_SIZE))

    @pytest.mark.parametrize("special", SPECIAL_TOKENS)
    def test_special_token_encodes_to_single_id(self, tokenizer, special_ids, special):
        assert run_bpe_encode(tokenizer, special) == [special_ids[special]]

    def test_special_tokens_inside_text(self, tokenizer, special_ids):
        body = "Hello world, my lord."
        ids = run_bpe_encode(tokenizer, f"<BOS>{body}<EOS>")
        assert ids[0] == special_ids["<BOS>"]
        assert ids[-1] == special_ids["<EOS>"]
        assert ids[1:-1] == run_bpe_encode(tokenizer, body)

    def test_adjacent_special_tokens(self, tokenizer, special_ids):
        pad = special_ids["<PAD>"]
        assert run_bpe_encode(tokenizer, "<PAD><PAD><PAD>") == [pad, pad, pad]

    def test_special_tokens_roundtrip(self, tokenizer):
        text = "<BOS>KING HENRY: What say you?<EOS><PAD><PAD>"
        assert run_bpe_decode(tokenizer, run_bpe_encode(tokenizer, text)) == text

    def test_special_tokens_never_learned_as_merges(self, vocab, special_ids):
        """The training corpus contains many ``<EOS>``, ``<PAD>`` … strings, but
        no regular (non-special) token may contain a full special-token string."""
        specials = {s.encode("utf-8") for s in SPECIAL_TOKENS}
        special_id_set = set(special_ids.values())
        for token_id, token in vocab.items():
            if token_id in special_id_set:
                continue
            for s in specials:
                assert s not in token, f"Regular token {token_id} ({token!r}) contains special token {s!r}"

    def test_special_lookalikes_are_not_special(self, tokenizer, special_ids):
        ids = run_bpe_encode(tokenizer, "<BOS <EOS > PAD>")
        assert not set(ids) & set(special_ids.values())


# ============================================================================
# Edge cases: Unicode, emoji, whitespace, mixed-language
# ============================================================================


class TestEdgeCases:
    @pytest.mark.parametrize("text", EDGE_CASES.values(), ids=EDGE_CASES.keys())
    def test_edge_case_roundtrip(self, tokenizer, text):
        assert run_bpe_decode(tokenizer, run_bpe_encode(tokenizer, text)) == text

    @pytest.mark.parametrize("text", EDGE_CASES.values(), ids=EDGE_CASES.keys())
    def test_unk_is_never_emitted(self, tokenizer, special_ids, text):
        assert special_ids["<UNK>"] not in run_bpe_encode(tokenizer, text)

    def test_whitespace_is_not_normalized(self, tokenizer):
        a = run_bpe_encode(tokenizer, "to be")
        b = run_bpe_encode(tokenizer, "to  be")
        c = run_bpe_encode(tokenizer, "to\tbe")
        assert a != b and a != c and b != c

    def test_decode_partial_utf8_does_not_raise(self, tokenizer):
        rocket = "🚀".encode()  # 4 bytes → ids 0..255 are raw bytes
        assert run_bpe_decode(tokenizer, [rocket[0]]) == "\ufffd"
        assert run_bpe_decode(tokenizer, list(rocket)) == "🚀"

    def test_decode_invalid_byte_does_not_raise(self, tokenizer):
        out = run_bpe_decode(tokenizer, [ord("a"), 0xFF, ord("b")])
        assert out == "a\ufffdb"
