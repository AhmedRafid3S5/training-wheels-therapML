"""Shared fixtures for Phase 2 (tokenization, attention, Mini-Llama)."""

import pytest

from tests.adapters import run_train_bpe
from tests.common import load_tinyshakespeare
from tests.phase_2.common import BPE_TRAIN_CHARS, HELDOUT_END, HELDOUT_START


@pytest.fixture(scope="session")
def bpe_train_text() -> str:
    return load_tinyshakespeare()[:BPE_TRAIN_CHARS]


@pytest.fixture(scope="session")
def heldout_text() -> str:
    return load_tinyshakespeare()[HELDOUT_START:HELDOUT_END]


@pytest.fixture(scope="session")
def train_bpe_cached(bpe_train_text):
    """Factory ``train(vocab_size, special_tokens=(), corpus=None)`` that trains
    each distinct tokenizer once per session (BPE training is the slow part)."""
    cache: dict = {}

    def train(vocab_size: int, special_tokens: tuple[str, ...] = (), corpus: str | None = None):
        corpus = bpe_train_text if corpus is None else corpus
        key = (vocab_size, tuple(special_tokens), corpus)
        if key not in cache:
            cache[key] = run_train_bpe(corpus, vocab_size, list(special_tokens) or None)
        return cache[key]

    return train
