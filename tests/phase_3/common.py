"""Helpers shared by the Phase 3 tests."""

import numpy as np
import torch

# Small Mini-Llama used by the fast unit tests.
SMALL_CONFIG = {
    "vocab_size": 128,
    "context_length": 64,
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 4,
    "num_kv_heads": 2,
    "d_ff": 172,
    "rope_theta": 10000.0,
}


class FixedLogitsModel(torch.nn.Module):
    """Always predicts the same next-token distribution, whatever the input."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)
        self.num_calls = 0

    def forward(self, idx):
        self.num_calls += 1
        return self.logits.expand(idx.shape[0], idx.shape[1], -1).clone()


class BigramModel(torch.nn.Module):
    """Next-token logits depend only on the current token: ``table[idx]``."""

    def __init__(self, vocab_size: int, seed: int = 0, scale: float = 1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("table", torch.randn(vocab_size, vocab_size, generator=g) * scale)
        self.num_calls = 0
        self.max_seen_len = 0

    def forward(self, idx):
        self.num_calls += 1
        self.max_seen_len = max(self.max_seen_len, idx.shape[1])
        return self.table[idx]


def reference_sampling_probs(
    logits: torch.Tensor, temperature: float, top_k: int | None = None, top_p: float | None = None
) -> np.ndarray:
    """Exact next-token distribution of the temperature → top-k → top-p pipeline
    for a single 1-D logits vector (written as a plain loop, for clarity)."""
    x = logits.double().numpy() / temperature
    order = np.argsort(-x, kind="stable")
    keep = np.ones_like(x, dtype=bool)
    if top_k is not None:
        keep[order[top_k:]] = False
    probs = np.where(keep, np.exp(x - x.max()), 0.0)
    probs /= probs.sum()
    if top_p is not None:
        mass_above = 0.0
        for idx in order:
            if not keep[idx]:
                continue
            if mass_above >= top_p:
                keep[idx] = False
            mass_above += probs[idx]
        probs = np.where(keep, probs, 0.0)
        probs /= probs.sum()
    return probs


def state_dict_bytes(model: torch.nn.Module) -> int:
    """Bytes taken by every tensor in ``model.state_dict()``. Tensors that share
    storage (e.g. tied weights) are counted once."""
    seen, total = set(), 0
    for t in model.state_dict().values():
        key = (t.data_ptr(), t.numel(), t.dtype)
        if key in seen:
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
    return total


def distinct_n(sequences: list[list[int]], n: int) -> float:
    """Fraction of n-grams across *sequences* that are unique (a standard
    diversity metric for generated text)."""
    ngrams = [tuple(s[i : i + n]) for s in sequences for i in range(len(s) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)
