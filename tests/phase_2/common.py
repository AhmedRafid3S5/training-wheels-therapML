"""Helpers shared by the Phase 2 tests."""

import time
from collections.abc import Callable

import numpy as np
import torch
from tqdm import tqdm

# Fixed TinyShakespeare slices so that every test sees the same data.
BPE_TRAIN_CHARS = 50_000  # training slice for the tokenizer tests
HELDOUT_START, HELDOUT_END = 1_000_000, 1_020_000  # never used for training

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


def make_attention_weights(d_model: int, num_heads: int, num_kv_heads: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random (q, k, v, o) projection weights for (grouped-query) attention."""
    g = torch.Generator().manual_seed(seed)
    d_head = d_model // num_heads
    kv_dim = num_kv_heads * d_head
    scale = d_model**-0.5
    return {
        "q_proj_weight": torch.randn(d_model, d_model, generator=g) * scale,
        "k_proj_weight": torch.randn(kv_dim, d_model, generator=g) * scale,
        "v_proj_weight": torch.randn(kv_dim, d_model, generator=g) * scale,
        "o_proj_weight": torch.randn(d_model, d_model, generator=g) * scale,
    }


def expand_kv_weight(weight: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """Turn a GQA K/V projection (num_kv_heads * d_head, d_model) into the
    equivalent MHA projection (num_heads * d_head, d_model) by repeating each KV
    head for every query head in its group (``repeat_interleave`` grouping)."""
    kv_dim, d_model = weight.shape
    d_head = kv_dim // num_kv_heads
    group = num_heads // num_kv_heads
    return weight.view(num_kv_heads, d_head, d_model).repeat_interleave(group, dim=0).reshape(-1, d_model)


def benchmark(fn: Callable[[], object], desc: str, n_iter: int = 50, n_warmup: int = 2) -> float:
    """Mean wall-clock seconds per call of *fn* (after *n_warmup* untimed calls)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in tqdm(range(n_iter), desc=desc):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))
