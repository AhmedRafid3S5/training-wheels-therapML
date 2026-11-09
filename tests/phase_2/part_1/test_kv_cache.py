"""Phase 2 · Part 1 · Task 9 — KV-Cache for Efficient Inference

Tests validate a KV-cached causal GQA + RoPE attention:

* prefill / token-by-token decoding give the same outputs as the uncached
  full-sequence computation,
* the cache stores RoPE-rotated K and plain V, growing by one step per token,
* GQA shrinks the cache by ``num_heads / num_kv_heads`` versus MHA,
* cached decoding is faster than recomputing the whole prefix every step
  (tokens/sec at sequence lengths 128, 256, 512).
"""

import time

import numpy as np
import pytest
import torch
from einops import rearrange

from tests.adapters import (
    get_kv_cache,
    get_kv_cached_attention,
    run_grouped_query_attention,
    run_kv_cached_attention,
    run_rope,
)
from tests.phase_2.common import make_attention_weights

D_MODEL, NUM_HEADS, NUM_KV_HEADS, THETA, MAX_SEQ = 64, 8, 2, 10000.0, 64
D_HEAD = D_MODEL // NUM_HEADS


@pytest.fixture
def weights():
    return make_attention_weights(D_MODEL, NUM_HEADS, NUM_KV_HEADS, seed=7)


@pytest.fixture
def x():
    return torch.randn(2, 32, D_MODEL, generator=torch.Generator().manual_seed(8))


def _new(w, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, d_model=D_MODEL, max_seq=MAX_SEQ):
    return get_kv_cached_attention(d_model, num_heads, num_kv_heads, max_seq, THETA, **w)


def _full(w, x, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, d_model=D_MODEL, max_seq=MAX_SEQ):
    return run_grouped_query_attention(
        d_model, num_heads, num_kv_heads, **w, in_features=x, theta=THETA, max_seq_len=max_seq
    )


def _feed(attn, x, chunk_sizes):
    outs, start = [], 0
    for size in chunk_sizes:
        outs.append(run_kv_cached_attention(attn, x[:, start : start + size]).detach())
        start += size
    return torch.cat(outs, dim=1)


# ============================================================================
# Accuracy
# ============================================================================


class TestKVCacheCorrectness:
    def test_prefill_matches_full(self, weights, x):
        cached = _feed(_new(weights), x, [x.shape[1]])
        np.testing.assert_allclose(cached.numpy(), _full(weights, x).detach().numpy(), atol=1e-5, rtol=1e-4)

    def test_token_by_token_matches_full(self, weights, x):
        cached = _feed(_new(weights), x, [1] * x.shape[1])
        np.testing.assert_allclose(cached.numpy(), _full(weights, x).detach().numpy(), atol=1e-5, rtol=1e-4)

    def test_prefill_then_decode_matches_full(self, weights, x):
        cached = _feed(_new(weights), x, [20] + [1] * 8 + [4])
        np.testing.assert_allclose(cached.numpy(), _full(weights, x).detach().numpy(), atol=1e-5, rtol=1e-4)

    def test_cache_grows_with_each_step(self, weights, x):
        attn = _new(weights)
        for n, size in zip((5, 6, 7, 10), (5, 1, 1, 3)):
            run_kv_cached_attention(attn, x[:, n - size : n])
            k, v = get_kv_cache(attn)
            assert k.shape == (x.shape[0], NUM_KV_HEADS, n, D_HEAD)
            assert v.shape == (x.shape[0], NUM_KV_HEADS, n, D_HEAD)

    def test_cached_values_are_plain_projections(self, weights, x):
        attn = _new(weights)
        _feed(attn, x, [10, 1, 1])
        _, v = get_kv_cache(attn)
        expected = rearrange(x[:, :12] @ weights["v_proj_weight"].T, "b s (h d) -> b h s d", h=NUM_KV_HEADS)
        np.testing.assert_allclose(v.detach().numpy(), expected.numpy(), atol=1e-5, rtol=1e-4)

    def test_cached_keys_have_rope_applied(self, weights, x):
        attn = _new(weights)
        _feed(attn, x, [10, 1, 1])
        k, _ = get_kv_cache(attn)
        raw = rearrange(x[:, :12] @ weights["k_proj_weight"].T, "b s (h d) -> b h s d", h=NUM_KV_HEADS)
        expected = run_rope(D_HEAD, THETA, MAX_SEQ, raw, torch.arange(12))
        np.testing.assert_allclose(k.detach().numpy(), expected.detach().numpy(), atol=1e-5, rtol=1e-4)

    def test_new_object_starts_with_empty_cache(self, weights, x):
        a = _feed(_new(weights), x[:, :4], [4])
        b = _feed(_new(weights), x[:, :4], [4])
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-6)


# ============================================================================
# Memory: MHA vs GQA
# ============================================================================


class TestKVCacheMemory:
    @pytest.mark.parametrize("num_kv_heads", [8, 4, 2, 1])
    def test_gqa_cache_is_smaller_than_mha(self, num_kv_heads, x):
        def cache_bytes(kv_heads):
            attn = _new(make_attention_weights(D_MODEL, NUM_HEADS, kv_heads), num_kv_heads=kv_heads)
            run_kv_cached_attention(attn, x)
            k, v = get_kv_cache(attn)
            return k.numel() * k.element_size() + v.numel() * v.element_size()

        mha, gqa = cache_bytes(NUM_HEADS), cache_bytes(num_kv_heads)
        print(f"\nKV-cache  MHA={mha:,} B  GQA(kv={num_kv_heads})={gqa:,} B  saving={mha / gqa:.0f}x")
        assert mha / gqa == pytest.approx(NUM_HEADS / num_kv_heads)


# ============================================================================
# Latency: generation with vs without cache
# ============================================================================


class TestKVCachePerformance:
    D_MODEL, NUM_HEADS, NUM_KV_HEADS, BATCH = 512, 8, 2, 4

    def _decode_tokens_per_sec(self, seq_len: int, use_cache: bool) -> float:
        """Generate *seq_len* positions one at a time, with or without the cache."""
        d, h, kv = self.D_MODEL, self.NUM_HEADS, self.NUM_KV_HEADS
        w = make_attention_weights(d, h, kv)
        x = torch.randn(self.BATCH, seq_len, d, generator=torch.Generator().manual_seed(0))

        def decode(n_steps: int):
            if use_cache:
                attn = get_kv_cached_attention(d, h, kv, seq_len, THETA, **w)
                for t in range(n_steps):
                    run_kv_cached_attention(attn, x[:, t : t + 1])
            else:  # recompute attention over the whole prefix at every step
                for t in range(1, n_steps + 1):
                    run_grouped_query_attention(d, h, kv, **w, in_features=x[:, :t], theta=THETA, max_seq_len=seq_len)

        with torch.no_grad():
            decode(8)  # warm-up
            t0 = time.perf_counter()
            decode(seq_len)
            elapsed = time.perf_counter() - t0
        return self.BATCH * seq_len / elapsed

    def test_kv_cache_speedup(self):
        print("\n  seq_len  no-cache tok/s   cache tok/s   speedup")
        speedups = {}
        for seq_len in (128, 256, 512):
            slow = self._decode_tokens_per_sec(seq_len, use_cache=False)
            fast = self._decode_tokens_per_sec(seq_len, use_cache=True)
            speedups[seq_len] = fast / slow
            print(f"  {seq_len:<8} {slow:>14.0f} {fast:>13.0f} {fast / slow:>9.1f}x")
        assert speedups[512] >= 3.0, f"KV-cache should be ≥3x faster at seq_len=512 (got {speedups[512]:.1f}x)"
        assert speedups[512] > speedups[128], "The KV-cache advantage should grow with sequence length"
