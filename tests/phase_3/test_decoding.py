"""Phase 3 · Task 2 — Decoding Strategies

Tests validate:

* the top-k and top-p (nucleus) logit filters, against hand-worked examples
  and a reference implementation,
* ``run_sample_next_token``: greedy at T = 0, and empirical sampling
  distributions that match temperature → top-k → top-p exactly,
* ``run_decode`` (your ``generate()``): greedy / temperature / top-k / top-p /
  combined decoding, reproducibility, context cropping, and that diversity
  grows with temperature (10 completions per setting),
* KV-cached decoding on your Mini-Llama gives the same tokens as uncached
  decoding, and is faster.

Toy models (bigram / fixed logits) check the decoding logic on its own. Your
Mini-Llama checks the KV-cache path.
"""

import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.adapters import get_mini_llama
from tests.adapters_0 import run_decode, run_sample_next_token, run_top_k_filter, run_top_p_filter
from tests.phase_3.common import (
    SMALL_CONFIG,
    BigramModel,
    FixedLogitsModel,
    distinct_n,
    reference_sampling_probs,
)

# Unsorted on purpose. For every top-p case below, p is at least 0.015 away from
# any cumulative probability, so float rounding cannot change the kept set.
LOGITS = torch.tensor([0.5, -1.0, 2.0, -0.3, 1.2, -2.5, 0.0, -1.5])


def _reference_top_k(logits, k):
    kth = torch.topk(logits, min(k, logits.shape[-1]), dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def _reference_top_p(logits, p):
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    mass_above = probs.cumsum(-1) - probs
    keep_sorted = mass_above < p
    keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
    return logits.masked_fill(~keep, float("-inf"))


def _assert_same_filter(actual, expected):
    assert actual.shape == expected.shape
    assert torch.equal(torch.isinf(actual), torch.isinf(expected)), "Different set of tokens was kept"
    finite = torch.isfinite(expected)
    np.testing.assert_allclose(actual[finite].numpy(), expected[finite].numpy(), rtol=0, atol=0)


# ============================================================================
# Top-k filter
# ============================================================================


class TestTopKFilter:
    @pytest.mark.parametrize("k", [1, 3, 10, 50])
    def test_matches_reference(self, k):
        logits = torch.randn(4, 7, 50, generator=torch.Generator().manual_seed(k))
        _assert_same_filter(run_top_k_filter(logits, k), _reference_top_k(logits, k))

    def test_hand_example(self):
        out = run_top_k_filter(LOGITS, 3)
        kept = torch.isfinite(out).nonzero().flatten().tolist()
        assert sorted(kept) == [0, 2, 4]  # logits 0.5, 2.0, 1.2
        assert torch.isneginf(out[torch.isinf(out)]).all(), "Filtered logits must be -inf"

    def test_k_larger_than_vocab_keeps_everything(self):
        assert torch.equal(run_top_k_filter(LOGITS, 100), LOGITS)

    def test_k_one_keeps_only_argmax(self):
        logits = torch.randn(16, 30, generator=torch.Generator().manual_seed(0))
        out = run_top_k_filter(logits, 1)
        assert (torch.isfinite(out).sum(-1) == 1).all()
        assert torch.equal(out.argmax(-1), logits.argmax(-1))

    def test_rows_are_independent(self):
        logits = torch.randn(8, 20, generator=torch.Generator().manual_seed(1))
        out = run_top_k_filter(logits, 5)
        for row_in, row_out in zip(logits, out):
            _assert_same_filter(run_top_k_filter(row_in, 5), row_out)

    def test_does_not_modify_input(self):
        logits = torch.randn(4, 20, generator=torch.Generator().manual_seed(2))
        before = logits.clone()
        run_top_k_filter(logits, 3)
        assert torch.equal(logits, before)


# ============================================================================
# Top-p (nucleus) filter
# ============================================================================


class TestTopPFilter:
    # probs: token 1 = 0.50, token 3 = 0.25, token 2 = 0.15, token 0 = 0.10
    PROBS = torch.tensor([0.10, 0.50, 0.15, 0.25])

    @pytest.mark.parametrize(
        "p, expected_kept",
        [(0.3, [1]), (0.6, [1, 3]), (0.8, [1, 2, 3]), (0.95, [0, 1, 2, 3])],
    )
    def test_hand_example(self, p, expected_kept):
        out = run_top_p_filter(torch.log(self.PROBS), p)
        assert sorted(torch.isfinite(out).nonzero().flatten().tolist()) == expected_kept

    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_matches_reference(self, p):
        logits = 2 * torch.randn(4, 6, 100, generator=torch.Generator().manual_seed(int(p * 10)))
        _assert_same_filter(run_top_p_filter(logits, p), _reference_top_p(logits, p))

    @pytest.mark.parametrize("p", [0.2, 0.5, 0.9])
    def test_kept_set_is_minimal(self, p):
        """Kept mass ≥ p, and removing the least probable kept token drops it below p."""
        logits = torch.randn(32, 40, generator=torch.Generator().manual_seed(3))
        out = run_top_p_filter(logits, p)
        probs = F.softmax(logits.double(), dim=-1)
        kept = torch.isfinite(out)
        kept_mass = (probs * kept).sum(-1)
        smallest_kept = torch.where(kept, probs, torch.inf).min(-1).values
        assert (kept_mass >= p - 1e-6).all()
        assert (kept_mass - smallest_kept < p).all()

    def test_tiny_p_keeps_only_argmax(self):
        logits = torch.randn(16, 30, generator=torch.Generator().manual_seed(4))
        out = run_top_p_filter(logits, 1e-9)
        assert (torch.isfinite(out).sum(-1) == 1).all()
        assert torch.equal(out.argmax(-1), logits.argmax(-1))

    def test_p_one_keeps_everything(self):
        logits = torch.randn(4, 30, generator=torch.Generator().manual_seed(5))
        assert torch.isfinite(run_top_p_filter(logits, 1.0)).all()

    def test_kept_logits_unchanged_and_input_not_modified(self):
        logits = torch.randn(4, 20, generator=torch.Generator().manual_seed(6))
        before = logits.clone()
        out = run_top_p_filter(logits, 0.7)
        assert torch.equal(logits, before)
        kept = torch.isfinite(out)
        assert torch.equal(out[kept], logits[kept])


# ============================================================================
# Sampling one token
# ============================================================================


class TestSampleNextToken:
    def test_temperature_zero_is_argmax(self):
        logits = torch.randn(3, 5, 40, generator=torch.Generator().manual_seed(0))
        out = run_sample_next_token(logits, temperature=0.0)
        assert out.shape == (3, 5) and out.dtype == torch.long
        assert torch.equal(out, logits.argmax(-1))
        # Greedy ignores the filters.
        assert torch.equal(run_sample_next_token(logits, 0.0, top_k=3, top_p=0.5), logits.argmax(-1))

    def test_output_shape_and_dtype(self):
        logits = torch.randn(2, 3, 4, 10)
        out = run_sample_next_token(logits, temperature=1.0, top_k=5, top_p=0.9)
        assert out.shape == (2, 3, 4) and out.dtype == torch.long
        assert ((out >= 0) & (out < 10)).all()

    @pytest.mark.parametrize(
        "temperature, top_k, top_p",
        [
            (1.0, None, None),
            (0.5, None, None),
            (1.5, None, None),
            (1.0, 3, None),
            (1.0, None, 0.7),
            (0.7, 5, 0.8),
            (1.5, 4, 0.9),
        ],
    )
    def test_distribution_matches_pipeline(self, temperature, top_k, top_p):
        n = 50_000
        torch.manual_seed(0)
        samples = run_sample_next_token(LOGITS.expand(n, -1), temperature, top_k=top_k, top_p=top_p)
        freqs = np.bincount(samples.numpy(), minlength=len(LOGITS)) / n
        expected = reference_sampling_probs(LOGITS, temperature, top_k, top_p)
        print(
            f"\nT={temperature} k={top_k} p={top_p}\n  empirical={np.round(freqs, 3)}\n  expected ={np.round(expected, 3)}"
        )
        assert (freqs[expected == 0] == 0).all(), "Sampled a token that should have been filtered out"
        np.testing.assert_allclose(freqs, expected, atol=0.012)

    def test_top_p_is_applied_after_temperature(self):
        """At T = 2 the distribution is flatter, so p = 0.6 needs more tokens than at T = 1."""
        n = 20_000
        torch.manual_seed(0)
        cold = set(run_sample_next_token(LOGITS.expand(n, -1), 1.0, top_p=0.6).tolist())
        hot = set(run_sample_next_token(LOGITS.expand(n, -1), 2.0, top_p=0.6).tolist())
        assert cold == {2, 4}
        assert hot == {0, 2, 4}

    def test_seed_reproducible(self):
        logits = torch.randn(64, 50, generator=torch.Generator().manual_seed(1))
        torch.manual_seed(42)
        a = run_sample_next_token(logits, 1.0, top_k=20, top_p=0.9)
        torch.manual_seed(42)
        b = run_sample_next_token(logits, 1.0, top_k=20, top_p=0.9)
        assert torch.equal(a, b)

    def test_numerically_stable(self):
        logits = torch.randn(256, 50, generator=torch.Generator().manual_seed(2)) * 1e4
        # A tiny temperature behaves like greedy instead of producing NaNs.
        assert torch.equal(run_sample_next_token(logits / 1e4, temperature=1e-4), logits.argmax(-1))
        out = run_sample_next_token(logits, temperature=1.0, top_p=0.9)
        assert torch.equal(out, logits.argmax(-1))


# ============================================================================
# generate(): toy models (no KV-cache)
# ============================================================================


def _decode(model, prompt, n, **kw):
    kw.setdefault("context_length", 64)
    kw.setdefault("use_kv_cache", False)
    return run_decode(model, prompt, n, **kw)


class TestDecodeToyModels:
    def test_returns_only_new_tokens(self):
        out = _decode(BigramModel(20), [1, 2, 3], 10, temperature=1.0, top_k=5, top_p=0.9)
        assert len(out) == 10
        assert all(isinstance(t, int) and 0 <= t < 20 for t in out)

    def test_greedy_follows_argmax(self):
        model = BigramModel(30, seed=1)
        out = _decode(model, [5], 15, temperature=0.0)
        expected, tok = [], 5
        for _ in range(15):
            tok = int(model.table[tok].argmax())
            expected.append(tok)
        assert out == expected

    @pytest.mark.parametrize("filters", [{"top_k": 1}, {"top_p": 1e-6}, {"top_k": 1, "top_p": 0.9}])
    def test_degenerate_filters_equal_greedy(self, filters):
        model = BigramModel(40, seed=2)
        greedy = _decode(model, [0], 25, temperature=0.0)
        torch.manual_seed(0)
        assert _decode(model, [0], 25, temperature=1.0, **filters) == greedy

    def test_combined_strategy_distribution(self):
        model = FixedLogitsModel(LOGITS)
        torch.manual_seed(0)
        n = 5_000
        out = _decode(model, [0], n, temperature=0.8, top_k=4, top_p=0.9, context_length=8)
        freqs = np.bincount(out, minlength=len(LOGITS)) / n
        expected = reference_sampling_probs(LOGITS, 0.8, top_k=4, top_p=0.9)
        print(f"\nempirical={np.round(freqs, 3)}  expected={np.round(expected, 3)}")
        assert (freqs[expected == 0] == 0).all()
        np.testing.assert_allclose(freqs, expected, atol=0.03)

    def test_seed_makes_sampling_reproducible(self):
        model = BigramModel(50, seed=3)
        torch.manual_seed(123)
        a = _decode(model, [0, 1], 30, temperature=0.9, top_k=10, top_p=0.95)
        torch.manual_seed(123)
        b = _decode(model, [0, 1], 30, temperature=0.9, top_k=10, top_p=0.95)
        assert a == b

    def test_crops_to_context_length(self):
        model = BigramModel(10)
        _decode(model, [1, 2, 3, 4, 5], 20, temperature=1.0, context_length=8)
        assert model.max_seen_len == 8, f"Model saw sequences of length {model.max_seen_len}, expected max 8"

    def test_diversity_grows_with_temperature(self):
        """10 completions from the same prompt for each T ∈ {0.1, 0.5, 1.0, 1.5}."""
        model = BigramModel(64, seed=4, scale=3.0)
        diversity = {}
        for temperature in (0.0, 0.1, 0.5, 1.0, 1.5):
            torch.manual_seed(0)
            completions = [_decode(model, [7], 40, temperature=temperature) for _ in range(10)]
            diversity[temperature] = (len({tuple(c) for c in completions}), distinct_n(completions, 2))
        print("\n  T     unique/10  distinct-2")
        for t, (u, d) in diversity.items():
            print(f"  {t:<5} {u:>9} {d:>11.3f}")
        assert diversity[0.0][0] == 1, "Greedy decoding must give 10 identical completions"
        d2 = [diversity[t][1] for t in (0.1, 0.5, 1.0, 1.5)]
        assert d2[0] < d2[1] < d2[2] < d2[3], f"distinct-2 should grow with temperature, got {d2}"


# ============================================================================
# generate(): Mini-Llama with KV-cache
# ============================================================================


KV_PROMPT = [3, 1, 4, 1, 5]


class TestDecodeKVCache:
    def _both(self, model, n, **kw):
        out = {}
        for use_cache in (False, True):
            torch.manual_seed(0)
            out[use_cache] = run_decode(
                model, KV_PROMPT, n, context_length=SMALL_CONFIG["context_length"], use_kv_cache=use_cache, **kw
            )
        return out[False], out[True]

    def test_greedy_cached_matches_uncached(self, small_llama):
        uncached, cached = self._both(small_llama, 40, temperature=0.0)
        assert len(cached) == 40
        assert cached == uncached

    @pytest.mark.parametrize("kw", [{"temperature": 1.0}, {"temperature": 0.8, "top_k": 20, "top_p": 0.9}])
    def test_sampling_cached_matches_uncached(self, small_llama, kw):
        uncached, cached = self._both(small_llama, 40, **kw)
        assert cached == uncached

    def test_cache_is_reset_between_calls(self, small_llama):
        kw = {"temperature": 0.0, "context_length": SMALL_CONFIG["context_length"], "use_kv_cache": True}
        a = run_decode(small_llama, KV_PROMPT, 20, **kw)
        b = run_decode(small_llama, [9, 9], 20, **kw)
        c = run_decode(small_llama, KV_PROMPT, 20, **kw)
        assert a == c, "A second call with the same prompt gave different tokens. Is the KV-cache reset?"
        assert b == run_decode(small_llama, [9, 9], 20, **{**kw, "use_kv_cache": False})


PERF_CONFIG = dict(SMALL_CONFIG, vocab_size=512, context_length=512, d_model=256, num_layers=4, num_heads=8, d_ff=688)


class TestDecodePerformance:
    def test_kv_cache_speedup(self):
        torch.manual_seed(0)
        model = get_mini_llama(**PERF_CONFIG).eval()
        prompt, n_new = list(range(16)), 400

        def tokens_per_sec(use_cache):
            kw = {"temperature": 0.0, "context_length": PERF_CONFIG["context_length"], "use_kv_cache": use_cache}
            run_decode(model, prompt, 8, **kw)  # warm-up
            t0 = time.perf_counter()
            run_decode(model, prompt, n_new, **kw)
            return n_new / (time.perf_counter() - t0)

        with torch.no_grad():
            slow, fast = tokens_per_sec(False), tokens_per_sec(True)
        print(
            f"\ngenerate {n_new} tokens: no-cache {slow:.0f} tok/s   KV-cache {fast:.0f} tok/s   ({fast / slow:.1f}x)"
        )
        assert fast >= 2.0 * slow, f"KV-cached generation should be ≥2x faster (got {fast / slow:.1f}x)"
