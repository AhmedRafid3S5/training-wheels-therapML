"""Phase 1 · Part 1 · Task 7 — Numerical Stability

Tests validate the learner's log-sum-exp implementation handles extreme values
without overflow/underflow and matches the naive computation for safe ranges.
"""

import time

import numpy as np
import pytest
from tqdm import tqdm

from tests.adapters import run_log_sum_exp


class TestLogSumExpAccuracy:
    def test_small_values(self):
        """Very small exponents must not underflow to -inf."""
        x = np.array([-1000.0, -1001.0, -1002.0])
        result = run_log_sum_exp(x)
        assert np.isfinite(result), "Result must be finite (underflow detected)"
        # Analytical: log(exp(-1000) + exp(-1001) + exp(-1002))
        #           = -1000 + log(1 + exp(-1) + exp(-2))
        expected = -1000.0 + np.log(1.0 + np.exp(-1.0) + np.exp(-2.0))
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_large_values(self):
        """Very large exponents must not overflow to inf."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = run_log_sum_exp(x)
        assert np.isfinite(result), "Result must be finite (overflow detected)"
        expected = 1002.0 + np.log(np.exp(-2.0) + np.exp(-1.0) + 1.0)
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_matches_naive_safe_range(self):
        """For moderate values, must equal naive log(sum(exp(x)))."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        expected = np.log(np.sum(np.exp(x)))
        result = run_log_sum_exp(x)
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_single_element(self):
        """log(sum(exp([a]))) = a."""
        np.testing.assert_allclose(run_log_sum_exp(np.array([5.0])), 5.0, atol=1e-12)

    def test_identical_elements(self):
        """log(sum(exp([a, a, a]))) = a + log(3)."""
        a = 42.0
        x = np.full(3, a)
        np.testing.assert_allclose(run_log_sum_exp(x), a + np.log(3.0), atol=1e-8)

    def test_mixed_extreme_values(self):
        """Mix of very large and very small values."""
        x = np.array([1000.0, -1000.0, 0.0])
        result = run_log_sum_exp(x)
        assert np.isfinite(result)
        # Dominated by exp(1000), so result ≈ 1000.0
        np.testing.assert_allclose(result, 1000.0, atol=1e-6)

    def test_all_zeros(self):
        x = np.zeros(5)
        np.testing.assert_allclose(run_log_sum_exp(x), np.log(5.0), atol=1e-8)


class TestLogSumExpPerformance:
    def test_lse_latency(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal(100_000)
        n_iter = 5000

        # Reference: scipy-style
        ref_times = []
        for _ in tqdm(range(n_iter), desc="Ref LSE"):
            t0 = time.perf_counter()
            c = x.max()
            _ = c + np.log(np.sum(np.exp(x - c)))
            ref_times.append(time.perf_counter() - t0)

        custom_times = []
        for _ in tqdm(range(n_iter), desc="Custom LSE"):
            t0 = time.perf_counter()
            _ = run_log_sum_exp(x)
            custom_times.append(time.perf_counter() - t0)

        ratio = np.mean(custom_times) / np.mean(ref_times)
        print(f"\nLSE  ref={np.mean(ref_times)*1e6:.0f}µs  custom={np.mean(custom_times)*1e6:.0f}µs  ratio={ratio:.1f}x")
        assert ratio <= 20.0, f"Custom LSE is {ratio:.1f}x slower (max 20x)"
