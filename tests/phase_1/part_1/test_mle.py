"""Phase 1 · Part 1 · Task 4 — Maximum Likelihood Estimation

Tests validate:
  • Correctness: gradient-descent MLE matches closed-form within tolerance.
  • Latency: convergence within a reasonable time budget.
"""

import time

import numpy as np
import pytest
from tqdm import tqdm

from tests.adapters import run_mle_gaussian


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

class TestMLEAccuracy:
    @pytest.mark.parametrize(
        "true_mu, true_sigma2, n_samples",
        [
            (0.0, 1.0, 5000),
            (5.0, 4.0, 5000),
            (-3.0, 0.25, 5000),
            (100.0, 9.0, 10000),
        ],
    )
    def test_mle_matches_closed_form(self, true_mu, true_sigma2, n_samples):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=true_mu, scale=np.sqrt(true_sigma2), size=n_samples)

        result = run_mle_gaussian(data)

        closed_mu = data.mean()
        closed_sigma2 = data.var()  # MLE variance (biased)

        assert abs(result["mu"] - closed_mu) < 0.5, (
            f"mu: expected ~{closed_mu:.4f}, got {result['mu']:.4f}"
        )
        assert abs(result["sigma2"] - closed_sigma2) < 1.0, (
            f"sigma2: expected ~{closed_sigma2:.4f}, got {result['sigma2']:.4f}"
        )

    def test_mle_output_keys(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=1000)
        result = run_mle_gaussian(data)
        assert "mu" in result and "sigma2" in result

    def test_mle_positive_variance(self):
        rng = np.random.default_rng(7)
        data = rng.normal(loc=2.0, scale=3.0, size=2000)
        result = run_mle_gaussian(data)
        assert result["sigma2"] > 0, "Variance must be positive"


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

class TestMLEPerformance:
    def test_mle_latency(self):
        rng = np.random.default_rng(99)
        data = rng.normal(loc=5.0, scale=2.0, size=5000)
        n_iter = 50

        times = []
        for _ in tqdm(range(n_iter), desc="MLE GD"):
            t0 = time.perf_counter()
            _ = run_mle_gaussian(data)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1e3
        print(f"\nMLE  avg={avg_ms:.1f}ms")
        # MLE gradient descent should finish within 5 seconds per call
        assert avg_ms < 5000, f"MLE took {avg_ms:.0f}ms per call (max 5000ms)"
