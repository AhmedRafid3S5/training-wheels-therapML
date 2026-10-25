"""Phase 1 · Part 1 · Task 8 — Optimization Deep Dive

Tests validate SGD, SGD+Momentum, and Adam optimizers (pure NumPy)
fitting a simple linear regression to synthetic data.
"""

import time

import numpy as np
import pytest
from tqdm import tqdm

from tests.adapters import (
    run_numpy_sgd_fit,
    run_numpy_sgd_momentum_fit,
    run_numpy_adam_fit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regression(n: int = 500, d: int = 3, seed: int = 42):
    """y = X @ w_true + noise."""
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    y = X @ w_true + rng.standard_normal(n) * 0.1
    return X, y, w_true


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

class TestNumpyOptimizersAccuracy:
    @pytest.fixture
    def regression_data(self):
        return _make_regression()

    def test_sgd_output_keys(self, regression_data):
        X, y, _ = regression_data
        result = run_numpy_sgd_fit(X, y, lr=0.01, n_steps=100)
        assert "weights" in result and "loss_history" in result

    def test_sgd_converges(self, regression_data):
        X, y, w_true = regression_data
        result = run_numpy_sgd_fit(X, y, lr=0.001, n_steps=5000)
        losses = result["loss_history"]
        assert losses[-1] < losses[0] * 0.01, (
            f"SGD did not converge: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_sgd_weights_close(self, regression_data):
        X, y, w_true = regression_data
        result = run_numpy_sgd_fit(X, y, lr=0.001, n_steps=10000)
        np.testing.assert_allclose(result["weights"], w_true, atol=0.5)

    def test_sgd_momentum_converges(self, regression_data):
        X, y, _ = regression_data
        result = run_numpy_sgd_momentum_fit(X, y, lr=0.001, momentum=0.9, n_steps=5000)
        losses = result["loss_history"]
        assert losses[-1] < losses[0] * 0.01

    def test_sgd_momentum_faster_than_vanilla(self, regression_data):
        """Momentum should reach a low loss in fewer effective steps."""
        X, y, _ = regression_data
        n_steps = 2000
        vanilla = run_numpy_sgd_fit(X, y, lr=0.001, n_steps=n_steps)
        momentum = run_numpy_sgd_momentum_fit(X, y, lr=0.001, momentum=0.9, n_steps=n_steps)
        assert momentum["loss_history"][-1] <= vanilla["loss_history"][-1] * 1.1, (
            "Momentum should converge at least as fast as vanilla SGD"
        )

    def test_adam_converges(self, regression_data):
        X, y, _ = regression_data
        result = run_numpy_adam_fit(X, y, lr=0.01, n_steps=3000)
        losses = result["loss_history"]
        assert losses[-1] < losses[0] * 0.01

    def test_adam_weights_close(self, regression_data):
        X, y, w_true = regression_data
        result = run_numpy_adam_fit(X, y, lr=0.01, n_steps=5000)
        np.testing.assert_allclose(result["weights"], w_true, atol=0.5)

    def test_loss_history_length(self, regression_data):
        X, y, _ = regression_data
        n_steps = 1000
        result = run_numpy_sgd_fit(X, y, lr=0.01, n_steps=n_steps)
        assert len(result["loss_history"]) == n_steps


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------

class TestNumpyOptimizersPerformance:
    def test_sgd_latency(self):
        X, y, _ = _make_regression(n=1000, d=10, seed=99)
        n_iter = 50
        times = []
        for _ in tqdm(range(n_iter), desc="SGD"):
            t0 = time.perf_counter()
            _ = run_numpy_sgd_fit(X, y, lr=0.001, n_steps=2000)
            times.append(time.perf_counter() - t0)
        avg_ms = np.mean(times) * 1e3
        print(f"\nSGD  avg={avg_ms:.1f}ms")
        assert avg_ms < 5000, f"SGD took {avg_ms:.0f}ms (max 5000ms)"

    def test_adam_latency(self):
        X, y, _ = _make_regression(n=1000, d=10, seed=99)
        n_iter = 50
        times = []
        for _ in tqdm(range(n_iter), desc="Adam"):
            t0 = time.perf_counter()
            _ = run_numpy_adam_fit(X, y, lr=0.01, n_steps=2000)
            times.append(time.perf_counter() - t0)
        avg_ms = np.mean(times) * 1e3
        print(f"\nAdam  avg={avg_ms:.1f}ms")
        assert avg_ms < 5000, f"Adam took {avg_ms:.0f}ms (max 5000ms)"
