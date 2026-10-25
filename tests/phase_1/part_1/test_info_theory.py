"""Phase 1 · Part 1 · Task 6 — Information Theory Toolkit

Tests validate entropy, KL divergence, cross-entropy, and mutual information
against known analytical values.  All quantities are in **bits** (log base 2).
"""

import time

import numpy as np
import pytest
from tqdm import tqdm

from tests.adapters import (
    run_entropy,
    run_kl_divergence,
    run_cross_entropy_info,
    run_mutual_information,
)


# ============================================================================
# Entropy
# ============================================================================

class TestEntropy:
    def test_fair_coin(self):
        """H(fair coin) = 1 bit."""
        H = run_entropy(np.array([0.5, 0.5]))
        np.testing.assert_allclose(H, 1.0, atol=1e-8)

    def test_certain_outcome(self):
        """H(deterministic) = 0."""
        H = run_entropy(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(H, 0.0, atol=1e-8)

    def test_uniform_distribution(self):
        """H(uniform over n) = log2(n)."""
        n = 8
        P = np.ones(n) / n
        H = run_entropy(P)
        np.testing.assert_allclose(H, np.log2(n), atol=1e-8)

    def test_entropy_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            P = rng.dirichlet(np.ones(5))
            assert run_entropy(P) >= -1e-12, "Entropy must be non-negative"


# ============================================================================
# KL Divergence
# ============================================================================

class TestKLDivergence:
    def test_same_distribution(self):
        """KL(P || P) = 0."""
        P = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(run_kl_divergence(P, P)) < 1e-8

    def test_nonnegative(self):
        """Gibbs' inequality: KL >= 0."""
        rng = np.random.default_rng(1)
        for _ in range(20):
            P = rng.dirichlet(np.ones(4))
            Q = rng.dirichlet(np.ones(4))
            assert run_kl_divergence(P, Q) >= -1e-8

    def test_known_value(self):
        """KL([0.5, 0.5] || [0.25, 0.75]) has a known closed-form."""
        P = np.array([0.5, 0.5])
        Q = np.array([0.25, 0.75])
        expected = 0.5 * np.log2(0.5 / 0.25) + 0.5 * np.log2(0.5 / 0.75)
        np.testing.assert_allclose(run_kl_divergence(P, Q), expected, atol=1e-8)

    def test_asymmetry(self):
        """KL is not symmetric: KL(P||Q) != KL(Q||P) in general."""
        P = np.array([0.9, 0.1])
        Q = np.array([0.1, 0.9])
        assert abs(run_kl_divergence(P, Q) - run_kl_divergence(Q, P)) > 0.01


# ============================================================================
# Cross-Entropy
# ============================================================================

class TestCrossEntropy:
    def test_cross_entropy_equals_entropy_when_same(self):
        """H(P, P) = H(P)."""
        P = np.array([0.3, 0.7])
        np.testing.assert_allclose(
            run_cross_entropy_info(P, P),
            run_entropy(P),
            atol=1e-8,
        )

    def test_cross_entropy_geq_entropy(self):
        """H(P, Q) >= H(P) (Gibbs' inequality)."""
        rng = np.random.default_rng(2)
        for _ in range(20):
            P = rng.dirichlet(np.ones(5))
            Q = rng.dirichlet(np.ones(5))
            assert run_cross_entropy_info(P, Q) >= run_entropy(P) - 1e-8

    def test_cross_entropy_known(self):
        P = np.array([0.5, 0.5])
        Q = np.array([0.25, 0.75])
        expected = -(0.5 * np.log2(0.25) + 0.5 * np.log2(0.75))
        np.testing.assert_allclose(run_cross_entropy_info(P, Q), expected, atol=1e-8)

    def test_relation_to_kl(self):
        """H(P, Q) = H(P) + KL(P||Q)."""
        P = np.array([0.6, 0.3, 0.1])
        Q = np.array([0.2, 0.5, 0.3])
        np.testing.assert_allclose(
            run_cross_entropy_info(P, Q),
            run_entropy(P) + run_kl_divergence(P, Q),
            atol=1e-8,
        )


# ============================================================================
# Mutual Information
# ============================================================================

class TestMutualInformation:
    def test_independent_variables(self):
        """I(X; Y) = 0 for independent X, Y."""
        # Joint = product of marginals → independent
        px = np.array([0.5, 0.5])
        py = np.array([0.3, 0.7])
        joint = np.outer(px, py)
        np.testing.assert_allclose(run_mutual_information(joint), 0.0, atol=1e-8)

    def test_perfect_correlation(self):
        """I(X; Y) = H(X) when Y = X."""
        joint = np.diag([0.25, 0.25, 0.25, 0.25])
        mi = run_mutual_information(joint)
        np.testing.assert_allclose(mi, np.log2(4), atol=1e-8)

    def test_nonnegative(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            raw = rng.dirichlet(np.ones(9)).reshape(3, 3)
            assert run_mutual_information(raw) >= -1e-8

    def test_known_joint(self):
        """Hand-computed example."""
        joint = np.array([[0.1, 0.2], [0.3, 0.4]])
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        expected = 0.0
        for i in range(2):
            for j in range(2):
                if joint[i, j] > 0:
                    expected += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
        np.testing.assert_allclose(run_mutual_information(joint), expected, atol=1e-8)


# ============================================================================
# Latency
# ============================================================================

class TestInfoTheoryPerformance:
    def test_entropy_latency(self):
        P = np.ones(10000) / 10000
        n_iter = 5000
        times = []
        for _ in tqdm(range(n_iter), desc="Entropy"):
            t0 = time.perf_counter()
            _ = run_entropy(P)
            times.append(time.perf_counter() - t0)
        avg_us = np.mean(times) * 1e6
        print(f"\nEntropy  avg={avg_us:.1f}µs")
        assert avg_us < 50_000, f"Entropy too slow: {avg_us:.0f}µs (max 50ms)"

    def test_mutual_information_latency(self):
        rng = np.random.default_rng(0)
        joint = rng.dirichlet(np.ones(10000)).reshape(100, 100)
        n_iter = 1000
        times = []
        for _ in tqdm(range(n_iter), desc="MI"):
            t0 = time.perf_counter()
            _ = run_mutual_information(joint)
            times.append(time.perf_counter() - t0)
        avg_us = np.mean(times) * 1e6
        print(f"\nMI  avg={avg_us:.1f}µs")
        assert avg_us < 100_000, f"MI too slow: {avg_us:.0f}µs (max 100ms)"
