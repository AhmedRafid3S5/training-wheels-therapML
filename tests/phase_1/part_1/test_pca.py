"""Phase 1 · Part 1 · Task 1 — PCA from Scratch

Tests validate:
  • Correctness: centering, covariance, eigendecomposition, projection,
    reconstruction, and reconstruction-error monotonicity.
  • Latency: custom PCA must run within 50× of sklearn-equivalent NumPy ops.
"""

import time

import numpy as np
import pytest
from tqdm import tqdm

from tests.adapters import run_pca


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_correlated_data(n: int = 500, d: int = 10, seed: int = 42) -> np.ndarray:
    """Synthetic data with clear principal directions."""
    rng = np.random.default_rng(seed)
    # Create data where first few dimensions carry most variance
    scales = np.array([10.0, 5.0, 2.0] + [0.1] * (d - 3))
    return rng.standard_normal((n, d)) * scales


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

class TestPCAAccuracy:
    @pytest.fixture
    def data(self):
        return _make_correlated_data()

    def test_pca_output_keys(self, data):
        """Returned dict must contain the required keys."""
        result = run_pca(data, k=3)
        for key in ("projected", "components", "eigenvalues", "mean", "reconstructed"):
            assert key in result, f"Missing key '{key}' in PCA output"

    def test_pca_projected_shape(self, data):
        k = 3
        result = run_pca(data, k=k)
        assert result["projected"].shape == (data.shape[0], k), (
            f"Expected projected shape {(data.shape[0], k)}, got {result['projected'].shape}"
        )

    def test_pca_components_shape(self, data):
        k = 3
        result = run_pca(data, k=k)
        assert result["components"].shape == (k, data.shape[1])

    def test_pca_eigenvalues_descending(self, data):
        result = run_pca(data, k=5)
        evals = result["eigenvalues"]
        assert np.all(evals[:-1] >= evals[1:]), "Eigenvalues must be in descending order"

    def test_pca_mean_subtracted(self, data):
        """After centering, the projected data should have near-zero column means."""
        result = run_pca(data, k=3)
        centered = data - result["mean"]
        np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-10)

    def test_pca_reconstruction_full_rank(self, data):
        """k = d should perfectly reconstruct the original data."""
        d = data.shape[1]
        result = run_pca(data, k=d)
        np.testing.assert_allclose(result["reconstructed"], data, atol=1e-8)

    def test_pca_reconstruction_error_decreases(self, data):
        """More components ⇒ lower reconstruction error."""
        errors = []
        for k in [1, 3, 5, data.shape[1]]:
            result = run_pca(data, k=k)
            err = np.linalg.norm(data - result["reconstructed"], "fro")
            errors.append(err)
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-8, (
                f"Error at k={[1,3,5,data.shape[1]][i]} ({errors[i]:.6f}) "
                f"should be >= error at k={[1,3,5,data.shape[1]][i+1]} ({errors[i+1]:.6f})"
            )

    def test_pca_known_simple(self):
        """Simple 2-D dataset where PC1 direction is obvious."""
        rng = np.random.default_rng(0)
        # Data lies mostly along the x-axis
        data = np.column_stack([rng.standard_normal(200) * 10, rng.standard_normal(200) * 0.1])
        result = run_pca(data, k=1)
        # The first component should be roughly aligned with x-axis
        dominant = np.abs(result["components"][0])
        assert dominant[0] > dominant[1], "First component should align with high-variance axis"

    def test_pca_components_orthonormal(self, data):
        """Principal components should be orthonormal."""
        result = run_pca(data, k=5)
        V = result["components"]  # (k, d)
        gram = V @ V.T
        np.testing.assert_allclose(gram, np.eye(V.shape[0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

class TestPCAPerformance:
    def test_pca_latency(self):
        data = _make_correlated_data(n=2000, d=50, seed=99)
        k = 10
        n_iter = 200

        # Reference: raw NumPy PCA
        ref_times = []
        for _ in tqdm(range(n_iter), desc="Ref PCA"):
            t0 = time.perf_counter()
            centered = data - data.mean(axis=0)
            cov = np.cov(centered, rowvar=False)
            vals, vecs = np.linalg.eigh(cov)
            idx = np.argsort(vals)[::-1][:k]
            _ = centered @ vecs[:, idx]
            ref_times.append(time.perf_counter() - t0)

        # Custom implementation
        custom_times = []
        for _ in tqdm(range(n_iter), desc="Custom PCA"):
            t0 = time.perf_counter()
            _ = run_pca(data, k=k)
            custom_times.append(time.perf_counter() - t0)

        ratio = np.mean(custom_times) / np.mean(ref_times)
        print(f"\nPCA  ref={np.mean(ref_times)*1e3:.2f}ms  custom={np.mean(custom_times)*1e3:.2f}ms  ratio={ratio:.1f}x")
        assert ratio <= 50.0, f"Custom PCA is {ratio:.1f}x slower than reference (max 50x)"
