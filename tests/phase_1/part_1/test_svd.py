"""Phase 1 · Part 1 · Task 2 — SVD & Low-Rank Approximation

Tests validate:
  • Correctness: decomposition identity, shapes, error monotonicity,
    Frobenius-norm computation.
  • Latency: within 50× of NumPy's built-in SVD for the same rank.
"""

import time

import numpy as np
import pytest
from tqdm import tqdm

from tests.adapters import run_truncated_svd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix(m: int = 100, n: int = 80, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n))


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

class TestSVDAccuracy:
    @pytest.fixture
    def matrix(self):
        return _make_matrix()

    def test_svd_output_keys(self, matrix):
        result = run_truncated_svd(matrix, rank=5)
        for key in ("U", "S", "Vt", "reconstructed"):
            assert key in result, f"Missing key '{key}'"

    def test_svd_shapes(self, matrix):
        r = 5
        result = run_truncated_svd(matrix, rank=r)
        m, n = matrix.shape
        assert result["U"].shape == (m, r)
        assert result["S"].shape == (r,)
        assert result["Vt"].shape == (r, n)
        assert result["reconstructed"].shape == (m, n)

    def test_svd_full_rank_reconstruction(self):
        """Full-rank SVD should reconstruct the matrix exactly."""
        mat = _make_matrix(m=20, n=15)
        r = min(mat.shape)
        result = run_truncated_svd(mat, rank=r)
        np.testing.assert_allclose(result["reconstructed"], mat, atol=1e-8)

    def test_svd_reconstruction_identity(self, matrix):
        """U @ diag(S) @ Vt must equal the returned reconstruction."""
        r = 10
        result = run_truncated_svd(matrix, rank=r)
        manual = result["U"] @ np.diag(result["S"]) @ result["Vt"]
        np.testing.assert_allclose(manual, result["reconstructed"], atol=1e-10)

    def test_svd_singular_values_descending(self, matrix):
        result = run_truncated_svd(matrix, rank=20)
        S = result["S"]
        assert np.all(S[:-1] >= S[1:] - 1e-12), "Singular values must be non-increasing"

    def test_svd_error_decreases_with_rank(self, matrix):
        errors = []
        for r in [1, 5, 10, 50]:
            result = run_truncated_svd(matrix, rank=r)
            err = np.linalg.norm(matrix - result["reconstructed"], "fro")
            errors.append(err)
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-8, (
                f"Frobenius error should decrease with rank: rank {[1,5,10,50][i]} "
                f"({errors[i]:.4f}) vs rank {[1,5,10,50][i+1]} ({errors[i+1]:.4f})"
            )

    def test_svd_rank1_approximation(self):
        """Rank-1 approx should equal outer product of first singular vectors."""
        mat = _make_matrix(m=30, n=20, seed=7)
        result = run_truncated_svd(mat, rank=1)
        expected = result["U"] * result["S"][0] @ result["Vt"]
        np.testing.assert_allclose(result["reconstructed"], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

class TestSVDPerformance:
    def test_svd_latency(self):
        mat = _make_matrix(m=500, n=400, seed=99)
        rank = 20
        n_iter = 200

        ref_times = []
        for _ in tqdm(range(n_iter), desc="Ref SVD"):
            t0 = time.perf_counter()
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            _ = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
            ref_times.append(time.perf_counter() - t0)

        custom_times = []
        for _ in tqdm(range(n_iter), desc="Custom SVD"):
            t0 = time.perf_counter()
            _ = run_truncated_svd(mat, rank=rank)
            custom_times.append(time.perf_counter() - t0)

        ratio = np.mean(custom_times) / np.mean(ref_times)
        print(f"\nSVD  ref={np.mean(ref_times)*1e3:.2f}ms  custom={np.mean(custom_times)*1e3:.2f}ms  ratio={ratio:.1f}x")
        assert ratio <= 50.0, f"Custom SVD is {ratio:.1f}x slower (max 50x)"
