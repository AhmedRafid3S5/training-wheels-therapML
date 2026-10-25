"""Phase 1 · Part 2 · Task 7 — Normalization & Residuals

Tests validate MyLayerNorm and MyRMSNorm against reference implementations
using pre-saved fixture data and PyTorch equivalents.
"""

import time

import numpy as np
import torch
import pytest
from tqdm import tqdm

from tests.adapters import run_layernorm, run_rmsnorm
from tests.common import FIXTURES_PATH


# ============================================================================
# LayerNorm
# ============================================================================

class TestLayerNorm:
    @pytest.fixture
    def ln_data(self):
        inp = torch.from_numpy(np.load(FIXTURES_PATH / "layernorm_input.npy"))
        gamma = torch.from_numpy(np.load(FIXTURES_PATH / "layernorm_gamma.npy"))
        beta = torch.from_numpy(np.load(FIXTURES_PATH / "layernorm_beta.npy"))
        return inp, gamma, beta

    def test_layernorm_matches_pytorch(self, ln_data):
        inp, gamma, beta = ln_data
        result = run_layernorm(inp, gamma, beta)
        # Compute reference with PyTorch
        norm_shape = inp.shape[1:]
        ln = torch.nn.LayerNorm(norm_shape, elementwise_affine=False)
        expected = ln(inp) * gamma + beta
        np.testing.assert_allclose(
            result.detach().numpy(), expected.detach().numpy(), atol=1e-5
        )

    def test_layernorm_shape(self, ln_data):
        inp, gamma, beta = ln_data
        result = run_layernorm(inp, gamma, beta)
        assert result.shape == inp.shape

    def test_layernorm_random(self):
        torch.manual_seed(42)
        inp = torch.randn(8, 64)
        gamma = torch.ones(64)
        beta = torch.zeros(64)
        result = run_layernorm(inp, gamma, beta)
        ln = torch.nn.LayerNorm(64, elementwise_affine=False)
        expected = ln(inp)
        np.testing.assert_allclose(
            result.detach().numpy(), expected.detach().numpy(), atol=1e-5
        )

    def test_layernorm_zero_mean_unit_var(self):
        """With gamma=1, beta=0, output should have ~0 mean, ~1 var per sample."""
        torch.manual_seed(7)
        inp = torch.randn(16, 128)
        gamma = torch.ones(128)
        beta = torch.zeros(128)
        result = run_layernorm(inp, gamma, beta)
        means = result.mean(dim=-1)
        vars_ = result.var(dim=-1, correction=0)
        np.testing.assert_allclose(means.detach().numpy(), 0.0, atol=1e-5)
        np.testing.assert_allclose(vars_.detach().numpy(), 1.0, atol=1e-4)


# ============================================================================
# RMSNorm
# ============================================================================

class TestRMSNorm:
    @pytest.fixture
    def rms_data(self):
        inp = torch.from_numpy(np.load(FIXTURES_PATH / "rmsnorm_input.npy"))
        gamma = torch.from_numpy(np.load(FIXTURES_PATH / "rmsnorm_gamma.npy"))
        return inp, gamma

    def test_rmsnorm_shape(self, rms_data):
        inp, gamma = rms_data
        result = run_rmsnorm(inp, gamma)
        assert result.shape == inp.shape

    def test_rmsnorm_known_computation(self, rms_data):
        """RMSNorm(x) = x / RMS(x) * gamma where RMS = sqrt(mean(x^2) + eps)."""
        inp, gamma = rms_data
        result = run_rmsnorm(inp, gamma)
        eps = 1e-5
        rms = torch.sqrt(inp.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected = (inp / rms) * gamma
        np.testing.assert_allclose(
            result.detach().numpy(), expected.detach().numpy(), atol=1e-5
        )

    def test_rmsnorm_random(self):
        torch.manual_seed(42)
        inp = torch.randn(8, 64)
        gamma = torch.ones(64)
        result = run_rmsnorm(inp, gamma)
        eps = 1e-5
        rms = torch.sqrt(inp.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected = inp / rms
        np.testing.assert_allclose(
            result.detach().numpy(), expected.detach().numpy(), atol=1e-5
        )


# ============================================================================
# Latency
# ============================================================================

class TestNormPerformance:
    def test_layernorm_latency(self):
        torch.manual_seed(99)
        inp = torch.randn(64, 256)
        gamma = torch.ones(256)
        beta = torch.zeros(256)
        n_iter = 3000

        ref_ln = torch.nn.LayerNorm(256)
        ref_times = []
        for _ in tqdm(range(n_iter), desc="Ref LN"):
            t0 = time.perf_counter()
            _ = ref_ln(inp)
            ref_times.append(time.perf_counter() - t0)

        custom_times = []
        for _ in tqdm(range(n_iter), desc="Custom LN"):
            t0 = time.perf_counter()
            _ = run_layernorm(inp, gamma, beta)
            custom_times.append(time.perf_counter() - t0)

        ratio = np.mean(custom_times) / np.mean(ref_times)
        print(f"\nLayerNorm  ratio={ratio:.1f}x")
        assert ratio <= 100.0, f"Custom LayerNorm is {ratio:.1f}x slower (max 100x)"

    def test_rmsnorm_latency(self):
        torch.manual_seed(99)
        inp = torch.randn(64, 256)
        gamma = torch.ones(256)
        n_iter = 3000

        # Reference: manual RMS norm
        ref_times = []
        for _ in tqdm(range(n_iter), desc="Ref RMS"):
            t0 = time.perf_counter()
            rms = torch.sqrt(inp.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            _ = inp / rms * gamma
            ref_times.append(time.perf_counter() - t0)

        custom_times = []
        for _ in tqdm(range(n_iter), desc="Custom RMS"):
            t0 = time.perf_counter()
            _ = run_rmsnorm(inp, gamma)
            custom_times.append(time.perf_counter() - t0)

        ratio = np.mean(custom_times) / np.mean(ref_times)
        print(f"\nRMSNorm  ratio={ratio:.1f}x")
        assert ratio <= 100.0, f"Custom RMSNorm is {ratio:.1f}x slower (max 100x)"
