"""Phase 1 · Part 2 · Task 4 — Loss Functions

Tests validate MyCrossEntropyLoss and MyKLDivergenceLoss
against PyTorch reference implementations.
"""

import time

import numpy as np
import torch
import pytest
from tqdm import tqdm

from tests.adapters import run_cross_entropy_loss, run_kl_divergence_loss
from tests.common import FIXTURES_PATH


# ============================================================================
# Cross-Entropy Loss
# ============================================================================

class TestCrossEntropyLoss:
    @pytest.fixture
    def ce_data(self):
        logits = torch.from_numpy(np.load(FIXTURES_PATH / "cross_entropy_logits.npy"))
        gt = torch.from_numpy(np.load(FIXTURES_PATH / "cross_entropy_ground_truth.npy"))
        return logits, gt

    def test_ce_matches_pytorch(self, ce_data):
        logits, gt = ce_data
        result = run_cross_entropy_loss(logits, gt)
        result_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result
        expected = torch.nn.functional.cross_entropy(logits, gt)
        np.testing.assert_allclose(result_np, expected.numpy(), atol=1e-4)

    def test_ce_scalar_output(self, ce_data):
        logits, gt = ce_data
        result = run_cross_entropy_loss(logits, gt)
        result_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result
        assert result_np.ndim <= 1

    def test_ce_numerical_stability(self, ce_data):
        """Loss should be invariant to adding a constant to all logits."""
        logits, gt = ce_data
        r1 = run_cross_entropy_loss(logits, gt)
        r2 = run_cross_entropy_loss(logits + 1e4, gt)
        r1_np = r1.detach().numpy() if isinstance(r1, torch.Tensor) else r1
        r2_np = r2.detach().numpy() if isinstance(r2, torch.Tensor) else r2
        np.testing.assert_allclose(r1_np, r2_np, atol=1e-4)

    def test_ce_random_inputs(self):
        torch.manual_seed(42)
        logits = torch.randn(16, 10)
        gt = torch.nn.functional.one_hot(torch.randint(0, 10, (16,)), 10).float()
        result = run_cross_entropy_loss(logits, gt)
        # Cross-entropy of random logits should be positive
        assert result.item() > 0

    def test_ce_tolerance_1e6(self):
        """PDF requirement: within 1e-6 of torch.nn.CrossEntropyLoss."""
        torch.manual_seed(7)
        logits = torch.randn(32, 20)
        targets = torch.nn.functional.one_hot(torch.randint(0, 20, (32,)), 20).float()
        result = run_cross_entropy_loss(logits, targets)
        expected = torch.nn.functional.cross_entropy(logits, targets)
        r_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result
        np.testing.assert_allclose(r_np, expected.numpy(), atol=1e-6)


# ============================================================================
# KL Divergence Loss
# ============================================================================

class TestKLDivergenceLoss:
    def test_kl_matches_pytorch(self):
        torch.manual_seed(42)
        logits = torch.randn(16, 10)
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        target_probs = torch.nn.functional.softmax(torch.randn(16, 10), dim=1)
        result = run_kl_divergence_loss(log_probs, target_probs)
        expected = torch.nn.functional.kl_div(log_probs, target_probs, reduction="batchmean")
        r_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result
        np.testing.assert_allclose(r_np, expected.numpy(), atol=1e-5)

    def test_kl_same_distribution_zero(self):
        torch.manual_seed(0)
        probs = torch.nn.functional.softmax(torch.randn(8, 5), dim=1)
        log_probs = torch.log(probs)
        result = run_kl_divergence_loss(log_probs, probs)
        r_val = result.item() if isinstance(result, torch.Tensor) else float(result)
        assert abs(r_val) < 1e-5, f"KL(P||P) should be ~0, got {r_val}"

    def test_kl_nonnegative(self):
        torch.manual_seed(1)
        for _ in range(10):
            log_p = torch.nn.functional.log_softmax(torch.randn(8, 6), dim=1)
            q = torch.nn.functional.softmax(torch.randn(8, 6), dim=1)
            result = run_kl_divergence_loss(log_p, q)
            r_val = result.item() if isinstance(result, torch.Tensor) else float(result)
            assert r_val >= -1e-5, f"KL divergence should be non-negative, got {r_val}"


# ============================================================================
# Latency
# ============================================================================

class TestLossPerformance:
    def test_ce_latency(self):
        torch.manual_seed(99)
        logits = torch.randn(128, 100)
        gt = torch.nn.functional.one_hot(torch.randint(0, 100, (128,)), 100).float()
        n_iter = 3000

        ref_times = []
        for _ in tqdm(range(n_iter), desc="Ref CE"):
            t0 = time.perf_counter()
            _ = torch.nn.functional.cross_entropy(logits, gt)
            ref_times.append(time.perf_counter() - t0)

        custom_times = []
        for _ in tqdm(range(n_iter), desc="Custom CE"):
            t0 = time.perf_counter()
            _ = run_cross_entropy_loss(logits, gt)
            custom_times.append(time.perf_counter() - t0)

        ratio = np.mean(custom_times) / np.mean(ref_times)
        print(f"\nCE  ratio={ratio:.1f}x")
        assert ratio <= 100.0, f"Custom CE is {ratio:.1f}x slower (max 100x)"
