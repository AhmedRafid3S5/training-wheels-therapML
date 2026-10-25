"""Phase 1 · Part 2 · Task 6 — Regularization (Dropout)

Tests validate MyDropout implementing Inverted Dropout:
scale active elements by 1/(1-p) during training.
"""

import time

import numpy as np
import torch
import pytest
from tqdm import tqdm

from tests.adapters import run_dropout


class TestDropoutAccuracy:
    def test_shape_preserved(self):
        x = torch.randn(2, 3, 4, 5)
        assert run_dropout(x, 0.5).shape == x.shape

    def test_shape_1d(self):
        x = torch.randn(100)
        assert run_dropout(x, 0.3).shape == x.shape

    def test_zero_prob_passthrough(self):
        x = torch.randn(100)
        torch.testing.assert_close(run_dropout(x, 0.0), x)

    def test_full_prob_all_zeros(self):
        x = torch.randn(1000)
        torch.testing.assert_close(run_dropout(x, 1.0), torch.zeros_like(x))

    def test_drop_ratio(self):
        x = torch.ones(10_000)
        for prob in [0.1, 0.3, 0.5, 0.7]:
            out = run_dropout(x, prob)
            actual_ratio = (out == 0).float().mean().item()
            assert abs(actual_ratio - prob) < 0.05, (
                f"prob={prob}: actual drop ratio {actual_ratio:.3f}"
            )

    def test_inverted_scaling(self):
        """Non-zero elements should be scaled by 1/(1-p)."""
        x = torch.ones(10_000)
        prob = 0.5
        out = run_dropout(x, prob)
        active = out[out != 0]
        expected_scale = 1.0 / (1.0 - prob)
        torch.testing.assert_close(
            active, torch.full_like(active, expected_scale), rtol=1e-5, atol=1e-5
        )

    def test_expected_value_preserved(self):
        """E[output] ≈ E[input] due to inverted scaling."""
        x = torch.ones(50_000)
        out = run_dropout(x, 0.5)
        assert abs(out.mean().item() - x.mean().item()) < 0.1

    def test_stochastic_over_runs(self):
        x = torch.randn(5_000)
        prob = 0.3
        ratios = []
        for _ in range(100):
            out = run_dropout(x, prob)
            ratios.append((out == 0).float().mean().item())
        avg = np.mean(ratios)
        assert abs(avg - prob) < 0.02, f"Average drop ratio {avg:.3f} vs expected {prob}"

    def test_scaling_multiple_probs(self):
        x = torch.ones(10_000)
        for prob in [0.2, 0.4, 0.6, 0.8]:
            out = run_dropout(x, prob)
            active = out[out != 0]
            if active.numel() > 0:
                expected = 1.0 / (1.0 - prob)
                torch.testing.assert_close(
                    active, torch.full_like(active, expected), rtol=1e-5, atol=1e-5
                )


class TestDropoutPerformance:
    def test_dropout_latency(self):
        x = torch.randn(256, 512)
        n_iter = 3000

        ref_times = []
        drop = torch.nn.Dropout(p=0.5)
        drop.train()
        for _ in tqdm(range(n_iter), desc="Ref Dropout"):
            t0 = time.perf_counter()
            _ = drop(x)
            ref_times.append(time.perf_counter() - t0)

        custom_times = []
        for _ in tqdm(range(n_iter), desc="Custom Dropout"):
            t0 = time.perf_counter()
            _ = run_dropout(x, 0.5)
            custom_times.append(time.perf_counter() - t0)

        ratio = np.mean(custom_times) / np.mean(ref_times)
        print(f"\nDropout  ratio={ratio:.1f}x")
        assert ratio <= 100.0, f"Custom Dropout is {ratio:.1f}x slower (max 100x)"
