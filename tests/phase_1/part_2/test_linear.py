"""Phase 1 · Part 2 · Task 2 — Linear Layers & Initialization

Tests validate MyLinear (xW^T + b) against PyTorch, and verify that
Kaiming/Xavier initialization keeps activation variance near 1.0.
"""

import time

import numpy as np
import torch
import pytest
from tqdm import tqdm

from tests.adapters import run_linear, run_kaiming_init, run_xavier_init
from tests.common import FIXTURES_PATH


class TestLinearAccuracy:
    @pytest.fixture
    def linear_data(self):
        input_data = np.load(FIXTURES_PATH / "linear_input.npy")
        weights = np.load(FIXTURES_PATH / "linear_weights.npy")
        d_out, d_in = weights.shape
        return {"input": input_data, "weights": weights, "d_in": d_in, "d_out": d_out}

    def test_linear_output_shape(self, linear_data):
        inp = torch.from_numpy(linear_data["input"])
        w = torch.from_numpy(linear_data["weights"])
        result = run_linear(linear_data["d_in"], linear_data["d_out"], w, inp)
        expected_shape = (linear_data["input"].shape[0], linear_data["d_out"])
        assert result.shape == expected_shape

    def test_linear_matches_pytorch(self, linear_data):
        inp = torch.from_numpy(linear_data["input"])
        w = torch.from_numpy(linear_data["weights"])
        result = run_linear(linear_data["d_in"], linear_data["d_out"], w, inp)
        expected = torch.nn.functional.linear(inp, w, bias=None)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), rtol=1e-5, atol=1e-8)

    def test_linear_random_sizes(self):
        torch.manual_seed(42)
        for d_in, d_out, batch in [(10, 5, 8), (64, 32, 16), (128, 256, 4)]:
            w = torch.randn(d_out, d_in)
            x = torch.randn(batch, d_in)
            result = run_linear(d_in, d_out, w, x)
            expected = torch.nn.functional.linear(x, w, bias=None)
            np.testing.assert_allclose(
                result.detach().numpy(), expected.numpy(), rtol=1e-5, atol=1e-7,
                err_msg=f"Mismatch for d_in={d_in}, d_out={d_out}"
            )


class TestInitialization:
    def test_kaiming_shape(self):
        w = run_kaiming_init(fan_in=256, fan_out=128)
        assert w.shape == (128, 256)

    def test_kaiming_variance(self):
        """Variance should be approximately 2 / fan_in."""
        fan_in = 512
        w = run_kaiming_init(fan_in=fan_in, fan_out=256)
        expected_var = 2.0 / fan_in
        actual_var = w.var().item()
        assert abs(actual_var - expected_var) / expected_var < 0.3, (
            f"Kaiming variance: expected ~{expected_var:.6f}, got {actual_var:.6f}"
        )

    def test_xavier_shape(self):
        w = run_xavier_init(fan_in=256, fan_out=128)
        assert w.shape == (128, 256)

    def test_xavier_variance(self):
        """Variance should be approximately 2 / (fan_in + fan_out)."""
        fan_in, fan_out = 512, 256
        w = run_xavier_init(fan_in=fan_in, fan_out=fan_out)
        expected_var = 2.0 / (fan_in + fan_out)
        actual_var = w.var().item()
        assert abs(actual_var - expected_var) / expected_var < 0.3, (
            f"Xavier variance: expected ~{expected_var:.6f}, got {actual_var:.6f}"
        )

    def test_kaiming_50_layer_variance(self):
        """Activation variance should stay near 1.0 through 50 ReLU layers."""
        torch.manual_seed(42)
        d = 256
        x = torch.randn(64, d)
        for layer_idx in range(50):
            w = run_kaiming_init(fan_in=d, fan_out=d)
            x = torch.nn.functional.relu(x @ w.T)
        var = x.var().item()
        # Variance should be in a reasonable range, not exploded or vanished
        assert 0.1 < var < 10.0, (
            f"After 50 Kaiming+ReLU layers, variance = {var:.4f} (expected ~1.0)"
        )


class TestLinearPerformance:
    def test_linear_latency(self):
        torch.manual_seed(99)
        d_in, d_out, batch = 256, 128, 64
        w = torch.randn(d_out, d_in)
        x = torch.randn(batch, d_in)
        n_iter = 2000

        ref_times = []
        for _ in tqdm(range(n_iter), desc="Ref Linear"):
            t0 = time.perf_counter()
            _ = torch.nn.functional.linear(x, w)
            ref_times.append(time.perf_counter() - t0)

        custom_times = []
        for _ in tqdm(range(n_iter), desc="Custom Linear"):
            t0 = time.perf_counter()
            _ = run_linear(d_in, d_out, w, x)
            custom_times.append(time.perf_counter() - t0)

        ratio = np.mean(custom_times) / np.mean(ref_times)
        print(f"\nLinear  ref={np.mean(ref_times)*1e6:.0f}µs  custom={np.mean(custom_times)*1e6:.0f}µs  ratio={ratio:.1f}x")
        assert ratio <= 100.0, f"Custom linear is {ratio:.1f}x slower (max 100x)"
