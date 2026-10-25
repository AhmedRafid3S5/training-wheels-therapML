"""Phase 1 · Part 2 · Task 3 — Activations

Tests validate MySigmoid, MySoftmax, MyReLU, MyGELU, MySwiGLU
against PyTorch reference implementations.
"""

import time

import numpy as np
import torch
import pytest
from tqdm import tqdm

from tests.adapters import run_sigmoid, run_relu, run_gelu, run_softmax, run_swiglu
from tests.common import FIXTURES_PATH


# ============================================================================
# Sigmoid
# ============================================================================

class TestSigmoid:
    def test_sigmoid_matches_pytorch(self):
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        result = run_sigmoid(x)
        expected = torch.sigmoid(x)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), rtol=1e-5, atol=1e-7)

    def test_sigmoid_range(self):
        x = torch.linspace(-10, 10, 100)
        result = run_sigmoid(x)
        assert (result >= 0).all() and (result <= 1).all()

    def test_sigmoid_at_zero(self):
        result = run_sigmoid(torch.tensor([0.0]))
        np.testing.assert_allclose(result.item(), 0.5, atol=1e-7)


# ============================================================================
# ReLU
# ============================================================================

class TestReLU:
    @pytest.fixture
    def relu_input(self):
        return torch.from_numpy(np.load(FIXTURES_PATH / "relu_input.npy"))

    def test_relu_matches_pytorch(self, relu_input):
        result = run_relu(relu_input)
        expected = torch.relu(relu_input)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), rtol=1e-5, atol=1e-8)

    def test_relu_negatives_zeroed(self):
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = run_relu(x)
        assert (result[:2] == 0).all()
        assert (result[3:] > 0).all()


# ============================================================================
# GELU
# ============================================================================

class TestGELU:
    @pytest.fixture
    def gelu_input(self):
        return torch.from_numpy(np.load(FIXTURES_PATH / "gelu_input.npy"))

    def test_gelu_matches_pytorch(self, gelu_input):
        result = run_gelu(gelu_input)
        expected = torch.nn.functional.gelu(gelu_input)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), rtol=1e-4, atol=1e-6)


# ============================================================================
# Softmax
# ============================================================================

class TestSoftmax:
    @pytest.fixture
    def softmax_input(self):
        return torch.from_numpy(np.load(FIXTURES_PATH / "softmax_input.npy"))

    def test_softmax_matches_pytorch(self, softmax_input):
        result = run_softmax(softmax_input, dim=1)
        expected = torch.nn.functional.softmax(softmax_input, dim=1)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), rtol=1e-5, atol=1e-8)

    def test_softmax_sums_to_one(self, softmax_input):
        result = run_softmax(softmax_input, dim=1)
        row_sums = result.sum(dim=1)
        np.testing.assert_allclose(row_sums.detach().numpy(), 1.0, atol=1e-6)

    def test_softmax_numerical_stability(self):
        """Adding a large constant should not change the result."""
        x = torch.tensor([[1000.0, 1001.0, 1002.0]])
        result = run_softmax(x, dim=1)
        assert torch.all(torch.isfinite(result)), "Softmax overflowed"
        expected = torch.nn.functional.softmax(x, dim=1)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), atol=1e-6)


# ============================================================================
# SwiGLU
# ============================================================================

class TestSwiGLU:
    @pytest.fixture
    def swiglu_data(self):
        data = {
            "input": np.load(FIXTURES_PATH / "swiglu_input.npy"),
            "w1_weight": np.load(FIXTURES_PATH / "swiglu_w1_weight.npy"),
            "w2_weight": np.load(FIXTURES_PATH / "swiglu_w2_weight.npy"),
            "w3_weight": np.load(FIXTURES_PATH / "swiglu_w3_weight.npy"),
            "expected": np.load(FIXTURES_PATH / "swiglu_output.npy"),
        }
        d_ff, d_model = data["w1_weight"].shape
        data["d_model"] = d_model
        data["d_ff"] = d_ff
        return data

    def test_swiglu_output(self, swiglu_data):
        inp = torch.from_numpy(swiglu_data["input"])
        w1 = torch.from_numpy(swiglu_data["w1_weight"])
        w2 = torch.from_numpy(swiglu_data["w2_weight"])
        w3 = torch.from_numpy(swiglu_data["w3_weight"])
        result = run_swiglu(swiglu_data["d_model"], swiglu_data["d_ff"], w1, w2, w3, inp)
        np.testing.assert_allclose(
            result.detach().numpy(), swiglu_data["expected"], rtol=1e-5, atol=1e-8
        )

    def test_swiglu_shape(self, swiglu_data):
        inp = torch.from_numpy(swiglu_data["input"])
        w1 = torch.from_numpy(swiglu_data["w1_weight"])
        w2 = torch.from_numpy(swiglu_data["w2_weight"])
        w3 = torch.from_numpy(swiglu_data["w3_weight"])
        result = run_swiglu(swiglu_data["d_model"], swiglu_data["d_ff"], w1, w2, w3, inp)
        assert result.shape == inp.shape


# ============================================================================
# Latency benchmarks
# ============================================================================

class TestActivationsPerformance:
    def _bench(self, name, ref_fn, custom_fn, x, n_iter=3000):
        ref_times = []
        for _ in tqdm(range(n_iter), desc=f"Ref {name}"):
            t0 = time.perf_counter()
            _ = ref_fn(x)
            ref_times.append(time.perf_counter() - t0)

        custom_times = []
        for _ in tqdm(range(n_iter), desc=f"Custom {name}"):
            t0 = time.perf_counter()
            _ = custom_fn(x)
            custom_times.append(time.perf_counter() - t0)

        ratio = np.mean(custom_times) / np.mean(ref_times)
        print(f"\n{name}  ratio={ratio:.1f}x")
        return ratio

    def test_relu_latency(self):
        x = torch.randn(256, 512)
        ratio = self._bench("ReLU", torch.relu, run_relu, x)
        assert ratio <= 100.0, f"Custom ReLU is {ratio:.1f}x slower (max 100x)"

    def test_softmax_latency(self):
        x = torch.randn(256, 512)
        ratio = self._bench(
            "Softmax",
            lambda t: torch.nn.functional.softmax(t, dim=1),
            lambda t: run_softmax(t, dim=1),
            x,
        )
        assert ratio <= 100.0, f"Custom Softmax is {ratio:.1f}x slower (max 100x)"

    def test_gelu_latency(self):
        x = torch.randn(256, 512)
        ratio = self._bench(
            "GELU",
            torch.nn.functional.gelu,
            run_gelu,
            x,
        )
        assert ratio <= 100.0, f"Custom GELU is {ratio:.1f}x slower (max 100x)"
