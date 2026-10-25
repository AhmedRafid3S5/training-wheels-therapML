"""Phase 1 · Part 2 · Task 1 — Autograd Intuition

Tests validate the learner's minimal scalar autograd engine (Value class)
with forward-pass correctness and reverse-mode gradient computation.
"""

import time
import math

import numpy as np
import pytest
from tqdm import tqdm

from tests.adapters import get_value_class


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

class TestAutograd:
    @pytest.fixture
    def Value(self):
        return get_value_class()

    # -- Forward pass --------------------------------------------------------

    def test_addition(self, Value):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert abs(c.data - 5.0) < 1e-8

    def test_multiplication(self, Value):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert abs(c.data - 6.0) < 1e-8

    def test_tanh(self, Value):
        a = Value(0.5)
        c = a.tanh()
        assert abs(c.data - math.tanh(0.5)) < 1e-6

    def test_chained_ops(self, Value):
        a = Value(2.0)
        b = Value(-3.0)
        c = a * b + Value(10.0)
        assert abs(c.data - 4.0) < 1e-8

    # -- Backward pass (gradients) -------------------------------------------

    def test_backward_simple_multiply(self, Value):
        """d/da (a * b) at a=3, b=4 → grad_a=4, grad_b=3."""
        a = Value(3.0)
        b = Value(4.0)
        c = a * b
        c.backward()
        assert abs(a.grad - 4.0) < 1e-6, f"Expected grad_a=4, got {a.grad}"
        assert abs(b.grad - 3.0) < 1e-6, f"Expected grad_b=3, got {b.grad}"

    def test_backward_addition(self, Value):
        """d/da (a + b) = 1 for both."""
        a = Value(2.0)
        b = Value(5.0)
        c = a + b
        c.backward()
        assert abs(a.grad - 1.0) < 1e-6
        assert abs(b.grad - 1.0) < 1e-6

    def test_backward_square(self, Value):
        """d/dx (x^2) at x=3 → 6."""
        x = Value(3.0)
        y = x * x
        y.backward()
        assert abs(x.grad - 6.0) < 1e-6, f"Expected grad=6, got {x.grad}"

    def test_backward_complex_graph(self, Value):
        """Test a small computation graph: (a*b + c) * tanh(d)."""
        a = Value(1.0)
        b = Value(2.0)
        c = Value(3.0)
        d = Value(0.5)

        e = a * b      # 2
        f = e + c       # 5
        g = d.tanh()    # tanh(0.5)
        h = f * g       # 5 * tanh(0.5)

        h.backward()

        # Numerical gradient check
        eps = 1e-5
        for val, name in [(a, "a"), (b, "b"), (c, "c"), (d, "d")]:
            # Reset
            a2 = Value(1.0); b2 = Value(2.0); c2 = Value(3.0); d2 = Value(0.5)
            vals = {"a": a2, "b": b2, "c": c2, "d": d2}

            def _forward(vs):
                return (vs["a"] * vs["b"] + vs["c"]) * vs["d"].tanh()

            orig = vals[name].data
            vals[name] = Value(orig + eps)
            h_plus = _forward(vals).data
            vals[name] = Value(orig - eps)
            h_minus = _forward(vals).data
            numerical_grad = (h_plus - h_minus) / (2 * eps)

            assert abs(val.grad - numerical_grad) < 1e-4, (
                f"Gradient mismatch for {name}: analytic={val.grad:.6f}, numerical={numerical_grad:.6f}"
            )

    def test_backward_tanh(self, Value):
        """d/dx tanh(x) = 1 - tanh(x)^2."""
        x = Value(0.8)
        y = x.tanh()
        y.backward()
        expected = 1.0 - math.tanh(0.8) ** 2
        assert abs(x.grad - expected) < 1e-6


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

class TestAutogradPerformance:
    def test_autograd_latency(self):
        Value = get_value_class()
        n_iter = 5000

        times = []
        for _ in tqdm(range(n_iter), desc="Autograd fwd+bwd"):
            t0 = time.perf_counter()
            a = Value(1.5)
            b = Value(2.5)
            c = a * b + Value(3.0)
            d = c.tanh()
            e = d * Value(4.0)
            e.backward()
            times.append(time.perf_counter() - t0)

        avg_us = np.mean(times) * 1e6
        print(f"\nAutograd fwd+bwd  avg={avg_us:.1f}µs")
        assert avg_us < 5000, f"Autograd too slow: {avg_us:.0f}µs per fwd+bwd (max 5ms)"
