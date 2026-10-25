"""Phase 1 · Part 2 · Task 5 — PyTorch-compatible Optimizers

Tests validate MySGD (with momentum) and MyAdamW against PyTorch equivalents
by running both on the same initialization for 1000 steps and comparing weights.
"""

import time

import numpy as np
import torch
import pytest
from tqdm import tqdm

from tests.adapters import get_sgd_cls, get_adam_cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optimize(opt_class, cfg, n_steps: int = 1000) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(model.parameters(), **cfg)

    for _ in range(n_steps):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()

    return model.weight.detach()


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

class TestSGDWithMomentum:
    def test_sgd_matches_pytorch(self):
        cfg = dict(lr=1e-3, weight_decay=0.01)
        pytorch_w = _optimize(torch.optim.SGD, cfg)
        custom_w = _optimize(get_sgd_cls(), cfg)
        assert torch.allclose(custom_w, pytorch_w, atol=1e-4), (
            f"Max diff: {(custom_w - pytorch_w).abs().max().item():.6f}"
        )

    def test_sgd_with_momentum(self):
        cfg = dict(lr=1e-3, momentum=0.9, weight_decay=0.01)
        pytorch_w = _optimize(torch.optim.SGD, cfg)
        custom_w = _optimize(get_sgd_cls(), cfg)
        assert torch.allclose(custom_w, pytorch_w, atol=1e-4), (
            f"Max diff: {(custom_w - pytorch_w).abs().max().item():.6f}"
        )


class TestAdamW:
    def test_adamw_matches_pytorch(self):
        cfg = dict(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        pytorch_w = _optimize(torch.optim.AdamW, cfg)
        custom_w = _optimize(get_adam_cls(), cfg)
        assert torch.allclose(custom_w, pytorch_w, atol=1e-4), (
            f"Max diff: {(custom_w - pytorch_w).abs().max().item():.6f}"
        )

    def test_adamw_100_steps(self):
        """PDF requirement: compare after 100 steps."""
        cfg = dict(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        pytorch_w = _optimize(torch.optim.AdamW, cfg, n_steps=100)
        custom_w = _optimize(get_adam_cls(), cfg, n_steps=100)
        assert torch.allclose(custom_w, pytorch_w, atol=1e-4)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

class TestOptimizerPerformance:
    def _bench_optimizer(self, name, opt_cls, cfg, n_iter=50):
        times = []
        for _ in tqdm(range(n_iter), desc=name):
            t0 = time.perf_counter()
            _ = _optimize(opt_cls, cfg, n_steps=500)
            times.append(time.perf_counter() - t0)
        return np.mean(times) * 1e3

    def test_sgd_latency(self):
        cfg = dict(lr=1e-3, weight_decay=0.01)
        ref_ms = self._bench_optimizer("Ref SGD", torch.optim.SGD, cfg)
        custom_ms = self._bench_optimizer("Custom SGD", get_sgd_cls(), cfg)
        ratio = custom_ms / ref_ms
        print(f"\nSGD  ref={ref_ms:.1f}ms  custom={custom_ms:.1f}ms  ratio={ratio:.1f}x")
        assert ratio <= 20.0, f"Custom SGD is {ratio:.1f}x slower (max 20x)"

    def test_adam_latency(self):
        cfg = dict(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        ref_ms = self._bench_optimizer("Ref AdamW", torch.optim.AdamW, cfg)
        custom_ms = self._bench_optimizer("Custom AdamW", get_adam_cls(), cfg)
        ratio = custom_ms / ref_ms
        print(f"\nAdamW  ref={ref_ms:.1f}ms  custom={custom_ms:.1f}ms  ratio={ratio:.1f}x")
        assert ratio <= 20.0, f"Custom AdamW is {ratio:.1f}x slower (max 20x)"
