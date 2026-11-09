"""Phase 2 · Part 1 · Task 4 — Scaled Dot-Product Attention

Tests validate Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V against
``torch.nn.functional.scaled_dot_product_attention`` and stored snapshots,
including boolean masking, extra batch/head dimensions and gradients.
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from tests.adapters import run_scaled_dot_product_attention
from tests.phase_2.common import benchmark


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# ============================================================================
# Snapshot tests (shared conftest fixtures q, k, v, mask)
# ============================================================================


class TestSDPASnapshots:
    def test_self_attention(self, numpy_snapshot, q, k, v, mask):
        output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
        numpy_snapshot.assert_match(output, atol=1e-6)

    def test_4d_self_attention(self, numpy_snapshot, q, k, v, mask):
        # (batch, heads, seq, d_k)
        q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
        mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)
        output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
        numpy_snapshot.assert_match(output, atol=1e-6)


# ============================================================================
# Accuracy vs PyTorch
# ============================================================================


class TestSDPA:
    def test_matches_pytorch_no_mask(self):
        Q, K, V = _rand(4, 10, 32, seed=1), _rand(4, 14, 32, seed=2), _rand(4, 14, 24, seed=3)
        result = run_scaled_dot_product_attention(Q, K, V)
        expected = F.scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), atol=1e-6, rtol=1e-5)

    def test_matches_pytorch_with_mask(self):
        Q, K, V = _rand(4, 10, 32, seed=1), _rand(4, 14, 32, seed=2), _rand(4, 14, 24, seed=3)
        mask = _rand(4, 10, 14, seed=4) > 0
        mask[..., 0] = True  # every query attends to at least one key
        result = run_scaled_dot_product_attention(Q, K, V, mask=mask)
        expected = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), atol=1e-6, rtol=1e-5)

    def test_matches_pytorch_5d(self):
        Q, K, V = _rand(2, 3, 4, 8, 16, seed=1), _rand(2, 3, 4, 8, 16, seed=2), _rand(2, 3, 4, 8, 16, seed=3)
        result = run_scaled_dot_product_attention(Q, K, V)
        expected = F.scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), atol=1e-6, rtol=1e-5)

    def test_scaling_by_sqrt_dk(self):
        """With d_k = 256 the unscaled scores would be ~16× too large."""
        Q, K, V = _rand(1, 6, 256, seed=1), _rand(1, 6, 256, seed=2), _rand(1, 6, 8, seed=3)
        result = run_scaled_dot_product_attention(Q, K, V)
        expected = torch.softmax(Q @ K.transpose(-1, -2) / 16.0, dim=-1) @ V
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), atol=1e-6, rtol=1e-5)

    def test_single_allowed_key_returns_its_value(self):
        Q, K, V = _rand(1, 5, 8, seed=1), _rand(1, 5, 8, seed=2), _rand(1, 5, 8, seed=3)
        mask = torch.eye(5, dtype=torch.bool).flip(0).unsqueeze(0)  # query i → key 4 - i only
        result = run_scaled_dot_product_attention(Q, K, V, mask=mask)
        np.testing.assert_allclose(result.detach().numpy(), V.flip(1).numpy(), atol=1e-6)

    def test_numerically_stable_for_large_scores(self):
        Q, K, V = _rand(2, 8, 16, seed=1) * 1e3, _rand(2, 8, 16, seed=2) * 1e3, _rand(2, 8, 16, seed=3)
        result = run_scaled_dot_product_attention(Q, K, V)
        assert torch.isfinite(result).all(), "SDPA overflowed on large scores — use a stable softmax"

    def test_gradients_match_pytorch(self):
        def grads(fn):
            Q, K, V = (_rand(2, 6, 8, seed=s).requires_grad_() for s in (1, 2, 3))
            mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
            fn(Q, K, V, mask).pow(2).sum().backward()
            return [t.grad for t in (Q, K, V)]

        ours = grads(lambda q, k, v, m: run_scaled_dot_product_attention(q, k, v, mask=m))
        ref = grads(lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m))
        for g_ours, g_ref in zip(ours, ref):
            np.testing.assert_allclose(g_ours.numpy(), g_ref.numpy(), atol=1e-5, rtol=1e-4)


# ============================================================================
# Latency benchmarks
# ============================================================================


class TestSDPAPerformance:
    def test_sdpa_latency(self):
        Q, K, V = _rand(8, 8, 256, 64, seed=1), _rand(8, 8, 256, 64, seed=2), _rand(8, 8, 256, 64, seed=3)
        mask = torch.tril(torch.ones(256, 256, dtype=torch.bool))

        with torch.no_grad():
            ref = benchmark(lambda: F.scaled_dot_product_attention(Q, K, V, attn_mask=mask), "Ref SDPA")
            custom = benchmark(lambda: run_scaled_dot_product_attention(Q, K, V, mask=mask), "Custom SDPA")
        ratio = custom / ref
        print(f"\nSDPA  ref={ref * 1e3:.2f}ms  custom={custom * 1e3:.2f}ms  ratio={ratio:.1f}x")
        assert ratio <= 20.0, f"Custom SDPA is {ratio:.1f}x slower (max 20x)"
