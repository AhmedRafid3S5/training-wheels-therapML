"""Phase 2 · Part 1 · Task 5 — Causal Self-Attention

Tests validate single-head causal self-attention with Q, K, V projections:
the result matches PyTorch, and no information flows from future tokens.
"""

import numpy as np
import torch
import torch.nn.functional as F

from tests.adapters import run_causal_self_attention
from tests.phase_2.common import benchmark

D_IN, D_K, D_V, SEQ = 32, 16, 24, 12


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(D_K, D_IN, generator=g) / D_IN**0.5,
        torch.randn(D_K, D_IN, generator=g) / D_IN**0.5,
        torch.randn(D_V, D_IN, generator=g) / D_IN**0.5,
    )


def _reference(wq, wk, wv, x):
    return F.scaled_dot_product_attention(x @ wq.T, x @ wk.T, x @ wv.T, is_causal=True)


class TestCausalSelfAttention:
    def test_matches_pytorch(self):
        wq, wk, wv = _weights()
        x = torch.randn(4, SEQ, D_IN, generator=torch.Generator().manual_seed(1))
        result = run_causal_self_attention(wq, wk, wv, x)
        assert result.shape == (4, SEQ, D_V)
        np.testing.assert_allclose(result.detach().numpy(), _reference(wq, wk, wv, x).numpy(), atol=1e-6, rtol=1e-5)

    def test_extra_leading_dims(self):
        wq, wk, wv = _weights()
        x = torch.randn(2, 3, SEQ, D_IN, generator=torch.Generator().manual_seed(2))
        result = run_causal_self_attention(wq, wk, wv, x)
        np.testing.assert_allclose(result.detach().numpy(), _reference(wq, wk, wv, x).numpy(), atol=1e-6, rtol=1e-5)

    def test_first_token_attends_only_to_itself(self):
        wq, wk, wv = _weights()
        x = torch.randn(2, SEQ, D_IN, generator=torch.Generator().manual_seed(3))
        result = run_causal_self_attention(wq, wk, wv, x)
        np.testing.assert_allclose(result[:, 0].detach().numpy(), (x[:, 0] @ wv.T).numpy(), atol=1e-6)

    def test_no_information_from_future_tokens(self):
        wq, wk, wv = _weights()
        x = torch.randn(1, SEQ, D_IN, generator=torch.Generator().manual_seed(4))
        t = 7
        x_perturbed = x.clone()
        x_perturbed[:, t:] += 10.0 * torch.randn(1, SEQ - t, D_IN, generator=torch.Generator().manual_seed(5))
        before = run_causal_self_attention(wq, wk, wv, x).detach()
        after = run_causal_self_attention(wq, wk, wv, x_perturbed).detach()
        np.testing.assert_allclose(after[:, :t].numpy(), before[:, :t].numpy(), atol=1e-6)
        assert not torch.allclose(after[:, t:], before[:, t:]), "Perturbed positions should change"


class TestCausalSelfAttentionPerformance:
    def test_causal_attention_latency(self):
        g = torch.Generator().manual_seed(0)
        wq, wk, wv = (torch.randn(128, 256, generator=g) for _ in range(3))
        x = torch.randn(16, 256, 256, generator=g)
        with torch.no_grad():
            ref = benchmark(lambda: _reference(wq, wk, wv, x), "Ref causal attn")
            custom = benchmark(lambda: run_causal_self_attention(wq, wk, wv, x), "Custom causal attn")
        ratio = custom / ref
        print(f"\nCausal attention  ref={ref * 1e3:.2f}ms  custom={custom * 1e3:.2f}ms  ratio={ratio:.1f}x")
        assert ratio <= 20.0, f"Custom causal attention is {ratio:.1f}x slower (max 20x)"
