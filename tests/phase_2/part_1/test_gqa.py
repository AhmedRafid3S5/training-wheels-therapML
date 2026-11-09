"""Phase 2 · Part 1 · Task 7 — Grouped Query Attention (GQA)

Tests validate causal GQA (fewer KV heads than query heads):

* matches ``F.scaled_dot_product_attention(enable_gqa=True)``,
* reduces to MHA when ``num_kv_heads == num_heads``,
* with RoPE, equals MHA whose K/V weights are repeated per group.

The MHA-vs-GQA memory comparison (KV-cache size) lives in ``test_kv_cache.py``.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from tests.adapters import (
    run_grouped_query_attention,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
)
from tests.phase_2.common import benchmark, expand_kv_weight, make_attention_weights

D_MODEL, NUM_HEADS, SEQ, THETA = 128, 8, 24, 10000.0
HEAD_CONFIGS = [(8, 4), (8, 2), (8, 1)]  # (num_heads, num_kv_heads); kv=1 is multi-query attention


def _reference_gqa(num_heads, num_kv_heads, w, x):
    q = rearrange(x @ w["q_proj_weight"].T, "... s (h d) -> ... h s d", h=num_heads)
    k = rearrange(x @ w["k_proj_weight"].T, "... s (h d) -> ... h s d", h=num_kv_heads)
    v = rearrange(x @ w["v_proj_weight"].T, "... s (h d) -> ... h s d", h=num_kv_heads)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return rearrange(o, "... h s d -> ... s (h d)") @ w["o_proj_weight"].T


class TestGQA:
    @pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_CONFIGS)
    def test_matches_pytorch_enable_gqa(self, num_heads, num_kv_heads):
        w = make_attention_weights(D_MODEL, num_heads, num_kv_heads)
        x = torch.randn(3, SEQ, D_MODEL, generator=torch.Generator().manual_seed(1))
        result = run_grouped_query_attention(D_MODEL, num_heads, num_kv_heads, **w, in_features=x)
        expected = _reference_gqa(num_heads, num_kv_heads, w, x)
        assert result.shape == x.shape
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), atol=1e-5, rtol=1e-4)

    def test_equals_mha_when_kv_heads_equal_heads(self):
        w = make_attention_weights(D_MODEL, NUM_HEADS, NUM_HEADS)
        x = torch.randn(2, SEQ, D_MODEL, generator=torch.Generator().manual_seed(2))
        gqa = run_grouped_query_attention(D_MODEL, NUM_HEADS, NUM_HEADS, **w, in_features=x)
        mha = run_multihead_self_attention(D_MODEL, NUM_HEADS, **w, in_features=x)
        np.testing.assert_allclose(gqa.detach().numpy(), mha.detach().numpy(), atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_CONFIGS)
    def test_with_rope_equals_expanded_mha_with_rope(self, num_heads, num_kv_heads):
        w = make_attention_weights(D_MODEL, num_heads, num_kv_heads)
        x = torch.randn(2, SEQ, D_MODEL, generator=torch.Generator().manual_seed(3))
        pos = torch.arange(SEQ)
        gqa = run_grouped_query_attention(
            D_MODEL, num_heads, num_kv_heads, **w, in_features=x, theta=THETA, max_seq_len=SEQ, token_positions=pos
        )
        w_mha = dict(w)
        w_mha["k_proj_weight"] = expand_kv_weight(w["k_proj_weight"], num_heads, num_kv_heads)
        w_mha["v_proj_weight"] = expand_kv_weight(w["v_proj_weight"], num_heads, num_kv_heads)
        mha = run_multihead_self_attention_with_rope(
            D_MODEL, num_heads, SEQ, THETA, **w_mha, in_features=x, token_positions=pos
        )
        np.testing.assert_allclose(gqa.detach().numpy(), mha.detach().numpy(), atol=1e-5, rtol=1e-4)

    def test_is_causal(self):
        w = make_attention_weights(D_MODEL, 8, 2)
        x = torch.randn(1, SEQ, D_MODEL, generator=torch.Generator().manual_seed(4))
        x2 = x.clone()
        x2[:, -1] += 5.0
        a = run_grouped_query_attention(D_MODEL, 8, 2, **w, in_features=x).detach()
        b = run_grouped_query_attention(D_MODEL, 8, 2, **w, in_features=x2).detach()
        np.testing.assert_allclose(a[:, :-1].numpy(), b[:, :-1].numpy(), atol=1e-6)

    def test_extra_leading_dims(self):
        w = make_attention_weights(D_MODEL, 8, 2)
        x = torch.randn(2, 3, SEQ, D_MODEL, generator=torch.Generator().manual_seed(5))
        result = run_grouped_query_attention(D_MODEL, 8, 2, **w, in_features=x)
        expected = _reference_gqa(8, 2, w, x)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), atol=1e-5, rtol=1e-4)


class TestGQAPerformance:
    def test_gqa_latency(self):
        d_model, num_heads, num_kv_heads = 256, 8, 2
        w = make_attention_weights(d_model, num_heads, num_kv_heads)
        x = torch.randn(16, 256, d_model)
        with torch.no_grad():
            ref = benchmark(lambda: _reference_gqa(num_heads, num_kv_heads, w, x), "Ref GQA")
            custom = benchmark(
                lambda: run_grouped_query_attention(d_model, num_heads, num_kv_heads, **w, in_features=x), "Custom GQA"
            )
        ratio = custom / ref
        print(f"\nGQA  ref={ref * 1e3:.2f}ms  custom={custom * 1e3:.2f}ms  ratio={ratio:.1f}x")
        assert ratio <= 20.0, f"Custom GQA is {ratio:.1f}x slower (max 20x)"
