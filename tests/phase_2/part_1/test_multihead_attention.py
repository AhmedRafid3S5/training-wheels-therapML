"""Phase 2 · Part 1 · Task 6 — Multi-Head Attention (MHA)

Tests validate causal multi-head self-attention against
``torch.nn.MultiheadAttention`` and a stored snapshot, and check that heads are
split from ``d_model`` in the expected order and computed independently.
"""

import numpy as np
import pytest
import torch

from tests.adapters import run_causal_self_attention, run_multihead_self_attention
from tests.phase_2.common import benchmark, make_attention_weights

D_MODEL, NUM_HEADS, SEQ = 64, 8, 20


def _torch_mha(d_model, num_heads, w, x):
    """Reference: torch.nn.MultiheadAttention with the same weights and a causal mask."""
    mha = torch.nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([w["q_proj_weight"], w["k_proj_weight"], w["v_proj_weight"]]))
        mha.out_proj.weight.copy_(w["o_proj_weight"])
    seq = x.shape[-2]
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)  # True = NOT allowed
    out, _ = mha(x, x, x, attn_mask=future, need_weights=False)
    return out.detach()


class TestMHASnapshot:
    def test_multihead_self_attention(self, numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
        sd, _ = ts_state_dict
        q, k, v, o = (sd[f"layers.0.attn.{n}_proj.weight"] for n in ("q", "k", "v", "output"))
        output = run_multihead_self_attention(
            d_model=d_model,
            num_heads=n_heads,
            q_proj_weight=q,
            k_proj_weight=k,
            v_proj_weight=v,
            o_proj_weight=o,
            in_features=in_embeddings,
        )
        numpy_snapshot.assert_match(output, atol=1e-6)


class TestMHA:
    @pytest.mark.parametrize("num_heads", [1, 2, 8])
    def test_matches_torch_multihead_attention(self, num_heads):
        w = make_attention_weights(D_MODEL, num_heads, num_heads)
        x = torch.randn(3, SEQ, D_MODEL, generator=torch.Generator().manual_seed(1))
        result = run_multihead_self_attention(D_MODEL, num_heads, **w, in_features=x)
        expected = _torch_mha(D_MODEL, num_heads, w, x)
        np.testing.assert_allclose(result.detach().numpy(), expected.numpy(), atol=1e-5, rtol=1e-4)

    def test_single_head_equals_causal_attention_plus_projection(self):
        w = make_attention_weights(D_MODEL, 1, 1)
        x = torch.randn(2, SEQ, D_MODEL, generator=torch.Generator().manual_seed(2))
        result = run_multihead_self_attention(D_MODEL, 1, **w, in_features=x)
        single = run_causal_self_attention(w["q_proj_weight"], w["k_proj_weight"], w["v_proj_weight"], x)
        expected = single @ w["o_proj_weight"].T
        np.testing.assert_allclose(result.detach().numpy(), expected.detach().numpy(), atol=1e-5, rtol=1e-4)

    def test_heads_are_independent_and_ordered(self):
        """With W_o = I, output columns [h·d_head, (h+1)·d_head) belong to head h.
        Zeroing head 3's V rows must zero exactly those columns."""
        d_head = D_MODEL // NUM_HEADS
        w = make_attention_weights(D_MODEL, NUM_HEADS, NUM_HEADS)
        w["o_proj_weight"] = torch.eye(D_MODEL)
        x = torch.randn(2, SEQ, D_MODEL, generator=torch.Generator().manual_seed(3))
        base = run_multihead_self_attention(D_MODEL, NUM_HEADS, **w, in_features=x).detach()

        w["v_proj_weight"] = w["v_proj_weight"].clone()
        w["v_proj_weight"][3 * d_head : 4 * d_head] = 0.0
        out = run_multihead_self_attention(D_MODEL, NUM_HEADS, **w, in_features=x).detach()

        head3 = slice(3 * d_head, 4 * d_head)
        np.testing.assert_allclose(out[..., head3].numpy(), 0.0, atol=1e-7)
        others = torch.ones(D_MODEL, dtype=torch.bool)
        others[head3] = False
        np.testing.assert_allclose(out[..., others].numpy(), base[..., others].numpy(), atol=1e-6)


class TestMHAPerformance:
    def test_mha_latency(self):
        d_model, num_heads = 256, 8
        w = make_attention_weights(d_model, num_heads, num_heads)
        x = torch.randn(16, 256, d_model)
        mha = torch.nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True)
        future = torch.triu(torch.ones(256, 256, dtype=torch.bool), diagonal=1)
        with torch.no_grad():
            ref = benchmark(lambda: mha(x, x, x, attn_mask=future, need_weights=False), "Ref MHA")
            custom = benchmark(
                lambda: run_multihead_self_attention(d_model, num_heads, **w, in_features=x), "Custom MHA"
            )
        ratio = custom / ref
        print(f"\nMHA  ref={ref * 1e3:.2f}ms  custom={custom * 1e3:.2f}ms  ratio={ratio:.1f}x")
        assert ratio <= 20.0, f"Custom MHA is {ratio:.1f}x slower (max 20x)"
