"""Phase 2 · Part 2 · Tasks 1–2 — TransformerBlock

Tests validate the Llama-style pre-norm block
(RMSNorm → GQA + RoPE → residual → RMSNorm → SwiGLU → residual):

* output matches a stored snapshot (reference checkpoint, MHA),
* GQA blocks equal MHA blocks with repeated K/V weights,
* residual connections and causality behave as expected.
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from tests.adapters import run_transformer_block
from tests.phase_2.common import benchmark, expand_kv_weight

D_MODEL, NUM_HEADS, D_FF, SEQ, THETA = 64, 8, 172, 16, 10000.0


def _block_weights(num_kv_heads: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    kv_dim = num_kv_heads * (D_MODEL // NUM_HEADS)
    rnd = lambda *s: torch.randn(*s, generator=g) * s[-1] ** -0.5
    return {
        "attn.q_proj.weight": rnd(D_MODEL, D_MODEL),
        "attn.k_proj.weight": rnd(kv_dim, D_MODEL),
        "attn.v_proj.weight": rnd(kv_dim, D_MODEL),
        "attn.output_proj.weight": rnd(D_MODEL, D_MODEL),
        "ln1.weight": 1.0 + 0.1 * torch.randn(D_MODEL, generator=g),
        "ffn.w1.weight": rnd(D_FF, D_MODEL),
        "ffn.w2.weight": rnd(D_MODEL, D_FF),
        "ffn.w3.weight": rnd(D_FF, D_MODEL),
        "ln2.weight": 1.0 + 0.1 * torch.randn(D_MODEL, generator=g),
    }


def _block(weights, x, num_kv_heads, num_heads=NUM_HEADS, d_model=D_MODEL, d_ff=D_FF):
    return run_transformer_block(d_model, num_heads, num_kv_heads, d_ff, x.shape[-2], THETA, weights, x)


class TestTransformerBlockSnapshot:
    def test_transformer_block(
        self, numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta
    ):
        block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}
        output = run_transformer_block(
            d_model=d_model,
            num_heads=n_heads,
            num_kv_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=n_keys,
            theta=theta,
            weights=block_weights,
            in_features=in_embeddings,
        )
        numpy_snapshot.assert_match(output, atol=1e-6)


class TestTransformerBlock:
    def test_output_shape(self):
        x = torch.randn(3, SEQ, D_MODEL)
        assert _block(_block_weights(2), x, 2).shape == x.shape

    def test_gqa_block_equals_expanded_mha_block(self):
        w_gqa = _block_weights(num_kv_heads=2)
        w_mha = dict(w_gqa)
        for key in ("attn.k_proj.weight", "attn.v_proj.weight"):
            w_mha[key] = expand_kv_weight(w_gqa[key], NUM_HEADS, 2)
        x = torch.randn(2, SEQ, D_MODEL, generator=torch.Generator().manual_seed(1))
        gqa = _block(w_gqa, x, num_kv_heads=2)
        mha = _block(w_mha, x, num_kv_heads=NUM_HEADS)
        np.testing.assert_allclose(gqa.detach().numpy(), mha.detach().numpy(), atol=1e-5, rtol=1e-4)

    def test_zero_output_projections_give_identity(self):
        """If both sub-layers output zero, the residual stream passes x through unchanged."""
        w = _block_weights(2)
        w["attn.output_proj.weight"] = torch.zeros_like(w["attn.output_proj.weight"])
        w["ffn.w2.weight"] = torch.zeros_like(w["ffn.w2.weight"])
        x = torch.randn(2, SEQ, D_MODEL)
        np.testing.assert_allclose(_block(w, x, 2).detach().numpy(), x.numpy(), atol=1e-6)

    def test_pre_norm_makes_sublayers_scale_invariant(self):
        """Pre-norm: block(c·x) − c·x == block(x) − x, because each sub-layer only
        sees RMSNorm(x). Fails for post-norm or missing normalization."""
        w = _block_weights(2)
        x = torch.randn(1, SEQ, D_MODEL, generator=torch.Generator().manual_seed(2))
        # Only the attention sub-layer: zero the FFN so the second residual branch vanishes.
        w["ffn.w2.weight"] = torch.zeros_like(w["ffn.w2.weight"])
        delta_1 = _block(w, x, 2) - x
        delta_5 = _block(w, 5.0 * x, 2) - 5.0 * x
        np.testing.assert_allclose(delta_5.detach().numpy(), delta_1.detach().numpy(), atol=1e-4, rtol=1e-3)

    def test_block_is_causal(self):
        w = _block_weights(2)
        x = torch.randn(1, SEQ, D_MODEL, generator=torch.Generator().manual_seed(3))
        x2 = x.clone()
        x2[:, 10:] += 3.0
        a, b = _block(w, x, 2).detach(), _block(w, x2, 2).detach()
        np.testing.assert_allclose(a[:, :10].numpy(), b[:, :10].numpy(), atol=1e-6)


# ============================================================================
# Latency
# ============================================================================


def _reference_block(w, x, num_heads, num_kv_heads):
    """Same-cost reference built from PyTorch built-ins (no RoPE)."""
    rms = lambda t, g: F.rms_norm(t, (t.shape[-1],), g, eps=1e-5)
    h_in = rms(x, w["ln1.weight"])
    q = rearrange(h_in @ w["attn.q_proj.weight"].T, "b s (h d) -> b h s d", h=num_heads)
    k = rearrange(h_in @ w["attn.k_proj.weight"].T, "b s (h d) -> b h s d", h=num_kv_heads)
    v = rearrange(h_in @ w["attn.v_proj.weight"].T, "b s (h d) -> b h s d", h=num_kv_heads)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    h = x + rearrange(a, "b h s d -> b s (h d)") @ w["attn.output_proj.weight"].T
    f_in = rms(h, w["ln2.weight"])
    return h + (F.silu(f_in @ w["ffn.w1.weight"].T) * (f_in @ w["ffn.w3.weight"].T)) @ w["ffn.w2.weight"].T


class TestTransformerBlockPerformance:
    def test_block_latency(self):
        d_model, num_heads, num_kv_heads, d_ff, seq = 256, 8, 2, 688, 256
        g = torch.Generator().manual_seed(0)
        kv_dim = num_kv_heads * d_model // num_heads
        w = {
            "attn.q_proj.weight": torch.randn(d_model, d_model, generator=g),
            "attn.k_proj.weight": torch.randn(kv_dim, d_model, generator=g),
            "attn.v_proj.weight": torch.randn(kv_dim, d_model, generator=g),
            "attn.output_proj.weight": torch.randn(d_model, d_model, generator=g),
            "ln1.weight": torch.ones(d_model),
            "ffn.w1.weight": torch.randn(d_ff, d_model, generator=g),
            "ffn.w2.weight": torch.randn(d_model, d_ff, generator=g),
            "ffn.w3.weight": torch.randn(d_ff, d_model, generator=g),
            "ln2.weight": torch.ones(d_model),
        }
        x = torch.randn(8, seq, d_model, generator=g)
        with torch.no_grad():
            ref = benchmark(lambda: _reference_block(w, x, num_heads, num_kv_heads), "Ref block", n_iter=30)
            custom = benchmark(
                lambda: run_transformer_block(d_model, num_heads, num_kv_heads, d_ff, seq, THETA, w, x),
                "Custom block",
                n_iter=30,
            )
        ratio = custom / ref
        print(f"\nTransformerBlock  ref={ref * 1e3:.2f}ms  custom={custom * 1e3:.2f}ms  ratio={ratio:.1f}x")
        assert ratio <= 20.0, f"Custom TransformerBlock is {ratio:.1f}x slower (max 20x)"
