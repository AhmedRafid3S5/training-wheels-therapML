"""Phase 2 · Part 1 · Task 8 — Rotary Positional Embeddings (RoPE)

Tests validate:

* the RoPE output and MHA-with-RoPE output against stored snapshots,
* the frequency schedule θ_i = theta^(−2i/d) with (2i, 2i+1) pairing,
* that RoPE is a rotation (norm-preserving, identity at position 0),
* the key property: q·k attention scores depend only on the *relative*
  distance between tokens, not on their absolute positions.
"""

import math

import numpy as np
import pytest
import torch
from einops import rearrange

from tests.adapters import run_multihead_self_attention_with_rope, run_rope

THETA = 10000.0


def _rope(x, positions, d_k=None, max_seq_len=2048):
    return run_rope(d_k or x.shape[-1], THETA, max_seq_len, x, positions)


class TestRoPESnapshots:
    def test_rope(self, numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):
        output = run_rope(d_model, theta, n_queries, in_embeddings, pos_ids)
        numpy_snapshot.assert_match(output, atol=1e-6)

    def test_multihead_self_attention_with_rope(
        self, numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
    ):
        sd, _ = ts_state_dict
        q, k, v, o = (sd[f"layers.0.attn.{n}_proj.weight"] for n in ("q", "k", "v", "output"))
        output = run_multihead_self_attention_with_rope(
            d_model=d_model,
            num_heads=n_heads,
            max_seq_len=n_keys,
            theta=theta,
            q_proj_weight=q,
            k_proj_weight=k,
            v_proj_weight=v,
            o_proj_weight=o,
            in_features=in_embeddings,
            token_positions=rearrange(pos_ids, "seq -> 1 seq"),
        )
        numpy_snapshot.assert_match(output, atol=1e-6)


class TestRoPE:
    @pytest.mark.parametrize("pos", [0, 1, 3, 17])
    def test_d2_rotates_by_position(self, pos):
        """For d_k = 2, θ_0 = 1, so [1, 0] at position p becomes [cos p, sin p]."""
        x = torch.tensor([[[1.0, 0.0]]])
        out = _rope(x, torch.tensor([pos]), max_seq_len=32)
        np.testing.assert_allclose(out[0, 0].numpy(), [math.cos(pos), math.sin(pos)], atol=1e-5)

    def test_frequency_schedule_and_pairing(self):
        """d_k = 4: pairs (0,1) and (2,3) rotate with θ_0 = 1 and θ_1 = theta^(-1/2)."""
        p = 5
        x = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
        out = _rope(x, torch.tensor([p]), max_seq_len=16)
        f1 = THETA ** (-2 * 1 / 4)
        expected = [math.cos(p), math.sin(p), math.cos(p * f1), math.sin(p * f1)]
        np.testing.assert_allclose(out[0, 0].numpy(), expected, atol=1e-5)

    def test_position_zero_is_identity(self):
        x = torch.randn(2, 1, 64)
        out = _rope(x, torch.tensor([0]))
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-6)

    def test_preserves_pairwise_norms(self):
        x = torch.randn(2, 32, 64)
        out = _rope(x, torch.arange(32))
        pair_norm = lambda t: t.view(*t.shape[:-1], -1, 2).norm(dim=-1)
        np.testing.assert_allclose(pair_norm(out).numpy(), pair_norm(x).numpy(), atol=1e-5, rtol=1e-5)

    def test_batched_positions(self):
        """token_positions of shape (batch, seq) — each row uses its own positions."""
        x = torch.randn(2, 6, 32)
        pos = torch.tensor([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]])
        out = _rope(x, pos)
        for b in range(2):
            np.testing.assert_allclose(out[b].numpy(), _rope(x[b : b + 1], pos[b])[0].numpy(), atol=1e-6)

    def test_relative_position_invariance(self):
        """⟨RoPE(q, m), RoPE(k, n)⟩ depends only on m − n."""
        g = torch.Generator().manual_seed(0)
        q, k = torch.randn(1, 1, 64, generator=g), torch.randn(1, 1, 64, generator=g)

        def score(m, n):
            return (_rope(q, torch.tensor([m])) * _rope(k, torch.tensor([n]))).sum().item()

        for offset in (0, 1, 3, 10):
            scores = [score(base + offset, base) for base in (0, 7, 50, 200)]
            np.testing.assert_allclose(scores, scores[0], rtol=1e-4, atol=1e-4)
        assert not math.isclose(score(3, 0), score(10, 0), rel_tol=1e-3), "Scores should vary with distance"
