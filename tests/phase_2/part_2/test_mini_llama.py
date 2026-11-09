"""Phase 2 · Part 2 · Tasks 1–2 — Mini-Llama 3 Architecture & Parameter Audit

Tests validate:

* the full LM forward pass against stored snapshots (reference checkpoint),
* a freshly built ``MiniLlama``: output shape, causality, sensible initial
  loss (≈ ln V), the exact parameter count implied by the architecture
  (RMSNorm, SwiGLU, GQA, no biases) and **weight tying**,
* the Model Parameter Audit utility.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.adapters import get_mini_llama, run_parameter_audit, run_transformer_lm

CONFIG = {
    "vocab_size": 512,
    "context_length": 64,
    "d_model": 128,
    "num_layers": 3,
    "num_heads": 8,
    "num_kv_heads": 2,
    "d_ff": 344,
    "rope_theta": 10000.0,
}


def expected_param_count(vocab_size, d_model, num_layers, num_heads, num_kv_heads, d_ff, **_) -> int:
    d_head = d_model // num_heads
    attn = 2 * d_model * d_model + 2 * d_model * num_kv_heads * d_head  # q, o + k, v
    ffn = 3 * d_model * d_ff  # w1, w2, w3
    norms = 2 * d_model  # ln1, ln2
    return vocab_size * d_model + num_layers * (attn + ffn + norms) + d_model  # tied embedding + ln_final


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return get_mini_llama(**CONFIG)


# ============================================================================
# Snapshot tests (reference checkpoint: untied, MHA)
# ============================================================================


class TestTransformerLMSnapshots:
    def test_transformer_lm(
        self, numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
    ):
        output = run_transformer_lm(
            vocab_size=vocab_size,
            context_length=n_keys,
            d_model=d_model,
            num_layers=n_layers,
            num_heads=n_heads,
            num_kv_heads=n_heads,
            d_ff=d_ff,
            rope_theta=theta,
            weights=ts_state_dict[0],
            in_indices=in_indices,
        )
        numpy_snapshot.assert_match(output, atol=1e-4, rtol=1e-2)

    def test_transformer_lm_truncated_input(
        self, numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
    ):
        output = run_transformer_lm(
            vocab_size=vocab_size,
            context_length=n_keys,
            d_model=d_model,
            num_layers=n_layers,
            num_heads=n_heads,
            num_kv_heads=n_heads,
            d_ff=d_ff,
            rope_theta=theta,
            weights=ts_state_dict[0],
            in_indices=in_indices[..., : in_indices.shape[-1] // 2],
        )
        numpy_snapshot.assert_match(output, atol=1e-4)


# ============================================================================
# Freshly initialised Mini-Llama
# ============================================================================


class TestMiniLlama:
    def test_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_output_shape(self, model):
        idx = torch.randint(0, CONFIG["vocab_size"], (4, CONFIG["context_length"]))
        assert model(idx).shape == (4, CONFIG["context_length"], CONFIG["vocab_size"])

    def test_accepts_shorter_sequences(self, model):
        idx = torch.randint(0, CONFIG["vocab_size"], (2, 7))
        assert model(idx).shape == (2, 7, CONFIG["vocab_size"])

    def test_is_causal(self, model):
        g = torch.Generator().manual_seed(1)
        idx = torch.randint(0, CONFIG["vocab_size"], (1, 32), generator=g)
        idx2 = idx.clone()
        idx2[0, 20:] = torch.randint(0, CONFIG["vocab_size"], (12,), generator=g)
        with torch.no_grad():
            a, b = model(idx), model(idx2)
        np.testing.assert_allclose(a[:, :20].numpy(), b[:, :20].numpy(), atol=1e-5)

    def test_exact_parameter_count(self, model):
        """Pins the architecture: tied embedding, GQA-sized K/V, SwiGLU with three
        matrices, RMSNorm gains only, and no biases anywhere."""
        n_params = sum(p.numel() for p in model.parameters())
        expected = expected_param_count(**CONFIG)
        assert n_params == expected, f"Got {n_params:,} parameters, expected {expected:,}"

    def test_weight_tying(self, model):
        """Exactly one (vocab_size, d_model) tensor must exist; it serves both as
        the token embedding and as the output projection."""
        V, d = CONFIG["vocab_size"], CONFIG["d_model"]
        tables = [p for p in model.parameters() if p.shape == (V, d)]
        assert len(tables) == 1, f"Expected a single shared (V, d) matrix, found {len(tables)}"
        idx = torch.randint(0, V, (2, 16))
        model.zero_grad()
        model(idx).sum().backward()
        # The output head touches every row, the embedding only rows in idx.
        # A tied matrix therefore gets non-zero gradient on every row.
        assert (tables[0].grad.abs().sum(dim=1) > 0).all()
        model.zero_grad()

    def test_initial_loss_is_near_ln_vocab(self):
        """At initialization predictions should be ≈ uniform: loss ≈ ln(V).
        A much larger loss means badly scaled initialization (e.g. N(0, 1)
        embeddings feeding a tied output head)."""
        torch.manual_seed(0)
        model = get_mini_llama(**CONFIG)
        V = CONFIG["vocab_size"]
        idx = torch.randint(0, V, (8, CONFIG["context_length"]))
        targets = torch.randint(0, V, (8, CONFIG["context_length"]))
        with torch.no_grad():
            loss = F.cross_entropy(model(idx).reshape(-1, V), targets.reshape(-1)).item()
        print(f"\nInitial loss {loss:.3f}  vs  ln(V) = {math.log(V):.3f}")
        assert abs(loss - math.log(V)) < 1.0, f"Initial loss {loss:.2f} is far from ln(V) = {math.log(V):.2f}"


# ============================================================================
# Model Parameter Audit
# ============================================================================


class TestParameterAudit:
    def test_rows_match_named_parameters(self, model):
        rows = run_parameter_audit(model)
        named = list(model.named_parameters())
        assert [r["name"] for r in rows] == [n for n, _ in named]
        for row, (_, p) in zip(rows, named):
            assert tuple(row["shape"]) == tuple(p.shape)
            assert row["num_params"] == p.numel()

    def test_percentages_sum_to_100(self, model):
        rows = run_parameter_audit(model)
        assert sum(r["percent"] for r in rows) == pytest.approx(100.0, abs=1e-6)

    def test_percentages_are_shares_of_total(self, model):
        rows = run_parameter_audit(model)
        total = sum(r["num_params"] for r in rows)
        assert total == expected_param_count(**CONFIG)
        for r in rows:
            assert r["percent"] == pytest.approx(100.0 * r["num_params"] / total)
