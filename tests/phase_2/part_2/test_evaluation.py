"""Phase 2 · Part 2 · Task 5 — Evaluation: Perplexity & Generation

Tests validate:

* perplexity = exp(cross-entropy),
* autoregressive generation: greedy decoding (T = 0), temperature sampling
  that matches softmax(logits / T), seeding, and context-window cropping.

Generation is tested against tiny hand-built "models" so that it is checked
independently of your Mini-Llama, then once end-to-end with ``get_mini_llama``.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.adapters import get_mini_llama, run_generate, run_perplexity

# ============================================================================
# Perplexity
# ============================================================================


class TestPerplexity:
    def test_matches_exp_cross_entropy(self):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(4, 16, 100, generator=g)
        targets = torch.randint(0, 100, (4, 16), generator=g)
        expected = math.exp(F.cross_entropy(logits.reshape(-1, 100), targets.reshape(-1)).item())
        assert run_perplexity(logits, targets) == pytest.approx(expected, rel=1e-5)

    def test_uniform_prediction_has_perplexity_vocab_size(self):
        logits = torch.zeros(2, 8, 321)
        targets = torch.randint(0, 321, (2, 8))
        assert run_perplexity(logits, targets) == pytest.approx(321.0, rel=1e-5)

    def test_confident_correct_prediction_has_perplexity_one(self):
        targets = torch.randint(0, 50, (3, 10))
        logits = F.one_hot(targets, 50).float() * 100.0
        assert run_perplexity(logits, targets) == pytest.approx(1.0, abs=1e-6)

    def test_shift_invariant_and_numerically_stable(self):
        """Adding a constant to every logit must not change perplexity (a naive
        exp() overflows here)."""
        g = torch.Generator().manual_seed(1)
        logits = torch.randn(2, 8, 30, generator=g)
        targets = torch.randint(0, 30, (2, 8), generator=g)
        assert run_perplexity(logits + 1e3, targets) == pytest.approx(run_perplexity(logits, targets), rel=1e-3)

    def test_2d_input(self):
        g = torch.Generator().manual_seed(2)
        logits, targets = torch.randn(32, 10, generator=g), torch.randint(0, 10, (32,), generator=g)
        expected = math.exp(F.cross_entropy(logits, targets).item())
        assert run_perplexity(logits, targets) == pytest.approx(expected, rel=1e-5)


# ============================================================================
# Generation
# ============================================================================


class FixedLogitsModel(torch.nn.Module):
    """Always predicts the same next-token distribution, whatever the input."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, idx):
        return self.logits.expand(idx.shape[0], idx.shape[1], -1).clone()


class BigramModel(torch.nn.Module):
    """next-token logits depend only on the current token: table[idx]."""

    def __init__(self, vocab_size: int, seed: int = 0):
        super().__init__()
        self.register_buffer(
            "table", torch.randn(vocab_size, vocab_size, generator=torch.Generator().manual_seed(seed))
        )
        self.max_seen_len = 0

    def forward(self, idx):
        self.max_seen_len = max(self.max_seen_len, idx.shape[1])
        return self.table[idx]


class TestGenerate:
    def test_returns_only_new_tokens(self):
        model = BigramModel(20)
        out = run_generate(model, [1, 2, 3], max_new_tokens=10, temperature=1.0, context_length=64)
        assert len(out) == 10
        assert all(isinstance(t, int) and 0 <= t < 20 for t in out)

    def test_greedy_follows_argmax(self):
        model = BigramModel(30, seed=1)
        out = run_generate(model, [5], max_new_tokens=15, temperature=0.0, context_length=64)
        expected, tok = [], 5
        for _ in range(15):
            tok = int(model.table[tok].argmax())
            expected.append(tok)
        assert out == expected

    @pytest.mark.parametrize("temperature", [1.0, 0.5, 2.0])
    def test_sampling_matches_softmax_over_temperature(self, temperature):
        logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
        model = FixedLogitsModel(logits)
        torch.manual_seed(0)
        n = 10_000
        out = run_generate(model, [0], max_new_tokens=n, temperature=temperature, context_length=8)
        freqs = np.bincount(out, minlength=4) / n
        expected = torch.softmax(logits / temperature, dim=-1).numpy()
        print(f"\nT={temperature}  empirical={np.round(freqs, 3)}  expected={np.round(expected, 3)}")
        np.testing.assert_allclose(freqs, expected, atol=0.03)

    def test_seed_makes_sampling_reproducible(self):
        model = BigramModel(50, seed=2)
        torch.manual_seed(123)
        a = run_generate(model, [0, 1], max_new_tokens=30, temperature=1.0, context_length=64)
        torch.manual_seed(123)
        b = run_generate(model, [0, 1], max_new_tokens=30, temperature=1.0, context_length=64)
        assert a == b

    def test_crops_to_context_length(self):
        model = BigramModel(10)
        run_generate(model, [1, 2, 3, 4, 5], max_new_tokens=20, temperature=1.0, context_length=8)
        assert model.max_seen_len == 8, f"Model saw sequences of length {model.max_seen_len}, expected max 8"

    def test_with_mini_llama(self):
        torch.manual_seed(0)
        model = get_mini_llama(
            vocab_size=128,
            context_length=16,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            d_ff=64,
            rope_theta=10000.0,
        )
        model.eval()
        prompt = [1, 2, 3]
        for temperature in (0.0, 0.1, 0.7, 1.0):
            out = run_generate(model, prompt, max_new_tokens=40, temperature=temperature, context_length=16)
            assert len(out) == 40 and all(0 <= t < 128 for t in out)
        # Greedy decoding is deterministic.
        a = run_generate(model, prompt, max_new_tokens=20, temperature=0.0, context_length=16)
        b = run_generate(model, prompt, max_new_tokens=20, temperature=0.0, context_length=16)
        assert a == b
