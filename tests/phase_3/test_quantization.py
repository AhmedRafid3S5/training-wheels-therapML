"""Phase 3 · Task 3 — Post-Training Quantization (naive INT8)

Tests validate:

* per-tensor absmax INT8 quantization: ``s = max|W| / 127``,
  ``W_q = round(W / s)`` stored as int8, ``W ≈ W_q · s``,
* error bounds: |W − W_q·s| ≤ s/2, and how a single outlier wrecks the
  precision of every other weight (the reason GPTQ / AWQ / per-channel
  schemes exist),
* the quantized Mini-Llama: int8 storage for every weight matrix, tied weights
  stay tied (≈4× smaller), the original model is untouched, outputs equal a
  "fake-quantized" fp32 model, KV-cached decoding still works, and
  inference is not drastically slower.

Perplexity before/after quantization on a *trained* model is checked in
``test_inference_integration.py``.
"""

import copy

import numpy as np
import pytest
import torch

from tests.adapters import get_mini_llama
from tests.adapters_0 import run_decode, run_dequantize_int8, run_quantize_int8, run_quantize_model_int8
from tests.phase_2.common import benchmark
from tests.phase_3.common import SMALL_CONFIG, state_dict_bytes


def _reference_quantize(w):
    scale = w.abs().max() / 127
    return torch.round(w / scale).to(torch.int8), scale


def _unique_2d_params(model):
    seen, out = set(), []
    for p in model.parameters():
        if p.dim() == 2 and id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


# ============================================================================
# Tensor-level quantization
# ============================================================================


class TestQuantizeTensor:
    @pytest.mark.parametrize("shape", [(100,), (64, 32), (4, 16, 8)])
    def test_dtypes_and_shapes(self, shape):
        w = torch.randn(shape, generator=torch.Generator().manual_seed(0))
        w_q, s = run_quantize_int8(w)
        assert w_q.dtype == torch.int8 and w_q.shape == w.shape
        assert s.dtype == torch.float32 and s.dim() == 0
        deq = run_dequantize_int8(w_q, s)
        assert deq.dtype == torch.float32 and deq.shape == w.shape

    def test_matches_absmax_formula(self):
        w = torch.randn(256, 128, generator=torch.Generator().manual_seed(1))
        w_q, s = run_quantize_int8(w)
        ref_q, ref_s = _reference_quantize(w)
        assert s.item() == pytest.approx(ref_s.item(), rel=1e-6)
        diff = (w_q.int() - ref_q.int()).abs()
        assert diff.max() <= 1, "Quantized values are off by more than one step"
        assert (diff > 0).float().mean() < 1e-3, "Too many quantized values differ from round(W / s)"

    def test_range_uses_full_int8_scale(self):
        w = torch.randn(64, 64, generator=torch.Generator().manual_seed(2))
        w[3, 5] = -10.0  # the absmax entry is negative
        w_q, _ = run_quantize_int8(w)
        assert w_q.min() >= -127 and w_q.max() <= 127
        assert w_q[3, 5] == -127

    def test_dequantize_is_wq_times_scale(self):
        w_q = torch.randint(-127, 128, (32, 16), dtype=torch.int8, generator=torch.Generator().manual_seed(3))
        s = torch.tensor(0.0123, dtype=torch.float32)
        torch.testing.assert_close(run_dequantize_int8(w_q, s), w_q.float() * s, rtol=0, atol=0)

    def test_roundtrip_error_at_most_half_a_step(self):
        w = torch.randn(512, 256, generator=torch.Generator().manual_seed(4)) * 0.02
        w_q, s = run_quantize_int8(w)
        err = (run_dequantize_int8(w_q, s) - w).abs()
        assert err.max() <= s.item() / 2 * (1 + 1e-4)  # float32 rounding slack

    def test_relative_error_is_small_for_gaussian_weights(self):
        w = torch.randn(512, 512, generator=torch.Generator().manual_seed(5))
        w_q, s = run_quantize_int8(w)
        rel = ((run_dequantize_int8(w_q, s) - w).norm() / w.norm()).item()
        print(f"\nrelative Frobenius error (Gaussian 512x512): {rel:.4f}")
        assert rel < 0.015

    def test_outlier_destroys_precision(self):
        """One large outlier inflates s, so every *other* weight loses precision."""
        w = torch.randn(256, 256, generator=torch.Generator().manual_seed(6))
        w_out = w.clone()
        w_out[0, 0] = 100.0
        rest = torch.ones_like(w, dtype=torch.bool)
        rest[0, 0] = False

        def err(x):
            x_q, s = run_quantize_int8(x)
            return (run_dequantize_int8(x_q, s) - x)[rest].abs().mean().item()

        clean, outlier = err(w), err(w_out)
        print(f"\nmean abs error on normal weights: clean={clean:.5f}  with outlier={outlier:.5f}")
        assert outlier > 10 * clean

    def test_all_zero_tensor(self):
        w_q, s = run_quantize_int8(torch.zeros(8, 8))
        deq = run_dequantize_int8(w_q, s)
        assert torch.isfinite(s) and torch.equal(deq, torch.zeros(8, 8))


# ============================================================================
# Quantizing the Mini-Llama
# ============================================================================


class TestQuantizedModel:
    @pytest.fixture
    def quantized(self, small_llama):
        return run_quantize_model_int8(small_llama).eval()

    @pytest.fixture
    def tokens(self):
        return torch.randint(0, SMALL_CONFIG["vocab_size"], (2, 32), generator=torch.Generator().manual_seed(0))

    def test_original_model_unchanged(self, small_llama):
        before = {k: v.clone() for k, v in small_llama.state_dict().items()}
        run_quantize_model_int8(small_llama)
        after = small_llama.state_dict()
        assert before.keys() == after.keys()
        for k, v in before.items():
            assert after[k].dtype == v.dtype and torch.equal(after[k], v), f"{k} was modified"

    def test_every_weight_matrix_stored_as_int8(self, small_llama, quantized):
        int8 = [t for t in quantized.state_dict().values() if t.dtype == torch.int8]
        for p in _unique_2d_params(small_llama):
            expected, _ = run_quantize_int8(p.detach())
            assert any(t.shape == expected.shape and torch.equal(t, expected) for t in int8), (
                f"No int8 tensor in the quantized state_dict matches the weight of shape {tuple(p.shape)}"
            )
        n_int8 = sum(t.numel() for t in {(t.data_ptr(), t.numel()): t for t in int8}.values())
        n_2d = sum(p.numel() for p in _unique_2d_params(small_llama))
        assert n_int8 == n_2d, f"{n_int8:,} int8 values stored but the model has {n_2d:,} weight-matrix values"

    def test_size_reduction(self, small_llama, quantized):
        fp32, int8 = state_dict_bytes(small_llama), state_dict_bytes(quantized)
        n_2d = sum(p.numel() for p in _unique_2d_params(small_llama))
        print(f"\nmodel size  fp32={fp32:,} B  int8={int8:,} B  reduction={fp32 / int8:.2f}x")
        # 4 bytes → 1 byte per weight-matrix entry, with the tied matrix stored once
        # and no fp32 copy kept.
        assert fp32 - int8 >= 0.98 * 3 * n_2d, (
            "Expected ~3 bytes saved per weight entry. Did you keep an fp32 copy or untie the embedding?"
        )
        assert fp32 / int8 >= 3.0

    def test_output_close_to_fp32(self, small_llama, quantized, tokens):
        with torch.no_grad():
            ref, out = small_llama(tokens), quantized(tokens)
        assert out.shape == ref.shape and out.dtype == torch.float32
        rel = ((out - ref).norm() / ref.norm()).item()
        print(f"\nrelative logit error int8 vs fp32: {rel:.4f}")
        assert rel < 0.05

    def test_equals_fake_quantized_fp32_model(self, small_llama, quantized, tokens):
        """The int8 model must compute exactly what an fp32 model computes with
        each weight matrix replaced by dequantize(quantize(W))."""
        fake = copy.deepcopy(small_llama)
        with torch.no_grad():
            for p in _unique_2d_params(fake):
                p.copy_(run_dequantize_int8(*run_quantize_int8(p)))
            np.testing.assert_allclose(quantized(tokens).numpy(), fake(tokens).numpy(), atol=1e-5, rtol=1e-4)

    def test_kv_cached_decoding_works(self, quantized):
        kw = {"temperature": 0.0, "context_length": SMALL_CONFIG["context_length"]}
        cached = run_decode(quantized, [1, 2, 3], 30, use_kv_cache=True, **kw)
        uncached = run_decode(quantized, [1, 2, 3], 30, use_kv_cache=False, **kw)
        assert len(cached) == 30 and cached == uncached


# ============================================================================
# Latency
# ============================================================================


class TestQuantizationPerformance:
    def test_quantize_tensor_speed(self):
        w = torch.randn(2048, 2048, generator=torch.Generator().manual_seed(0))
        ours = benchmark(lambda: run_quantize_int8(w), "run_quantize_int8", n_iter=20)
        ref = benchmark(lambda: _reference_quantize(w), "reference", n_iter=20)
        print(f"\nquantize 2048x2048: ours={ours * 1e3:.2f} ms  ref={ref * 1e3:.2f} ms  ratio={ours / ref:.1f}x")
        assert ours <= 10 * ref

    def test_inference_speed_change(self):
        cfg = dict(SMALL_CONFIG, vocab_size=1024, context_length=128, d_model=256, num_layers=4, num_heads=8, d_ff=688)
        torch.manual_seed(0)
        model = get_mini_llama(**cfg).eval()
        quantized = run_quantize_model_int8(model).eval()
        x = torch.randint(0, cfg["vocab_size"], (8, cfg["context_length"]), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            fp32 = benchmark(lambda: model(x), "fp32 forward", n_iter=20)
            int8 = benchmark(lambda: quantized(x), "int8 forward", n_iter=20)
        print(f"\nforward (8x128): fp32={fp32 * 1e3:.1f} ms  int8={int8 * 1e3:.1f} ms  ratio={int8 / fp32:.2f}x")
        assert int8 <= 4 * fp32, "On-the-fly dequantization should not make inference >4x slower"
