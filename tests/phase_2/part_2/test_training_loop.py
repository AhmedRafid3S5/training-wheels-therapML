"""Phase 2 · Part 2 · Task 3 — The Training Loop

Tests validate the training-loop building blocks:

* learning-rate schedule: linear warmup (first 10%) + cosine decay,
* gradient clipping (matches ``torch.nn.utils.clip_grad_norm_``),
* mixed precision: the model works under ``torch.autocast`` (bf16 on CPU),
  trains with fp16 + ``GradScaler`` and is benchmarked fp32 vs. bf16
  (CUDA-only tests are skipped without a GPU).
"""

import math
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.adapters import get_adam_cls, get_mini_llama, run_gradient_clipping, run_lr_schedule

MAX_LR, MIN_LR, TOTAL = 1e-3, 1e-4, 1000
WARMUP = int(0.1 * TOTAL)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _expected_lr(step):
    if step < WARMUP:
        return MAX_LR * step / WARMUP
    if step <= TOTAL:
        progress = (step - WARMUP) / (TOTAL - WARMUP)
        return MIN_LR + 0.5 * (1 + math.cos(math.pi * progress)) * (MAX_LR - MIN_LR)
    return MIN_LR


# ============================================================================
# Learning-rate schedule
# ============================================================================


class TestLRSchedule:
    @pytest.mark.parametrize("step", [0, 1, 50, 99, 100, 101, 250, 550, 999, 1000, 1001, 5000])
    def test_values(self, step):
        assert run_lr_schedule(step, TOTAL, MAX_LR, MIN_LR) == pytest.approx(_expected_lr(step), rel=1e-6, abs=1e-12)

    def test_warmup_is_linear_and_increasing(self):
        lrs = [run_lr_schedule(s, TOTAL, MAX_LR, MIN_LR) for s in range(WARMUP + 1)]
        diffs = np.diff(lrs)
        assert (diffs > 0).all()
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-6)

    def test_peak_at_end_of_warmup(self):
        lrs = [run_lr_schedule(s, TOTAL, MAX_LR, MIN_LR) for s in range(TOTAL + 1)]
        assert int(np.argmax(lrs)) == WARMUP
        assert lrs[WARMUP] == pytest.approx(MAX_LR)

    def test_cosine_decay_is_monotonic_to_min(self):
        lrs = [run_lr_schedule(s, TOTAL, MAX_LR, MIN_LR) for s in range(WARMUP, TOTAL + 1)]
        assert (np.diff(lrs) < 0).all()
        assert lrs[-1] == pytest.approx(MIN_LR)
        # Halfway through the cosine phase the LR is the midpoint of max and min.
        mid = WARMUP + (TOTAL - WARMUP) // 2
        assert run_lr_schedule(mid, TOTAL, MAX_LR, MIN_LR) == pytest.approx((MAX_LR + MIN_LR) / 2, rel=1e-3)

    def test_custom_warmup_fraction(self):
        assert run_lr_schedule(200, 1000, 1.0, 0.0, warmup_frac=0.2) == pytest.approx(1.0)
        assert run_lr_schedule(100, 1000, 1.0, 0.0, warmup_frac=0.2) == pytest.approx(0.5)


# ============================================================================
# Gradient clipping
# ============================================================================


def _params_with_grads(scale: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(*s, generator=g)) for s in [(5, 3), (7,), (2, 4, 6)]]
    for p in params:
        p.grad = torch.randn(p.shape, generator=g) * scale
    return params


class TestGradientClipping:
    @pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
    def test_matches_pytorch(self, scale):
        ours, ref = _params_with_grads(scale), _params_with_grads(scale)
        norm_ours = run_gradient_clipping(ours, max_norm=1.0)
        norm_ref = torch.nn.utils.clip_grad_norm_(ref, max_norm=1.0)
        assert float(norm_ours) == pytest.approx(norm_ref.item(), rel=1e-5)
        for a, b in zip(ours, ref):
            np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), atol=1e-6, rtol=1e-5)

    def test_clipped_norm_equals_max_norm(self):
        params = _params_with_grads(100.0)
        run_gradient_clipping(params, max_norm=1.0)
        total = torch.sqrt(sum(p.grad.pow(2).sum() for p in params)).item()
        assert total == pytest.approx(1.0, rel=1e-4)

    def test_small_gradients_untouched(self):
        params = _params_with_grads(1e-4)
        before = [p.grad.clone() for p in params]
        run_gradient_clipping(params, max_norm=1.0)
        for p, b in zip(params, before):
            np.testing.assert_allclose(p.grad.numpy(), b.numpy(), rtol=1e-6)

    def test_skips_params_without_grad(self):
        params = _params_with_grads(100.0)
        frozen = torch.nn.Parameter(torch.randn(3))
        norm = run_gradient_clipping(params + [frozen], max_norm=1.0)
        assert frozen.grad is None
        assert math.isfinite(float(norm))


# ============================================================================
# Mixed precision
# ============================================================================

SMALL = {
    "vocab_size": 256,
    "context_length": 32,
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 4,
    "num_kv_heads": 2,
    "d_ff": 172,
    "rope_theta": 10000.0,
}


def _batch(cfg, batch_size, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, cfg["vocab_size"], (batch_size, cfg["context_length"] + 1), generator=g).to(device)
    return idx[:, :-1], idx[:, 1:]


def _loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))


class TestMixedPrecision:
    def test_bf16_autocast_forward_close_to_fp32(self):
        torch.manual_seed(0)
        model = get_mini_llama(**SMALL)
        x, y = _batch(SMALL, 4)
        with torch.no_grad():
            loss_fp32 = _loss(model, x, y).item()
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss_bf16 = _loss(model, x, y).item()
        print(f"\nloss fp32={loss_fp32:.4f}  bf16={loss_bf16:.4f}")
        assert loss_bf16 == pytest.approx(loss_fp32, rel=0.02)

    def test_bf16_autocast_backward_gives_finite_fp32_grads(self):
        torch.manual_seed(0)
        model = get_mini_llama(**SMALL)
        x, y = _batch(SMALL, 4)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = _loss(model, x, y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.dtype == torch.float32, f"{name} master weights must stay fp32"
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name} has missing/non-finite grad"

    @requires_cuda
    def test_fp16_gradscaler_training(self):
        torch.manual_seed(0)
        model = get_mini_llama(**SMALL).cuda()
        opt = get_adam_cls()(model.parameters(), lr=3e-3, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda")
        x, y = _batch(SMALL, 16, device="cuda")
        losses = []
        for _ in range(50):
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = _loss(model, x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            run_gradient_clipping(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        assert all(math.isfinite(v) for v in losses)
        assert losses[-1] < losses[0] - 0.5, (
            f"fp16 training did not reduce the loss: {losses[0]:.3f} → {losses[-1]:.3f}"
        )

    @requires_cuda
    def test_fp32_vs_bf16_training_speed(self):
        cfg = dict(SMALL, vocab_size=8192, context_length=256, d_model=512, num_layers=4, num_heads=8, d_ff=1376)

        def steps_per_sec(dtype):
            torch.manual_seed(0)
            model = get_mini_llama(**cfg).cuda()
            opt = get_adam_cls()(model.parameters(), lr=1e-4)
            x, y = _batch(cfg, 16, device="cuda")

            def step():
                opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
                    loss = _loss(model, x, y)
                loss.backward()
                opt.step()

            for _ in range(3):
                step()
            torch.cuda.synchronize()
            t0, n = time.perf_counter(), 20
            for _ in range(n):
                step()
            torch.cuda.synchronize()
            return n / (time.perf_counter() - t0)

        fp32, bf16 = steps_per_sec(torch.float32), steps_per_sec(torch.bfloat16)
        print(f"\nsteps/sec  fp32={fp32:.2f}  bf16={bf16:.2f}  speedup={bf16 / fp32:.2f}x")
        assert bf16 >= 0.9 * fp32, "bf16 autocast should not be meaningfully slower than fp32 on GPU"
