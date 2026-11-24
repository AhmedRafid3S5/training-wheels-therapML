"""Phase 3 · Tasks 2–3 — Integration Test: Inference with a Trained Mini-Llama

Trains the same small Mini-Llama as the Phase 2 integration test on
TinyShakespeare, with your BPE tokenizer, optimizer, LR schedule and gradient
clipping (CPU-friendly, uses CUDA if available). Then:

* **Decoding:** 10 completions per strategy (greedy, T ∈ {0.1, 0.5, 1.0, 1.5},
  top-k, top-p, combined), all with your KV-cache. Greedy gives identical
  completions, diversity grows with temperature, and low-temperature samples
  read like text.
* **Quantization:** the INT8 model is ≈4× smaller, its validation perplexity
  is within 3% of fp32, it agrees with the fp32 model's top-1 prediction on
  ≥95% of positions, and it still generates text. Size, perplexity and speed
  are printed as a table.
"""

import time

import pytest
import torch
import torch.nn.functional as F

from tests.adapters import (
    get_adam_cls,
    get_mini_llama,
    run_bpe_decode,
    run_bpe_encode,
    run_gradient_clipping,
    run_lr_schedule,
    run_perplexity,
    run_train_bpe,
)
from tests.adapters_0 import run_decode, run_quantize_model_int8
from tests.common import load_tinyshakespeare
from tests.phase_2.common import BPE_TRAIN_CHARS, HELDOUT_END, HELDOUT_START
from tests.phase_3.common import distinct_n, state_dict_bytes

VOCAB_SIZE = 512
TRAIN_CHARS = 300_000
CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "context_length": 64,
    "d_model": 128,
    "num_layers": 4,
    "num_heads": 4,
    "num_kv_heads": 2,
    "d_ff": 344,
    "rope_theta": 10000.0,
}
STEPS, BATCH_SIZE, MAX_LR, MIN_LR = 600, 32, 3e-3, 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = "KING HENRY:\n"
N_COMPLETIONS, N_NEW = 10, 48
STRATEGIES = {
    "greedy": {"temperature": 0.0},
    "T=0.1": {"temperature": 0.1},
    "T=0.5": {"temperature": 0.5},
    "T=1.0": {"temperature": 1.0},
    "T=1.5": {"temperature": 1.5},
    "top-k=20": {"temperature": 1.0, "top_k": 20},
    "top-p=0.9": {"temperature": 1.0, "top_p": 0.9},
    "combined": {"temperature": 0.8, "top_k": 40, "top_p": 0.9},
}


def _batches(data, ctx, batch_size, generator):
    while True:
        starts = torch.randint(0, len(data) - ctx - 1, (batch_size,), generator=generator)
        x = torch.stack([data[s : s + ctx] for s in starts])
        y = torch.stack([data[s + 1 : s + ctx + 1] for s in starts])
        yield x.to(DEVICE), y.to(DEVICE)


def _val_windows(data, ctx):
    n = (len(data) - 1) // ctx
    return data[: n * ctx].view(n, ctx), data[1 : n * ctx + 1].view(n, ctx)


@torch.no_grad()
def _val_logits(model, x):
    return torch.cat([model(x[i : i + 64]).float().cpu() for i in range(0, len(x), 64)])


def _letter_ratio(text):
    return sum(ch.isalpha() or ch.isspace() for ch in text) / max(len(text), 1)


@pytest.fixture(scope="module")
def trained():
    text = load_tinyshakespeare()
    tokenizer = run_train_bpe(text[:BPE_TRAIN_CHARS], VOCAB_SIZE)
    train_data = torch.tensor(run_bpe_encode(tokenizer, text[:TRAIN_CHARS]), dtype=torch.long)
    val_data = torch.tensor(run_bpe_encode(tokenizer, text[HELDOUT_START:HELDOUT_END]), dtype=torch.long)

    torch.manual_seed(0)
    model = get_mini_llama(**CONFIG).to(DEVICE)
    optimizer = get_adam_cls()(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.1)
    batches = _batches(train_data, CONFIG["context_length"], BATCH_SIZE, torch.Generator().manual_seed(0))
    model.train()
    for step in range(STEPS):
        lr = run_lr_schedule(step, STEPS, MAX_LR, MIN_LR)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = next(batches)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"\nstep {step:4d}  loss {loss.item():.3f}", end="")
    model = model.cpu().eval()
    return {"model": model, "tokenizer": tokenizer, "val": _val_windows(val_data, CONFIG["context_length"])}


@pytest.fixture(scope="module")
def quantized(trained):
    return run_quantize_model_int8(trained["model"]).eval()


@pytest.fixture(scope="module")
def completions(trained):
    """{strategy: 10 completions (token ids)} generated with the KV-cache."""
    prompt = run_bpe_encode(trained["tokenizer"], PROMPT)
    assert len(prompt) + N_NEW <= CONFIG["context_length"]
    out = {}
    with torch.no_grad():
        for name, kw in STRATEGIES.items():
            torch.manual_seed(0)
            out[name] = [
                run_decode(
                    trained["model"],
                    prompt,
                    N_NEW,
                    context_length=CONFIG["context_length"],
                    use_kv_cache=True,
                    **kw,
                )
                for _ in range(N_COMPLETIONS)
            ]
    return prompt, out


# ============================================================================
# Decoding strategies on the trained model
# ============================================================================


class TestDecodingIntegration:
    def test_samples(self, trained, completions):
        prompt, out = completions
        tok = trained["tokenizer"]
        print("\n  strategy     unique/10  distinct-2  letters")
        for name, comps in out.items():
            texts = [run_bpe_decode(tok, prompt + c) for c in comps]
            assert all(len(c) == N_NEW for c in comps)
            assert all(t.startswith(PROMPT) for t in texts)
            ratio = sum(_letter_ratio(t) for t in texts) / len(texts)
            print(f"  {name:<12} {len({tuple(c) for c in comps}):>9} {distinct_n(comps, 2):>11.3f} {ratio:>8.2f}")
        for name in ("greedy", "T=0.5", "T=1.5", "combined"):
            print(f"\n--- {name} (completion 1 of 10) ---\n{run_bpe_decode(tok, prompt + out[name][0])}")

    def test_greedy_is_deterministic(self, completions):
        _, out = completions
        assert len({tuple(c) for c in out["greedy"]}) == 1

    def test_diversity_grows_with_temperature(self, completions):
        _, out = completions
        d = {name: distinct_n(out[name], 2) for name in ("T=0.1", "T=1.0", "T=1.5")}
        assert d["T=0.1"] < d["T=1.0"] < d["T=1.5"], f"distinct-2 should grow with temperature: {d}"
        assert len({tuple(c) for c in out["T=1.0"]}) == N_COMPLETIONS

    def test_low_temperature_is_coherent(self, trained, completions):
        prompt, out = completions
        for name in ("greedy", "T=0.5", "combined"):
            texts = [run_bpe_decode(trained["tokenizer"], prompt + c) for c in out[name]]
            ratio = sum(_letter_ratio(t) for t in texts) / len(texts)
            assert ratio > 0.75, f"{name}: only {ratio:.0%} letters/whitespace"


# ============================================================================
# INT8 quantization of the trained model
# ============================================================================


class TestQuantizationIntegration:
    def test_size_perplexity_speed_report(self, trained, quantized):
        model, (x, y) = trained["model"], trained["val"]

        def report(m):
            t0 = time.perf_counter()
            logits = _val_logits(m, x)
            elapsed = time.perf_counter() - t0
            return state_dict_bytes(m), run_perplexity(logits, y), x.numel() / elapsed, logits

        size32, ppl32, speed32, logits32 = report(model)
        size8, ppl8, speed8, logits8 = report(quantized)
        agreement = (logits32.argmax(-1) == logits8.argmax(-1)).float().mean().item()
        print("\n            size (KB)   val ppl   tokens/s")
        print(f"  fp32  {size32 / 1024:>12.1f} {ppl32:>9.3f} {speed32:>10.0f}")
        print(f"  int8  {size8 / 1024:>12.1f} {ppl8:>9.3f} {speed8:>10.0f}")
        print(
            f"  size {size32 / size8:.2f}x smaller, perplexity {100 * (ppl8 / ppl32 - 1):+.2f}%, "
            f"speed {speed8 / speed32:.2f}x, top-1 agreement {agreement:.1%}"
        )

        assert size32 / size8 >= 3.5
        assert ppl8 <= 1.03 * ppl32, f"INT8 perplexity {ppl8:.3f} is >3% worse than fp32 {ppl32:.3f}"
        assert agreement >= 0.95

    def test_quantized_model_generates_text(self, trained, quantized):
        tok = trained["tokenizer"]
        prompt = run_bpe_encode(tok, PROMPT)
        with torch.no_grad():
            new = run_decode(
                quantized, prompt, N_NEW, temperature=0.0, context_length=CONFIG["context_length"], use_kv_cache=True
            )
        text = run_bpe_decode(tok, prompt + new)
        print(f"\n--- int8 greedy ---\n{text}")
        assert len(new) == N_NEW and text.startswith(PROMPT)
        assert _letter_ratio(text) > 0.75
