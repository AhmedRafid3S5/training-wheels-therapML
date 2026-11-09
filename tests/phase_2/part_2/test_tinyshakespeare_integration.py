"""Phase 2 · Part 2 · Tasks 3–5 — Integration Test: Mini-Llama on TinyShakespeare

End-to-end check that every piece works together:

    your BPE tokenizer (Part 1)  →  your MiniLlama  →  your MyAdamW (Phase 1)
    + your LR schedule + your gradient clipping  →  your perplexity / generation

A small model is trained for a few hundred steps on a TinyShakespeare slice
(CPU-friendly; uses CUDA if available). It must:

* start at loss ≈ ln(V) (random-noise stage),
* steadily reduce the training loss with stable, finite gradient norms,
* reach a held-out perplexity clearly better than a unigram baseline,
* produce decodable text samples at T ∈ {0.1, 0.7, 1.0}.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.adapters import (
    get_adam_cls,
    get_mini_llama,
    run_bpe_decode,
    run_bpe_encode,
    run_generate,
    run_gradient_clipping,
    run_lr_schedule,
    run_perplexity,
)
from tests.common import load_tinyshakespeare

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


def _batches(data: torch.Tensor, ctx: int, batch_size: int, generator: torch.Generator):
    while True:
        starts = torch.randint(0, len(data) - ctx - 1, (batch_size,), generator=generator)
        x = torch.stack([data[s : s + ctx] for s in starts])
        y = torch.stack([data[s + 1 : s + ctx + 1] for s in starts])
        yield x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def _val_perplexity(model, data: torch.Tensor, ctx: int) -> float:
    n = (len(data) - 1) // ctx
    x = data[: n * ctx].view(n, ctx).to(DEVICE)
    y = data[1 : n * ctx + 1].view(n, ctx).to(DEVICE)
    model.eval()
    logits = torch.cat([model(x[i : i + 64]) for i in range(0, n, 64)])
    model.train()
    return run_perplexity(logits.float().cpu(), y.cpu())


def _unigram_perplexity(train: torch.Tensor, val: torch.Tensor) -> float:
    counts = torch.bincount(train, minlength=VOCAB_SIZE).double() + 1.0  # add-one smoothing
    log_p = torch.log(counts / counts.sum())
    return math.exp(-log_p[val].mean().item())


@pytest.fixture(scope="module")
def run(train_bpe_cached, heldout_text):
    tokenizer = train_bpe_cached(VOCAB_SIZE)
    train_data = torch.tensor(run_bpe_encode(tokenizer, load_tinyshakespeare()[:TRAIN_CHARS]), dtype=torch.long)
    val_data = torch.tensor(run_bpe_encode(tokenizer, heldout_text), dtype=torch.long)

    torch.manual_seed(0)
    model = get_mini_llama(**CONFIG).to(DEVICE)
    optimizer = get_adam_cls()(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.1)
    batches = _batches(train_data, CONFIG["context_length"], BATCH_SIZE, torch.Generator().manual_seed(0))

    losses, grad_norms = [], []
    model.train()
    for step in range(STEPS):
        lr = run_lr_schedule(step, STEPS, MAX_LR, MIN_LR)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = next(batches)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        grad_norms.append(float(run_gradient_clipping(model.parameters(), max_norm=1.0)))
        optimizer.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"\nstep {step:4d}  loss {loss.item():.3f}  lr {lr:.2e}  grad_norm {grad_norms[-1]:.3f}", end="")

    val_ppl = _val_perplexity(model, val_data, CONFIG["context_length"])
    unigram_ppl = _unigram_perplexity(train_data, val_data)
    print(f"\nval perplexity {val_ppl:.2f}   unigram baseline {unigram_ppl:.2f}")
    return {
        "model": model.eval(),
        "tokenizer": tokenizer,
        "losses": losses,
        "grad_norms": grad_norms,
        "val_ppl": val_ppl,
        "unigram_ppl": unigram_ppl,
    }


class TestTinyShakespeareIntegration:
    def test_initial_loss_is_near_ln_vocab(self, run):
        assert abs(run["losses"][0] - math.log(VOCAB_SIZE)) < 1.0

    def test_training_loss_decreases(self, run):
        losses = run["losses"]
        start, end = sum(losses[:10]) / 10, sum(losses[-50:]) / 50
        print(f"\ntrain loss: first-10 avg {start:.3f} → last-50 avg {end:.3f}")
        assert end < start - 1.5, f"Training loss barely moved: {start:.3f} → {end:.3f}"

    def test_gradient_norms_are_healthy(self, run):
        norms = run["grad_norms"]
        assert all(math.isfinite(n) for n in norms), "Non-finite gradient norm encountered"
        assert max(norms) < 100.0, f"Exploding gradients: max norm {max(norms):.1f}"
        assert min(norms[: len(norms) // 2]) > 1e-4, "Vanishing gradients early in training"

    def test_val_perplexity_beats_unigram_baseline(self, run):
        assert run["val_ppl"] < 0.5 * run["unigram_ppl"], (
            f"Validation perplexity {run['val_ppl']:.2f} is not clearly better than the "
            f"unigram baseline {run['unigram_ppl']:.2f} — the model is not using context"
        )

    @pytest.mark.parametrize("temperature", [0.1, 0.7, 1.0])
    def test_generation_samples(self, run, temperature):
        tok = run["tokenizer"]
        prompt = run_bpe_encode(tok, "KING HENRY:\n")
        torch.manual_seed(0)
        new_ids = run_generate(run["model"], prompt, max_new_tokens=100, temperature=temperature, context_length=64)
        text = run_bpe_decode(tok, prompt + new_ids)
        print(f"\n--- T={temperature} ---\n{text}\n")
        assert len(new_ids) == 100
        assert isinstance(text, str) and text.startswith("KING HENRY:")
        # A trained model should mostly produce letters and spaces, not byte soup.
        letters = sum(ch.isalpha() or ch.isspace() for ch in text)
        assert letters / len(text) > 0.7
