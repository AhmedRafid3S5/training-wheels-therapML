# Training Modules - TherapML

A structured training program for the TherapML team, covering foundational ML mathematics through hands-on PyTorch implementation. Fork this repo, implement each task inside `therapml/`, wire your code through `tests/adapters.py`, and run the automated test suite to validate both **correctness** and **performance**.

---

## Project Structure

```
therapml/                            # Your implementation goes here
├── __init__.py
tests/
├── adapters.py                      # Glue layer — connect your code to the tests
├── conftest.py                      # Shared pytest fixtures & snapshot utilities
├── common.py                        # Shared constants (FIXTURES_PATH)
├── test_basic_tensor.py             # Pre-phase: tensor multiply & dot product
├── fixtures/                        # Pre-generated .npy data, reference checkpoint, tinyshakespeare.txt
├── _snapshots/                      # Reference outputs for RoPE / attention / transformer snapshot tests
│
└── phase_1/
│   ├── part_1/                      # Week 1 — Mathematical Foundations (NumPy)
│   │   ├── test_pca.py              # Task 1: PCA from scratch
│   │   ├── test_svd.py              # Task 2: SVD & low-rank approximation
│   │   ├── test_mle.py              # Task 4: Maximum likelihood estimation
│   │   ├── test_info_theory.py      # Task 6: Entropy, KL, cross-entropy, MI
│   │   ├── test_numerical_stability.py  # Task 7: Log-sum-exp stability
│   │   └── test_optimizers.py       # Task 8: SGD, Momentum, Adam (NumPy)
│   │
│   └── part_2/                      # Week 2 — PyTorch Toolbox
│       ├── test_autograd.py         # Task 1: Scalar autograd engine (Value class)
│       ├── test_linear.py           # Task 2: MyLinear + Kaiming/Xavier init
│       ├── test_activations.py      # Task 3: Sigmoid, ReLU, GELU, Softmax, SwiGLU
│       ├── test_loss.py             # Task 4: CrossEntropy, KL divergence
│       ├── test_optimizers.py       # Task 5: MySGD (momentum) + MyAdamW
│       ├── test_dropout.py          # Task 6: Inverted dropout
│       ├── test_normalization.py    # Task 7: LayerNorm, RMSNorm
│       └── test_mnist_integration.py  # Task 8: MNIST ≥97% with custom components
│
└── phase_2/
    ├── conftest.py                  # Shared fixtures (TinyShakespeare slices, cached BPE training)
    ├── common.py                    # Shared helpers (random attention weights, GQA→MHA expansion, benchmark)
    │
    ├── part_1/                      # Week 3 — Tokenization & Attention
    │   ├── test_bpe.py              # Task 1: BPE train / encode / decode, round-trip
    │   ├── test_vocab_size.py       # Task 2: Vocabulary-size experiment
    │   ├── test_special_tokens.py   # Task 3: <PAD>/<BOS>/<EOS>/<UNK> + Unicode/emoji edge cases
    │   ├── test_scaled_dot_product_attention.py  # Task 4: softmax(QKᵀ/√d_k)V
    │   ├── test_causal_attention.py # Task 5: Causal self-attention with Q/K/V projections
    │   ├── test_multihead_attention.py  # Task 6: MHA vs torch.nn.MultiheadAttention
    │   ├── test_gqa.py              # Task 7: Grouped Query Attention
    │   ├── test_rope.py             # Task 8: Rotary Positional Embeddings
    │   └── test_kv_cache.py         # Task 9: KV-cache correctness, memory, speed-up
    │
    └── part_2/                      # Week 4 — Mini-Llama from Scratch
        ├── test_transformer_block.py  # Tasks 1–2: Pre-norm RMSNorm + GQA/RoPE + SwiGLU block
        ├── test_mini_llama.py       # Tasks 1–2: Full LM, weight tying, parameter count & audit
        ├── test_training_loop.py    # Task 3: LR schedule, gradient clipping, mixed precision
        ├── test_evaluation.py       # Task 5: Perplexity, temperature sampling / generation
        └── test_tinyshakespeare_integration.py  # Tasks 3–5: End-to-end training on TinyShakespeare
```

---

## How It Works

Every test imports thin adapter functions from `tests/adapters.py`. Each adapter starts as `raise NotImplementedError`. Your job:

1. **Implement** the logic inside `therapml/` (create whatever modules and classes you need).
2. **Wire** your code into `tests/adapters.py` — replace `raise NotImplementedError` with imports and calls to your implementation.
3. **Run** the tests to check both accuracy and latency.

> **Rule:** Never put implementation logic in `tests/adapters.py`. It should contain only imports and one-liner wrappers.

---

## Getting Started

### 1. Fork & clone

```bash
git clone <your-fork-url>
cd training-wheels
```

### 2. Set up the environment

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 3. Run the tests

```bash
# Run everything
pytest -v

# Run a specific phase/part
pytest tests/phase_1/part_1/ -v        # Week 1 — Math foundations
pytest tests/phase_1/part_2/ -v        # Week 2 — PyTorch toolbox
pytest tests/phase_2/part_1/ -v        # Week 3 — Tokenization & attention
pytest tests/phase_2/part_2/ -v        # Week 4 — Mini-Llama

# Run a single task
pytest tests/phase_1/part_1/test_pca.py -v

# Skip performance benchmarks during development
pytest tests/phase_1/part_1/ -v -k "not Performance"

# Skip the slow end-to-end training tests too
pytest tests/phase_2/ -v -k "not Performance and not Integration"
```

> Run `pytest` from the repository root. Snapshot tests load reference outputs from `tests/_snapshots/` using a relative path.

---

## Phase 1 — Task Reference

### Part 1: Mathematical Foundations (Week 1)

All implementations in this part use **NumPy only** (no PyTorch).

| Test File | Task | What You Implement | Key Acceptance Criteria |
|---|---|---|---|
| `test_pca.py` | PCA from Scratch | `run_pca(data, k)` | Correct centering, eigenvalue ordering, reconstruction error ↓ with k, ≤50× latency |
| `test_svd.py` | SVD & Low-Rank | `run_truncated_svd(matrix, rank)` | U·S·Vᵀ = reconstruction, error ↓ with rank, ≤50× latency |
| `test_mle.py` | MLE Gaussian | `run_mle_gaussian(data)` | μ, σ² match closed-form within tolerance |
| `test_info_theory.py` | Info Theory Toolkit | `run_entropy`, `run_kl_divergence`, `run_cross_entropy_info`, `run_mutual_information` | H(fair coin)=1 bit, KL≥0, H(P,Q)=H(P)+KL(P‖Q) |
| `test_numerical_stability.py` | Log-Sum-Exp | `run_log_sum_exp(x)` | No overflow/underflow on extreme inputs, ≤20× latency |
| `test_optimizers.py` | Optimizer Comparison | `run_numpy_sgd_fit`, `run_numpy_sgd_momentum_fit`, `run_numpy_adam_fit` | Convergence on linear regression, momentum beats vanilla SGD |

### Part 2: The PyTorch Toolbox (Week 2)

All implementations use **PyTorch** and must be compatible with `torch.autograd`.

| Test File | Task | What You Implement | Key Acceptance Criteria |
|---|---|---|---|
| `test_autograd.py` | Autograd Engine | `get_value_class()` → `Value` with +, \*, tanh, backward | Gradients match numerical check |
| `test_linear.py` | Linear + Init | `run_linear`, `run_kaiming_init`, `run_xavier_init` | Matches `F.linear`; Kaiming variance ≈ 2/fan\_in; 50-layer variance stable |
| `test_activations.py` | Activations | `run_sigmoid`, `run_relu`, `run_gelu`, `run_softmax`, `run_swiglu` | Match PyTorch equivalents; softmax numerically stable |
| `test_loss.py` | Loss Functions | `run_cross_entropy_loss`, `run_kl_divergence_loss` | Within 10⁻⁶ of `nn.CrossEntropyLoss`; shift-invariant; KL≥0 |
| `test_optimizers.py` | SGD + AdamW | `get_sgd_cls()`, `get_adam_cls()` | Match `torch.optim` weights after 100 & 1000 steps |
| `test_dropout.py` | Inverted Dropout | `run_dropout(input, prob)` | Correct drop ratio, 1/(1−p) scaling, E[out]=E[in] |
| `test_normalization.py` | LayerNorm + RMSNorm | `run_layernorm`, `run_rmsnorm` | Match PyTorch; zero-mean & unit-var for LayerNorm |
| `test_mnist_integration.py` | MNIST Classifier | `get_custom_mnist_model`, `get_custom_loss_fn`, `get_custom_optimizer` | **≥97% test accuracy** using only your custom components |

---

## Phase 2 — Task Reference

### Part 1: Tokenization & Attention Mechanisms (Week 3)

All attention adapters take plain weight tensors (no biases) and must support autograd and arbitrary leading batch dimensions.

| Test File | Task | What You Implement | Key Acceptance Criteria |
|---|---|---|---|
| `test_bpe.py` | BPE from Scratch | `run_train_bpe`, `run_bpe_encode`, `run_bpe_decode`, `get_bpe_vocab` | Exact vocab size; ids 0–255 = raw bytes; merges follow pair frequency; `decode(encode(s)) == s` for every test string; ≥1.8 bytes/token on held-out text (V=1000); training ≤30 s, encoding ≥10 KB/s |
| `test_vocab_size.py` | Vocabulary Size Experiment | *(reuses the BPE adapters)* | Compression ratio ↑ and tokens/sentence ↓ as V ∈ {256, 512, 1000, 2000} grows; rare words still round-trip |
| `test_special_tokens.py` | Special Tokens & Edge Cases | `special_tokens=["<PAD>","<BOS>","<EOS>","<UNK>"]` | Each special token is one atomic id after the merges and is never learned as a merge; Unicode/emoji/whitespace round-trip losslessly; `<UNK>` never emitted; invalid UTF-8 decodes to U+FFFD |
| `test_scaled_dot_product_attention.py` | Scaled Dot-Product Attention | `run_scaled_dot_product_attention(Q, K, V, mask)` | Matches `F.scaled_dot_product_attention` + snapshots (values & gradients); stable for large scores; ≤20× latency |
| `test_causal_attention.py` | Causal Self-Attention | `run_causal_self_attention` | Matches PyTorch; no information flows from future tokens; ≤20× latency |
| `test_multihead_attention.py` | Multi-Head Attention | `run_multihead_self_attention` | Matches `torch.nn.MultiheadAttention` + snapshot; heads split in order and independent; ≤20× latency |
| `test_gqa.py` | Grouped Query Attention | `run_grouped_query_attention` | Matches `F.scaled_dot_product_attention(enable_gqa=True)`; equals MHA when `num_kv_heads == num_heads`; with RoPE equals MHA with repeated K/V weights; ≤20× latency |
| `test_rope.py` | RoPE | `run_rope`, `run_multihead_self_attention_with_rope` | Snapshots; θᵢ = θ^(−2i/d) with (2i, 2i+1) pairs; norm-preserving; q·k score depends only on relative position |
| `test_kv_cache.py` | KV-Cache | `get_kv_cached_attention`, `run_kv_cached_attention`, `get_kv_cache` | Prefill / token-by-token output = uncached output; K cached post-RoPE, V raw; GQA cache is `num_heads/num_kv_heads`× smaller; ≥3× faster decoding at seq_len 512 |

**Conventions the Part 1 tests rely on** (also documented in `tests/adapters.py`):

- **BPE:** byte-level. Token id `i < 256` is `bytes([i])`. Merges take ids 256, 257, … in the order they are learned, and special tokens come last. `vocab_size` counts all of them. Pre-tokenization and tie-breaking are up to you. Encoding must be lossless, so do not normalize whitespace.
- **Masks:** boolean, `True` = "may attend" (same as `F.scaled_dot_product_attention`).
- **Head layout:** projection rows are ordered head by head. In GQA, query head `i` uses KV head `i // (num_heads // num_kv_heads)`.
- **RoPE:** RoFormer-style interleaved pairs `(x[2i], x[2i+1])`. Positions broadcast against the leading dimensions.

### Part 2: The Assembly — Mini-Llama from Scratch (Week 4)

| Test File | Task | What You Implement | Key Acceptance Criteria |
|---|---|---|---|
| `test_transformer_block.py` | TransformerBlock | `run_transformer_block` | Snapshot; GQA block = MHA block with repeated K/V; pre-norm residual structure; causal; ≤20× latency vs. a PyTorch-built-ins block |
| `test_mini_llama.py` | MiniLlama + Parameter Audit | `run_transformer_lm`, `get_mini_llama`, `run_parameter_audit` | Snapshots (full & truncated input); **exact parameter count** (tied embedding, GQA, SwiGLU, RMSNorm, no biases); single shared embedding/output matrix; initial loss within 1 nat of ln V; audit rows/percentages correct |
| `test_training_loop.py` | Training Loop | `run_lr_schedule`, `run_gradient_clipping` | Linear warmup (10%) → cosine decay to `min_lr`; clipping matches `clip_grad_norm_` and returns the pre-clip norm; model works under bf16 autocast (CPU). *GPU only:* fp16 + `GradScaler` training and fp32 vs. bf16 speed |
| `test_evaluation.py` | Evaluation | `run_perplexity`, `run_generate` | Perplexity = exp(CE), shift-invariant; `temperature=0` is greedy; samples follow softmax(logits/T); seeded generation is reproducible; context window cropping |
| `test_tinyshakespeare_integration.py` | End-to-End Training | *(reuses everything)* | Your BPE (V=512) + your `MiniLlama` + your `MyAdamW` (Phase 1) + schedule + clipping, 600 steps on TinyShakespeare: loss starts ≈ ln V and drops by >1.5 nats, gradient norms stay healthy, **validation perplexity < 0.5× the unigram baseline**, and samples at T ∈ {0.1, 0.7, 1.0} decode to text |

**Weight-dict layout for `run_transformer_block` / `run_transformer_lm`** (block keys are prefixed with `layers.{i}.` in the LM):

| Key | Shape | Meaning |
|---|---|---|
| `attn.q_proj.weight` | `(d_model, d_model)` | Query projection |
| `attn.k_proj.weight`, `attn.v_proj.weight` | `(num_kv_heads · d_head, d_model)` | Key / value projections (GQA) |
| `attn.output_proj.weight` | `(d_model, d_model)` | Output projection |
| `ln1.weight`, `ln2.weight` | `(d_model,)` | RMSNorm gains (eps = 1e-5) before attention / FFN |
| `ffn.w1.weight`, `ffn.w3.weight` | `(d_ff, d_model)` | SwiGLU gate / up projections |
| `ffn.w2.weight` | `(d_model, d_ff)` | SwiGLU down projection: `w2(SiLU(w1·x) ⊙ w3·x)` |
| `token_embeddings.weight`, `lm_head.weight` | `(vocab_size, d_model)` | LM only |
| `ln_final.weight` | `(d_model,)` | LM only — final RMSNorm |

> The reference checkpoint in `tests/fixtures/ts_tests/` is **untied**, so `run_transformer_lm` must be able to load a separate `lm_head`. The model returned by `get_mini_llama` must be **tied**.

> The ablation study (Task 6) is a written deliverable and has no automated tests.

---

## Test Philosophy

Every test file validates two dimensions:

- **Accuracy** — Correctness against PyTorch references, analytical closed-forms, or known values.
- **Latency** — Performance benchmarks comparing your implementation to a reference. Tests assert your code runs within a reasonable multiple (typically 20–100×) of the optimized baseline.

Where no PyTorch reference exists (BPE tokenizer, KV-cache), the latency tests use generous absolute budgets or relative speed-ups instead. Tests that need a CUDA GPU are skipped automatically on CPU-only machines.

During development, skip latency tests with `-k "not Performance"` and focus on getting the math right first.

---

## Dependencies

Managed via [uv](https://docs.astral.sh/uv/). Key packages:

| Package | Purpose |
|---|---|
| `torch ≥2.10` | PyTorch framework |
| `numpy ≥2.4` | Numerical computing |
| `pytest ≥9.0` | Test runner |
| `jaxtyping ≥0.3` | Tensor shape annotations |
| `einops ≥0.8` | Tensor reshaping utilities |
| `tqdm ≥4.67` | Progress bars in benchmarks |