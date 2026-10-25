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
├── fixtures/                        # Pre-generated .npy data for deterministic tests
├── _snapshots/                      # Reserved for future phases
│
└── phase_1/
    ├── part_1/                      # Week 1 — Mathematical Foundations (NumPy)
    │   ├── test_pca.py              # Task 1: PCA from scratch
    │   ├── test_svd.py              # Task 2: SVD & low-rank approximation
    │   ├── test_mle.py              # Task 4: Maximum likelihood estimation
    │   ├── test_info_theory.py      # Task 6: Entropy, KL, cross-entropy, MI
    │   ├── test_numerical_stability.py  # Task 7: Log-sum-exp stability
    │   └── test_optimizers.py       # Task 8: SGD, Momentum, Adam (NumPy)
    │
    └── part_2/                      # Week 2 — PyTorch Toolbox
        ├── test_autograd.py         # Task 1: Scalar autograd engine (Value class)
        ├── test_linear.py           # Task 2: MyLinear + Kaiming/Xavier init
        ├── test_activations.py      # Task 3: Sigmoid, ReLU, GELU, Softmax, SwiGLU
        ├── test_loss.py             # Task 4: CrossEntropy, KL divergence
        ├── test_optimizers.py       # Task 5: MySGD (momentum) + MyAdamW
        ├── test_dropout.py          # Task 6: Inverted dropout
        ├── test_normalization.py    # Task 7: LayerNorm, RMSNorm
        └── test_mnist_integration.py  # Task 8: MNIST ≥97% with custom components
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

# Run a single task
pytest tests/phase_1/part_1/test_pca.py -v

# Skip performance benchmarks during development
pytest tests/phase_1/part_1/ -v -k "not Performance"
```

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

## Test Philosophy

Every test file validates two dimensions:

- **Accuracy** — Correctness against PyTorch references, analytical closed-forms, or known values.
- **Latency** — Performance benchmarks comparing your implementation to a reference. Tests assert your code runs within a reasonable multiple (typically 20–100×) of the optimized baseline.

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