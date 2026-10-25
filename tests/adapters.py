"""
Adapter module — glue layer between learner implementations (in therapml/)
and the automated test suite.

Learners implement all logic inside ``therapml/`` and only fill in these
thin wrapper functions so the tests can call them in a uniform way.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from jaxtyping import Float
from torch import Tensor


# ============================================================================
# Pre-phase: Tensor Operations (kept from original repo)
# ============================================================================

def run_tensor_multiply(arr1: Float[list, "b x y"], arr2: Float[list, "b y z"]) -> Float[list, "b x z"]:
    raise NotImplementedError


def run_tensor_dot(arr1: Float[list, "..."], arr2: Float[list, "..."], dim: int):
    raise NotImplementedError


# ============================================================================
# Phase 1 — Part 1: Mathematical Foundations  (Week 1, NumPy-only)
# ============================================================================

# --- Task 1: PCA -----------------------------------------------------------

def run_pca(data: NDArray, k: int) -> dict:
    """Run PCA on *data* (n_samples, n_features) and project to *k* components.

    Must return a dict with keys:
        ``projected``     – (n_samples, k) projected data
        ``components``    – (k, n_features) principal component row-vectors
        ``eigenvalues``   – (k,) top-k eigenvalues in descending order
        ``mean``          – (n_features,) per-feature mean used for centering
        ``reconstructed`` – (n_samples, n_features) reconstruction from k components
    """
    raise NotImplementedError


# --- Task 2: SVD & Low-Rank Approximation ----------------------------------

def run_truncated_svd(matrix: NDArray, rank: int) -> dict:
    """Truncated SVD of *matrix* (m, n) at the given *rank*.

    Must return a dict with keys:
        ``U``             – (m, rank)
        ``S``             – (rank,)
        ``Vt``            – (rank, n)
        ``reconstructed`` – (m, n) low-rank reconstruction U @ diag(S) @ Vt
    """
    raise NotImplementedError


# --- Task 4: Maximum Likelihood Estimation ----------------------------------

def run_mle_gaussian(data: NDArray, lr: float = 0.01, n_steps: int = 5000) -> dict:
    """Fit a 1-D Gaussian to *data* via gradient-descent MLE.

    Must return a dict with keys:
        ``mu``    – estimated mean  (float)
        ``sigma2``– estimated variance  (float)
    """
    raise NotImplementedError


# --- Task 6: Information Theory Toolkit -------------------------------------

def run_entropy(P: NDArray) -> float:
    """Shannon entropy H(P) in *bits* (log base 2)."""
    raise NotImplementedError


def run_kl_divergence(P: NDArray, Q: NDArray) -> float:
    """KL divergence D_KL(P || Q) in *bits*."""
    raise NotImplementedError


def run_cross_entropy_info(P: NDArray, Q: NDArray) -> float:
    """Cross-entropy H(P, Q) in *bits*.

    Named ``_info`` to avoid collision with the PyTorch loss adapter.
    """
    raise NotImplementedError


def run_mutual_information(joint: NDArray) -> float:
    """Mutual information I(X; Y) from a 2-D joint distribution table, in *bits*."""
    raise NotImplementedError


# --- Task 7: Numerical Stability -------------------------------------------

def run_log_sum_exp(x: NDArray) -> float:
    """Numerically stable log-sum-exp: log(sum(exp(x)))."""
    raise NotImplementedError


# --- Task 8: Optimizers (pure NumPy, linear regression) ---------------------

def run_numpy_sgd_fit(
    X: NDArray, y: NDArray, lr: float, n_steps: int,
) -> dict:
    """Vanilla SGD on linear regression.

    Returns ``{'weights': NDArray, 'loss_history': list[float]}``.
    """
    raise NotImplementedError


def run_numpy_sgd_momentum_fit(
    X: NDArray, y: NDArray, lr: float, momentum: float, n_steps: int,
) -> dict:
    """SGD + momentum on linear regression.

    Returns ``{'weights': NDArray, 'loss_history': list[float]}``.
    """
    raise NotImplementedError


def run_numpy_adam_fit(
    X: NDArray, y: NDArray, lr: float, n_steps: int,
) -> dict:
    """Adam on linear regression.

    Returns ``{'weights': NDArray, 'loss_history': list[float]}``.
    """
    raise NotImplementedError


# ============================================================================
# Phase 1 — Part 2: The PyTorch Toolbox  (Week 2)
# ============================================================================

# --- Task 1: Autograd Intuition --------------------------------------------

def get_value_class() -> Any:
    """Return the learner's ``Value`` class (scalar autograd engine)."""
    raise NotImplementedError


# --- Task 2: Linear Layers & Initialization --------------------------------

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) -> Float[Tensor, "... d_out"]:
    raise NotImplementedError


def run_kaiming_init(fan_in: int, fan_out: int) -> Float[Tensor, "fan_out fan_in"]:
    """Return a weight matrix initialized with Kaiming He Normal."""
    raise NotImplementedError


def run_xavier_init(fan_in: int, fan_out: int) -> Float[Tensor, "fan_out fan_in"]:
    """Return a weight matrix initialized with Xavier Glorot."""
    raise NotImplementedError


# --- Task 3: Activations ---------------------------------------------------

def run_sigmoid(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    raise NotImplementedError


def run_relu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    raise NotImplementedError


def run_gelu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    raise NotImplementedError


# --- Task 4: Loss Functions ------------------------------------------------

def run_cross_entropy_loss(
    logits: Float[Tensor, "batch output_dim"],
    ground_truth: Float[Tensor, "batch output_dim"],
) -> Float[Tensor, ""]:
    raise NotImplementedError


def run_kl_divergence_loss(
    input_log_probs: Float[Tensor, "batch d"],
    target_probs: Float[Tensor, "batch d"],
) -> Float[Tensor, ""]:
    """KL-divergence loss.  *input_log_probs* are **log-probabilities**,
    *target_probs* are **probabilities**."""
    raise NotImplementedError


# --- Task 5: Optimizers (PyTorch-compatible) --------------------------------

def get_sgd_cls() -> Any:
    """Return the learner's SGD optimizer class (with momentum support)."""
    raise NotImplementedError


def get_adam_cls() -> Any:
    """Return the learner's AdamW optimizer class."""
    raise NotImplementedError


# --- Task 6: Regularization ------------------------------------------------

def run_dropout(input: Float[Tensor, "..."], prob: float) -> Float[Tensor, "..."]:
    raise NotImplementedError


# --- Task 7: Normalization & Residuals --------------------------------------

def run_layernorm(
    input: Float[Tensor, "batch ..."],
    gamma: Float[Tensor, "batch ..."],
    beta: Float[Tensor, "batch ..."],
) -> Float[Tensor, "batch ..."]:
    raise NotImplementedError


def run_rmsnorm(
    input: Float[Tensor, "batch ..."],
    gamma: Float[Tensor, "batch ..."],
) -> Float[Tensor, "batch ..."]:
    raise NotImplementedError


# --- Task 8: MNIST Integration Test -----------------------------------------

def get_custom_mnist_model() -> Any:
    """Return an *untrained* nn.Module built entirely from custom components
    (MyLinear, MyReLU, MyDropout, MyLayerNorm).
    """
    raise NotImplementedError


def get_custom_loss_fn() -> Any:
    """Return the learner's MyCrossEntropyLoss callable."""
    raise NotImplementedError


def get_custom_optimizer(params, lr: float = 1e-3) -> Any:
    """Return the learner's MyAdamW optimizer wrapping *params*."""
    raise NotImplementedError
