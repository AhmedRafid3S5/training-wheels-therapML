"""
Adapter module — glue layer between learner implementations (in therapml/)
and the automated test suite.

Learners implement all logic inside ``therapml/`` and only fill in these
thin wrapper functions so the tests can call them in a uniform way.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from therapml.pca import pca
from therapml.svd import truncated_svd
from therapml.mle_gaussian import grad_des_mle
from therapml.logsumexp import logsumexp
from therapml.dropout import inverted_dropout
from therapml.norm import myLayerNorm, myRMSnorm
from therapml.information import Entropy
from therapml.optimizers import Optimizers
from therapml.autograd import Value
from therapml.activations import Activation
from therapml.linear import MyLinear
from therapml.loss import Loss
from therapml.adamw import MyAdam
from therapml.tensor_sgd import MySGD
from therapml.custom_mlp import CustomMLP, MyCrossEntropyLoss
from therapml.adamw import MyAdam
from therapml.tokenizer import BasicTokenizer

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
    return pca(data,k)


# --- Task 2: SVD & Low-Rank Approximation ----------------------------------

def run_truncated_svd(matrix: NDArray, rank: int) -> dict:
    """Truncated SVD of *matrix* (m, n) at the given *rank*.

    Must return a dict with keys:
        ``U``             – (m, rank)
        ``S``             – (rank,)
        ``Vt``            – (rank, n)
        ``reconstructed`` – (m, n) low-rank reconstruction U @ diag(S) @ Vt
    """
    return truncated_svd(matrix,rank)


# --- Task 4: Maximum Likelihood Estimation ----------------------------------

def run_mle_gaussian(data: NDArray, lr: float = 0.01, n_steps: int = 5000) -> dict:
    """Fit a 1-D Gaussian to *data* via gradient-descent MLE.

    Must return a dict with keys:
        ``mu``    – estimated mean  (float)
        ``sigma2``– estimated variance  (float)
    """
    return grad_des_mle(data, lr, n_steps)


# --- Task 6: Information Theory Toolkit -------------------------------------

def run_entropy(P: NDArray) -> float:
    """Shannon entropy H(P) in *bits* (log base 2)."""
    return Entropy.shanonEntropy(P)


def run_kl_divergence(P: NDArray, Q: NDArray) -> float:
    """KL divergence D_KL(P || Q) in *bits*."""
    return Entropy.klDivergence(P,Q)


def run_cross_entropy_info(P: NDArray, Q: NDArray) -> float:
    """Cross-entropy H(P, Q) in *bits*.

    Named ``_info`` to avoid collision with the PyTorch loss adapter.
    """
    return Entropy.cross_entropy(P,Q)


def run_mutual_information(joint: NDArray) -> float:
    """Mutual information I(X; Y) from a 2-D joint distribution table, in *bits*."""
    return Entropy.mutual_information(joint)


# --- Task 7: Numerical Stability -------------------------------------------

def run_log_sum_exp(x: NDArray) -> float:
    """Numerically stable log-sum-exp: log(sum(exp(x)))."""
    return logsumexp(x)


# --- Task 8: Optimizers (pure NumPy, linear regression) ---------------------

def run_numpy_sgd_fit(
    X: NDArray, y: NDArray, lr: float, n_steps: int,
) -> dict:
    """Vanilla SGD on linear regression.

    Returns ``{'weights': NDArray, 'loss_history': list[float]}``.
    """
    return Optimizers.sgd(X,y,lr,n_steps)


def run_numpy_sgd_momentum_fit(
    X: NDArray, y: NDArray, lr: float, momentum: float, n_steps: int,
) -> dict:
    """SGD + momentum on linear regression.

    Returns ``{'weights': NDArray, 'loss_history': list[float]}``.
    """
    return Optimizers.sgd_momentum(X,y,lr,momentum,n_steps)


def run_numpy_adam_fit(
    X: NDArray, y: NDArray, lr: float, n_steps: int,
) -> dict:
    """Adam on linear regression.

    Returns ``{'weights': NDArray, 'loss_history': list[float]}``.
    """
    return Optimizers.sgd_adam_fit(X,y,lr,n_steps)


# ============================================================================
# Phase 1 — Part 2: The PyTorch Toolbox  (Week 2)
# ============================================================================

# --- Task 1: Autograd Intuition --------------------------------------------

def get_value_class() -> Any:
    """Return the learner's ``Value`` class (scalar autograd engine)."""
    return Value


# --- Task 2: Linear Layers & Initialization --------------------------------

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) :
    layer = MyLinear(d_in,d_out, bias =True)
    with torch.no_grad():
        layer.weight.copy_(weights)
        layer.bias.fill_(0.0)

    return layer(in_features) #executes a forward pass via nn internal __call__ method
    


def run_kaiming_init(fan_in: int, fan_out: int) -> Float[Tensor, "fan_out fan_in"]:
    """Return a weight matrix initialized with Kaiming He Normal."""
    return MyLinear.runKaiming(fan_in,fan_out)


def run_xavier_init(fan_in: int, fan_out: int) -> Float[Tensor, "fan_out fan_in"]:
    """Return a weight matrix initialized with Xavier Glorot."""
    return MyLinear.runXavier(fan_in, fan_out)


# --- Task 3: Activations ---------------------------------------------------

def run_sigmoid(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return Activation.mySigmoid(in_features)


def run_relu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return Activation.myReLU(in_features)


def run_gelu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return Activation.myGELU(in_features)


def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    return Activation.mySoftmax(in_features,dim)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    return Activation.mySwiGLU(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features)


# --- Task 4: Loss Functions ------------------------------------------------

def run_cross_entropy_loss(
    logits: Float[Tensor, "batch output_dim"],
    ground_truth: Float[Tensor, "batch output_dim"],
) -> Float[Tensor, ""]:
    return Loss._crossentropyloss(logits,ground_truth)


def run_kl_divergence_loss(
    input_log_probs: Float[Tensor, "batch d"],
    target_probs: Float[Tensor, "batch d"],
) -> Float[Tensor, ""]:
    """KL-divergence loss.  *input_log_probs* are **log-probabilities**,
    *target_probs* are **probabilities**."""
    return Loss.kldivergenceloss(input_log_probs, target_probs)


# --- Task 5: Optimizers (PyTorch-compatible) --------------------------------

def get_sgd_cls() -> Any:
    """Return the learner's SGD optimizer class (with momentum support)."""
    return MySGD


def get_adam_cls() -> Any:
    """Return the learner's AdamW optimizer class."""
    return MyAdam


# --- Task 6: Regularization ------------------------------------------------

def run_dropout(input: Float[Tensor, "..."], prob: float) -> Float[Tensor, "..."]:
    return inverted_dropout(input, prob)


# --- Task 7: Normalization & Residuals --------------------------------------

def run_layernorm(
    input: Float[Tensor, "batch ..."],
    gamma: Float[Tensor, "batch ..."],
    beta: Float[Tensor, "batch ..."],
) -> Float[Tensor, "batch ..."]:
    return myLayerNorm(input, gamma, beta)


def run_rmsnorm(
    input: Float[Tensor, "batch ..."],
    gamma: Float[Tensor, "batch ..."],
) -> Float[Tensor, "batch ..."]:
    return myRMSnorm(input, gamma)


# --- Task 8: MNIST Integration Test -----------------------------------------

def get_custom_mnist_model() -> Any:
    """Return an *untrained* nn.Module built entirely from custom components
    (MyLinear, MyReLU, MyDropout, MyLayerNorm).
    """
    return CustomMLP()


def get_custom_loss_fn() -> Any:
    """Return the learner's MyCrossEntropyLoss callable."""
    return MyCrossEntropyLoss()


def get_custom_optimizer(params, lr: float = 1e-3) -> Any:
    """Return the learner's MyAdamW optimizer wrapping *params*."""
    return MyAdam(params, lr=lr)


# ============================================================================
# Phase 2 — Part 1: Tokenization & Attention Mechanisms  (Week 3)
# ============================================================================

# --- Tasks 1–3: Byte Pair Encoding (BPE) tokenizer --------------------------
#
# Conventions the tests rely on (byte-level BPE, as in GPT-2 / Llama 3):
#   * Token ids 0–255 are the 256 raw bytes:  ``vocab[i] == bytes([i])``.
#   * Every learned merge gets the next free id (256, 257, ...) in the order it
#     was learned, and its bytes are the concatenation of the two merged tokens.
#   * Special tokens get the ids *after* the merges. ``vocab_size`` counts
#     everything (256 bytes + merges + special tokens).
#   * Pre-tokenization (e.g. a GPT-2 style regex split) is up to you, and so is
#     tie-breaking between equally frequent pairs.
#   * Encoding is lossless: ``decode(encode(text)) == text`` for *every* string.
#     Do not normalize whitespace or Unicode inside ``encode``.

def run_train_bpe(corpus: str, vocab_size: int, special_tokens: list[str] | None = None) -> Any:
    """Train a byte-level BPE tokenizer on *corpus* and return the trained tokenizer object.

    The returned object is opaque to the tests. It is only passed back into
    ``run_bpe_encode``, ``run_bpe_decode`` and ``get_bpe_vocab``.

    Args:
        corpus: Training text.
        vocab_size: Final vocabulary size *including* the 256 byte tokens and
            all special tokens. ``vocab_size == 256`` (with no special tokens)
            means "learn zero merges".
        special_tokens: e.g. ``["<PAD>", "<BOS>", "<EOS>", "<UNK>"]``. Each one
            must become a single token id, must never be split or merged with
            surrounding text, and must not be learned as part of any merge even
            if it appears in *corpus*.
    """
    bpe = BasicTokenizer()
    bpe.train(corpus,vocab_size)
    return bpe


def run_bpe_encode(tokenizer: Any, text: str) -> list[int]:
    """Encode *text* into a list of token ids using *tokenizer*.

    Occurrences of special-token strings in *text* must map to their single
    special-token id. Because the tokenizer is byte-level, every string is
    representable, so ``<UNK>`` must never be emitted.
    """
    return tokenizer.encode(text)


def run_bpe_decode(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode token ids back into a string.

    Byte sequences that are not valid UTF-8 (e.g. half of an emoji) must not
    raise. Decode them with ``errors="replace"`` so they become U+FFFD.
    """
    return tokenizer.decode(token_ids)


def get_bpe_vocab(tokenizer: Any) -> dict[int, bytes]:
    """Return the full vocabulary as ``{token_id: token_bytes}``.

    Special tokens map to their UTF-8 encoding (e.g. ``b"<EOS>"``).
    """
    return tokenizer.get_vocab()


# --- Task 4: Scaled Dot-Product Attention -----------------------------------

def run_scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """softmax(Q Kᵀ / √d_k) V.

    *mask* follows the ``torch.nn.functional.scaled_dot_product_attention``
    boolean convention: ``True`` means "may attend", ``False`` positions are
    set to −∞ before the softmax. Any number of leading batch/head dimensions
    must be supported.
    """
    raise NotImplementedError


# --- Task 5: Causal Self-Attention (single head) ----------------------------

def run_causal_self_attention(
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    in_features: Float[Tensor, "... seq d_in"],
) -> Float[Tensor, "... seq d_v"]:
    """Single-head causal self-attention with Q, K, V projections (no bias,
    no output projection). Token *i* may only attend to tokens ``0..i``.
    """
    raise NotImplementedError


# --- Task 6: Multi-Head Attention -------------------------------------------

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_model d_model"],
    k_proj_weight: Float[Tensor, "d_model d_model"],
    v_proj_weight: Float[Tensor, "d_model d_model"],
    o_proj_weight: Float[Tensor, "d_model d_model"],
    in_features: Float[Tensor, "... seq d_model"],
) -> Float[Tensor, "... seq d_model"]:
    """Causal multi-head self-attention (no bias, no RoPE).

    The rows of each projection weight are ordered head by head, i.e.
    ``q_proj_weight == torch.cat([W_q_head0, ..., W_q_head{h-1}], dim=0)``
    where each ``W_q_head_i`` has shape ``(d_model // num_heads, d_model)``.
    All heads must be computed in parallel (no Python loop over heads).
    """
    raise NotImplementedError


# --- Task 7: Grouped Query Attention ----------------------------------------

def run_grouped_query_attention(
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    q_proj_weight: Float[Tensor, "d_model d_model"],
    k_proj_weight: Float[Tensor, "kv_dim d_model"],
    v_proj_weight: Float[Tensor, "kv_dim d_model"],
    o_proj_weight: Float[Tensor, "d_model d_model"],
    in_features: Float[Tensor, "... seq d_model"],
    theta: float | None = None,
    max_seq_len: int | None = None,
    token_positions: Int[Tensor, "... seq"] | None = None,
) -> Float[Tensor, "... seq d_model"]:
    """Causal grouped-query self-attention.

    ``d_head = d_model // num_heads`` and ``kv_dim = num_kv_heads * d_head``.
    Query heads are split into ``num_kv_heads`` contiguous groups. Query head
    ``i`` uses KV head ``i // (num_heads // num_kv_heads)``, the same as
    ``repeat_interleave`` and ``F.scaled_dot_product_attention(enable_gqa=True)``.

    If *theta* is not ``None``, apply RoPE to Q and K at *token_positions*
    (default: ``arange(seq)``). If *theta* is ``None``, use no positional
    encoding.
    """
    raise NotImplementedError


# --- Task 8: Rotary Positional Embeddings -----------------------------------

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, "... seq d_k"],
    token_positions: Int[Tensor, "... seq"],
) -> Float[Tensor, "... seq d_k"]:
    """Apply RoPE to *in_query_or_key*.

    Use the RoFormer pairing: dimensions ``(2i, 2i+1)`` form one 2-D pair, which
    is rotated by angle ``pos · θ_i`` with ``θ_i = theta^(−2i / d_k)``.
    *token_positions* broadcasts against the leading dimensions of the input.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_model d_model"],
    k_proj_weight: Float[Tensor, "d_model d_model"],
    v_proj_weight: Float[Tensor, "d_model d_model"],
    o_proj_weight: Float[Tensor, "d_model d_model"],
    in_features: Float[Tensor, "... seq d_model"],
    token_positions: Int[Tensor, "... seq"],
) -> Float[Tensor, "... seq d_model"]:
    """Causal multi-head self-attention with RoPE applied to Q and K (per head,
    with ``d_k = d_model // num_heads``)."""
    raise NotImplementedError


# --- Task 9: KV-Cache --------------------------------------------------------

def get_kv_cached_attention(
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_model d_model"],
    k_proj_weight: Float[Tensor, "kv_dim d_model"],
    v_proj_weight: Float[Tensor, "kv_dim d_model"],
    o_proj_weight: Float[Tensor, "d_model d_model"],
) -> Any:
    """Return a *fresh* (empty-cache) causal GQA + RoPE attention object that
    keeps a KV-cache between calls. Same math as ``run_grouped_query_attention``
    with *theta* set."""
    raise NotImplementedError


def run_kv_cached_attention(
    attn: Any,
    in_features: Float[Tensor, "batch new_seq d_model"],
) -> Float[Tensor, "batch new_seq d_model"]:
    """Feed *new* tokens through the cached attention object.

    The new tokens sit at positions ``cache_len .. cache_len + new_seq - 1``
    (used for RoPE and for the causal mask). They attend to every cached token
    and causally to each other. Their K/V are appended to the cache. Calling
    this once with the whole sequence (prefill) or once per token (decoding)
    must give the same outputs as the uncached computation.
    """
    raise NotImplementedError


def get_kv_cache(attn: Any) -> tuple[Tensor, Tensor]:
    """Return the cached ``(K, V)`` tensors, each of shape
    ``(batch, num_kv_heads, cached_seq_len, d_head)``. K is stored *after* RoPE
    and V never has RoPE applied."""
    raise NotImplementedError


# ============================================================================
# Phase 2 — Part 2: The Assembly — Mini-Llama from Scratch  (Week 4)
# ============================================================================
#
# Weight-dict layout shared by ``run_transformer_block`` and
# ``run_transformer_lm`` (block keys are prefixed with ``layers.{i}.`` in the LM):
#
#   attn.q_proj.weight       (d_model, d_model)
#   attn.k_proj.weight       (num_kv_heads * d_head, d_model)
#   attn.v_proj.weight       (num_kv_heads * d_head, d_model)
#   attn.output_proj.weight  (d_model, d_model)
#   ln1.weight               (d_model,)    RMSNorm gain before attention
#   ffn.w1.weight            (d_ff, d_model)  gate projection
#   ffn.w2.weight            (d_model, d_ff)  down projection
#   ffn.w3.weight            (d_ff, d_model)  up projection
#   ln2.weight               (d_model,)    RMSNorm gain before the FFN
#
# Block math (pre-norm, RMSNorm eps = 1e-5, causal, RoPE on Q and K):
#   h   = x + Attn(RMSNorm(x))
#   out = h + w2( SiLU(w1 · RMSNorm(h)) ⊙ (w3 · RMSNorm(h)) )

# --- Tasks 1–2: Architecture & Assembly -------------------------------------

def run_transformer_block(
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch seq d_model"],
) -> Float[Tensor, "batch seq d_model"]:
    """Run your ``TransformerBlock`` with the given *weights* loaded (see the
    layout above). Token positions are ``arange(seq)``."""
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, "batch seq"],
) -> Float[Tensor, "batch seq vocab_size"]:
    """Run your ``MiniLlama`` with the given *weights* loaded and return logits.

    Besides the per-layer keys (``layers.{i}.<block key>``), *weights* contains:
        ``token_embeddings.weight`` (vocab_size, d_model)
        ``ln_final.weight``         (d_model,)
        ``lm_head.weight``          (vocab_size, d_model)

    The reference checkpoint used here is *untied* (``lm_head`` differs from
    the embedding), so your model needs a way to load an untied output head
    (e.g. a ``tie_weights`` flag).
    """
    raise NotImplementedError


def get_mini_llama(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    d_ff: int,
    rope_theta: float,
) -> nn.Module:
    """Return a freshly initialised, **weight-tied** ``MiniLlama``.

    ``model(in_indices)`` with ``in_indices`` of shape ``(batch, seq)`` (``seq ≤
    context_length``) must return logits of shape ``(batch, seq, vocab_size)``.
    The model must contain exactly the parameters listed in the layout above
    (no biases), plus the token embedding (shared with the output head) and the
    final RMSNorm gain.
    """
    raise NotImplementedError


def run_parameter_audit(model: nn.Module) -> list[dict]:
    """Model Parameter Audit. Return one row per parameter tensor (the order of
    ``model.named_parameters()``) as a dict with keys:

        ``name``       – str, the parameter name
        ``shape``      – tuple[int, ...]
        ``num_params`` – int
        ``percent``    – float, share of the total parameter count in [0, 100]

    Your utility may also print the table. The tests only check the return value.
    """
    raise NotImplementedError


# --- Task 3: The Training Loop ----------------------------------------------

def run_lr_schedule(step: int, total_steps: int, max_lr: float, min_lr: float, warmup_frac: float = 0.1) -> float:
    """Linear warmup followed by cosine decay. With ``W = int(warmup_frac * total_steps)``:

        step <  W              : max_lr · step / W
        W ≤ step ≤ total_steps : min_lr + ½(1 + cos(π · (step − W) / (total_steps − W))) · (max_lr − min_lr)
        step >  total_steps    : min_lr
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    """Clip the gradients of *parameters* in place so their combined L2 norm is
    at most *max_norm*. Return the total norm **before** clipping (you will log
    it as the gradient-health metric). Parameters with ``grad is None`` are
    skipped."""
    raise NotImplementedError


# --- Task 5: Evaluation -----------------------------------------------------

def run_perplexity(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> float:
    """Perplexity = exp(mean token-level cross-entropy, in nats)."""
    raise NotImplementedError


def run_generate(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    context_length: int,
) -> list[int]:
    """Generate autoregressively from *model* and return only the
    *max_new_tokens* new token ids.

    * At each step, call ``model(ids)`` with ``ids`` of shape ``(1, seq)``. Keep
      only the last ``context_length`` tokens when the running sequence gets
      longer than that. Sample from the logits at the last position.
    * ``temperature == 0`` means greedy (argmax) decoding. Otherwise sample from
      ``softmax(logits / temperature)`` using the global torch RNG, so that
      ``torch.manual_seed`` makes generation reproducible.
    """
    raise NotImplementedError
