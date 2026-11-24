"""
Adapter module for Phase 3 — glue layer between learner implementations (in
therapml/) and the Phase 3 test suite.

Same rules as ``tests/adapters.py``: implement all logic inside ``therapml/`` and
only fill in these thin wrapper functions. Phase 3 builds on your Phase 1–2 code,
so the Phase 3 tests also import from ``tests/adapters.py`` (``get_mini_llama``,
``run_perplexity``, the BPE adapters, ``get_adam_cls``, ``run_lr_schedule`` and
``run_gradient_clipping``). Those must already be wired up.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from jaxtyping import Float, Int
from torch import Tensor, nn

from therapml.decode_strategies import *

# ============================================================================
# Phase 3: The LLM Lifecycle & Inference  (Week 5)
# ============================================================================
#
# Task 1 (Pipeline Mapping) is a diagram deliverable and has no adapters.

# --- Task 2: Decoding Strategies --------------------------------------------
#
# Sampling pipeline used by every sampling adapter, in this order:
#
#   1. temperature : logits / T          (T == 0 means greedy: return argmax and skip 2–4)
#   2. top-k       : keep the k largest logits, set the rest to −∞
#   3. top-p       : on softmax of the logits from step 2, keep the smallest set
#                    of most-probable tokens whose cumulative probability ≥ p,
#                    set the rest to −∞
#   4. sample      : one draw from softmax of the result, using the *global*
#                    torch RNG (so ``torch.manual_seed`` makes it reproducible)
#
# ``top_k=None`` and ``top_p=None`` disable the corresponding filter.


def run_top_k_filter(logits: Float[Tensor, "... vocab"], k: int) -> Float[Tensor, "... vocab"]:
    """Keep the *k* largest logits along the last dimension and set every other
    entry to ``-inf``. Kept logits are returned unchanged. ``k >= vocab`` keeps
    everything. Each row is filtered independently. Do not modify *logits* in
    place."""
    return top_k(logits,k)


def run_top_p_filter(logits: Float[Tensor, "... vocab"], p: float) -> Float[Tensor, "... vocab"]:
    """Nucleus filter along the last dimension.

    Sort tokens by probability (``softmax(logits)``), most probable first. Keep
    a token if the total probability of the tokens ranked *above* it is ``< p``.
    The most probable token is therefore always kept, and the kept set is the
    smallest one whose probability mass reaches *p*. Everything else becomes
    ``-inf``. Kept logits are returned unchanged. ``p >= 1.0`` keeps everything.
    Each row is filtered independently. Do not modify *logits* in place.
    """
    return top_p(logits,p)

def run_sample_next_token(
    logits: Float[Tensor, "... vocab"],
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Int[Tensor, "..."]:
    """Pick one token id per row of *logits* with the pipeline described above.

    Returns a ``torch.long`` tensor of shape ``logits.shape[:-1]``. The tests
    call this with many identical rows at once to check the sampling
    distribution, so it must be vectorised over the leading dimensions.
    """
    return sample_next_token(logits,
                             temperature,
                             top_k,
                             top_p)


def run_decode(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    context_length: int,
    use_kv_cache: bool,
) -> list[int]:
    """Your ``generate()``: autoregressive decoding that supports temperature,
    top-k and top-p at the same time. Return only the *max_new_tokens* new ids.

    * Each step samples one token from the logits at the last position, using
      the same pipeline as ``run_sample_next_token`` (one RNG draw per step).
    * ``use_kv_cache=False``: call ``model(ids)`` with ``ids`` of shape
      ``(1, seq)`` every step, keeping only the last *context_length* tokens.
      This path must work for *any* module that maps ``(batch, seq)`` ids to
      ``(batch, seq, vocab)`` logits. The tests use tiny toy models here.
    * ``use_kv_cache=True``: *model* is a Mini-Llama from ``get_mini_llama``
      (or ``run_quantize_model_int8``). Run the prompt through the model once
      (prefill), then feed one new token per step, reusing your KV-cache from
      Phase 2. The tests guarantee ``len(prompt_ids) + max_new_tokens <=
      context_length`` on this path. The model's cache must be empty again
      (or reset) at the start of every call.

    Both paths must produce the same tokens for the same seed.
    """
    raise NotImplementedError


# --- Task 3: Post-Training Quantization -------------------------------------
#
# Naive symmetric per-tensor INT8 (absmax) quantization:
#
#   s   = max(|W|) / 127                      stored as float32 (0-dim tensor)
#   W_q = round(W / s)                        stored as torch.int8, in [-127, 127]
#   W  ≈ W_q · s                              dequantization
#
# Use ``torch.round`` (round half to even). For an all-zero tensor, pick any
# non-zero scale, so that the tensor dequantizes to zeros without NaNs.


def run_quantize_int8(weight: Float[Tensor, "..."]) -> tuple[Tensor, Tensor]:
    """Quantize *weight* and return ``(W_q, s)``. ``W_q`` has the same shape as
    *weight* and dtype ``torch.int8``. ``s`` is a 0-dim ``torch.float32``
    tensor."""
    raise NotImplementedError


def run_dequantize_int8(w_q: Tensor, scale: Tensor) -> Float[Tensor, "..."]:
    """Return the ``torch.float32`` approximation ``W_q · s``."""
    raise NotImplementedError


def run_quantize_model_int8(model: nn.Module) -> nn.Module:
    """Return an INT8-quantized copy of a Mini-Llama from ``get_mini_llama``.

    * Every 2-D weight matrix (token embedding / output head, attention and
      FFN projections) is stored as an int8 tensor plus one float32 scale.
      Both must be registered as buffers or parameters, so they show up in
      ``state_dict()``. Do not keep the float copy around.
    * 1-D tensors (RMSNorm gains) stay float32.
    * The tied embedding / output head is quantized once and stays shared.
    * Weights are dequantized on the fly in ``forward``. The returned model
      has the same interface as the original, including the KV-cache path
      used by ``run_decode(..., use_kv_cache=True)``.
    * *model* itself must not be modified.
    """
    raise NotImplementedError


# --- Task 4: Model Serving — Local Deployment -------------------------------


def run_stream_decode(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    context_length: int,
    use_kv_cache: bool,
) -> Iterator[int]:
    """Streaming version of ``run_decode`` (your naive PyTorch serving loop).

    Same arguments and same tokens as ``run_decode``, but a *generator* that
    yields each token id as soon as it has been sampled. No model call may
    happen before the caller asks for the first token, and only one decoding
    step may run per ``next()``. That is what makes time-to-first-token
    measurable.
    """
    raise NotImplementedError


def run_load_test(
    send_request: Callable[[str], Iterable[Any]],
    prompts: list[str],
    concurrency: int,
) -> dict:
    """Your load-testing script.

    ``send_request(prompt)`` is a blocking, thread-safe callable that returns an
    iterator streaming the response. Each yielded item counts as **one token**.
    Send every prompt in *prompts* with at most *concurrency* requests in
    flight at a time (e.g. with a ``ThreadPoolExecutor``). Measure all times
    with ``time.perf_counter``.

    For each request, time is measured from the moment ``send_request`` is
    called, not from when the prompt was queued:

        ``ttft``           – seconds until the first item arrives
        ``latency``        – seconds until the stream is exhausted
        ``num_tokens``     – number of items received
        ``tokens_per_sec`` – ``num_tokens / latency``

    Return a dict with:

        ``requests``   – list of the per-request dicts above, in the order of *prompts*
        ``total_time`` – wall-clock seconds from the first request start to the last request end
        ``throughput`` – total tokens over all requests / ``total_time``
    """
    raise NotImplementedError


def get_server_client(base_url: str, model: str, max_tokens: int) -> Callable[[str], Iterator[str]]:
    """*Optional* — only used when ``PHASE3_SERVER_URL`` is set.

    Return a ``send_request(prompt)`` callable for your running vLLM or TGI
    server. It should send a streaming completion request for *model* with at
    most *max_tokens* new tokens and ``temperature=0``, and yield one text
    chunk per streamed token (e.g. from the OpenAI-compatible
    ``POST {base_url}/v1/completions`` endpoint with ``"stream": true``).
    """
    raise NotImplementedError
