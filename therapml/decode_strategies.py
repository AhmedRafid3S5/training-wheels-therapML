
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from jaxtyping import Float, Int
from torch import Tensor, nn
import torch

def top_k(logits: Float[Tensor, "... vocab"], k: int):
    vocab_size = logits.size(-1)
    if k >= vocab_size:
        return logits.clone()

    k = max(1,k)
    # find top k along vocubulary direction
    top_values, _ = torch.topk(logits, k , dim = -1)
    kth_cutoff = top_values[..., -1:]

    return logits.masked_fill(logits< kth_cutoff, float("-inf"))

def top_p(logits: Float[Tensor, "... vocab"], p: float):
    probs = torch.softmax(logits, dim = -1)
    probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    cumilative_probs = torch.cumsum(probs, dim=-1) - probs
    remove_mask = cumilative_probs >= p
    original_remove_mask = torch.zeros_like(remove_mask).scatter_(
        dim=-1, index=sorted_indices, src=remove_mask
    )

    #original_remove_mask applise to the actual logits thensor
    return logits.masked_fill(original_remove_mask, float("-inf"))

def sample_next_token(
        logits: Float[Tensor, "... vocab"],
        temperature: float,
        top_k_val : int | None,
        top_p_val : float | None,
        ):
    if temperature == 0:
        return logits.argmax(dim = -1)
    else:
        scaled_logits = logits / temperature
    if top_k_val is not None:
        scaled_logits = top_k(scaled_logits, top_k_val) # type: ignore
    if top_p_val is not None:
        scaled_logits = top_p(scaled_logits, top_p_val) # type: ignore

    #convert to probabilities

    probs = torch.softmax(scaled_logits, dim=-1)

    # Flatten leading dimensions for torch.multinomial
    orig_shape = logits.shape[:-1]
    vocab_size = logits.size(-1)
    probs_2d = probs.reshape(-1, vocab_size)
    # Sample exactly 1 token per row using global RNG
    sampled_2d = torch.multinomial(probs_2d, num_samples=1).squeeze(-1)

    return sampled_2d.reshape(orig_shape).long()


