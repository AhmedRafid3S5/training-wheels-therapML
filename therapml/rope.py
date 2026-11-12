import torch
from torch import Tensor

def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
):

    """Apply Rotary Positional Encoding to a key or query vector of a token"""

    device = in_query_or_key.device
    dtype = in_query_or_key.dtype

    two_i = torch.arange(0, d_k,2, device=device, dtype=dtype)
    freqs = theta ** -(two_i/ d_k) # a list of frequencies
    angles = token_positions.unsqueeze(-1).to(dtype) * freqs
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)

    cos  = torch.cos(angles)
    sin  = torch.sin(angles)

    x_swapped = torch.empty_like(in_query_or_key)
    x_swapped[..., 0::2] = -in_query_or_key[...,1::2]
    x_swapped[..., 1::2] =  in_query_or_key[...,0::2]

    output = (in_query_or_key * cos) + (x_swapped * sin)

    return output 


