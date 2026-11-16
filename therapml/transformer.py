import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from therapml.norm import myRMSnorm
from therapml.attention import *
from therapml.activations import Activation
from therapml.rope import rope

# Prenormalization with RMSNorm applied before each sub layer

class RMSNorm(nn.Module):
    """
        Root-Mean-Square Layer Normalisation.
    
        For a vector x of length d:
            RMS(x) = sqrt( mean(x²) + ε )
            output  = (x / RMS(x)) * Y
    
        Y is a learned scale parameter initialised to 1 (no shift/bias term).
        """

    
    def __init__(self, d_model, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # Initialised to ones so the first pass is identity-like
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):

        return myRMSnorm(x, self.weight)


class GroupedQueryAttention(nn.Module):

     """
        Causal Grouped-Query Self-Attention with Rotary Positional Embeddings.
    
        Key ideas:
          - num_heads       Q heads
          - num_kv_heads    K and V heads  (must divide num_heads)
          - Each group of (num_heads // num_kv_heads) Q heads shares one K/V head.
          - RoPE is applied PER HEAD after splitting into (batch, heads, seq, head_dim).
          - No bias in any projection.
        """

     def __init__(self,d_model,num_heads,num_kv_heads,max_seq_len,theta):
         super().__init__()
         #assert divisibility of parameters
         assert d_model % num_heads == 0
         assert num_heads % num_kv_heads == 0

         self.num_heads    = num_heads
         self.num_kv_heads = num_kv_heads
         self.head_dim     = d_model // num_heads
         self.num_groups   = num_heads // num_kv_heads   # how many Q heads per KV head
         self.theta        = theta
         self.d_model      = d_model
         self.max_seq_len  = max_seq_len

         kv_dim = num_kv_heads * self.head_dim

         #initializing projection matrix Wq,Wv,Wk and Output
         self.q_proj = nn.Linear(d_model, d_model, bias=False)
         self.k_proj = nn.Linear(d_model, kv_dim,  bias=False)
         self.v_proj = nn.Linear(d_model, kv_dim,  bias=False)
         self.output_proj = nn.Linear(d_model, d_model, bias=False)

     def forward(self, x, token_positions):
         return grouped_query_attention(
             self.d_model,
             self.num_heads,
             self.num_kv_heads,
             self.q_proj.weight,
             self.k_proj.weight,
             self.v_proj.weight,
             self.output_proj.weight,
             x,
             self.theta,
             self.max_seq_len,
             token_positions,
         )

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate weight
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up projection

    def forward(self, x):
        # SwiGLU(x) = (SiLU(x @ W1^T) * (x @ W3^T)) @ W2^T
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):

    """
    Data-flow:
            x  →  RMSNorm  →  GQA + RoPE  →  + x  →  h
            h  →  RMSNorm  →  SwiGLU FFN  →  + h  →  out
    
    """

    def __init__(self,
                 d_model,
                 num_heads,
                 num_kv_heads,
                 d_ff,
                 max_seq_len,
                 theta) -> None:
        super().__init__()

        #define the sublayers here
        #NOTE: This only helps set up the size of each module
        self.ln1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions):
        #Modules are wired up in sequence in the forward def
        h = x + self.attn(self.ln1(x), token_positions)
        out = h + self.ffn(self.ln2(h))
        return out

