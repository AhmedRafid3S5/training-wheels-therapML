import torch
import math
from torch import Tensor
import torch.nn.functional as F
from typing import Any
from therapml.rope import rope

#Stateful KV Cache Class
#NOTE: The sole job of this class is to hold a growing tensor of K and V
class CachedGQA:
    def __init__(
            self,
            d_model: int, num_heads: int, num_kv_heads: int,
            max_seq_len: int, theta: float,
            q_proj_weight: Tensor, k_proj_weight: Tensor, 
            v_proj_weight: Tensor, o_proj_weight: Tensor
    ):

        """Constructor allocated persistent memory for K and V"""
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        self.q_proj = q_proj_weight
        self.k_proj = k_proj_weight
        self.v_proj = v_proj_weight
        self.o_proj = o_proj_weight
        
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads

        #cache memory allocated for K and V
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0

def get_cached_attention(
    d_model: int, num_heads: int, num_kv_heads: int, max_seq_len: int, theta: float,
    q_proj_weight: Tensor, k_proj_weight: Tensor, v_proj_weight: Tensor, o_proj_weight: Tensor   
    ):
    """Return a fresh (empty-cache) causal GQA + RoPE attention object."""

    return CachedGQA(
        d_model, num_heads, num_kv_heads, max_seq_len, theta,
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight
    )

def run_cached_attention(attn: CachedGQA, in_features: Tensor):
    """Feed new tokens through the cached attention object."""

    #enforce all calculation to hardware that holds the in_features Tensor
    device = in_features.device
    d_type = in_features.dtype
    new_seq = in_features.size(-2)

    # in_features is the new incoming tokens
    # attn is the CachedGQA object passed to this function
    Q = F.linear(in_features,attn.q_proj)
    K = F.linear(in_features,attn.k_proj)
    V = F.linear(in_features,attn.v_proj)

    # Unflatten and transpose to separate the heads
    Q = Q.unflatten(-1, (attn.num_heads, attn.head_dim)).transpose(-2, -3)
    K = K.unflatten(-1, (attn.num_kv_heads, attn.head_dim)).transpose(-2, -3)
    V = V.unflatten(-1, (attn.num_kv_heads, attn.head_dim)).transpose(-2, -3)

    #Extract absolute positions of the updated cache size
    positions = torch.arange(attn.cache_len, attn.cache_len + new_seq, device = device)
    Q = rope(attn.head_dim,attn.theta,0,Q,positions)
    K = rope(attn.head_dim,attn.theta,0,K,positions)

    #Append new K and V to historical cache
    if attn.k_cache is None: # the cache is empty and this is the first KV calculation
       batch_size = in_features.size(0)

       #Pre allocate maxmium seq len during initial memory allocation
       attn.k_cache = torch.zeros( # type: ignore
           batch_size,attn.num_kv_heads, attn.max_seq_len, attn.head_dim,
           device=device, dtype=d_type
       )

       attn.v_cache = torch.zeros_like(attn.k_cache) ## type: ignore

    start = attn.cache_len
    end = start + new_seq

    #IN-PLACE assignment for new K , works even when cache is empty
    attn.k_cache[..., start:end, :] = K # type: ignore
    attn.v_cache[..., start:end, :] = V # type: ignore

    #extract the active portion of the cache, discarding the unallocated trailing zeros
    active_k_cache = attn.k_cache[..., :end,:] # type: ignore
    active_v_cache = attn.v_cache[..., :end,:] # type: ignore

    full_seq = active_k_cache.size(-2)
    # 4. GQA Duplication on the ACTIVE CACHE
    K_full = torch.repeat_interleave(active_k_cache, repeats=attn.num_groups, dim=-3)
    V_full = torch.repeat_interleave(active_v_cache, repeats=attn.num_groups, dim=-3)

    # Q only has 'new_seq' rows. K_full has 'full_seq' columns.
    # scores shape becomes: [..., num_heads, new_seq, full_seq]
    scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    full_mask = torch.ones(full_seq, full_seq, dtype=torch.bool, device=device).tril()
    mask = full_mask[attn.cache_len : full_seq, :]
    
    scores = scores.masked_fill(~mask, float('-inf'))

    attn_weights = torch.softmax(scores,dim=-1)
    out = torch.matmul(attn_weights,V_full)
    out = out.transpose(-2,-3).contiguous()
    out = out.flatten(-2,-1)

    #update internal state
    attn.cache_len = full_seq

    return F.linear(out, attn.o_proj)

def get_caches(attn: CachedGQA):
    """Returns the caches K and V tensors"""
    #only return the active cache !!

    if attn.k_cache is None:
        return (None, None)

    return (attn.k_cache[...,:attn.cache_len,:],attn.v_cache[...,:attn.cache_len,:]) # type: ignore


def benchmark_generation(
    seq_lens: list[int] = [128, 256, 512],
    d_model: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    batch_size: int = 4,
    theta: float = 10000.0,
    save_path: str = "kv_cache_benchmark.png",
):
    """
    Benchmark token-by-token generation speed: with KV-cache vs. without KV-cache
    for specified sequence lengths (e.g. 128, 256, 512).
    Generates and saves a tokens/sec vs. sequence length comparison plot.
    """
    import time
    import matplotlib.pyplot as plt
    from therapml.attention import grouped_query_attention

    d_head = d_model // num_heads
    kv_dim = num_kv_heads * d_head
    scale = d_model ** -0.5

    # Deterministic projection weights
    g = torch.Generator().manual_seed(42)
    q_proj_weight = torch.randn(d_model, d_model, generator=g) * scale
    k_proj_weight = torch.randn(kv_dim, d_model, generator=g) * scale
    v_proj_weight = torch.randn(kv_dim, d_model, generator=g) * scale
    o_proj_weight = torch.randn(d_model, d_model, generator=g) * scale

    tok_per_sec_cached = []
    tok_per_sec_uncached = []

    print("\n" + "=" * 68)
    print(f"{'Seq Len':<10} | {'No-Cache (tok/s)':<18} | {'With-Cache (tok/s)':<20} | {'Speedup':<10}")
    print("-" * 68)

    for seq_len in seq_lens:
        x = torch.randn(batch_size, seq_len, d_model, generator=torch.Generator().manual_seed(123))

        # --- 1. With KV-Cache ---
        # Warmup (8 steps)
        attn_warm = CachedGQA(d_model, num_heads, num_kv_heads, seq_len, theta,
                              q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
        for t in range(min(8, seq_len)):
            run_cached_attention(attn_warm, x[:, t : t + 1])

        # Timed run: generate seq_len tokens one-by-one
        attn = CachedGQA(d_model, num_heads, num_kv_heads, seq_len, theta,
                         q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(seq_len):
                run_cached_attention(attn, x[:, t : t + 1])
        elapsed_cached = time.perf_counter() - t0
        rate_cached = (batch_size * seq_len) / elapsed_cached
        tok_per_sec_cached.append(rate_cached)

        # --- 2. Without KV-Cache (recompute entire prefix at each step) ---
        # Warmup (8 steps)
        for t in range(1, min(8, seq_len) + 1):
            grouped_query_attention(d_model, num_heads, num_kv_heads,
                                   q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
                                   x[:, :t], theta=theta, max_seq_len=seq_len)

        # Timed run: recompute attention over the whole prefix at each step
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(1, seq_len + 1):
                grouped_query_attention(d_model, num_heads, num_kv_heads,
                                       q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
                                       x[:, :t], theta=theta, max_seq_len=seq_len)
        elapsed_uncached = time.perf_counter() - t0
        rate_uncached = (batch_size * seq_len) / elapsed_uncached
        tok_per_sec_uncached.append(rate_uncached)

        speedup = rate_cached / rate_uncached
        print(f"{seq_len:<10} | {rate_uncached:<18.1f} | {rate_cached:<20.1f} | {speedup:<10.2f}x")

    print("=" * 68)

    # --- Plotting ---
    plt.figure(figsize=(9, 5.5))
    plt.plot(seq_lens, tok_per_sec_cached, marker='o', linewidth=2.5, color='#2ca02c', label='With KV-Cache')
    plt.plot(seq_lens, tok_per_sec_uncached, marker='s', linewidth=2.5, color='#d62728', label='Without KV-Cache (Recompute)')

    for i, s in enumerate(seq_lens):
        speedup = tok_per_sec_cached[i] / tok_per_sec_uncached[i]
        plt.annotate(
            f"{speedup:.1f}x speedup\n({tok_per_sec_cached[i]:.0f} tok/s)",
            xy=(s, tok_per_sec_cached[i]),
            xytext=(0, 12),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#2ca02c',
        )
        plt.annotate(
            f"{tok_per_sec_uncached[i]:.0f} tok/s",
            xy=(s, tok_per_sec_uncached[i]),
            xytext=(0, -22),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            color='#d62728',
        )

    plt.title("Generation Speed: KV-Cache vs. Full Prefix Recomputation", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Sequence Length", fontsize=11)
    plt.ylabel("Generation Speed (Tokens / Second)", fontsize=11)
    plt.xticks(seq_lens)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    benchmark_generation()


def benchmark_generation(
    seq_lens: list[int] = [128, 256, 512],
    d_model: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    batch_size: int = 4,
    theta: float = 10000.0,
    save_path: str = "kv_cache_benchmark.png",
):
    """
    Benchmark token-by-token generation speed: with KV-cache vs. without KV-cache
    for specified sequence lengths (e.g. 128, 256, 512).
    Generates and saves a tokens/sec vs. sequence length comparison plot.
    """
    import time
    import matplotlib.pyplot as plt
    from therapml.attention import grouped_query_attention

    d_head = d_model // num_heads
    kv_dim = num_kv_heads * d_head
    scale = d_model ** -0.5

    # Deterministic projection weights
    g = torch.Generator().manual_seed(42)
    q_proj_weight = torch.randn(d_model, d_model, generator=g) * scale
    k_proj_weight = torch.randn(kv_dim, d_model, generator=g) * scale
    v_proj_weight = torch.randn(kv_dim, d_model, generator=g) * scale
    o_proj_weight = torch.randn(d_model, d_model, generator=g) * scale

    tok_per_sec_cached = []
    tok_per_sec_uncached = []

    print("\n" + "=" * 68)
    print(f"{'Seq Len':<10} | {'No-Cache (tok/s)':<18} | {'With-Cache (tok/s)':<20} | {'Speedup':<10}")
    print("-" * 68)

    for seq_len in seq_lens:
        x = torch.randn(batch_size, seq_len, d_model, generator=torch.Generator().manual_seed(123))

        # --- 1. With KV-Cache ---
        # Warmup (8 steps)
        attn_warm = CachedGQA(d_model, num_heads, num_kv_heads, seq_len, theta,
                              q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
        for t in range(min(8, seq_len)):
            run_cached_attention(attn_warm, x[:, t : t + 1])

        # Timed run: generate seq_len tokens one-by-one
        attn = CachedGQA(d_model, num_heads, num_kv_heads, seq_len, theta,
                         q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(seq_len):
                run_cached_attention(attn, x[:, t : t + 1])
        elapsed_cached = time.perf_counter() - t0
        rate_cached = (batch_size * seq_len) / elapsed_cached
        tok_per_sec_cached.append(rate_cached)

        # --- 2. Without KV-Cache (recompute entire prefix at each step) ---
        # Warmup (8 steps)
        for t in range(1, min(8, seq_len) + 1):
            grouped_query_attention(d_model, num_heads, num_kv_heads,
                                   q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
                                   x[:, :t], theta=theta, max_seq_len=seq_len)

        # Timed run: recompute attention over the whole prefix at each step
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(1, seq_len + 1):
                grouped_query_attention(d_model, num_heads, num_kv_heads,
                                       q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
                                       x[:, :t], theta=theta, max_seq_len=seq_len)
        elapsed_uncached = time.perf_counter() - t0
        rate_uncached = (batch_size * seq_len) / elapsed_uncached
        tok_per_sec_uncached.append(rate_uncached)

        speedup = rate_cached / rate_uncached
        print(f"{seq_len:<10} | {rate_uncached:<18.1f} | {rate_cached:<20.1f} | {speedup:<10.2f}x")

    print("=" * 68)

    # --- Plotting ---
    plt.figure(figsize=(9, 5.5))
    plt.plot(seq_lens, tok_per_sec_cached, marker='o', linewidth=2.5, color='#2ca02c', label='With KV-Cache')
    plt.plot(seq_lens, tok_per_sec_uncached, marker='s', linewidth=2.5, color='#d62728', label='Without KV-Cache (Recompute)')

    for i, s in enumerate(seq_lens):
        speedup = tok_per_sec_cached[i] / tok_per_sec_uncached[i]
        plt.annotate(
            f"{speedup:.1f}x speedup\n({tok_per_sec_cached[i]:.0f} tok/s)",
            xy=(s, tok_per_sec_cached[i]),
            xytext=(0, 12),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#2ca02c',
        )
        plt.annotate(
            f"{tok_per_sec_uncached[i]:.0f} tok/s",
            xy=(s, tok_per_sec_uncached[i]),
            xytext=(0, -22),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            color='#d62728',
        )

    plt.title("Generation Speed: KV-Cache vs. Full Prefix Recomputation", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Sequence Length", fontsize=11)
    plt.ylabel("Generation Speed (Tokens / Second)", fontsize=11)
    plt.xticks(seq_lens)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    benchmark_generation()
