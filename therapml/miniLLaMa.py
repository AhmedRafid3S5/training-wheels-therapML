import torch
import torch.nn as nn
from torch import Tensor

from therapml.transformer import TransformerBlock, RMSNorm, LayerNorm

class MiniLlama(nn.Module):

    def __init__(self,
                 vocab_size:     int,
                 context_length: int,
                 d_model :       int,
                 num_layers :    int,
                 num_heads :     int,
                 num_kv_heads:   int,
                 d_ff :          int,
                 rope_theta:     float,
                 tie_weights:    bool = True,
                 use_MHA_GELU:   bool = False):

        super().__init__()

        self.context_length = context_length
        self.tie_weights = tie_weights
        # a massive 2D look up table of rows= vocab_size, col= d_model (d dimensional space)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        # token_embeddings are fully trainable parameters
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)

        #Stack of Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                MHA_GELU=use_MHA_GELU
            )
            for _ in range(num_layers)
        ])
        if use_MHA_GELU:
           self.ln_final = RMSNorm(d_model)
        else:
           self.ln_final =  LayerNorm(d_model)
        # convert the n dimensional vector into a list of logits of the length vocab_size 
        # turn the [1, d_model] vector into [1, vocab_size]
        # via matrix multiplication with a [d_model, vocab_size] matrix for projection
        self.lm_head =  nn.Linear(d_model, vocab_size, bias = False)

        if tie_weights:
            #use the same matrix for up and down projections
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, in_indices: Tensor) :
         """
                in_indices : (batch, seq)  — integer token IDs, 0 ≤ id < vocab_size
                Returns    : (batch, seq, vocab_size)  — logits for each position
        """
         batch, seq = in_indices.shape
         #use the embeddings lookup table to convert token id into an embedding vector
         x = self.token_embeddings(in_indices) # shape: [batch, seq, d_model]
         # calculate absolute position of each token in the sequence for RoPE
         token_positions = torch.arange(seq, device=in_indices.device)

         #pass through every transformer block
         for tf_block in self.layers:
             x = tf_block(x, token_positions) # shape: [batch, seq, d_model]

         x = self.ln_final(x)
         raw_logits = self.lm_head(x)

         return raw_logits

    def load_weights(self, weights: dict[str, Tensor]) -> None:
        if self.tie_weights:
            # Detach lm_head from the shared tensor before loading
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.data.clone())

        # Use PyTorch's built-in loader; strict=False tolerates minor key mismatches
        missing, unexpected = self.load_state_dict(weights, strict=False)
        if missing:
            print(f"[MiniLlama.load_weights] Missing keys: {missing}")
        if unexpected:
            print(f"[MiniLlama.load_weights] Unexpected keys: {unexpected}")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Audit Utility
# ─────────────────────────────────────────────────────────────────────────────

def parameter_audit(model: nn.Module, print_table: bool = True) -> list[dict]:
    """
    Inspect every parameter tensor in the model and return a list of rows.

    Each row is a dict with keys:
        name        – str, the dotted parameter name (e.g. "layers.0.ln1.weight")
        shape       – tuple[int, ...], e.g. (d_model,)
        num_params  – int, total elements (product of shape)
        percent     – float, percentage of the model's total parameters [0, 100]

    Also prints a formatted table when print_table=True.
    """
    # Collect all named parameters
    params = list(model.named_parameters())
    total  = sum(p.numel() for _, p in params)

    rows = []
    for name, p in params:
        num = p.numel()
        rows.append({
            "name":       name,
            "shape":      tuple(p.shape),
            "num_params": num,
            "percent":    100.0 * num / total,
        })

    if print_table:
        # Determine column widths dynamically for a tidy table
        max_name  = max(len(r["name"]) for r in rows)
        max_shape = max(len(str(r["shape"])) for r in rows)

        header = (
            f"{'Parameter':<{max_name}}  "
            f"{'Shape':<{max_shape}}  "
            f"{'# Params':>12}  "
            f"{'% of Total':>10}"
        )
        sep = "─" * len(header)

        print("\n" + sep)
        print(header)
        print(sep)
        for r in rows:
            print(
                f"{r['name']:<{max_name}}  "
                f"{str(r['shape']):<{max_shape}}  "
                f"{r['num_params']:>12,}  "
                f"{r['percent']:>9.3f}%"
            )
        print(sep)
        print(f"{'TOTAL':<{max_name}}  {'':>{max_shape}}  {total:>12,}  {'100.000%':>10}")
        print(sep + "\n")

    return rows
