import torch
import math
from torch import Tensor
import torch.nn.functional as F
from therapml.rope import rope

def scaled_attention(Q,K,V,mask):

    #get d_k for scale
    # -1 means last dim of Tensor, which corresponds to the column in a 2D matrix
    d_k = Q.size(-1)
    # .transpose(-2, -1) swaps the final two dimensions (length and features) 
    # while leaving all arbitrary leading batch/head dimensions untouched.
    scores = torch.matmul(Q, K.transpose(-2,-1))
    scores = scores / math.sqrt(d_k) # normalization for numeric stabaility

    if mask is not None:
        # ~mask inverts the boolean tensor (True becomes False).
        # masked_fill_ replaces values where the condition is True with -infinity.
        scores = scores.masked_fill(~mask,float('-inf'))

    attention_weights = torch.softmax(scores,dim=-1)
    output = torch.matmul(attention_weights,V)

    return output # output = softmax(QK_T)*V

def causal_attention(Wq,Wk,Wv,in_features):

    #calculate Q,K,V using the weights and the in_features embeddings
    Q = F.linear(in_features, Wq)
    K = F.linear(in_features, Wk)
    V = F.linear(in_features, Wv)

    seq_len = in_features.size(-2)
    #create the mask for causualattention
    # torch.ones creates a matrix of 1s.
    # .tril() zeroes out everything strictly above the main diagonal.
    # > 0 converts it to a boolean mask (True for allowed, False for masked)
    mask = torch.ones(seq_len,seq_len, device=in_features.device).tril() > 0

    return scaled_attention(Q,K,V,mask)

def multihead_attention(Wq,Wk,Wv,Wo,in_features,d_model,num_heads,token_pos:Tensor, theta=None):

    head_dim = d_model // num_heads
    seq_len = in_features.size(-2)

    #compute Q,K,V using the input and learned weights
    Q = F.linear(in_features, Wq)
    K = F.linear(in_features, Wk)
    V = F.linear(in_features, Wv)

    

    #change tensor orientation
    #1st arg = -1 means last dim is the target which is d_model
    #2nd arg = (num_heads, head_dim) is a tuple defining the new dimensions that will replace whatever is in -1


    Q = Q.unflatten(-1, (num_heads,head_dim)).transpose(-2,-3)
    K = K.unflatten(-1, (num_heads,head_dim)).transpose(-2,-3)
    V = V.unflatten(-1, (num_heads,head_dim)).transpose(-2,-3)

    # rope is applied per head, so apply rope after splitting q,k and v
    if theta is not None:
        pos = token_pos.unsqueeze(-2) if token_pos.ndim < Q.ndim -1 else token_pos
        Q = rope(head_dim, theta,0, Q, pos)
        K = rope(head_dim, theta,0, K, pos)

    scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(head_dim)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device).tril()
    scores = scores.masked_fill(~mask, float('-inf'))

    #calculate attention
    attn = torch.softmax(scores, dim=-1) # dim=-1 applies softmax across the columns of the matrix
    out  = torch.matmul(attn, V)

    #recombine heads
    out = out.transpose(-2,-3).contiguous()
    out = out.flatten(-2,-1)

    output_proj = F.linear(out, Wo)

    return output_proj



def grouped_query_attention(
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    theta: float | None = None,
    max_seq_len: int | None = None,
    token_positions: Tensor | None = None,
):
    head_dim = d_model // num_heads
    num_groups = num_heads // num_kv_heads #num of kv groups per head
    seq_len = in_features.size(-2)
    device = in_features.device
    dtype = in_features.dtype

    Q = F.linear(in_features, q_proj_weight)
    K = F.linear(in_features, k_proj_weight)
    V = F.linear(in_features, v_proj_weight)

    # Q gets split per head
    # G KV blocks per head
    Q = Q.unflatten(-1,(num_heads,head_dim)).transpose(-2,-3)
    K = K.unflatten(-1,(num_kv_heads, head_dim)).transpose(-2,-3)
    V = V.unflatten(-1, (num_kv_heads, head_dim)).transpose(-2,-3)

    if theta is not None:
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=device)

        Q = rope(head_dim,theta,seq_len,Q,token_positions)
        K = rope(head_dim,theta,seq_len,K,token_positions)

    # 4. Grouped-Query Attention: Duplicate K and V heads to match Q heads
    # repeat_interleave duplicates the data along the 'head' dimension (dim=-3)
    K = torch.repeat_interleave(K, repeats=num_groups, dim=-3)
    V = torch.repeat_interleave(V, repeats=num_groups, dim=-3)

    scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(head_dim)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    scores = scores.masked_fill(~mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn,V)

    out = out.transpose(-2,-3).contiguous()
    out = out.flatten(-2,-1)

    return F.linear(out, o_proj_weight)
    