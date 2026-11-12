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