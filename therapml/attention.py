import torch
import math
from torch import Tensor
import torch.nn.functional as F

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