from typing import Any,Dict
import numpy as np
from sympy import Float
from torch import Tensor
import torch
from therapml.logsumexp import logsumexp


class Loss:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _crossentropyloss(logits,ground_truth):
        #max logit
        max_logits = torch.max(logits,dim=-1, keepdim=True).values
        log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(logits-max_logits),dim=-1,keepdim=True))
        log_probs = logits - log_sum_exp

        sample_losses = -torch.sum(ground_truth*log_probs, dim=-1)
        return torch.mean(sample_losses)

    @staticmethod
    def kldivergenceloss( input_log_probs,target_probs):
        P = target_probs
        logQ = input_log_probs
        mask = P > 0.0
        kl_elementwise = torch.zeros_like(P)
        kl_elementwise[mask] = P[mask]*(torch.log(P[mask]) - logQ[mask])
        return torch.sum(kl_elementwise) / P.shape[0] # returns avg loss across D dimensions for each sample N
