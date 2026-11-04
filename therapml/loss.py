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
