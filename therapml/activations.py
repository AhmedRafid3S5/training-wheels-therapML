from typing import Any, Dict
import torch
from torch import Tensor
from jaxtyping import Float


class Activation:
    def __init__(self) -> None:
        pass

    @staticmethod
    def mySigmoid(in_features: Float[Tensor, "..."]) :
        pass

    @staticmethod
    def mySoftmax(in_features: Float[Tensor, "..."], dim: int = -1):
        max_val = torch.max(in_features,dim=dim,keepdim=True).values
        exp_features = torch.exp(in_features - max_val)
        sum_exp = torch.sum(exp_features, dim=dim,keepdim=True)
        return exp_features / sum_exp

    @staticmethod
    def myReLU(in_features: Float[Tensor, "..."]) :
        return torch.maximum(in_features,torch.tensor(0.0))

    @staticmethod
    def myGELU(in_features: Float[Tensor, "..."]):
        pass

    @staticmethod
    def mySwiGLU(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
        in_features: Float[Tensor, " ... d_model"],
    ):
        pass

