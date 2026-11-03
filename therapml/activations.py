import math
from typing import Any, Dict
import torch
from torch import Tensor
from jaxtyping import Float


class Activation:
    def __init__(self) -> None:
        pass

    @staticmethod
    def mySigmoid(in_features: Float[Tensor, "..."]) :
        return 1.0 / ( 1.0 + torch.exp(-in_features))

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
        #NOTE : tanh approximation does not meet the test tolerance requirements
        # fast approximation of gelu using tanh
        #const1 = math.sqrt(2/math.pi)
        #const2 = 0.044715
        #return 0.5*(in_features)*(1+torch.tanh(const1*(in_features + const2*torch.pow(in_features,3))))
        const = (1.0 / math.sqrt(2))
        return 0.5*in_features*(1.0 + torch.erf(const*in_features))
    @staticmethod
    def mySwiGLU(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
        in_features: Float[Tensor, " ... d_model"],
    ) -> Float[Tensor, " ... d_model"]:
        # Swish(x) = x * sigmoid(x)
        # 1. Gate projection: Swish(x @ W1^T)
        gate = torch.nn.functional.linear(in_features, w1_weight)
        swish_gate = gate * torch.sigmoid(gate)

        # 2. Value/up projection: x @ W3^T
        value = torch.nn.functional.linear(in_features, w3_weight)

        # 3. Element-wise gating: Swish(x @ W1^T) * (x @ W3^T)
        gated = swish_gate * value

        # 4. Down projection back to d_model: gated @ W2^T
        return torch.nn.functional.linear(gated, w2_weight)

