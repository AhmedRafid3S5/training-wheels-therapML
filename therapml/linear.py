import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

class MyLinear(nn.Module):
    """Custom implementation, inheriting nn Module class"""

    def __init__(self,d_in, d_out, bias = True):
        super().__init__()

        #random init of w and b without any initialization technique
        self.weight = nn.Parameter(torch.randn(d_out,d_in))
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output

    @staticmethod
    def runKaiming(f_in,f_out):
        weight = nn.Parameter((torch.randn((f_out,f_in))*math.sqrt(2)) / f_in**0.5) 
        return weight

    @staticmethod
    def runXavier(f_in, f_out):
        #torch.randn(dout,din) defines the shape of thw weight matrix, assigned random values sampled from ~N(0, var(x))
        #we broadcast the scaling factor across each element of this matrix using the pytorch broadcasting
        weight = nn.Parameter((torch.randn((f_out,f_in))) / f_in**0.5) 
        return weight