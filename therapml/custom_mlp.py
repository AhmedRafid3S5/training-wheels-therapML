"""

__call__ turns an instance of a class into a callable object 
(meaning you can call it with parentheses () just like a standard function).

"""

from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from therapml.linear import MyLinear
from therapml.activations import Activation
from therapml.dropout import inverted_dropout
from therapml.norm import myLayerNorm
from therapml.loss import Loss
from therapml.adamw import MyAdam

#Layer Wrappers
#Custom class inherits the nn.Module parent class
#Each class must contain a `forward` function that pyTorch can call
class MyRelu(nn.Module):
    # x ---> Relu ---> y = max(x,0)
    def forward(self,x: Tensor) -> Tensor:
        return Activation.myReLU(x)

class MyDropout(nn.Module):
    def __init__(self, p: float= 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0.0:
            #ONLY APPLY DROPOUT DURING TRAINING
            return inverted_dropout(input=x,prob=self.p)
        # when not in training i.e in inference mode, pass the values as is
        return x

class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta  = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x:Tensor) ->Tensor:
            return myLayerNorm(x,self.gamma, self.beta)

class CustomMLP(nn.Module):
    """
       MLP ARCHITECTURE
       INPUT -> Layer 1 -> Layer 2 -> Output
       784   -> 256     -> 128     -> 10

    """

    # we initialize and configure each module in the constuctor
    def __init__(self, 
                 in_features: int = 784,
                 hidden_1: int = 256,
                 hidden_2: int = 128,
                 num_classes: int = 10) -> None:
        super().__init__()  
        ## Configure modules
        #Layer 1
        self.f1 = MyLinear(in_features,hidden_1)
        self.ln1= MyLayerNorm(hidden_1) 
        self.act1 = MyRelu()
        self.drop1 = MyDropout(0.1)

        #Layer 2
        self.f2 = MyLinear(hidden_1,hidden_2)
        self.ln2 = MyLayerNorm(hidden_2)
        self.act2 = MyRelu()
        self.drop2 = MyDropout(0.1)

        #Layer 3
        self.f3 = MyLinear(hidden_2,num_classes)

        ## Initialize weights with Kaiming
        with torch.no_grad():
          self.f1.weight.copy_(MyLinear.runKaiming(in_features, hidden_1))
          self.f2.weight.copy_(MyLinear.runKaiming(hidden_1,hidden_2))
          self.f3.weight.copy_(MyLinear.runKaiming(hidden_2,num_classes))

    def forward(self,x: Tensor) -> Tensor:
        ## Wiring up the modules happens here
        x = self.f1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.f2(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.f3(x)

        return x

class MyCrossEntropyLoss:
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        if targets.ndim == 1 or targets.shape[-1] != logits.shape[-1]:
            targets = torch.nn.functional.one_hot(targets, num_classes=logits.shape[-1]).float()
        return Loss._crossentropyloss(logits, targets)



             
