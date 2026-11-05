import math

import torch
from torch.optim import Optimizer
from typing import Iterable, Tuple

#inherit Optimizer to get zero_grad and parameter group management
class MyAdam(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float,float] = (0.9,0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01, #for L2 requralization, decoupled
          ):
          
          defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

          #pass defaults to superclass
          super().__init__(params, defaults)

    @torch.no_grad() #we are updating weights not calculating grads
    def step(self): # type: ignore
        for group in self.param_groups:
             for p in group["params"]:
                  if p.grad is None: #eg. frozen layer
                       continue

                  state = self.state[p]

                  if len(state) == 0:
                       #very first time we are updating this specific parameter
                       state["step"] = 0
                       state["exp_avg"] = torch.zeros_like(p)
                       state["exp_avg_sq"] = torch.zeros_like(p)

                  state["step"] += 1
                  beta1, beta2 = group["betas"]

                  # Decoupled Weight Decay
                  # p is a tensor, .mul_ is inplace tensor multiplication
                  p.mul_(1.0 - group["lr"] * group["weight_decay"])

                  # Update moving averages
                  state["exp_avg"].mul_(beta1).add_(p.grad, alpha=1.0-beta1)
                  state["exp_avg_sq"].mul_(beta2).addcmul_(p.grad,p.grad, value=1.0 - beta2)

                  # Bias correction
                  bias_correction1 = 1.0 - beta1 ** state["step"]
                  bias_correction2 = 1.0 - beta2 ** state["step"]

                  denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                  step_size = group["lr"] / bias_correction1
                  p.addcdiv_(state["exp_avg"], denom, value=-step_size) 
        



