import math

import torch
from torch.optim import Optimizer
from typing import Iterable, Tuple

class MySGD(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
       

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self): #type: ignore

            """Single optimizaion step"""
            """EQN:
            V(t) = momentum*V(t-1)  + 1.0* param_grad
            """
            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                lr = group['lr']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    d_p = p.grad

                    #Weight Decay (L2 Penalty)
                    if weight_decay != 0.0:
                        # use .add() to create a new tensor rather than modifying p.grad in-place
                        d_p = d_p.add(p, alpha=weight_decay) #scalar multiple = weight_decay

                    #Applying Momentum
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach() # buf = d_p simply creates a pointer referencing d_p
                                                                                             # torch.clone creates a memory independent copy
                                                                                             # detach ensure there is no OOM since the clone also tracks autograd history
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1.0) #alpha is a builtin scalar multiple

                        d_p = buf

                    p.add_(d_p, alpha=-lr)