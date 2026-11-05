import math
from typing import Any

import matplotlib.pyplot as plt
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
        # Xavier Normal: Var = 2 / (fan_in + fan_out)
        weight = nn.Parameter((torch.randn((f_out, f_in)) * math.sqrt(2.0)) / math.sqrt(f_in + f_out))
        return weight


if __name__ == "__main__":
    torch.manual_seed(42)
    n_layers = 50
    d = 256
    batch_size = 64

    # -------------------------------------------------------------
    # Matched Schemes (Proper Theoretical Pairings)
    # -------------------------------------------------------------
    # 1. Kaiming He Normal with ReLU (Var ≈ 1.0)
    x_kaiming_relu = torch.randn(batch_size, d)
    kaiming_relu_vars = [x_kaiming_relu.var().item()]
    for _ in range(n_layers):
        w = MyLinear.runKaiming(d, d)
        x_kaiming_relu = torch.nn.functional.relu(x_kaiming_relu @ w.T)
        kaiming_relu_vars.append(x_kaiming_relu.var().item())

    # 2. Xavier (Glorot) Normal with Tanh (Var ≈ 1.0)
    x_xavier_tanh = torch.randn(batch_size, d)
    xavier_tanh_vars = [x_xavier_tanh.var().item()]
    for _ in range(n_layers):
        w = MyLinear.runXavier(d, d)
        x_xavier_tanh = torch.tanh(x_xavier_tanh @ w.T)
        xavier_tanh_vars.append(x_xavier_tanh.var().item())

    # -------------------------------------------------------------
    # Mismatched Schemes (To Observe Variance Collapse & Inflation)
    # -------------------------------------------------------------
    # 3. Xavier with ReLU: Lacks factor of 2 for ReLU's half-zeroing -> Signal collapses/vanishes!
    x_xavier_relu = torch.randn(batch_size, d)
    xavier_relu_vars = [x_xavier_relu.var().item()]
    for _ in range(n_layers):
        w = MyLinear.runXavier(d, d)
        x_xavier_relu = torch.nn.functional.relu(x_xavier_relu @ w.T)
        xavier_relu_vars.append(x_xavier_relu.var().item())

    # 4. Kaiming with Tanh: Includes factor of 2, but Tanh has gain ≈ 1.0 -> Signal over-amplifies/saturates!
    x_kaiming_tanh = torch.randn(batch_size, d)
    kaiming_tanh_vars = [x_kaiming_tanh.var().item()]
    for _ in range(n_layers):
        w = MyLinear.runKaiming(d, d)
        x_kaiming_tanh = torch.tanh(x_kaiming_tanh @ w.T)
        kaiming_tanh_vars.append(x_kaiming_tanh.var().item())

    # -------------------------------------------------------------
    # Visual Plots: Activation Variance Across 50 Layers
    # -------------------------------------------------------------
    layers = list(range(n_layers + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left Plot: Linear Scale
    ax1.plot(layers, kaiming_relu_vars, label="Kaiming + ReLU (Matched)", color="forestgreen", linewidth=2.5)
    ax1.plot(layers, xavier_tanh_vars, label="Xavier + Tanh (Matched)", color="dodgerblue", linewidth=2.5)
    ax1.plot(layers, xavier_relu_vars, label="Xavier + ReLU (Mismatched - Vanishes)", color="darkorange", linestyle="--", linewidth=2)
    ax1.plot(layers, kaiming_tanh_vars, label="Kaiming + Tanh (Mismatched - Saturates)", color="crimson", linestyle="--", linewidth=2)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.8, label="Ideal Variance (1.0)")
    ax1.set_title("50-Layer MLP: Variance Comparison (Linear Scale)")
    ax1.set_xlabel("Layer Index (0 = Input)")
    ax1.set_ylabel("Activation Variance")
    ax1.set_ylim(-0.1, 2.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Right Plot: Log Scale (Highlights exponential vanishing of Xavier + ReLU)
    ax2.plot(layers, kaiming_relu_vars, label="Kaiming + ReLU (Matched)", color="forestgreen", linewidth=2.5)
    ax2.plot(layers, xavier_tanh_vars, label="Xavier + Tanh (Matched)", color="dodgerblue", linewidth=2.5)
    ax2.plot(layers, xavier_relu_vars, label="Xavier + ReLU (Vanishes exponentially)", color="darkorange", linestyle="--", linewidth=2)
    ax2.plot(layers, kaiming_tanh_vars, label="Kaiming + Tanh (Over-amplifies)", color="crimson", linestyle="--", linewidth=2)
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.8, label="Ideal Variance (1.0)")
    ax2.set_yscale("log")
    ax2.set_title("50-Layer MLP: Vanishing vs Stability (Log Scale)")
    ax2.set_xlabel("Layer Index (0 = Input)")
    ax2.set_ylabel("Activation Variance (log scale)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.show()