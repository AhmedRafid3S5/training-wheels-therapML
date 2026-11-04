import math
from typing import Any, Dict
import torch
from torch import Tensor
from jaxtyping import Float
import numpy as np
import matplotlib.pyplot as plt
import copy


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

if __name__ == "__main__":

  np.random.seed(42)
  
  # Activation functions and their derivatives
  def relu(x):
      return np.maximum(0, x)
  def d_relu(x):
      return (x > 0).astype(float)
  
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))
  def d_sigmoid(x):
      s = sigmoid(x)
      return s * (1 - s)
  
  #10-layer MLP
  num_layers = 10
  input_size = 1
  hidden_size = 5
  output_size = 1
  
  # Initialize identical weights and biases for fair comparison
  weights = []
  biases = []
  
  W_first = np.random.randn(hidden_size, input_size) 
  b_first = np.random.randn(hidden_size, 1)
  weights.append(W_first)
  biases.append(b_first)
  
  for _ in range(num_layers - 2):
      weights.append(np.random.randn(hidden_size, hidden_size))
      biases.append(np.random.randn(hidden_size, 1))
  
  weights.append(np.random.randn(output_size, hidden_size))
  biases.append(np.random.randn(output_size, 1))
  
  # Deep copy for two separate networks
  weights_sig = copy.deepcopy(weights)
  biases_sig = copy.deepcopy(biases)
  weights_relu = copy.deepcopy(weights)
  biases_relu = copy.deepcopy(biases)
  
  def forward_backward(x, y, weights, biases, activation, d_activation):
      # Forward Pass
      activations = [x]
      pre_activations = []
      
      a = x
      for i in range(num_layers):
          z = weights[i] @ a + biases[i]
          pre_activations.append(z)
          
          # Output layer is always linear for regression
          if i == num_layers - 1:
              a = z
          else:
              a = activation(z)
          activations.append(a)
          
      y_pred = activations[-1]
      
      # Backward Pass (Error at output layer: MSE loss derivative)
      delta = (y_pred - y) # shape (1,1)
      
      grad_magnitudes = []
      
      # Backpropagate through the layers
      for i in reversed(range(num_layers)):
          # Calculate weight gradient magnitude (L2 norm) for monitoring
          dW = delta @ activations[i].T
          grad_magnitudes.append(np.linalg.norm(dW))
          
          # Propagate error to previous layer
          if i > 0:
              delta = weights[i].T @ delta
              delta = delta * d_activation(pre_activations[i-1])
              
      # Reverse the list so index 0 is Layer 1
      grad_magnitudes.reverse() 
      return grad_magnitudes
  
  # Test with a single data point
  x = np.array([[0.5]])
  y = np.array([[1.0]])
  
  # Get gradient magnitudes layer by layer
  grad_sig = forward_backward(x, y, weights_sig, biases_sig, sigmoid, d_sigmoid)
  grad_relu = forward_backward(x, y, weights_relu, biases_relu, relu, d_relu)
  
  # Plotting
  plt.figure(figsize=(10, 6))
  layers = np.arange(1, num_layers + 1)
  
  plt.plot(layers, grad_sig, marker='o', color='crimson', label='Sigmoid Network', linewidth=2)
  plt.plot(layers, grad_relu, marker='s', color='dodgerblue', label='ReLU Network', linewidth=2)
  
  plt.yscale('log')
  plt.title('Gradient Magnitude Flowing Backwards (Vanishing Gradient Problem)', fontsize=14)
  plt.xlabel('Layer Index (1 = First Hidden Layer, 10 = Output Layer)', fontsize=12)
  plt.ylabel('Gradient L2 Norm (Log Scale)', fontsize=12)
  plt.xticks(layers)
  plt.grid(True, linestyle='--', alpha=0.7)
  plt.legend(fontsize=12)
  plt.tight_layout()
  plt.show()