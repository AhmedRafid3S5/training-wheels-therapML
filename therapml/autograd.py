from typing import Any, Dict, Set, Tuple
import math
import numpy as np


class Value:
    #NOTE: = () is a tuple placeholder
    #        _op is a string used to identify the type of operation
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None  #pass lambda: None so that call onto _backwards() of a leaf does not crash the program
        self._prev = set(_children) #a set() is enforced to ensure that each child is registered once, also so that topological sort
        self._op = _op              #processes each unique dependency only once

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self,other), '+')

        def _backward():
            other.grad += out.grad # for z = a + b, delZ/dela = delL/delZ * 1.0 
            self.grad  += out.grad # _backward() is an internal function
        out._backward  = _backward# calling _backward witout bracktes binds the function to a variable
        # binding a function , _backward remembers the memory references, recording the maths history
        return out

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data * self.data, (self,other), _op='*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t,(self,))

        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        topology = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topology.append(v)

        #build the topology
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topology):
            v._backward()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure project root is in sys.path
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import the same regression data generator used in test_optimizers
    from tests.phase_1.part_1.test_optimizers import _make_regression

    # Generate a simple 1D linear regression: y = w_true * x + noise
    # Using d=1 so we can train a single scalar weight with our Value class
    X_np, y_np, w_true = _make_regression(n=100, d=1, seed=42)
    X_flat = X_np.flatten()
    y_flat = y_np.flatten()

    print(f"True weight: {w_true[0]:.4f}")
    print(f"Dataset: {len(X_flat)} samples")
    print("-" * 50)

    # Initialize learnable parameter
    lr = 0.01
    n_epochs = 1000

    w = Value(0.0)  # start from zero

    for epoch in range(n_epochs):
        # --- Forward pass: compute MSE loss over all samples ---
        # Loss = (1/N) * sum( (w*x_i - y_i)^2 )
        #      = (1/N) * sum( (w*x_i - y_i) * (w*x_i - y_i) )
        total_loss = Value(0.0)
        for x_i, y_i in zip(X_flat, y_flat):
            pred = w * x_i           # w * x_i  (scalar)
            diff = pred + (-y_i)     # pred - y_i (using __add__ with negative constant)
            sq   = diff * diff       # (pred - y_i)^2
            total_loss = total_loss + sq

        # Mean loss
        mse = total_loss * (1.0 / len(X_flat))

        # --- Backward pass ---
        mse.backward()

        # --- Gradient descent update ---
        # Manually update w.data (NOT through the Value graph)
        w_new_data = w.data - lr * w.grad
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}  |  loss = {mse.data:.6f}  |  w = {w.data:.4f}  |  grad = {w.grad:.4f}")

        # Reset w for next epoch (create a fresh Value to avoid stale graph)
        w = Value(w_new_data)

    print("-" * 50)
    print(f"Learned weight: {w.data:.4f}")
    print(f"True weight:    {w_true[0]:.4f}")
    print(f"Error:          {abs(w.data - w_true[0]):.6f}")
