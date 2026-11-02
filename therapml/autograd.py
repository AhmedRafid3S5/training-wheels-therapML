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
