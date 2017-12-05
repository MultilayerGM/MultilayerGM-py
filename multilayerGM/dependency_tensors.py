import random as _rand
from itertools import accumulate
from .distributions import categorical

from abc import ABC, abstractmethod


class DependencyTensor(ABC):
    def __init__(self, shape, aspect_types):
        self.shape = shape
        self.aspect_types = aspect_types

    @abstractmethod
    def getrandneighbour(self, node):
        rnode = (_rand.randint(i) for i in self.shape)
        return rnode

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DependencyTensor:
            if any("getrandneighbour" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented


class UniformMultiplex(DependencyTensor):
    def __init__(self, nodes, layers, p):
        super().__init__((nodes, layers), 'r')
        self.p = p

    def getrandneighbour(self, node):

        if _rand.random() > self.p:
            # returning the same state node for resampling from null
            return node
        else:
            n = _rand.randrange(0, self.shape[1]-1)
            if n >= node[1]:
                n += 1
            return (node[0], n)


class UniformTemporal(DependencyTensor):
    def __init__(self, nodes, layers, p):
        super().__init__((nodes, layers), 'o')
        self.p = p

    def getrandneighbour(self, node):
        if node[1] == 0 or _rand.random() > self.p:
            # returning the same state node for resampling from null
            return node
        else:
            return (node[0], node[1]-1)


class BlockMultiplex(DependencyTensor):
    def __init__(self, nodes, layers, n_blocks, p_in, p_out):
        super().__init__((nodes, layers), 'r')


class MultiAspect(DependencyTensor):
    def __init__(self, tensors, alpha):
        nodes = tensors[0].shape[0]
        for t in tensors:
            if t.shape[0] != nodes:
                raise ValueError("All aspects need to have the same number of nodes")
        shape =[nodes]
        shape += [t.shape[1] for t in tensors]
        super().__init__(tuple(shape), "".join(t.aspect_types for t in tensors))
        self.calpha = list(accumulate(alpha))
        a_sum = self.calpha[-1]
        if a_sum != 1:
            self.calpha[:] = [a/a_sum for a in self.calpha]
        self.alpha = [a/a_sum for a in alpha]
        self.tensors= tensors

    def getrandneighbour(self, node):
        i = categorical(self.calpha)
        n, a_i = self.tensors[i].getrandneighbour((node[0], node[i+1]))
        node = list(node) # convert to list for index assignment
        node[0] = n
        node[i+1] = a_i
        return tuple(node)

