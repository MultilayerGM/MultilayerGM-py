import random as _rand
from itertools import accumulate, chain
from .distributions import categorical
from collections import defaultdict

from abc import ABC, abstractmethod


class DependencyTensor(ABC):
    """
    Base class for implementing dependency tensors.

    At a minimum, a concrete implementation of a dependency tensor needs to override the `getrandneighbour` method which
    implicitly specifies the interlayer dependencies.

    """
    def __init__(self, shape, aspect_types, state_nodes=None):
        """
        Initialise dependency tensor

        :param shape: [n, a_1, ..., a_d] specify number of nodes and number of layers for each aspect
        :param aspect_types: specify sampling type for each aspect ('o' for ordered or 'r' for random).
                             `aspect_types` can be a string or list of strings such that aspect_types[i] specifies the
                             type of the ith aspect. Note that `len(aspect_types) = len(shape)-1`.
        :param state_nodes: List of state-nodes (optional, defaults to all state nodes)
        """
        self.shape = shape
        self.aspect_types = aspect_types
        if state_nodes is None:
            self._state_nodes = {l: [(n, *l) for n in range(shape[0])] for l in SubscriptIterator(self.shape[1:])}
        else:
            self._state_nodes = {l: [] for l in SubscriptIterator(self.shape[1:])}
            for n in state_nodes:
                if all(mi > ni >= 0 for mi, ni in zip(self.shape, n)):
                    n_t = tuple(n)
                    self._state_nodes[n[1:]].append(n)
                else:
                    raise ValueError("Provided state nodes do not match `shape`")

    def state_nodes(self, layer=None):
        """
        Return list of state nodes

        :param layer: (optional, if specified only return state-nodes for this layer)
        """
        if layer is None:
            return chain.from_iterable(self._state_nodes.values())
        else:
            return self._state_nodes[layer]

    def ordered_aspect_layers(self):
        """
        List all layers corresponding to ordered aspects (indeces corresponding to unordered (random) aspects are fixed
        at 0. In particular, this method always includes the all-zero layer (0,...,0) even when there are no ordered
        aspects.
        :return: Iterator
        """
        shape = list(self.shape[1:])
        for i in range(len(shape)):
            if self.aspect_types[i] == 'r':
                shape[i] = 1
        return SubscriptIterator(shape)

    def random_aspect_layers(self):
        """
        List all layers corresponding to random aspects (indeces corresponding to ordered aspects are fixed
        at 0. In particular, this method always includes the all-zero layer (0,...,0) even when there are no unordered
        (random) aspects.
        :return: Iterator
        """
        shape = list(self.shape[1:])
        for i in range(len(shape)):
            if self.aspect_types[i] == 'o':
                shape[i] = 1
        return SubscriptIterator(shape)

    def number_of_layers(self, aspect_type=None):
        """
        Total number of layers
        :param aspect_type: optionally only count layers of a particular type
                            ('o' to return the number of classes of order-equivalent layers and or 'r' to return the
                            size of each class).
        :return: number of layers
        """
        if aspect_type is None:
            def check(t):
                return True
        else:
            def check(t):
                return t == aspect_type

        n = 1
        for s, t in zip(self.shape[1:], self.aspect_types):
            if check(t):
                n *= s
        return n

    @abstractmethod
    def getrandneighbour(self, node):
        """
        Return either a randomly sampled neighbour of the input state-node or the input state-node itself. This method
        is called to determine the updated meso-set assignment of a state-node. If the state-node returned by this
        method is not the same as the input state-node, the input state-node's meso-set assignment is updated by copying
        the assignment of the output state-node. If the input and output state-nodes are the same, the input state-nodes
        assignment is updated by sampling from the null-distribution.
        :param node: input state-node
        :return: output state-node
        """
        if _rand.random() > 0.5:
            # returning the same state node with probability 0.5 for resampling from null-distribution
            return node
        else:
            # return state-node sampled uniformly from all state-nodes in other layers (only an example)
            rnode = (_rand.randint(self.shape[0]), *(rn if rn < n else rn+1
                                                     for rn, n in zip((_rand.randint(i-1) for i in self.shape[1:]),
                                                                      node[1:])))
            return rnode


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
                raise ValueError("All tensors need to have the same number of nodes")
            if len(t.shape) != 2:
                raise ValueError("All tensors need to have exactly one aspect")
        shape =[nodes]
        shape += [t.shape[1] for t in tensors]
        super().__init__(tuple(shape), "".join(t.aspect_types for t in tensors))
        self.calpha = list(accumulate(alpha))
        a_sum = self.calpha[-1]
        if a_sum != 1:
            self.calpha[:] = [a/a_sum for a in self.calpha]
        self.alpha = [a/a_sum for a in alpha]
        self.tensors = tensors

    def getrandneighbour(self, node):
        i = categorical(self.calpha)
        n, a_i = self.tensors[i].getrandneighbour((node[0], node[i+1]))
        node = list(node) # convert to list for index assignment
        node[0] = n
        node[i+1] = a_i
        return tuple(node)


class SubscriptIterator:
    def __init__(self, shape):
        self.shape = shape
        self.state = [0 for _ in self.shape]
        if self.state:
            self.state[-1] = -1

    def __iter__(self):
        return self

    def __next__(self):
        self._update_state(len(self.shape)-1)
        return tuple(self.state)

    def _update_state(self, index):
        if index < 0:
            raise StopIteration
        else:
            if self.state[index] < self.shape[index] - 1:
                self.state[index] += 1
            else:
                self.state[index] = 0
                self._update_state(index - 1)
