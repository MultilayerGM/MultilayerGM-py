import random as _rand
from .distributions import categorical, dirichlet
from itertools import accumulate


def sample_partition(dependency_tensor, null_distribution,
                     steps=100,
                     initial_partition=None,
                     state_nodes=None,
                     ):
    if state_nodes is None:
        state_nodes = get_state_nodes(dependency_tensor)
    state_nodes = [tuple(n) for n in state_nodes]
    node_buckets = OrderedAspectBuckets(dependency_tensor, state_nodes)
    if initial_partition is None:
        # partition = _np.zeros(dependency_tensor.shape)
        partition = dict()
        for node in state_nodes:
            partition[node] = null_distribution(node)
    else:
        partition = initial_partition

    for nodes in node_buckets:
        for it in range(steps):
            _rand.shuffle(nodes)
            for node in nodes:
                update_node = dependency_tensor.getrandneighbour(node)
                if update_node == node:
                    partition[node] = null_distribution(node)
                else:
                    partition[node] = partition[update_node]

    return partition


def dirichlet_null(layers, theta, n):
    weights = dict()
    for layer in SubscriptIterator(layers):
        weights[layer] = list(accumulate(dirichlet(theta, n)))
    def null(node):
        return categorical(weights[node[1:]])
    return null


def get_ordered_aspects(dependency_tensor):
    shape = dependency_tensor.shape[1:]
    for i in range(len(shape)):
        if dependency_tensor.aspect_types[i] == 'r':
            shape[i] = 0
    return SubscriptIterator(shape)


def get_state_nodes(dependency_tensor):
    return SubscriptIterator(dependency_tensor.shape)


def number_of_layers(dependency_tensor, aspect_type=None):
    if aspect_type is None:
        def check(t):
            return True
    else:
        def check(t):
            return t == aspect_type

    n = 1
    for s, t in zip(dependency_tensor.shape[1:], dependency_tensor.aspect_types):
        if check(t):
            n *= s
    return n


class SubscriptIterator:
    def __init__(self, shape):
        self.shape = shape
        self.state = [0 for _ in self.shape]
        self.state[0] = -1

    def __iter__(self):
        return self

    def __next__(self):
        self._update_state(0)
        return tuple(self.state)

    def _update_state(self, index):
        if self.state[index] < self.shape[index] - 1:
            self.state[index] += 1
        else:
            if index >= len(self.shape) - 1:
                raise StopIteration
            else:
                self.state[index] = 0
                self._update_state(index + 1)


class OrderedAspectBuckets:
    """
    divide state nodes into buckets based on ordered aspects
    """
    def __init__(self, dependency_tensor, state_nodes):
        self.index = [i+1 for i, t in enumerate(dependency_tensor.aspect_types) if t == 'o']
        self.shape = [1]
        for i in self.index:
            self.shape.append(self.shape[-1]*dependency_tensor.shape[i])
        self.buckets = [[] for _ in range(self.shape[-1])]
        for node in state_nodes:
            self.buckets[self.map(node)].append(node)

    def map(self, node):
        m = 0
        for i, s in zip(self.index, self.shape[:-1]):
            m += node[i] * s
        return m

    def __iter__(self):
        return (b for b in self.buckets)
