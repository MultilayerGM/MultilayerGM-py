import random as _rand
from .distributions import categorical, dirichlet
from .dependency_tensors import SubscriptIterator
from itertools import accumulate
import numpy as _np


def sample_partition(dependency_tensor, null_distribution,
                     updates=100,
                     initial_partition=None
                     ):
    if initial_partition is None:
        partition = {node: null_distribution(node) for node in dependency_tensor.state_nodes()}
    else:
        partition = {node: initial_partition[node] for node in dependency_tensor.state_nodes()}

    random_layers = list(dependency_tensor.random_aspect_layers())
    if len(random_layers) <= 1:
        n_updates = 1
    else:
        n_updates = updates * len(random_layers)
    for ordered_layer in dependency_tensor.ordered_aspect_layers():
        for it in range(n_updates):
            random_layer = _rand.choice(random_layers)
            layer = tuple(o+r for o, r in zip(ordered_layer, random_layer))
            for node in dependency_tensor.state_nodes(layer):
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






