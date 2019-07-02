import random as _rand
from .distributions import categorical, dirichlet
from .dependency_tensors import SubscriptIterator
from itertools import accumulate
import numpy as _np


def sample_partition(dependency_tensor, null_distribution,
                     updates=100,
                     initial_partition=None
                     ):
    """
    Sample partition for a multilayer network with specified interlayer dependencies

    :param dependency_tensor: dependency tensor
    :param null_distribution: null distribution (function that takes a state-node as input and returns a random mesoset
                              assignment
    :param updates: expected number of (pseudo-)Gibbs-sampling updates per state-node (has no effect for fully ordered
                    dependency tensor. (optional, default=100)
    :param initial_partition: mapping of state-nodes to initial meso-set assignment.
                              (optional, default=sampled from null distribution)

    :return: sampled partition as a mapping (dict) from state-nodes to meso-set assignments.
    """
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
    """
    Samples meso-set assignment probabilities from a Dirichlet distribution and returns a categorical null-distribution
    based on these probabilities.

    :param layers: [a_1,...a_d] Number of layers for each aspect
    :param theta: concentration parameter for the dirichlet distribution
    :param n: number of meso-sets
    :return: null distribution function: (state-node) -> random meso-set
    """
    weights = dict()
    for layer in SubscriptIterator(layers):
        weights[layer] = list(accumulate(dirichlet(theta, n)))

    def null(node):
        return categorical(weights[node[1:]])

    return null



