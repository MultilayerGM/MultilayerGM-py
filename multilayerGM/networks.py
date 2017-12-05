import numpy as _np
from collections import defaultdict as _dfdict
from collections.abc import MutableSet
from itertools import chain


class IdentityMap:
    def __getitem__(self, item):
        return item


class MultiNet:
    def __init__(self, nodes=0, aspects=None, neighbors=None, directed=False):
        self.directed = directed
        self.aspects = aspects
        self.n_nodes = nodes
        if self.aspects is not None:
            self.aspects = list(self.aspects)
        self._neighbors = _dfdict(set)
        if not neighbors is None:
            self.add_from_neighbors_dict(neighbors)

    def add_from_neighbors_dict(self, neighbors_dict):
        for node1, neighbors in neighbors_dict.items():
            for node2 in neighbors:
                self.add_edge(node1, node2)

    def add_edge(self, node1, node2):
        self[node1].add(node2)

    def add_node(self, node):
        self[node]

    def __contains__(self, item):
        return tuple(item) in self._neighbors

    def __getitem__(self, node1):
        node1 = tuple(node1)

        def update_shape(node):
            if self.aspects is None:
                self.aspects = [x + 1 for x in node[1:]]
            else:
                if len(node) != len(self.aspects) + 1:
                    raise(ValueError('Mismatched number of aspects for node'))
                else:
                    self.aspects[:] = [a if x < a else x + 1 for a, x in zip(self.aspects, node[1:])]
            if node[0] >= self.n_nodes:
                self.n_nodes = node[0] + 1
        if node1 not in self._neighbors:
            update_shape(node1)

        neighbors = self._neighbors[node1]

        class NeighborSet(MutableSet):

            def __contains__(self, node2):
                return node2 in neighbors

            def __iter__(self):
                return iter(neighbors)

            def __len__(self):
                return len(neighbors)

            def __repr__(self):
                return 'NeighborSet: ' + repr(neighbors)

            def add(self1, node2):
                node2 = tuple(node2)
                if node2 not in neighbors:
                    if node2 not in self:
                        self[node2] # initialize node2
                    neighbors.add(node2)
                    if not self.directed:
                        self._neighbors[node2].add(node1)

            def discard(self1, node2):
                node2 = tuple(node2)
                neighbors.discard(node2)
                if not self.directed:
                    self._neighbors[node2].discard(node1)

        return NeighborSet()

    def __setitem__(self, key, value):
        value = [tuple(v) for v in value]
        if all(len(v) == len(self.aspects)+1 for v in value):
            self[key].clear()
            for v in value:
                self[key].add(v)
        else:
            raise(ValueError('Mismatched number of aspects for node'))

    def nodes(self, layer=None):
        if layer is None:
            return self._neighbors.keys()
        else:
            return (tuple(chain([i], layer)) for i in range(self.n_nodes) if chain([i], layer) in self._neighbors)

    def edges(self, layer1=None, layer2=None):
        if layer1 is None:
            if layer2 is None:
                for node1, neighbors in self._neighbors.items():
                    for node2 in neighbors:
                        yield (node1, node2)
            else:
                for node1, neighbors in self._neighbors.items():
                    for node2 in neighbors:
                        if list(node2[1:]) == list(layer2):
                            yield (node1, node2)
        else:
            for i in range(self.n_nodes):
                node1 = tuple(chain([i], layer1))
                if node1 in self._neighbors:
                    if layer2 is None:
                        for node2 in self._neighbors[node1]:
                            yield (node1, node2)
                    else:
                        for node2 in self._neighbors[node1]:
                            if list(node2[1:]) == list(layer2):
                                yield (node1, node2)

    def is_edge(self, node1, node2):
        if node1 in self._neighbors and node2 in self._neighbors[node1]:
            return True
        else:
            return False


def DCSBM_benchmark(mu, nodes=1000, k_min=5, k_max=70, t_k=-2, alpha=1.5, communities=50):
    degrees = power_law_sample(nodes,t=t_k, x_max=k_max,x_min=k_min)
    partition = _np.zeros(nodes, dtype=int)
    block_matrix = build_block_matrix(partition, degrees * mu)
    neighbors = block_model_sampler(block_matrix=block_matrix, partition=partition, degrees=degrees * mu)
    partition = sorted(
        _np.random.choice(communities, p=_np.random.dirichlet([alpha] * communities, 1).ravel(), size=nodes))
    block_matrix = build_block_matrix(partition, degrees * (1-mu))
    neighbors = block_model_sampler(block_matrix=block_matrix, partition=partition, degrees=degrees * (1-mu),
                                    neighbors=neighbors)
    return neighbors, partition


def multilayer_DCSBM_network(partition, mu=0.1, k_min=5, k_max=70, t_k=-2):
    layer_partitions = _dfdict(dict)
    neighbors = MultiNet()
    if isinstance(partition, _np.ndarray):
        for node, p in _np.ndenumerate(partition):
            layer_partitions[tuple(node[1:])][tuple(node)] = p
    else:
        for node, p in partition.items():
            layer_partitions[tuple(node[1:])][tuple(node)] = p

    for p in layer_partitions.values():
        n_nodes = len(p)
        nodes_l = list(p.keys())
        degrees = power_law_sample(n_nodes, t=t_k, x_max=k_max, x_min=k_min)
        # random edges
        p_l = _np.zeros(n_nodes, dtype=int)
        block_matrix = build_block_matrix(p_l, degrees * mu)
        neighbors = block_model_sampler(block_matrix=block_matrix, partition=p_l, degrees=degrees * mu,
                                        neighbors=neighbors, nodes=nodes_l)
        # community edges
        p_l = _np.fromiter(p.values(), dtype=int)
        block_matrix = build_block_matrix(p_l, degrees * (1 - mu))
        neighbors = block_model_sampler(block_matrix=block_matrix, partition=p_l, degrees=degrees * (1 - mu),
                                        neighbors=neighbors, nodes=nodes_l)
    return neighbors


def block_model_sampler(block_matrix, partition, degrees, neighbors=None, nodes=IdentityMap()):
    max_reject = 100
    partition = _np.array(partition)
    degrees = _np.array(degrees)
    n_nodes = len(partition)
    if n_nodes != len(degrees):
        raise ValueError('length mismatch between partition and node degrees')

    n_groups = max(partition)+1

    # look up map for group memberships
    partition_map = [[] for _ in range(n_groups)]
    for i, g in enumerate(partition):
        partition_map[g].append(i)

    group_sizes = [len(g) for g in partition_map]

    sigma = [degrees[g] / sum(degrees[g]) for g in partition_map]

    if neighbors is None:
        neighbors = _dfdict(set)

    for g1 in range(n_groups):
        for g2 in range(g1,n_groups):
            w = block_matrix[g1,g2]
            if w > 0:
                m = _np.random.poisson(w)
                if g1 == g2:
                    dense = 2*m > group_sizes[g1]*(group_sizes[g1]-1)
                    m = int(_np.ceil(m / 2))
                else:
                    dense = 2*m > group_sizes[g1]*group_sizes[g2]

                if dense:
                    probabilities = w * _np.outer(sigma[g1], sigma[g2])
                    probabilities[probabilities > 1] = 1
                    if g1 == g2:
                        for i in range(group_sizes[g1]):
                            ni = _np.extract(probabilities[i, i + 1:] > _np.random.rand(1, group_sizes[g1] - i - 1),
                                             _np.arange(i + 1, group_sizes[g1]))
                            for j in ni:
                                neighbors[nodes[partition_map[g1][i]]].add(nodes[partition_map[g2][j]])
                                neighbors[nodes[partition_map[g2][j]]].add(nodes[partition_map[g1][i]])
                    else:
                        for i in range(group_sizes[g1]):
                            ni = _np.flatnonzero(probabilities[i, :] > _np.random.rand([1, group_sizes[g2]]))
                            for j in ni:
                                neighbors[nodes[partition_map[g1][i]]].add(nodes[partition_map[g2][j]])
                                neighbors[nodes[partition_map[g2][j]]].add(nodes[partition_map[g1][i]])

                else:
                    for e in range(m):
                        is_neighbor = True
                        reject_count = 0
                        while is_neighbor and reject_count <= max_reject:
                            node1 = nodes[_np.random.choice(partition_map[g1], 1, p=sigma[g1])[0]]
                            node2 = nodes[_np.random.choice(partition_map[g2], 1, p=sigma[g2])[0]]
                            is_neighbor = node1 in neighbors[node2] or node1 == node2
                            reject_count += 1
                        if reject_count > max_reject:
                            print('rejection threshold reached')
                            break
                        neighbors[node1].add(node2)
                        neighbors[node2].add(node1)

    return neighbors


def build_block_matrix(partition, degrees):
    nc = max(partition) + 1
    block_matrix = _np.zeros([nc, nc])
    for i, g in enumerate(partition):
        block_matrix[g, g] += degrees[i]
    return block_matrix


def power_law_sample(n, t, x_min, x_max):
    y = _np.random.rand(n)
    if t != -1:
        return ((x_max**(t+1) - x_min**(t+1)) * y + x_min**(t+1))**(1/(t+1))
    else:
        return x_min*(x_max/x_min)**y



