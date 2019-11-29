import numpy as _np
from collections import defaultdict as _dfdict
import nxmultilayer as nxm


def multilayer_DCSBM_network(partition, mu=0.1, k_min=5, k_max=70, t_k=-2, degrees=None):
    """
    Generate multilayer benchmark networks with planted community structure using a DCSBM model. The mixing parameter
    `mu` determines the strength of the planted community structure. For `mu=0`, all edges are constrained to fall
    within communities. For `mu>0`, a fraction `mu` of edges is sampled independently of the planted communities.
    For `mu=1`, all edges are independent of the planted communities and the generated network has no community structure.

    The expected degrees are sampled from a truncated powerlaw distribution.

    The sampled multilayer network is returned as a MultilayerGraph which adds some functionality on top of a NetworkX
    Graph (see https://github.com/LJeub/nxMultilayerNet )

    :param partition: Partition to plant
    :param mu: Fraction of random edges (default: 0.1)
    :param k_min: Minimum cutoff for distribution of expected degrees
    :param k_max: Maximum cutoff for distribution of expected degrees
    :param t_k: Exponent for distribution of expected degrees
    :param degrees: (Optional) specify degree sequence as mapping of state-node -> degree (if specified, `k_min`,
                    `k_max`, and `t_k` have no effect and the specified degrees are used instead)

    :return: generated multilayer network
    """
    layer_partitions = _dfdict(dict)
    neighbors = _dfdict(set)
    if isinstance(partition, _np.ndarray):
        for node, p in _np.ndenumerate(partition):
            layer_partitions[tuple(node[1:])][tuple(node)] = p
    else:
        for node, p in partition.items():
            layer_partitions[tuple(node[1:])][tuple(node)] = p

    fixed_degrees = degrees

    for p in layer_partitions.values():
        n_nodes = len(p)
        nodes_l = list(p.keys())
        if fixed_degrees is None:
            degrees = power_law_sample(n_nodes, t=t_k, x_max=k_max, x_min=k_min)
        else:
            degrees = _np.array([fixed_degrees[node] for node in nodes_l])
        # random edges
        no_partition = _np.zeros(n_nodes, dtype=int)
        block_matrix = build_block_matrix(no_partition, degrees * mu)
        neighbors = block_model_sampler(block_matrix=block_matrix, partition=no_partition, degrees=degrees * mu,
                                        neighbors=neighbors, nodes=nodes_l)
        # community edges
        p_l = _np.fromiter(p.values(), dtype=int)
        block_matrix = build_block_matrix(p_l, degrees * (1 - mu))
        neighbors = block_model_sampler(block_matrix=block_matrix, partition=p_l, degrees=degrees * (1 - mu),
                                        neighbors=neighbors, nodes=nodes_l)
    multinet = nxm.MultilayerGraph(n_aspects=len(nodes_l[0])-1)
    if isinstance(partition, _np.ndarray):
        for node, p in _np.ndenumerate(partition):
            multinet.add_node(node, mesoset=p)
    else:
        for node, p in partition.items():
            multinet.add_node(node, mesoset=p)

    for node1, adj in neighbors.items():
        for node2 in adj:
            multinet.add_edge(node1, node2)

    return multinet


class IdentityMap:
    def __getitem__(self, item):
        return item


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



