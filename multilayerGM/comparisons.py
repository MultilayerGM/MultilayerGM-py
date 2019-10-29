from collections import defaultdict as _dfdict
from collections import Counter as _Counter
from math import log2
from statistics import mean


def layer_partitions(partition):
    """
    Construct induced partitions for each layer from an input multilayer partition.

    :param partition: Input partition as mapping of state-node to mesoset

    :return: mapping of layer to induced partition
    """
    partitions = _dfdict(dict)
    for node, p in partition.items():
        partitions[tuple(node[1:])][node[0]] = p
    return partitions


def nmi(partition1, partition2):
    """
    Compute NMI between two partitions. If the input partitions are multilayer, this function computes the multilayer
    NMI.

    :param partition1: first input partition as mapping of node to mesoset
    :param partition2: second input partition as mapping of node to mesoset

    :return: NMI value (normalised by joint entropy)
    """
    n = len(partition1)
    if len(partition2) != n:
        raise ValueError("partitions need to have the same number of elements")
    p12 = _Counter((partition1[key], partition2[key]) for key in partition1)
    h12 = sum((p/n) * log2(p/n) for p in p12.values())

    p1 = _Counter(partition1.values())
    h1 = sum((p/n) * log2(p/n) for p in p1.values())

    p2 = _Counter(partition2.values())
    h2 = sum((p/n) * log2(p/n) for p in p2.values())

    return (h1 + h2 - h12) / h12


def mean_nmi(partition1, partition2):
    """
    Compute mean NMI between induced partitions for a pair of multilayer partitions.

    :param partition1: first input partition as mapping of state-node to mesoset
    :param partition2: second input partition as mapping of state-node to mesoset

    :return: mean NMI value (normalised by joint entropy)
    """
    layer_partitions1 = layer_partitions(partition1)
    layer_partitions2 = layer_partitions(partition2)
    return mean(nmi(layer_partitions1[layer], layer_partitions2[layer]) for layer in layer_partitions1)


def nmi_tensor(partition):
    """
    Compute NMI between all pairs of induced partitions for an input multilayer partition.

    :param partition: input partition as mapping of state-node to mesoset
    :return: nmi values as mapping dict(layer1: dict(layer2: nmi))
    """
    lp = layer_partitions(partition)
    return {l1: {l2: nmi(lp[l1], lp[l2]) for l2 in lp} for l1 in lp}
