import json
from scipy.io import savemat
import numpy as np
import nxmultilayer as nxm
from ast import literal_eval


def save_json_edgelist(multinet, filename):
    """
    Export edgelist for multilayer network in JSON format

    :param multinet: Multilayer network in nxm.MultilayerGraph or nxm.MultilayerDiGraph format
    :param filename: filepath for output
    """
    with open(filename, 'w') as f:
        json.dump(list(multinet.to_directed(as_view=True).edges()), f)


def save_json_partition(partition, filename):
    """
    Export partition in JSON format (note that node keys are converted to string)

    :param partition: map from state nodes to mesoset assignments
    :param filename: filepath for output
    """
    partition = {repr(node): p for node, p in partition}
    with open(filename, 'w') as f:
        json.dump(partition, f)


def load_json_multinet(edgelist, partition=None):
    """
    Load multilayer network from JSON edgelist and optionally load partition data as well

    :param edgelist: filename for edgelist data
    :param partition: (optional) filename for partition data
    :return: nxm.MultilayerDiGraph (if partition is provided each node has a 'mesoset' attribute)
    """
    with open(edgelist) as f:
        edges = json.load(f)
    if edges:
        mg = nxm.MultilayerDiGraph(n_aspects=len(edges[0][0]))
        for source, target in edges:
            mg.add_edge(source, target)
        if partition is not None:
            with open(partition) as f:
                partition_data = json.load(f)
            partition_data = {literal_eval(key): value for key, value in partition_data.items()}
            for node, p in partition_data:
                mg.add_node(node, mesoset=p)
    else:
        mg = nxm.MultilayerDiGraph()
    return mg


def load_JSON_partition(filename):
    """
    Load partition from JSON data
    :param filename: path
    :return: dict: Mapping from state nodes to mesosets
    """
    with open(filename) as f:
        partition_data = json.load(f)
    partition_data = {literal_eval(key): value for key, value in partition_data.items()}
    return partition_data


def save_matlab(multinet, file):
    """
    Save multilayer network and partition in MATLAB format. Output file contains three variables:
    source: start node for each edge
    target: end node for each edge
    partition: mesoset assignment for each state node

    (Note that this function assumes that state-nodes are integer tuples)

    :param multinet: Multilayer network (each state node should have a 'mesoset' attribute)
    :param file: Output path
    :return:
    """
    multinet = multinet.to_directed(as_view=True)
    source = np.row_stack([e[0] for e in multinet.edges()]) + 1 # one-based indeces for MATLAB
    target = np.row_stack([e[1] for e in multinet.edges()]) + 1

    partition = np.ndarray(shape=tuple(len(a) for a in multinet.aspects), dtype=float)
    for node, p in multinet.nodes(data='mesoset'):
        partition[node]=p
    savemat(file, {'source': source, 'target': target, 'partition': partition})
