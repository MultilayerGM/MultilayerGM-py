import json
from scipy.io import savemat
import numpy as np

def export_json_edgelist(multinet, file):
    with open(file, 'w') as f:
        json.dump(list(multinet.edges()), f)


def export_matlab_edgelist(multinet, file):
    source = np.row_stack(e[0] for e in multinet.edges()) + 1 # one-based indeces for MATLAB
    target = np.row_stack(e[1] for e in multinet.edges()) + 1
    savemat(file, {'source': source, 'target': target})


def export_matlab_partition(partition, file):
    if isinstance(partition, np.ndarray):
        savemat(file, {'partition': partition})
    else:
        shape = np.row_stack(partition.keys()).max(0) + 1
        np_partition = np.ndarray(shape=shape, dtype=float)
        for node, p in partition.items():
            np_partition[node] = p
        savemat(file, {'partition': np_partition})


