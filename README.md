# multilayerGM (Version 0.1.0)

This code implements the method for generating multilayer networks with 
planted mesoscale structure of

>M. Bazzi, L. G. S. Jeub, A. Arenas, S. D. Howison, and M. A. Porter. 
Generative benchmark models for mesoscale structure in multilayer networks. 
arXiv:1608.06196 [cs.SI], 2016.

If you use results based on this code in an academic publication please cite 
this paper and cite the code as
>L. G. S. Jeub. A Python framework for generating multilayer networks with planted 
mesoscale structure. <https://github.com/MultilayerGM/MultilayerGM-py>, 2019 


## Installation

This package supports installation with setuptools. The easiest way to 
install the latest version of this package and its dependencies is to use 
```
pip install git+https://github.com/MultilayerGM/MultilayerGM-py.git@master
```
To install a particular version, replace `master` above by the appropriate commit
identifier, e.g., use `v0.1.0` to install version 0.1.0.


## Basic usage

The examples below assume that the package has been imported as
```python
import multilayerGM as gm
```

The model generates a multilayer network in two steps: 
1.  Generate a multilayer partition with desired dependencies between layers

2.  Generate a multilayer network that reflects this partition

### Generate partition

The main interface for generating partitions is 
```python
partition = gm.sample_partition(dependency_tensor=dt, null_distribution=null)
```
and supports generating partitions and supports multiple aspects with a mix of
ordered and unordered aspects. The type of multilayer network is determined by 
the choice of interlayer dependency tensor. Different dependency tensors are 
implemented in `gm.dependency_tensors`.

The null distribution determines the random component of the generating process 
and is largely responsible for determining the expected mesoset sizes. `null` 
should be a function that takes a state node as input and returns a random mesoset
assignment. 

Currently, the only implemented choice (which is the one we use for the numerical
examples in the paper) for the null distribution independently samples the 
mesoset distribution for each layer from a symmetric dirichlet distribution 
with concentration parameter `theta` and number of mesosets `n_sets`. 

```python
null = gm.dirichlet_null(layers=dt.shape[1:], theta=theta, n_sets=n_sets)
```

However, it is straight-forward to substitute different null distributions.


#### Multiplex network with uniform dependencies:

To set up a dependency tensor for generating multiplex networks with uniform
interlayer dependencies use
```python
dt = gm.dependency_tensors.UniformMultiplex(n_nodes, n_layers, p)
```
where `n_nodes` is the desired number of nodes, `n_layers` is the desired number 
of layers and `p` is the copying probability. With probability `p`, the state node
that is being updated copies its mesoset assignment from a different state node 
representing the same physical node chosen uniformly at random. This is an unordered
dependency tensor (i.e., `dt.aspect_types='r'`) and the partition will be sampled using pseudo-Gibbs sampling.


#### Temporal network

To set up a dependency tensor for generating temporal networks use
```python
dt = gm.dependency_tensors.Temporal(n_nodes, n_layers, p)
```
where `n_nodes` is the desired number of nodes, `n_layers` is the desired number 
of layers and `p` specifies the copying probabilities. A state node in layer `l` that is being updated copies its 
mesoset assignment from the state node representing the same physical node in the previous layer with probability 
`p[l-1]`. Otherwise the state node
gets a new mesoset assignment by sampling from the null distribution. Mesoset assignments
for state nodes in the first layer are always sampled from the null distribution. One can also specify a single 
probability for `p` to generate a temporal network with uniform dependencies. This
is an ordered dependency tensor (i.e., `dt.aspect_types='o'`) and the partition
will be sampled sequentially.


#### Networks with multiple aspects

To generate multilayer networks with multiple aspects but without direct inter-aspect
dependencies use
```python
dt = gm.dependency_tensors.MultiAspect(tensors, weights)
```
where `tensors` is a list of single-aspect dependency tensors and `weights` is a list of 
corresponding weights. An update for this dependency tensor proceeds in two steps. First, 
an aspect is selected with probability proportional to its weight and then the update 
proceeds according to the dependency tensor for this aspect. The dependency tensors in 
`tensors` can be a mix of ordered and unordered tensors and accordingly the partition will be sampled
using a mix of sequential and pseudo-Gibbs sampling.


#### Custom dependency tensor

To generate networks that do not fit in the above categories, it is possible
to define a fully-custom dependency tensor. To do so, the custom dependency 
tensor should be a subclass of `gm.dependency_tensors.DependencyTensor` or 
otherwise make sure to implement the same interface. Subclasses of `gm.dependency_tensors.DependencyTensor`
have to at a minimum define a `getrandneighbour` method which determines the copying 
process (see the inline documentation). In most cases one would also need to extent the 
constructor (i.e., define a custom `__init__` method) to store additional parameters. 
Note that `gm.dependency_tensors.DependencyTensor` makes it possible to specify
the state nodes that are included in the tensor such that one construct networks 
that are not fully interconnected. However, if one wants to use this feature, one
has to ensure that the `getrandneighbour` method can only return state nodes that
are actually included in the tensor.


### Generate network

Currently the only implemented network model is the DCSBM based benchmark model
that we use for the numerical examples in the paper. To generate a network from
a multilayer partition using this model use
```python
multinet = gm.multilayer_DCSBM_network(partition, mu=0.1, k_min=5, k_max=70, t_k=-2)
```
Here `partition` is a multilayer partition defined as a mapping from state nodes to 
mesoset assignments (such as the output of `gm.sample_partition()`) and `mu` is a parameter 
that determines the strength of the planted community structure (for `mu=0` communities 
are disconnected from each other and for `mu=1` the network has no communities). The distribution
of expected degrees for the network is a truncated powerlaw with minimum cutoff `k_min`, 
maximum cutoff `k_max`, and exponent `t_k`.

The network is returned as `MultilayerGraph` which extends the NetworkX `Graph` class
with some multilayer network functionality (see https://github.com/LJeub/nxMultilayerNet for more details).
Each node has a `'mesoset'` attribute, such that the planted partition can be recovered 
using
```python
partition = dict(multinet.nodes(data='mesoset'))
```

### Save results

This package includes functions to export the results to JSON and MATLAB '.mat' format. 

To export a partition to JSON (note that state nodes are converted from tuple to string) use
```python
gm.export.save_json_partition(partition, filename)
```

To load the partition data (converting state nodes back to tuples) use
```python
partition = gm.export.load_json_partition(filename)
```

To export the multilayer network as a JSON edgelist use
```python
gm.export.save_json_edgelist(multinet, filename)
```

To load the network data (optionally specify a filename for partition data 
to restore `'mesoset'` node attribute) use
```python
multinet = gm.export.load_json_multinet(edgelist, partition=None)
```