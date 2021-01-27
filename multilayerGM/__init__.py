from . import dependency_tensors
from . import export
from . import comparisons
from .networks import multilayer_DCSBM_network
from .partitions import sample_partition, DirichletNull, dirichlet_null
from pkg_resources import get_distribution, DistributionNotFound


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
