from importlib.metadata import PackageNotFoundError, version

from . import (
    constants,
    dense,
    engine_select,
    kernel,
    multipair,
    polarization,
    rotations,
    simulator,
    sphere,
    utils,
)
from .beam import Beam
from .dense import (
    DenseSphericalTransform,
    clear_dense_matrix_cache,
    dense_cache_nbytes,
    dense_compute_alm,
    precompute_dense_matrix,
)
from .engine_select import resolve_engine
from .kernel import (
    clear_kernel_cache,
    kernel_cache_nbytes,
    precompute_kernel,
)
from .polarization import PairStokesBeam, PolarizedSky, polarized_convolve
from .simulator import Simulator
from .sky import Sky

# isort: split
from . import alm

__author__ = "Christian Hellum Bye"
try:
    __version__ = version("croissant-sim")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
