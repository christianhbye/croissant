from importlib.metadata import PackageNotFoundError, version

from . import (
    constants,
    dense,
    multipair,
    polarization,
    rotations,
    simulator,
    utils,
)
from .beam import Beam
from .dense import DenseSphericalTransform, dense_compute_alm
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
