from importlib.metadata import PackageNotFoundError, version

from . import constants, multipair, rotations, simulator, utils
from .beam import Beam
from .simulator import Simulator
from .sky import Sky
from .sphere import clear_dense_matrix_cache, precompute_dense_matrix

# isort: split
from . import alm

__author__ = "Christian Hellum Bye"
try:
    __version__ = version("croissant-sim")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
