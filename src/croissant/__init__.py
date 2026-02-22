from importlib.metadata import PackageNotFoundError, version

from . import constants, rotations, simulator, multipair, utils
from .beam import Beam
from .simulator import Simulator
from .sky import Sky

# isort: split
from . import alm

__author__ = "Christian Hellum Bye"
try:
    __version__ = version("croissant")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
