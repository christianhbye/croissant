__author__ = "Christian Hellum Bye"
__version__ = "3.1.0"

from . import constants, dpss, sphtransform
from .healpix import Alm, HealpixMap
from .beam import Beam
from .rotations import Rotator
from .simulator import Simulator
from .sky import Sky

# enable double precision
from jax import config
config.update("jax_enable_x64", True)
