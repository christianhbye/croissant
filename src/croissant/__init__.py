__author__ = "Christian Hellum Bye"
__version__ = "5.0.0"

# enable double precision
from jax import config

config.update("jax_enable_x64", True)

from . import alm, constants, rotations, simulator, utils
from .beam import Beam
from .simulator import Simulator
