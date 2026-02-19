__author__ = "Christian Hellum Bye"
__version__ = "5.0.0"

# enable double precision
from jax import config

config.update("jax_enable_x64", True)

from . import constants
from . import utils
from . import alm, rotations, simulator
