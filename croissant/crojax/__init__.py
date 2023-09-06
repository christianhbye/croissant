# enable double precision
from jax import config

config.update("jax_enable_x64", True)

from .healpix import Alm
