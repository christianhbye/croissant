__author__ = "Christian Hellum Bye"
__version__ = "4.1.1"

import warnings

warnings.warn(
    "The NumPy/healpy-based version of 'croissant' is deprecated and will be"
    " removed in version 5.0.0. Starting with v5.0.0, 'croissant' will be"
    " exclusively JAX-native. If you require the NumPy backend, please pin"
    " your dependency to 'croissant<5.0.0'.",
    FutureWarning,
    stacklevel=2
)

from . import constants
from . import core
from . import dpss
from . import jax
from . import utils
from .core import *  # noqa F403
