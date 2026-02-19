import warnings

from . import *

warnings.warn(
    "The croissant.jax interface is deprecated and will be removed in a future release. Please use the croissant interface directly instead.",
    FutureWarning,
    stacklevel=2,
)
