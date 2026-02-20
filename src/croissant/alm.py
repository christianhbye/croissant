import warnings

from .utils import (
    getidx,
    getlm,
    is_real,
    lmax_from_shape,
    reduce_lmax,
    shape_from_lmax,
    total_power,
)

warnings.warn(
    "The alm module is deprecated and will be removed in a future version. "
    "Use the utils module instead.",
    DeprecationWarning,
    stacklevel=2,
)
