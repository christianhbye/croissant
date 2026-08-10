"""
Explicit SPICE kernel management for croissant's lunar frames.

Croissant historically relied on lunarsky (< 1.0) furnishing the lunar
frame kernels into the SPICE kernel pool as an import side effect.
lunarsky >= 1.0 evaluates lunar orientation natively and no longer
touches SPICE, so croissant furnishes the kernels its own ``spiceypy``
calls need. The kernels are unmodified NAIF originals vendored in
``croissant/data`` (public-domain US government data).
"""

from importlib.resources import files
from threading import Lock

import spiceypy as spice

# Kernels required to evaluate the MOON_ME frame with spice.pxform:
# - moon_pa_de421_1900-2050.bpc: DE421 lunar principal-axes orientation
# - moon_080317.tf: defines the MOON_PA / MOON_ME frame family
# - moon_assoc_me.tf: associates the MOON_ME alias with MOON_ME_DE421
LUNAR_KERNELS = (
    "moon_pa_de421_1900-2050.bpc",
    "moon_080317.tf",
    "moon_assoc_me.tf",
)

_furnish_lock = Lock()
_furnished = False


def furnish_lunar_kernels():
    """
    Furnish the SPICE kernels that define the MOON_ME frame.

    Idempotent and thread-safe: the kernels are loaded into the global
    SPICE kernel pool at most once per process. Call this before any
    direct ``spiceypy`` computation involving MOON_ME; it is invoked
    automatically by the rotation helpers in ``croissant.rotations``.
    """
    global _furnished
    with _furnish_lock:
        if _furnished:
            return
        data = files("croissant") / "data"
        for name in LUNAR_KERNELS:
            spice.furnsh(str(data / name))
        _furnished = True
