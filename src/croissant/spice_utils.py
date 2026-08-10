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


def _loaded_kernel_files():
    """Return the file paths currently in the SPICE kernel pool."""
    return {spice.kdata(i, "ALL")[0] for i in range(spice.ktotal("ALL"))}


def furnish_lunar_kernels():
    """
    Furnish the SPICE kernels that define the MOON_ME frame.

    Idempotent and thread-safe. The SPICE kernel pool itself is the
    source of truth: only kernels missing from the pool are furnished,
    so MEPA computations keep working even if user code clears or
    reloads the pool (e.g. with ``spice.kclear``) between croissant
    calls. Invoked automatically by the rotation helpers in
    ``croissant.rotations``.
    """
    with _furnish_lock:
        data = files("croissant") / "data"
        loaded = _loaded_kernel_files()
        for name in LUNAR_KERNELS:
            path = str(data / name)
            if path not in loaded:
                spice.furnsh(path)
