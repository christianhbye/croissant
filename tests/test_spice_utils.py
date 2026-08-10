import subprocess
import sys

import numpy as np
import spiceypy as spice

from croissant import spice_utils

# Runs in a subprocess so clearing the SPICE kernel pool cannot affect
# other tests, and so no previously-imported package can have furnished
# kernels as a side effect (lunarsky < 1.0 did this at import time).
_COLD_POOL_SCRIPT = """
import numpy as np
import spiceypy as spice

spice.kclear()

from croissant.rotations import get_mepa_rotation_matrix

R1 = get_mepa_rotation_matrix(0.0)
np.testing.assert_allclose(R1 @ R1.T, np.eye(3), atol=1e-12)
np.testing.assert_allclose(np.linalg.det(R1), 1.0, atol=1e-12)

# User code may clear the shared kernel pool between croissant calls;
# the pool, not croissant-side bookkeeping, is the source of truth, so
# the next uncached MEPA call must re-furnish and succeed.
spice.kclear()
get_mepa_rotation_matrix.cache_clear()
R2 = get_mepa_rotation_matrix(0.0)
np.testing.assert_allclose(R2, R1, atol=1e-15)
"""


def test_mepa_frame_from_cold_spice_pool():
    result = subprocess.run(
        [sys.executable, "-c", _COLD_POOL_SCRIPT],
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert result.returncode == 0, result.stderr


def test_furnish_lunar_kernels_idempotent():
    spice_utils.furnish_lunar_kernels()
    nloaded = spice.ktotal("ALL")
    spice_utils.furnish_lunar_kernels()
    assert spice.ktotal("ALL") == nloaded


def test_furnished_kernels_define_moon_me():
    spice_utils.furnish_lunar_kernels()
    R = np.array(spice.pxform("J2000", "MOON_ME", 0.0))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
