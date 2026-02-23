import warnings

import numpy as np

# sidereal days in seconds
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
sidereal_day_earth = 23.9345 * 3600
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
sidereal_day_moon = 655.720 * 3600

sidereal_day = {"earth": sidereal_day_earth, "moon": sidereal_day_moon}

Y00 = 1 / np.sqrt(4 * np.pi)  # the 0,0 spherical harmonic function


def _get_pix_weigths():
    warnings.warn(
        "The constant PIX_WEIGHTS_NSIDE is deprecated and will be removed in "
        "a future version. It was used for healpy routines which are no "
        "longer used in croissant",
        DeprecationWarning,
        stacklevel=2,
    )
    return (32, 64, 128, 512, 1024, 2048, 4096)


PIX_WEIGHTS_NSIDE = _get_pix_weigths()
