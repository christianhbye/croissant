import warnings

import numpy as np

# sidereal days in seconds
# Earth: one turn of the Earth Rotation Angle (IAU 2000), the rotation
# about the CIRS pole that the Earth simulation frame is built on
sidereal_day_earth = 86400 / 1.00273781191135448
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
sidereal_day_moon = 655.720 * 3600

sidereal_day = {"earth": sidereal_day_earth, "moon": sidereal_day_moon}

Y00 = 1 / np.sqrt(4 * np.pi)  # the 0,0 spherical harmonic function


_PIX_WEIGHTS_NSIDE = (32, 64, 128, 512, 1024, 2048, 4096)


def __getattr__(name):
    if name == "PIX_WEIGHTS_NSIDE":
        warnings.warn(
            "The constant PIX_WEIGHTS_NSIDE is deprecated and will be removed "
            "in a future version of croissant. It was used for healpy "
            "routines which are no longer used in croissant",
            FutureWarning,
            stacklevel=2,
        )
        return _PIX_WEIGHTS_NSIDE
    raise AttributeError(f"module {__name__} has no attribute {name}")
