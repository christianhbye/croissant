import numpy as np

# nside's for which healpix has computed pixel weights:
PIX_WEIGHTS_NSIDE = [32, 64, 128, 256, 512, 1024, 2048, 4096]

sidereal_day = 86164.0905  # sidereal day in seconds

Y00 = 1 / np.sqrt(4 * np.pi)  # the 0,0 spherical harmonic function
