import numpy as np

# nside's for which healpix has computed pixel weights:
PIX_WEIGHTS_NSIDE = (32, 64, 128, 256, 512, 1024, 2048, 4096)

# sidereal days in seconds
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
sidereal_day_earth = 23.9345 * 3600
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
sidereal_day_moon = 655.720 * 3600

Y00 = 1 / np.sqrt(4 * np.pi)  # the 0,0 spherical harmonic function
