from math import pi, sqrt

# sidereal days in seconds
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
sidereal_day_earth = 23.9345 * 3600
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
sidereal_day_moon = 655.720 * 3600

sidereal_day = {"earth": sidereal_day_earth, "moon": sidereal_day_moon}

Y00 = 1 / sqrt(4 * pi)  # the 0,0 spherical harmonic function
