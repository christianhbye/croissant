from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from healpy import Rotator
import numpy as np

def topo_to_radec(phi, theta, time, loc):
    """
    Convert topocentric coordinates to ra/dec at given time. Useful for
    antenna beams.
    """
    if np.isscalar(phi):
        phi = np.array([phi])
    else:
        phi = np.array(phi)
    if np.isscalar(theta):
        theta = np.array([theta])
    else:
        theta = np.array(theta)
    # Allow loc to be earth location object and time to be Time object
    lat, lon = loc
    loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg)
    obstime = Time(time)
    azs = [ph*u.degree for ph in phi]
    alts = [(90-th)*u.degree for th in theta]
    altaz = AltAz(alt=alts, az=azs, location=loc, obstime=obstime)
    icrs = altaz.transform_to("icrs")
    ra = icrs.ra.deg
    dec = icrs.dec.deg
    return ra, dec

def rotate_map(sky_map, from_coords="galactic", to_coords="equatorial"):
    coords = {"galactic": "G", "ecliptic": "E", "equatorial": "C"}
    fc = from_coords.lower()
    tc = to_coords.lower()
    if not all([fc, tc] in coords):
        raise ValueError(
            f"Invalid coordinate system name, must be in {list{coords.keys()}}"
        )
    rot = Rotator(coord=[coords[fc], coords[tc]])
    rotated_map = rot.rotate_map_alms(sky_map)
    return rotated_map
