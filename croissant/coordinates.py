from astropy.coordinates import AltAz, EarthLocation, ICRS
from astropy.time import Time
from astropy import units
from healpy import Rotator
import numpy as np


def topo_to_radec(phi, theta, time, loc, grid=True):
    """
    Convert topocentric coordinates to ra/dec at given time. Useful for
    antenna beams.

    Parameters
    ----------
    phi : array-like
        The azimuth(s) in radians.
    theta : array-like
        The zenith angle(s) in radians.
    time :
    loc :
    grid : bool
        If True then phi and theta are assumed to be coordinate axes of a grid.
        This has no effect if phi and theta have size 1.
    return_matrix : bool
        If True, return the rotation matrix. Else, return the coordinates.

    Returns
    -------
    ra : np.ndarray
        The right acension(s) in radians.
    dec : np.ndarray
        The declination(s) in radians.

    """
    phi = np.squeeze(phi).reshape(-1)
    theta = np.squeeze(theta).reshape(-1)
    if grid:  # phi and theta are coordinate axis
        phi, theta = np.meshgrid(phi, theta)
        phi = np.flatten(phi)
        theta = np.flatten(theta)
    # Allow loc to be earth location object and time to be Time object
    if type(loc) != EarthLocation:
        lat, lon, alt = loc
        loc = EarthLocation(
            lat=lat * units.deg, lon=lon * units.deg, height=alt * units.m
        )
    if type(time) != Time:
        time = Time(time, scale="utc", location=loc)
    azs = phi * units.rad
    alts = (np.pi / 2 - theta) * units.rad
    altaz = AltAz(alt=alts, az=azs, location=loc, obstime=time)
    icrs = altaz.transform_to(ICRS())
    ra = icrs.ra.rad
    dec = icrs.dec.rad
    return ra, dec


def _hp_rotate(from_coords, to_coords):
    coords = {"galactic": "G", "ecliptic": "E", "equitorial": "C"}
    fc = from_coords.lower()
    tc = to_coords.lower()
    if fc not in coords or tc not in coords:
        raise ValueError(
            f"Invalid coordinate system name, must be in {list[coords.keys()]}"
        )
    rot = Rotator(coord=[coords[fc], coords[tc]])
    return rot


def rotate_map(sky_map, from_coords="galactic", to_coords="equitorial"):
    rot = _hp_rotate(from_coords, to_coords)
    rotated_map = np.empty_like(sky_map)
    for i, m in enumerate(sky_map):  # each frequency
        rm = rot.rotate_map_alms(m)
        rotated_map[i] = rm
    return rotated_map


def rotate_alm(alm, from_coords="galactic", to_coords="equitorial"):
    rot = _hp_rotate(from_coords, to_coords)
    rotated_alm = np.empty_like(alm)
    for i, a in enumerate(alm):  # for each frequency
        ra = rot.rotate_alm(a)
        rotated_alm[i] = ra
    return rotated_alm
