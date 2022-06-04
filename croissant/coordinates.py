from astropy.coordinates import AltAz, EarthLocation, ICRS
from astropy.time import Time
from astropy import units
from healpy import nside2npix, pix2ang, Rotator
import numpy as np


def topo2radec(phi, theta, time, loc, grid=True):
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


def radec2topo(ra, dec, time, loc):
    """
    Convert the ra/dec-combination(s) to topocentric coordinates.

    Parameters
    ----------
    ra : array-like
        The right ascension(s) in degrees.
    dec : array-like
        The declination(s) in degrees.
    time : array-like, str, or astropy.time.Time object
        The time to compute the transformation at. Must be able to initialize
        an astropy.time.Time object.
    loc : array-like or astropy.coordinates.EarthLocation object
        The location of the topocentric coordinates. Must be able to initialize
        an astropy.coordinates.EarthLocation object.

    Returns
    -------
    theta : np.ndarray
        Colatitudes in radians.
    phi : np.ndarray
        Azimuths in radians.

    """
    if type(loc) != EarthLocation:
        lat, lon, alt = loc
        loc = EarthLocation(
            lat=lat*units.deg, lon=lon*units.deg, height=alt*units.m
        )
    if type(time) != Time:
        time = Time(time, scale="utc", location=loc)

    icrs = ICRS(ra=ra*units.deg, dec=dec*units.deg)
    # transform to altaz
    altaz = icrs.transform_to(AltAz(location=loc, obstime=time))
    theta = np.pi/2 - altaz.alt.rad
    phi = altaz.az.rad
    return theta, phi

def healpix2lonlat(nside, pix=None)
    """
    Compute the longtitudes and latitudes of the pixel centers of a healpix
    map.

    Parameters
    ----------
    nside : int
        The nside of the healpix map.
    pix : array-like (optional)
        Which pixels to get the longtitudes and latitudes of. Defaults to all
        pixels.

    Returns
    -------
    lon : scalar or np.ndarray
        The longtitude(s) in degrees.
    lat : scalar or np.ndarray
        The latitude(s) in degrees.

    """
    if pix is None:
        pix = np.arange(nside2npix(nside))
    lon, lat = pix2ang(nside, pix, nest=False, lonlat=True)
    return lon, lat

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
