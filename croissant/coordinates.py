from astropy.coordinates import AltAz, EarthLocation, ICRS
from astropy.time import Time
from astropy import units
from healpy import npix2nside, Rotator
import numpy as np

from .constants import PIX_WEIGHTS_NSIDE


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
            lat=lat * units.deg, lon=lon * units.deg, height=alt * units.m
        )
    if type(time) != Time:
        time = Time(time, scale="utc", location=loc)

    icrs = ICRS(ra=ra * units.deg, dec=dec * units.deg)
    # transform to altaz
    altaz = icrs.transform_to(AltAz(location=loc, obstime=time))
    theta = np.pi / 2 - altaz.alt.rad
    phi = altaz.az.rad
    return theta, phi


def _hp_rotate(from_coords, to_coords):
    coords = {"galactic": "G", "ecliptic": "E", "equatorial": "C"}
    fc = from_coords.lower()
    tc = to_coords.lower()
    if fc not in coords or tc not in coords:
        raise ValueError(
            f"Invalid coordinate system name, must be in {list(coords.keys())}"
        )
    rot = Rotator(coord=[coords[fc], coords[tc]])
    return rot


def rotate_map(sky_map, from_coords="galactic", to_coords="equitorial"):
    rot = _hp_rotate(from_coords, to_coords)
    sky_map = np.array(sky_map, copy=True, dtype=np.float64)
    npix = sky_map.shape[-1]
    nside = npix2nside(npix)
    use_pix_weights = nside in PIX_WEIGHTS_NSIDE
    if sky_map.ndim == 1:
        rotated_map = rot.rotate_map_alms(
            sky_map, use_pixel_weights=use_pix_weights
        )
    elif sky_map.ndim == 2:
        rotated_map = np.empty_like(sky_map)
        for i, m in enumerate(sky_map):  # each frequency
            rm = rotate_map(m, from_coords=from_coords, to_coords=to_coords)
            rotated_map[i] = rm
    else:
        raise ValueError("sky_map must be a 1d map or a (2d) list of maps.")
    return rotated_map


def rotate_alm(alm, from_coords="galactic", to_coords="equitorial"):
    rot = _hp_rotate(from_coords, to_coords)
    alm = np.array(alm, copy=True, dtype=np.complex128)
    if alm.ndim == 1:
        rotated_alm = rot.rotate_alm(alm)
    elif alm.ndim == 2:
        rotated_alm = np.empty_like(alm)
        for i, a in enumerate(alm):  # for each frequency
            rotated_alm[i] = rotate_alm(
                a, from_coords=from_coords, to_coords=to_coords
            )
    else:
        raise ValueError(f"alm must have 1 or 2 dimensions, not {alm.ndim}.")
    return rotated_alm
