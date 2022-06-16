from astropy.coordinates import AltAz, EarthLocation, ICRS
from astropy import units
from healpy import Alm, npix2nside, Rotator
from lunarsky import LunarTopo, MCMF, MoonLocation, Time
import numpy as np

from .constants import PIX_WEIGHTS_NSIDE


def topo2radec(theta, phi, time, loc, grid=True):
    """
    Convert topocentric coordinates to ra/dec at given time. Useful for
    antenna beams.

    Parameters
    ----------
    phi : array-like
        The azimuth(s) in radians.
    theta : array-like
        The zenith angle(s) in radians.
    time : array-like, str, or Time instance
        The time to compute the transformation at. Must be able to initialize
        an astropy.time.Time object.
    loc : array-like (lat, lon, alt) or EarthLocation instance
        The location of the topocentric coordinates. Must be able to initialize
        an astropy.coordinates.EarthLocation object. 
    grid : bool
        If True then phi and theta are assumed to be coordinate axes of a grid.
        This has no effect if phi and theta have size 1.

    Returns
    -------
    ra : np.ndarray
        The right acension(s) in degrees.
    dec : np.ndarray
        The declination(s) in degrees.

    """
    phi = np.ravel(phi)
    theta = np.ravel(theta)
    if grid:  # phi and theta are coordinate axis
        phi, theta = np.meshgrid(phi, theta)
        phi = phi.ravel()
        theta = theta.ravel()
    # Allow loc to be earth location object and time to be Time object
    if not isinstance(loc, EarthLocation):
        lat, lon, alt = loc
        loc = EarthLocation(
            lat=lat * units.deg, lon=lon * units.deg, height=alt * units.m
        )
    
    time = Time(time, scale="utc", location=loc)
    azs = phi * units.rad
    alts = (np.pi / 2 - theta) * units.rad
    altaz = AltAz(alt=alts, az=azs, location=loc, obstime=time)
    icrs = altaz.transform_to(ICRS())
    ra = icrs.ra.deg
    dec = icrs.dec.deg
    return ra, dec


def radec2topo(ra, dec, time, loc):
    """
    Convert ra/dec-combination(s) to topocentric coordinates.

    Parameters
    ----------
    ra : array-like
        The right ascension(s) in degrees.
    dec : array-like
        The declination(s) in degrees.
    time : array-like, str, or Time instance
        The time to compute the transformation at. Must be able to initialize
        an astropy.time.Time object.
    loc : array-like (lat, lon, alt) or EarthLocation instance
        The location of the topocentric coordinates. Must be able to initialize
        an astropy.coordinates.EarthLocation object.

    Returns
    -------
    theta : np.ndarray
        Colatitudes in radians.
    phi : np.ndarray
        Azimuths in radians.

    """
    if not isinstance(loc, EarthLocation):
        lat, lon, alt = loc
        loc = EarthLocation(
            lat=lat * units.deg, lon=lon * units.deg, height=alt * units.m
        )
    
    time = Time(time, scale="utc", location=loc)

    icrs = ICRS(ra=ra * units.deg, dec=dec * units.deg)
    # transform to altaz
    altaz = icrs.transform_to(AltAz(location=loc, obstime=time))
    theta = np.pi / 2 - altaz.alt.rad
    phi = altaz.az.rad
    return theta, phi

def topo2mcmf(theta, phi, time, loc, grid=True):
    """
    Convert topocentric coordinates to mcmf at given time. Useful for
    antenna beams.

    Parameters
    ----------
    phi : array-like
        The azimuth(s) in radians.
    theta : array-like
        The zenith angle(s) in radians.
    time : array-like, str, or Time instance
        The time to compute the transformation at. Must be able to initialize
        an astropy.time.Time object.
    loc : array-like (lat, lon, alt) or EarthLocation instance
        The location of the topocentric coordinates. Must be able to initialize
        an astropy.coordinates.EarthLocation object. 
    grid : bool
        If True then phi and theta are assumed to be coordinate axes of a grid.
        This has no effect if phi and theta have size 1.

    Returns
    -------
    lon : np.ndarray
        MCMF longtitude in degrees.
    lat : np.ndarray
        MCMF latitude in degrees.

    """
    phi = np.ravel(phi)
    theta = np.ravel(theta)
    if grid:  # phi and theta are coordinate axis
        phi, theta = np.meshgrid(phi, theta)
        phi = phi.ravel()
        theta = theta.ravel()
    # Allow loc to be earth location object and time to be Time object
    if not isinstance(loc, MoonLocation):
        lat, lon, alt = loc
        loc = MoonLocation(
            lat=lat * units.deg, lon=lon * units.deg, height=alt * units.m
        )
    
    time = Time(time, scale="utc", location=loc)
    azs = phi * units.rad
    alts = (np.pi / 2 - theta) * units.rad
    altaz = LunarTopo(alt=alts, az=azs, location=loc, obstime=time)
    mcmf = altaz.transform_to(MCMF())
    lon = mcmf.ra.deg
    lat = mcmf.dec.deg
    return lon, lat

def mcmf2topo(ra, dec, time, loc):
    """
    Convert ra/dec-combination(s) to topocentric coordinates.

    Parameters
    ----------
    ra : array-like
        The right ascension(s) in degrees.
    dec : array-like
        The declination(s) in degrees.
    time : array-like, str, or Time instance
        The time to compute the transformation at. Must be able to initialize
        an astropy.time.Time object.
    loc : array-like (lat, lon, alt) or EarthLocation instance
        The location of the topocentric coordinates. Must be able to initialize
        an astropy.coordinates.EarthLocation object.

    Returns
    -------
    theta : np.ndarray
        Colatitudes in radians.
    phi : np.ndarray
        Azimuths in radians.

    """
    if not isinstance(loc, MoonLocation):
        lat, lon, alt = loc
        loc = MoonLocation(
            lat=lat * units.deg, lon=lon * units.deg, height=alt * units.m
        )
    
    time = Time(time, scale="utc", location=loc)

    mcmf = MCMF(ra=ra * units.deg, dec=dec * units.deg)
    # transform to altaz
    altaz = mcmf.transform_to(LunarTopo(location=loc, obstime=time))
    theta = np.pi / 2 - altaz.alt.rad
    phi = altaz.az.rad
    return theta, phi


def rot_coords(
    axis1, axis2, from_coords, to_coords, time=None, loc=None, lonlat=False
):
    """
    Wrapper for the other coordinate transform functions. Rotates coordinates 
    from one coordinate system to another.
    Supported coordinate system conversions are
        topocentric <-> mcmf (asssumed topocentric at a MoonLocation)
        topocentric <-> equatorial  (topocenric at an EarthLocation)

    Parameters
    ----------
    axis1 : 1d-array
        Colatitudes in radians (if lonlat = False) or longtitudes in degrees
        (if lonlat = True).
    
    axis2 : 1d-array
        Azimuth angles in radians (if lonlat = False) or latitudes in degrees
        (if lonlat = True).
    
    from_coords : str
        Coordinate system to transform from.
    
    to_coords : str
        Coordinate system to transform to.
    
    time : str or astropy.time.Time instance or lunarsky.time.Time instance
        The time of the coordinate transform. Must be able to initialize a Time
        object. Required if transforming to or from topocentric coordinates.
    
    loc: array-like (lat, lon, alt) or astropy.coordinates.EarthLocation
         instance or lunarsky.moon.MoonLocation instance
        The location of the coordinate transform. Required if transforming
        to or from topocentric coordinates. If array-like it must have the
        form (latitude, longtitude, altitude).
    
    lonlat : bool
        If True, input and out are longtitudes and latitudes in degrees.
        Otherwise, they are colatitudes (polar angle) and azimuths in radians.

    Returns
    -------
    rot_axis1 : np.1darray
        Colatitudes in radians in new coordinate system (if lonlat = False) or
        longtitudes in degrees (in lonlat = True).
    
    rot_axis2: np.1darray
        Azimuths in radians in new coordinate system (if lonlat = False) or 
        latitudes in degrees (if lonlat = True).
        
    """
    from_coords = from_coords.lower()
    to_coords = to_coords.lower()

    if from_coords == to_coords:
        return axis1, axis2

    elif from_coords == "topocentric":
        if to_coords == "equatorial":
            func = topo2radec
        elif to_coords == "mcmf":
            func = topo2mcmf
        else:
            raise ValueError(
                "Coordinate transform must be topocentric <-> equatorial or "
                f"topocentric <-> mcmf, not {from_coords} to {to_coords}."
            )

        if lonlat:
            phi = np.deg2rad(axis1)
            theta = np.pi / 2 - np.deg2rad(axis2)
        else:
            theta = axis1
            phi = axis2
        ra, dec = func(theta, phi, time, loc, grid=False)
        if lonlat:
            return ra, dec
        else:
            rot_axis1 = np.pi/2 - np.deg2rad(dec)
            rot_axis2 = np.deg2rad(ra)
            return rot_axis1, rot_axis2

    elif to_coords == "topocentric":
        if from_coords == "equatorial":
            func = radec2topo
        elif from_coords == "mcmf":
            func = mcmf2topo
        else:
            raise ValueError(
                "Coordinate transform must be topocentric <-> equatorial or "
                f"topocentric <-> mcmf, not {from_coords} to {to_coords}."
            )

        if lonlat:
            ra = axis1
            dec = axis2
        else:
            dec = 90 - np.rad2deg(axis1)
            ra = np.rad2deg(axis2)
        theta, phi = func(ra, dec, time, loc)
        if lonlat:
            rot_axis1 = np.rad2deg(phi)
            rot_axis2 = 90 - np.rad2deg(theta)
            return rot_axis1, rot_axis2
        else:
            return theta, phi



def hp_rotate(from_coords, to_coords):
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
    rot = hp_rotate(from_coords, to_coords)
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


#XXX ADD MCMF support
def rotate_alm(alm, from_coords="galactic", to_coords="equatorial"):
    rot = hp_rotate(from_coords, to_coords)
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


def rot_alm_z(phi, lmax):
    """
    Get the coefficients that rotate alms around the z-axis by phi
    (measured counterclockwise).

    Parameters
    ----------
    phi : array-like
        The angle(s) to rotate the azimuth by in radians.
    lmax : int
        The maximum ell of the alm.

    Returns
    -------
     phase : np.ndarray
        The coefficients that rotate the alms by phi. Will have shape
        (alm.size) if phi is a scalar or (phi.size, alm.size) if phi
        is an array.

    """

    phi = np.reshape(phi, (-1, 1))
    emms = Alm.getlm(lmax)[1].reshape(1, -1)
    phase = np.exp(1j * emms * phi)
    return np.squeeze(phase)
