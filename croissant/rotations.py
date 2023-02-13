from astropy.coordinates import AltAz, EarthLocation, ICRS
from astropy import units
import healpy as hp
from lunarsky import LunarTopo, MCMF, MoonLocation, Time
import numpy as np
from spiceypy import pxform

from .constants import PIX_WEIGHTS_NSIDE


#XXX
# here we just provide some rotation matrices / euler angles and provide
# a thin wrapper for healpy.Rotator

class Rotator(hp.Rotator):

    def __init__



def get_euler(from_coords="galactic", to_coords="mcmf", time=None):
    """
    Compute the (ZYX) Euler angles needed to describe a rotation between MCMF,
    galactic and equatorial coordinates.

    Parameters
    ----------
    time : str or astropy.timing.Time instance
        The time of the coordinate transform.

    from_coords : str (optional)
        Coordinate system to convert from.

    to_coords : str (optional)
        Coordinate system to convert to.

    Returns
    -------
    euler_angles : tup
        The ZYX Euler angles in radians that describe the coordinate
        transformation.

    """
    fc = from_coords.lower()
    tc = to_coords.lower()
    coords = {"galactic": "GALACTIC", "equatorial": "J2000", "mcmf": "MOON_ME"}
    if fc not in coords or tc not in coords:
        raise ValueError(
            f"Invalid coordinate system name, must be in {list(coords.keys())}"
        )
    if time is None:
        et = 0
    else:  # get epoch time
        et = Time(time) - Time("J2000", scale="tt")
        et = et.sec
    # rotation matrix
    rot_mat = pxform(coords[fc], coords[tc], et)
    # get euler angles
    beta = -np.arcsin(rot_mat[0, 2])
    alpha = np.arctan2(
        rot_mat[1, 2] / np.cos(beta), rot_mat[2, 2] / np.cos(beta)
    )
    gamma = np.arctan2(
        rot_mat[0, 1] / np.cos(beta), rot_mat[0, 0] / np.cos(beta)
    )
    euler_angles = (gamma, -beta, alpha)
    return euler_angles


def hp_rotate(from_coords, to_coords, time=None):
    """
    Parameters
    ----------
    time : str or astropy.timing.Time instance
        The time of the coordinate transform.
    """
    hp_coords = {"galactic": "G", "equatorial": "C"}
    fc = from_coords.lower()
    tc = to_coords.lower()
    if "mcmf" in [fc, tc]:
        euler_angles = get_euler(from_coords=fc, to_coords=tc, time=time)
        rot = hp.Rotator(rot=euler_angles, deg=False, eulertype="ZYX")
    elif fc in hp_coords and tc in hp_coords:
        rot = hp.Rotator(coord=[hp_coords[fc], hp_coords[tc]])
    else:
        raise ValueError(
            f"Invalid coordinate system, must be in {list(hp_coords.keys())}"
        )
    return rot


def rotate_map(hp_map, from_coords="galactic", to_coords="mcmf"):
    rot = hp_rotate(from_coords, to_coords)
    hp_map = np.array(hp_map, copy=True, dtype=np.float64)
    npix = hp_map.shape[-1]
    nside = hp.npix2nside(npix)
    use_pix_weights = nside in PIX_WEIGHTS_NSIDE
    if hp_map.ndim == 1:
        rotated_map = rot.rotate_map_alms(
            hp_map, use_pixel_weights=use_pix_weights
        )
    elif hp_map.ndim == 2:
        rotated_map = np.empty_like(hp_map)
        for i, m in enumerate(hp_map):  # iterate over frequency axis
            rm = rotate_map(m, from_coords=from_coords, to_coords=to_coords)
            rotated_map[i] = rm
    else:
        raise ValueError("hp_map must be a 1d map or a (2d) list of maps.")
    return rotated_map


def rotate_alm(alm, from_coords="galactic", to_coords="mcmf"):
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
    emms = hp.Alm.getlm(lmax)[1].reshape(1, -1)
    phase = np.exp(1j * emms * phi)
    return np.squeeze(phase)
