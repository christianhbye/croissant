from astropy.coordinates import AltAz
from lunarsky import LunarTopo
from s2fft import rotate_flms
import jax

from ..utils import get_rot_mat, rotmat_to_euler
from .alm import lmax_from_shape


@jax.jit
def rotate_alm(alm, from_frame, to_frame, dl_array=None):
    """
    Transform a spherical harmonic decomposition from one coordinate system to
    another. This is a wrapper around the s2fft.rotate_flms function that
    computes the Euler angles from the input and output coordinate systems.

    Parameters
    ----------
    alm : jnp.ndarray
        The alm array to transform.
    from_frame : str or astropy frame
        The coordinate system of the input alm.
    to_frame : str or astropy frame
        The coordinate system of the output alm.
    dl_array : jnp.ndarray
        Precomputed array of reduced Wigner d-function values. These
        can be computed with the s2fft.generate_rotate_dls function.

    Returns
    -------
    alm_rot : jnp.ndarray
        The alm array in the ``to_frame'' coordinate system.

    """
    rmat = get_rot_mat(from_frame, to_frame)
    euler = rotmat_to_euler(rmat)
    lmax = lmax_from_shape(alm.shape)
    alm_rot = rotate_flms(alm, lmax + 1, euler, dl_array=dl_array)
    return alm_rot


@jax.jit
def gal2eq(alm, dl_array=None):
    """
    Transform a spherical harmonic decomposition from Galactic to Equatorial
    coordinates.

    Parameters
    ----------
    alm : jnp.ndarray
        The alm array to transform.
    dl_array : jnp.ndarray
        Precomputed array of reduced Wigner d-function values. These
        can be computed with the s2fft.generate_rotate_dls function.

    Returns
    -------
    alm_rot : jnp.ndarray
        The alm array in Equatorial coordinates.

    """
    return rotate_alm(alm, "galactic", "fk5", dl_array=dl_array)


@jax.jit
def gal2mcmf(alm, dl_array=None):
    """
    Transform a spherical harmonic decomposition from Galactic to MCMF
    coordinates (moon equivalent of equatorial coordinates).

    Parameters
    ----------
    alm : jnp.ndarray
        The alm array to transform.
    dl_array : jnp.ndarray
        Precomputed array of reduced Wigner d-function values. These
        can be computed with the s2fft.generate_rotate_dls function.

    Returns
    -------
    alm_rot : jnp.ndarray
        The alm array in MCMF coordinates.

    """
    return rotate_alm(alm, "galactic", "mcmf", dl_array=dl_array)


@jax.jit
def topo2eq(alm, loc, time, dl_array=None):
    """
    Transform a spherical harmonic decomposition from topocentric on Earth to
    equatorial coordinates.

    Parameters
    ----------
    alm : jnp.ndarray
        The alm array to transform.
    loc : astropy.coordinates.EarthLocation
        The location of the observer.
    time : astropy.time.Time
        The time of the observation.
    dl_array : jnp.ndarray
        Precomputed array of reduced Wigner d-function values. These
        can be computed with the s2fft.generate_rotate_dls function.

    Returns
    -------
    alm_rot : jnp.ndarray
        The alm array in Equatorial coordinates.

    """
    topo = AltAz(location=loc, obstime=time)
    return rotate_alm(alm, topo, "fk5", loc, time, dl_array=dl_array)


@jax.jit
def topo2mcmf(alm, loc, time, dl_array=None):
    """
    Transform a spherical harmonic decomposition from topocentric on Moon to
    equatorial coordinates.

    Parameters
    ----------
    alm : jnp.ndarray
        The alm array to transform.
    loc : lunarsky.MoonLocation
        The location of the observer.
    time : lunarsky.Time
        The time of the observation.
    dl_array : jnp.ndarray
        Precomputed array of reduced Wigner d-function values. These
        can be computed with the s2fft.generate_rotate_dls function.

    Returns
    -------
    alm_rot : jnp.ndarray
        The alm array in MCMF coordinates.

    """
    topo = LunarTopo(location=loc, obstime=time)
    return rotate_alm(alm, topo, "mcmf", loc, time, dl_array=dl_array)
