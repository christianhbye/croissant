import warnings

import jax.numpy as jnp
import numpy as np
import s2fft
from astropy import units
from lunarsky import Time

from . import rotations
from .constants import Y00


def _future_warning(func):
    """
    Decorator to add a FutureWarning to a function.

    Parameters
    ----------
    func : callable
        The function to add the FutureWarning to.

    Returns
    -------
    wrapper : callable
        The wrapped function that raises a FutureWarning when called.

    """

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is has been moved to the rotations module "
            "and will be removed from the utils module in a future release.",
            FutureWarning,
        )
        return func(*args, **kwargs)

    return wrapper


@_future_warning
def get_rot_mat(from_frame, to_frame):
    """
    Get the rotation matrix that transforms from one frame to another.

    Parameters
    ----------
    from_frame : str or astropy frame
        The coordinate frame to transform from.
    to_frame : str or astropy frame
        The coordinate frame to transform to.

    Returns
    -------
    rmat : np.ndarray
        The rotation matrix.

    """
    return rotations.get_rot_mat(from_frame, to_frame)


@_future_warning
def rotmat_to_euler(mat, eulertype="ZYX"):
    """
    Convert a rotation matrix to Euler angles in the specified convention.

    Parameters
    ----------
    mat : np.ndarray
        The rotation matrix.
    eulertype : str, either ``ZYX'' or ``ZYZ''.
        The Euler angle convention to use.

    Returns
    -------
    eul : tup
        The Euler angles in the specified convention.

    Notes
    -----
    ``ZYX'' is the default healpy convention, what you would make ``rot''
    when you call healpy.Rotator(rot, euletype="ZYX"). Wikipedia refers
    to this as Tait-Bryan angles X1-Y2-Z3.

    ``ZYZ'' is the convention typically used for Wigner D matrices, which
    s2fft uses. Wkipidia calls it Euler angles Z1-Y2-Z3. This would be
    used in s2fft.utils.rotation.rotate_flms.


    """
    return rotations.rotmat_to_euler(mat, eulertype=eulertype)


@_future_warning
def rotmat_to_eulerZYX(mat):
    """
    Convert a rotation matrix to Euler angles in the ZYX convention. This is
    sometimes referred to as Tait-Bryan angles X1-Y2-Z3.

    Parameters
    ----------
    mat : np.ndarray
        The rotation matrix.

    Returns
    --------
    eul : tup
        The Euler angles in the order yaw, -pitch, roll. This is the input
        healpy.rotator.Rotator expects when ``eulertype'' is ZYX.

    """
    return rotations.rotmat_to_eulerZYX(mat)


@_future_warning
def rotmat_to_eulerZYZ(mat):
    """
    Convert a rotation matrix to Euler angles in the ZYZ convention. This is
    sometimes referred to as Euler angles Z1-Y2-Z3.

    Parameters
    ----------
    mat : np.ndarray
        The rotation matrix.

    Returns
    --------
    eul : tup
        The Euler angles in the order alpha, beta, gamma. This is the input
        s2fft.utils.rotation.rotate_flms expects.

    """
    return rotations.rotmat_to_eulerZYZ(mat)


def hp_npix2nside(npix):
    """
    Calculate the nside of a HEALPix map from the number of pixels.

    Parameters
    ----------
    npix : int
        The number of pixels in the map.

    Returns
    -------
    nside : int
        The nside of the map.

    """
    nside = int(np.sqrt(npix / 12))
    return nside


def time_array(t_start=None, t_end=None, N_times=None, delta_t=None):
    """
    Generate an array of evenly sampled times to run the simulation at.

    Parameters
    ----------
    t_start : str or astropy.time.Time or lunarsky.Time
        The start time of the simulation.
    t_end : str or astropy.time.Time or lunarsky.Time
        The end time of the simulation.
    N_times : int
        The number of times to run the simulation at.
    delta_t : float or astropy.units.Quantity
        The time step between each time in the simulation.

    Returns
    -------
    times : astropy.time.Time or lunarsky.Time or astropy.units.Quantity
        The evenly sampled times to run the simulation at.

    """

    if t_start is not None:
        t_start = Time(t_start, scale="utc")

    try:
        dt = np.arange(N_times) * delta_t
    except TypeError:
        t_end = Time(t_end, scale="utc")
        total_time = (t_end - t_start).sec
        if N_times is None:
            try:
                delta_t = delta_t.to_value("s")
            except AttributeError:
                warnings.warn(
                    "delta_t is not an astropy.units.Quantity. Assuming "
                    "units of seconds.",
                    UserWarning,
                )
            dt = np.arange(0, total_time + delta_t, delta_t)
        else:
            dt = np.linspace(0, total_time, N_times)
        dt = dt * units.s

    if t_start is None:
        times = dt
    else:
        times = t_start + dt

    return times


def generate_phi(lmax=None, sampling="mw", nside=None):
    """
    Generate an array of phi values for a given lmax and sampling scheme.

    Parameters
    ----------
    lmax : int
        The maximum spherical harmonic degree to support. Required if
        `sampling`` is not "healpix".
    sampling : str
        The type of sampling. Supported schemes are from s2fft and
        include {"mw", "mwss", "dh", "gl", "healpix"}.
    nside : int
        The nside of the HEALPix map. Only required if sampling is
        "healpix".

    Returns
    -------
    phi : np.ndarray
        An array of phi values for the given lmax and sampling scheme.

    """
    L = lmax + 1
    if sampling != "healpix":
        if lmax is None:
            raise ValueError(
                "lmax must be provided if sampling is not healpix."
            )
        phi = s2fft.sampling.s2_samples.phis_equiang(L, sampling=sampling)
    else:
        ntheta = s2fft.sampling.s2_samples.ntheta(
            L=L, sampling=sampling, nside=nside
        )
        ts = np.arange(ntheta).astype(np.float64)
        phi = np.concatenate(
            [s2fft.sampling.s2_samples.phis_ring(t, nside) for t in ts],
            axis=0,
        )
    return phi


def getidx(lmax, ell, emm):
    """
    Get the index of the alm array for a given l and m.

    Parameters
    ----------
    lmax : int
        The maximum l value.
    ell : int or jnp.ndarray
        The value of l.
    emm : int or jnp.ndarray
        The value of m.

    Returns
    -------
    l_ix : int or jnp.ndarray
       The l index (which is the same as the input ell).
    m_ix : int or jnp.ndarray
        The m index.

    """
    return ell, emm + lmax


def total_power(alm, lmax):
    """
    Compute the integral of a signal (such as an antenna beam) given
    the spherical harmonic coefficients. This is needed to normalize the
    visibilities. Only the monoopole component will integrate to
    a non-zero value.

    Parameters
    ----------
    alm : jnp.ndarray
        The spherical harmonic coefficients. The last two dimensions must
        correspond to the ell and emm indices respectively.
    lmax : int
        The maximum l value.

    Returns
    -------
    power : float
        The total power of the signal.

    """
    # get the index of the monopole component
    lix, mix = getidx(lmax, 0, 0)
    monopole = alm[..., lix, mix]
    return jnp.real(monopole) / Y00


def getlm(lmax, ix):
    """
    Get the l and m corresponding to the index of the alm array.

    Parameters
    ----------
    lmax : int
        The maximum l value.

    ix : jnp.ndarray
        The indices of the alm array. The first row corresponds to the l
        index, and the second row corresponds to the m index. Multiple
        indices can be passed in as an array with shape (2, n).

    Returns
    -------
    ell : jnp.ndarray
        The value of l. Has shape (n,).
    emm : jnp.ndarray
        The value of m. Has shape (n,).

    """
    ell = ix[0]
    emm = ix[1] - lmax
    return ell, emm


def lmax_from_shape(shape):
    """
    Get the lmax from the shape of the alm array.

    Parameters
    ----------
    shape : tuple
        The shape of the alm array. The last two dimensions must correspond
        to the ell and emm indices respectively.

    Returns
    -------
    lmax : int
        The maximum l value.

    """
    return shape[-2] - 1


def is_real(alm):
    """
    Check if the an array of alms correspond to a real-valued
    signal. Mathematically, this is true if the coefficients satisfy
    alm(l, m) = (-1)^m * conj(alm(l, -m)).

    Parameters
    ----------
    alm : jnp.ndarray
        The spherical harmonics coefficients. The last two dimensions
        must correspond to the ell and emm indices respectively.

    Returns
    -------
    is_real : bool
        True if the coefficients correspond to a real-valued signal.

    """
    lmax = lmax_from_shape(alm.shape)
    emm = jnp.arange(1, lmax + 1)  # positive ms
    # reshape emm to broadcast with alm by adding 1 or 2 dimensions
    emm = emm.reshape((1,) * (alm.ndim - 1) + emm.shape)
    # get alms for negative m, in reverse order (i.e., increasing abs(m))
    neg_m = alm[..., :lmax][..., ::-1]
    # get alms for positive m
    pos_m = alm[..., lmax + 1 :]
    return jnp.allclose(neg_m, (-1) ** emm * jnp.conj(pos_m)).item()


def reduce_lmax(alm, new_lmax):
    """
    Reduce the maximum l value of the alm.

    Parameters
    ----------
    alm : jnp.ndarray
        The alm array. Last two dimensions must correspond to the ell and
        emm indices.
    new_lmax : int
        The new maximum l value. Must be less than the lmax of alm.

    Returns
    -------
    new_alm : jnp.ndarray
        The alm array with the new maximum l value.

    Raises
    ------
    ValueError
        If new_lmax is greater than the lmax of alm.


    """
    lmax = lmax_from_shape(alm.shape)
    d = lmax - new_lmax  # number of ell values to remove
    if d < 0:
        raise ValueError(
            "new_lmax must be less than the lmax of alm. Got "
            f"new_lmax={new_lmax} and lmax={lmax}."
        )
    elif d == 0:
        return alm
    return alm[..., :-d, d:-d]


def shape_from_lmax(lmax):
    """
    Get the shape of the alm array given the maximum l value.

    Parameters
    ----------
    lmax : int
        The maximum l value.

    Returns
    -------
    shape : tup

    """
    return (lmax + 1, 2 * lmax + 1)


def lmax_from_ntheta(ntheta, sampling):
    """
    Get the lmax corresponding to a given number of theta samples and
    sampling scheme.

    Parameters
    ----------
    ntheta : int
        The number of theta samples or pixels if sampling is "healpix".
    sampling : str
        The sampling scheme. Supported schemes are from s2fft and include
        {"mw", "mwss", "dh", "gl", "healpix"}.

    Returns
    -------
    lmax : int
        The maximum l value corresponding to the given number of theta
        samples and sampling scheme.

    Raises
    ------
    ValueError
        If the sampling scheme is not supported.

    """
    if sampling in {"mw", "gl"}:
        lmax = ntheta - 1
    elif sampling == "mwss":
        lmax = ntheta - 2
    elif sampling == "dh":
        lmax = ntheta // 2 - 1
    elif sampling == "healpix":
        nside = hp_npix2nside(ntheta)
        lmax = 2 * nside
    else:
        raise ValueError(
            f"Unsupported sampling scheme {sampling}. Supported schemes are "
            "from s2fft and include {'mw', 'mwss', 'dh', 'gl', 'healpix'}."
        )
    return lmax
