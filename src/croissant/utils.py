import warnings

import jax.numpy as jnp
import numpy as np
import s2fft
from astropy import units
from lunarsky import Time

from .constants import Y00


def valid_nside(nside):
    """
    Check if the nside of a HEALPix map is valid.

    Parameters
    ----------
    nside : int
        The nside of the map.

    Returns
    -------
    valid : bool
        True if nside is valid, False otherwise. This is true if nside
        is a positive integer and a power of 2.

    """
    valid = nside > 0 and (nside & (nside - 1)) == 0
    return valid


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

    Raises
    ------
    ValueError
        If npix is not a positive integer and a multiple of 12. Note
        that this only checks that the nside can be computed, but does
        guarantee that nside is valid. Use the function `valid_nside`
        to check if nside is valid.

    """
    if npix <= 0 or npix % 12 != 0:
        raise ValueError(
            "npix must be a positive integer and a multiple of 12. Got "
            f"npix={npix}."
        )
    nside = int(np.sqrt(npix / 12))
    return nside


def hp_valid_npix(npix):
    """
    Check if the number of pixels in a HEALPix map is valid.

    Parameters
    ----------
    npix : int
        The number of pixels in the map.

    Returns
    -------
    valid : bool
        True if npix is valid, False otherwise. This is true if npix is
        a positive integer and can be expressed as npix = 12 * nside^2,
        where nside is a power of 2. See the functions `valid_nside`
        and `hp_npix2nside`.

    """
    try:
        nside = hp_npix2nside(npix)
    except ValueError:
        return False
    return valid_nside(nside)


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

    Raises
    ------
    ValueError
        If the required parameters are not provided for the given sampling
        scheme. If sampling is "healpix", nside must be provided. If
        sampling is not "healpix", lmax must be provided.

    """
    if sampling != "healpix":
        if lmax is None:
            raise ValueError(
                "lmax must be specified if sampling is not 'healpix'. Got "
                f"lmax={lmax} and sampling={sampling}."
            )
        L = lmax + 1
        phi = s2fft.sampling.s2_samples.phis_equiang(L, sampling=sampling)
    else:
        if nside is None:
            raise ValueError(
                "nside must be specified if sampling is 'healpix'. Got "
                f"nside={nside} and sampling={sampling}."
            )
        ntheta = s2fft.sampling.s2_samples.ntheta(
            sampling=sampling, nside=nside
        )
        ts = np.arange(ntheta)
        phi = np.concatenate(
            [s2fft.sampling.s2_samples.phis_ring(t, nside) for t in ts],
            axis=0,
        )

    return phi


def generate_theta(lmax=None, sampling="mw", nside=None):
    """
    Generate an array of theta values for a given lmax and sampling
    scheme. This is a convenience wrapper around ``thetas'' from
    s2fft.sampling.s2_samples to match the interface of
    ``generate_phi''.

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
    theta : np.ndarray
        An array of theta values for the given lmax and sampling scheme.

    Raises
    ------
    ValueError
        If the required parameters are not provided for the given sampling
        scheme. If sampling is "healpix", nside must be provided. If
        sampling is not "healpix", lmax must be provided.

    """
    if sampling == "healpix":
        L = None  # does not matter
        if nside is None:
            raise ValueError(
                "nside must be specified if sampling is 'healpix'. Got "
                f"nside={nside} and sampling={sampling}."
            )
    elif lmax is None:
        raise ValueError(
            "lmax must be specified if sampling is not 'healpix'. Got "
            f"lmax={lmax} and sampling={sampling}."
        )
    else:
        L = lmax + 1

    theta = s2fft.sampling.s2_samples.thetas(
        L=L, sampling=sampling, nside=nside
    )

    # if sampling is healpix we get one theta per ring only
    if sampling == "healpix":
        ts = np.arange(len(theta))
        nring = np.array(
            [s2fft.sampling.s2_samples.nphi_ring(t, nside) for t in ts],
        )
        theta = np.repeat(theta, nring)

    return theta


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
    visibilities. Only the monopole component will integrate to
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


def __getattr__(name):
    """
    Redirect attribute access to the rotations module for deprecated
    functions.
    """
    if name in {
        "get_rot_mat",
        "rotmat_to_euler",
        "rotmat_to_eulerZYX",
        "rotmat_to_eulerZYZ",
    }:
        warnings.warn(
            f"utils.{name} has been moved to the rotations module "
            "and will be removed from the utils module in a future release.",
            FutureWarning,
            stacklevel=2,
        )
        from . import rotations

        return getattr(rotations, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
