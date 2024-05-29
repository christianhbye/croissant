import jax.numpy as jnp
import s2fft
from ..constants import Y00


def alm2map(alm, spin=0, nside=None, sampling="mw", precomps=None, spmd=True):
    """
    Construct a map on the sphere from the alm array. This is a wrapper
    around s2fft.inverse provided for convenience.

    Parameters
    ----------
    alm : jnp.ndarray
        The alm array. Must have shape (lmax+1, 2*lmax+1). Use
        jax.vmap to vectorize over multiple alms.
    spin : int
        Harmonic spin of the map. Must be 0 or 1.
    nside : int
        The nside of the healpix map to construct. Required if sampling
        is "healpix".
    sampling : str
        Sampling scheme on the sphere. Must be in
        {"mw", "mwss", "dh", "healpix"}. Passed to s2fft.inverse.
    precomps : list
        Precomputed values for the s2fft.inverse function. Passed to
        s2fft.inverse.
    spmd : bool
        Map the computation over all available devices. Passed to
        s2fft.inverse.

    Returns
    -------
    m : jnp.ndarray
        The map(s) corresponding to the alm.

    """
    L = lmax_from_shape(alm.shape) + 1
    m = s2fft.inverse_jax(
        alm,
        L,
        spin=spin,
        nside=nside,
        sampling=sampling,
        reality=is_real(alm),
        spmd=spmd,
        L_lower=0,
    )
    return m


def map2alm(
    m,
    lmax,
    spin=0,
    nside=None,
    sampling="mw",
    reality=True,
    precomps=None,
    spmd=True,
):
    """
    Construct the alm array from a map on the sphere. This is a wrapper
    around s2fft.forward provided for convenience.

    Parameters
    ----------
    m : jnp.ndarray
        The map on the sphere. Use jax.vmap to vectorize over multiple
        maps.
    lmax : int
        The maximum l value. Note that s2fft uses L which is lmax+1.
    spin : int
        Harmonic spin of the map. Must be 0 or 1.
    nside : int
        The nside of the healpix map. Required if sampling is "healpix".
    sampling : str
        Sampling scheme on the sphere. Must be in
        {"mw", "mwss", "dh", "gl", "healpix"}. Passed to s2fft.forward.
    reality : bool
        True if the map is real-valued. Passed to s2fft.forward.
    precomps : list
        Precomputed values for the s2fft.forward function. Passed to
        s2fft.forward.
    spmd : bool
        Map the computation over all available devices. Passed to
        s2fft.forward.

    Returns
    -------
    alm : jnp.ndarray
        The alm array corresponding to the map.

    """
    L = lmax + 1
    alm = s2fft.forward_jax(
        m,
        L,
        spin=spin,
        nside=nside,
        sampling=sampling,
        reality=reality,
        spmd=spmd,
        L_lower=0,
    )
    return alm


def total_power(alm):
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

    Returns
    -------
    power : float
        The total power of the signal.

    """
    lmax = lmax_from_shape(alm.shape)
    # get the index of the monopole component
    lix, mix = getidx(lmax, 0, 0)
    monopole = alm[..., lix, mix]
    return 4 * jnp.pi * jnp.real(monopole) * Y00


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

    Raises
    ------
    IndexError
        If l,m don't satisfy abs(m) <= l <= lmax.
    """
    return ell, emm + lmax


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
    return jnp.all(neg_m == (-1) ** emm * jnp.conj(pos_m)).item()


def reduce_lmax(alm, new_lmax):
    """
    Reduce the maximum l value of the alm.

    Parameters
    ----------
    alm : jnp.ndarray
        The alm array. Last two dimensions must correspond to the ell and
        emm indices.
    new_lmax : int
        The new maximum l value. Must be less than or equal to alm lmax.

    Returns
    -------
    new_alm : jnp.ndarray
        The alm array with the new maximum l value.

    Raises
    ------
    ValueError
        If new_lmax is greater than the current lmax.

    """
    lmax = lmax_from_shape(alm.shape)
    d = lmax - new_lmax  # number of ell values to remove
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
