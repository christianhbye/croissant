from functools import partial

import jax
import jax.numpy as jnp

from .constants import Y00


@jax.jit
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


@partial(jax.jit, static_argnums=(1,))
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


@jax.jit
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
