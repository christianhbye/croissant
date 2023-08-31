import numpy as np
import jax
import healpy as hp
from .constants import PIX_WEIGHTS_NSIDE


def alm2map(alm, nside, lmax=None, method="numpy"):
    """
    Compute the healpix map from the spherical harmonics coefficients.

    Parameters
    ----------
    alm : array-like
        The spherical harmonics coefficients in the healpy convention. Shape
        ([nfreq], hp.Alm.getsize(lmax)).
        Note if method="jax": must be a jnp array with 2 dimensions, i.e.,
        must have a frequency axis even if nfreq=1.
    nside : int
        The nside of the output map(s).
    lmax : int
        The lmax of the spherical harmonics transform. Defaults to 3*nside-1.
    method : "numpy" or "jax"

    Returns
    -------
    map : np.ndarray or jnp.ndarray
        The healpix map. Shape ([nfreq], hp.nside2npix(nside)).

    Raises
    ------
    ValueError :
        If method is not "numpy" or "jax".
    """
    if method == "numpy":
        return alm2map_numpy(alm, nside, lmax=lmax)
    elif method == "jax":
        return alm2map_jax(alm, nside, lmax=lmax)
    else:
        raise ValueError("method must be ``numpy'' or ``jax''.")


def alm2map_numpy(alm, nside, lmax=None):
    alm = np.array(alm, copy=True)
    if alm.ndim == 1:
        return hp.alm2map(alm, nside, lmax=lmax)
    else:
        npix = hp.nside2npix(nside)
        nfreqs = alm.shape[0]
        hp_map = np.empty((nfreqs, npix))
        for i in range(nfreqs):
            map_i = hp.alm2map(alm[i], nside, lmax=lmax)
            hp_map[i] = map_i
        return hp_map


@jax.jit
def alm2map_jax(alm, nside, lmax=None):
    return jax.vmap(
        hp.alm2map,
        in_axes=(
            0,
            None,
        ),
    )(alm, nside, lmax=lmax)


def map2alm(data, lmax=None, method="numpy"):
    """
    Compute the spherical harmonics coefficents of a healpix map.

    Parameters
    ----------
    data : np.ndarray or jnp.ndarray
        The healpix map(s). Shape ([nfreq], hp.nside2npix(nside)). Note if
        method="jax": must be a jnp array with 2 dimensions, i.e., must have a
        frequency axis even if nfreq=1.
    lmax : int
        The lmax of the spherical harmonics transform. Defaults to 3*nside-1.
    method : "numpy" or "jax"

    Returns
    -------
    alm : np.ndarray or jnp.ndarray
        The spherical harmonics coefficients in the healpy convention. Shape
        ([nfreq], hp.Alm.getsize(lmax)).

    Raises
    ------
    ValueError :
        If method is not "numpy" or "jax".
    """
    npix = data.shape[-1]
    nside = hp.npix2nside(npix)
    use_pix_weights = nside in PIX_WEIGHTS_NSIDE
    use_ring_weights = not use_pix_weights
    kwargs = {
        "lmax": lmax,
        "pol": False,
        "use_weights": use_ring_weights,
        "use_pixel_weights": use_pix_weights,
    }
    if method == "numpy":
        return map2alm_numpy(data, **kwargs)
    elif method == "jax":
        return map2alm_jax(data, lmax=lmax)
    else:
        raise ValueError("method must be ``numpy'' or ``jax''.")


def map2alm_numpy(data, **kwargs):
    """
    Compute the spherical harmonics coefficents of a healpix map.

    Parameters
    ----------
    data : np.ndarray
        The healpix map(s). Shape ([nfreq], hp.nside2npix(nside)).

    kwargs are passed to hp.map2alm.

    Returns
    -------
    alm : np.ndarray
        The spherical harmonics coefficients in the healpy convention. Shape
        ([nfreq], hp.Alm.getsize(lmax)).
    """
    if data.ndim == 1:
        alm = hp.map2alm(data, **kwargs)
    else:
        # compute the alms of the first map to determine the size of the array
        alm0 = hp.map2alm(data[0], **kwargs)
        alm = np.empty((len(data), alm0.size), dtype=alm0.dtype)
        alm[0] = alm0
        for i in range(1, len(data)):
            alm[i] = hp.map2alm(data[i], **kwargs)
    return alm


@jax.jit
def map2alm_jax(data, **kwargs):
    """
    Compute the spherical harmonics coefficents of a healpix map.

    Parameters
    ----------
    data : jnp.ndarray
        The healpix map(s). Shape (nfreq, hp.nside2npix(nside)).

    kwargs are passed to hp.map2alm.

    Returns
    -------
    alm : jnp.ndarray
        The spherical harmonics coefficients in the healpy convention. Shape
        (nfreq, hp.Alm.getsize(lmax)).
    """
    return jax.vmap(hp.map2alm, in_axes=(0,))(data, **kwargs)
