import numpy as np
import healpy as hp
from ..constants import PIX_WEIGHTS_NSIDE


def alm2map(alm, nside, lmax=None):
    """
    Compute the healpix map from the spherical harmonics coefficients.

    Parameters
    ----------
    alm : array-like
        The spherical harmonics coefficients in the healpy convention. Shape
        ([nfreq], hp.Alm.getsize(lmax)).
    nside : int
        The nside of the output map(s).
    lmax : int
        The lmax of the spherical harmonics transform. Defaults to 3*nside-1.

    Returns
    -------
    map : np.ndarray
        The healpix map. Shape ([nfreq], hp.nside2npix(nside)).

    """
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


def map2alm(data, lmax=None):
    """
    Compute the spherical harmonics coefficents of a healpix map.

    Parameters
    ----------
    data : array-like
        The healpix map(s). Shape ([nfreq], hp.nside2npix(nside)).
    lmax : int
        The lmax of the spherical harmonics transform. Defaults to 3*nside-1.

    Returns
    -------
    alm : np.ndarray
        The spherical harmonics coefficients in the healpy convention. Shape
        ([nfreq], hp.Alm.getsize(lmax)).

    """
    data = np.array(data, copy=True)
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
