import healpy as hp
import numpy as np

from . import constants

def alm2map(alm, nside, lmax=None, mmax=None):
    alm = np.array(alm, copy=True)
    if alm.ndim == 1:
        alm.shape = (1, -1)
    nfreqs = alm.shape[0]
    npix = hp.nside2npix(nside)
    hp_map = np.empty((nfreqs, npix))
    for i in range(nfreqs):
        map_i = hp.alm2map(alm[i], nside, lmax=lmax, mmax=mmax)
        hp_map[i] = map_i
    return np.squeeze(hp_map)

def map2alm(data, lmax=None, mmax=None):
    """
    Compute the spherical harmonics coefficents of a healpix map.
    """

    data = np.array(data)
    npix = data.shape[-1]
    nside = hp.npix2nside(npix)
    use_pix_weights = nside in constants.PIX_WEIGHTS_NSIDE
    use_ring_weights = not use_pix_weights
    kwargs = {
        "lmax": lmax,
        "mmax": mmax,
        "pol": False,
        "use_weights": use_ring_weights,
        "use_pixel_weights": use_pix_weights,
    }
    if data.ndim == 1:
        alm = hp.map2alm(data, **kwargs)
    elif data.ndim == 2:
        alm = np.empty(
            (len(data), hp.Alm.getsize(lmax, mmax=lmax)), dtype=np.complex128
        )
        for i in range(len(data)):
            alm[i] = hp.map2alm(data[i], **kwargs)
    else:
        raise ValueError("Input data must be a map or list of maps.")
    return alm


