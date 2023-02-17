import healpy as hp
import numpy as np

from .constants import PIX_WEIGHTS_NSIDE


def alm2map(alm, nside, lmax=None):
    alm = np.array(alm, copy=True)
    if alm.ndim == 1:
        alm.shape = (1, -1)
    nfreqs = alm.shape[0]
    npix = hp.nside2npix(nside)
    hp_map = np.empty((nfreqs, npix))
    for i in range(nfreqs):
        map_i = hp.alm2map(alm[i], nside, lmax=lmax)
        hp_map[i] = map_i
    return np.squeeze(hp_map)


def map2alm(data, lmax=None):
    """
    Compute the spherical harmonics coefficents of a healpix map.
    """

    data = np.array(data)
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
    elif data.ndim == 2:
        # compute the alms of the first map to determine the size of the array
        alm0 = hp.map2alm(data[0], **kwargs)
        alm = np.empty((len(data), alm0.size), dtype=alm0.dtype)
        alm[0] = alm0
        for i in range(1, len(data)):
            alm[i] = hp.map2alm(data[i], **kwargs)
    else:
        raise ValueError("Input data must be a map or list of maps.")
    return alm
