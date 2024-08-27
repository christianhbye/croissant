import numpy as np
import healpy as hp
from ..constants import PIX_WEIGHTS_NSIDE


def alm2map(alm, nside, lmax=None, polarized=False):
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
    polarized : bool
        If true, alm's are assumed to be TEB and output will be I, Q, U maps.
        I is spin-0 and Q, U are spin-2. Multiple frequency maps are not yet
        supported in this case.

     Returns
     -------
     map : np.ndarray
         The healpix map or sequence of maps.

    """
    return hp.alm2map(alm, nside, lmax=lmax, pol=polarized)


def map2alm(data, lmax=None, polarized=False):
    """
    Compute the spherical harmonics coefficents of a healpix map.

    Parameters
    ----------
    data : array-like
        The healpix map(s). Shape ([nfreq], hp.nside2npix(nside)).
    lmax : int
        The lmax of the spherical harmonics transform. Defaults to 3*nside-1.
    polarized : bool
        If true, map is assumed to be I, Q, U maps. I is spin-0 and Q, U are
        spin-2. Output alm's will be TEB.
        Multiple frequency maps are not yet supported in this case.

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
    return hp.map2alm(
        data,
        lmax=lmax,
        pol=polarized,
        use_weights=use_ring_weights,
        use_pixel_weights=use_pix_weights,
    )
