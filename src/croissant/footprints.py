"""
Predict the size of a precomputed transform without building it.

Croissant's engines precompute different amounts of the pixels-to-alm
map, and both the automatic engine policy and the benchmarks need to know
what a choice would cost before paying for it. These helpers are pure
arithmetic over the transform's geometry; they import nothing from
croissant, so any module may use them.
"""

import numpy as np
import s2fft

_COMPLEX_ITEMSIZE = np.dtype(np.complex128).itemsize


def transform_lmax(lmax, sampling, nside=None):
    """
    Band-limit a transform must actually be performed at.

    s2fft's HEALPix FFT requires ``L >= 2 * nside`` even when only lower
    modes are wanted, the same floor ``croissant.dense`` handles at
    ``dense.py:52``.

    Parameters
    ----------
    lmax : int
        Requested maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.

    Returns
    -------
    int
        The band-limit to transform at, always ``>= lmax``.

    """
    if sampling != "healpix":
        return int(lmax)
    if nside is None:
        raise ValueError("nside is required for HEALPix transforms.")
    return max(int(lmax), 2 * int(nside) - 1)


def _ntheta(lmax, sampling, nside=None):
    """Number of latitude rings for a sampling scheme."""
    if sampling == "healpix":
        if nside is None:
            raise ValueError("nside is required for HEALPix transforms.")
        return 4 * int(nside) - 1
    return s2fft.sampling.s2_samples.ntheta(L=lmax + 1, sampling=sampling)


def _npix(lmax, sampling, nside=None):
    """Number of spatial samples for a sampling scheme."""
    if sampling == "healpix":
        if nside is None:
            raise ValueError("nside is required for HEALPix transforms.")
        return 12 * int(nside) ** 2
    L = lmax + 1
    return s2fft.sampling.s2_samples.ntheta(
        L=L, sampling=sampling
    ) * s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling)


def kernel_nbytes(lmax, sampling, nside=None, reality=False):
    """
    Predict a Wigner-d kernel's memory footprint.

    Reports the footprint at the band-limit the kernel would really be
    built at, i.e. after applying the HEALPix ``L >= 2 * nside`` floor.
    Reporting the requested ``lmax`` instead would under-predict by
    ``(2 * nside / (lmax + 1)) ** 2`` whenever a caller asks for a low
    band-limit on a high-resolution map.

    Parameters
    ----------
    lmax : int
        Requested maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    reality : bool
        Whether the kernel is built for a real field. Real kernels store
        only ``m >= 0``, halving the last axis.

    Returns
    -------
    int
        Size in bytes of the complex128 kernel.

    """
    L = transform_lmax(lmax, sampling, nside=nside) + 1
    nm = L if reality else 2 * L - 1
    return _ntheta(lmax, sampling, nside) * L * nm * _COMPLEX_ITEMSIZE


def dense_nbytes(lmax, sampling, nside=None, spin=0, reality=True):
    """
    Predict the dense analysis operator's memory footprint.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    spin : int
        Spin weight of the field.
    reality : bool
        Whether the field is real. Real scalar fields store only the
        independent ``m >= 0`` coefficients.

    Returns
    -------
    int
        Size in bytes of the complex128 operator.

    """
    L = lmax + 1
    if spin == 0 and reality:
        ncoeff = (lmax + 1) * (lmax + 2) // 2
    else:
        ncoeff = L * L - spin * spin
    return ncoeff * _npix(lmax, sampling, nside) * _COMPLEX_ITEMSIZE
