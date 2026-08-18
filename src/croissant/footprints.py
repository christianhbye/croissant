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
_REAL_ITEMSIZE = np.dtype(np.float64).itemsize


def transform_lmax(lmax, sampling, nside=None):
    """
    Band-limit a transform must actually be performed at.

    s2fft's HEALPix FFT requires ``L >= 2 * nside`` even when only lower
    modes are wanted. This is the single definition of that floor;
    ``croissant.dense`` and the engine policy both import it, so an s2fft
    version that relaxes the constraint is a one-line change here rather
    than a hunt through every module that builds a transform.

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


def _kernel_itemsize(sampling):
    """
    Item size, in bytes, of a Wigner-d kernel s2fft would build.

    Only the HEALPix kernel is complex128: it carries the per-ring
    phase shifts needed because HEALPix rings start at a ring-dependent
    phi offset rather than all starting at phi=0. The equiangular
    samplings (``mw``, ``mwss``, ``dh``, ``gl``) have no such offset
    and their kernels are real, float64. Measured directly (build a
    kernel and inspect ``.dtype``) at L=16, spin=0, reality=True,
    forward=True: healpix (nside=8) is complex128; mw, mwss, dh and gl
    are all float64. Spin != 0 and ``forward=False`` do not change
    this -- only the sampling scheme does.
    """
    return _COMPLEX_ITEMSIZE if sampling == "healpix" else _REAL_ITEMSIZE


def _kernel_ntheta(lmax, sampling, nside=None):
    """
    Leading (theta) axis length of a Wigner-d kernel s2fft would build.

    This is a property of the *kernel* s2fft's precompute path
    constructs, not of the sampling's ordinary data grid --
    ``s2fft.sampling.s2_samples.ntheta`` (used for the data grid in
    :func:`_npix`) gives the wrong answer for ``mw`` and ``mwss``.
    Measured by building a kernel with
    ``s2fft.precompute_transforms.construct.spin_spherical_kernel`` at
    L (= transform_lmax + 1) in ``{5, 8, 16, 17, 32}`` and comparing
    ``.shape[0]`` against ``s2fft.sampling.s2_samples.ntheta``, to tell
    a coincidence at one L from a rule:

    ================  =================  =======================
    sampling          kernel leading axis  ordinary ntheta(L=16)
    ================  =================  =======================
    healpix (nside=8)  4*nside - 1 (= 31)  31 (same formula)
    mwss                2*L + 1 (= 33)     17
    mw                  2*L + 1 (= 33)     16
    dh                  2*L (= 32)         32 (same formula)
    gl                  L (= 16)           16 (same formula)
    ================  =================  =======================

    ``dh``, ``gl`` and ``healpix`` match their ordinary ``ntheta``;
    ``mw`` and ``mwss`` share ``2*L + 1`` instead, despite having
    different ordinary ``ntheta`` values (``L`` and ``L + 1``): s2fft's
    Turok/Bosch kernel recursion extends their theta axis to cover the
    range the recursion needs, rather than just the sampling's own ring
    count. This holds regardless of ``spin``.

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
        Leading axis length of the kernel s2fft would build.

    """
    if sampling == "healpix":
        if nside is None:
            raise ValueError("nside is required for HEALPix transforms.")
        return 4 * int(nside) - 1
    L = transform_lmax(lmax, sampling, nside=nside) + 1
    if sampling in ("mw", "mwss"):
        return 2 * L + 1
    return s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling)


def spatial_shape(lmax, sampling, nside=None):
    """
    Shape of one map's spatial axes for a sampling scheme.

    HEALPix stores a single flat pixel axis; the equiangular schemes
    store a (theta, phi) grid.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.

    Returns
    -------
    tuple of int
        ``(npix,)`` for ``"healpix"``, ``(ntheta, nphi)`` otherwise.

    """
    if sampling == "healpix":
        if nside is None:
            raise ValueError("nside is required for HEALPix transforms.")
        return (12 * int(nside) ** 2,)
    L = lmax + 1
    return (
        s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling),
        s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling),
    )


def _npix(lmax, sampling, nside=None):
    """Total number of spatial samples for a sampling scheme."""
    return int(np.prod(spatial_shape(lmax, sampling, nside=nside)))


def kernel_nbytes(lmax, sampling, nside=None, spin=0, reality=False):
    """
    Predict a Wigner-d kernel's memory footprint.

    Reports the footprint at the band-limit the kernel would really be
    built at, i.e. after applying the HEALPix ``L >= 2 * nside`` floor.
    Reporting the requested ``lmax`` instead would under-predict by
    ``(2 * nside / (lmax + 1)) ** 2`` whenever a caller asks for a low
    band-limit on a high-resolution map.

    ``reality`` is downgraded to ``False`` whenever ``spin != 0``, even
    if the caller passes ``reality=True``. This is the one place that
    still absorbs the contradiction rather than raising on it, and
    deliberately so: the transform entry points
    (``kernel.precompute_kernel``, ``kernel.kernel_compute_alm``,
    ``sphere.compute_alm``) reject the pair, but a PREDICTOR's job is to
    describe whatever the transform would do, not to police its caller
    -- ``resolve_engine`` must be able to size any configuration it is
    handed without first validating it.

    s2fft's real-field precompute path is only valid for spin 0: a spin
    field has no ``m -> -m`` Hermitian symmetry to exploit, so its
    kernel always stores the full ``m`` range. Predicting sizes for spin
    fields without applying this rule under-predicts by very close to 2x
    (the ratio of ``2 * L - 1`` to ``L``) -- this has been the source of
    a recurring, silent mismatch between predicted and built kernel
    sizes, so the rule is enforced once here rather than left for every
    caller to repeat correctly.

    Parameters
    ----------
    lmax : int
        Requested maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    spin : int
        Spin weight of the field.
    reality : bool
        Whether the kernel is built for a real field. Real kernels store
        only ``m >= 0``, halving the last axis. Defaults to False to
        match :func:`dense_nbytes` and, more importantly, the transform
        it predicts: :func:`croissant.sphere.compute_alm` assumes
        nothing about the caller's data, so a predictor that assumed a
        real field would under-predict the default transform by 2x.
        Callers that know their own field is real pass ``reality=True``
        here exactly as they do there. Ignored (treated as ``False``)
        whenever ``spin != 0``; see above.

    Returns
    -------
    int
        Size in bytes of the kernel: complex128 for ``"healpix"``,
        float64 for the equiangular samplings (``"mw"``, ``"mwss"``,
        ``"dh"``, ``"gl"``); see :func:`_kernel_itemsize`.

    """
    reality = bool(reality) and spin == 0
    L = transform_lmax(lmax, sampling, nside=nside) + 1
    nm = L if reality else 2 * L - 1
    ntheta = _kernel_ntheta(lmax, sampling, nside)
    return ntheta * L * nm * _kernel_itemsize(sampling)


def dense_nbytes(lmax, sampling, nside=None, spin=0, reality=False):
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
        independent ``m >= 0`` coefficients. Defaults to False, matching
        :func:`kernel_nbytes` and the transform this predicts.

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
