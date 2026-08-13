"""
Precomputed-kernel spherical harmonic analysis.

This engine caches the Wigner-d kernel that carries the theta-to-ell
stage of the transform and contracts it per call, leaving only the
per-ring FFT to be recomputed. It sits between ``engine="s2fft"``, which
precomputes nothing, and ``engine="dense"``, which materialises the whole
``ncoeff x npix`` operator: the kernel is ``O(nside**3)`` where the dense
operator is ``O(nside**4)``, which is what makes moderate and high nside
reachable.

Two s2fft constraints are load-bearing and must not be relaxed:

1. Kernels are built with the *numpy* ``spin_spherical_kernel``, never
   ``spin_spherical_kernel_jax``. The jax builder routes through the
   Price-McEwen recursion, which hits exact Wigner-d zero nodes at
   HEALPix's rational cos(theta) values for spin != 0; the NaNs are then
   masked to zero, silently dropping those modes (~7e-2 relative error
   at spin 2). The numpy builder uses Turok's recursion and is exact.
2. ``iter > 0`` is never passed to s2fft's precompute transform. Its
   refinement branch builds an inverse kernel with the broken jax
   builder and diverges for spin != 0. Croissant runs the refinement
   iteration itself in :func:`kernel_compute_alm`.
"""

from collections import OrderedDict
from functools import partial
from threading import Lock

import jax
import jax.numpy as jnp
import s2fft
import s2fft.precompute_transforms

from .footprints import kernel_nbytes, transform_lmax

__all__ = [
    "clear_kernel_cache",
    "kernel_compute_alm",
    "kernel_nbytes",
    "precompute_kernel",
    "transform_lmax",
]

_KERNEL_CACHE_MAXSIZE = 8
_KERNEL_CACHE = OrderedDict()
_KERNEL_CACHE_LOCK = Lock()


def precompute_kernel(
    lmax, sampling, nside=None, spin=0, reality=False, forward=True
):
    """
    Build and cache the Wigner-d kernel for one transform configuration.

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
        Whether the kernel is for a real field. This is a BUILD
        parameter, not only an apply-time one: with ``reality=True``
        s2fft's precompute path slices ``ftm`` to ``m >= 0`` and expects
        a kernel whose last axis is ``L`` rather than ``2L - 1``.
        Building with one value and applying with the other raises
        ``ValueError: Size of label 'm' ... does not match previous
        terms``, so it is part of the cache key and callers must pass the
        same value here and at apply time.
    forward : bool
        Build the analysis kernel if True, the synthesis kernel if
        False. The synthesis kernel is only needed for iterative
        refinement.

    Returns
    -------
    jax.Array
        Kernel of shape ``(ntheta, L, L)`` when ``reality`` is True and
        ``(ntheta, L, 2L - 1)`` otherwise, where ``L`` is
        ``transform_lmax(...) + 1``.

    """
    key = (
        int(lmax),
        str(sampling),
        None if nside is None else int(nside),
        int(spin),
        bool(reality),
        bool(forward),
    )
    with _KERNEL_CACHE_LOCK:
        if key in _KERNEL_CACHE:
            _KERNEL_CACHE.move_to_end(key)
            return _KERNEL_CACHE[key]

    # NOTE: the numpy builder, deliberately. See the module docstring.
    built = s2fft.precompute_transforms.construct.spin_spherical_kernel(
        L=transform_lmax(lmax, sampling, nside=nside) + 1,
        spin=int(spin),
        reality=bool(reality),
        sampling=sampling,
        nside=nside,
        forward=bool(forward),
    )
    array = jnp.asarray(built)

    if isinstance(array, jax.core.Tracer):
        # Built for the first time from inside an active jax trace
        # (e.g. SphBase's jitted compute_alm on first use, or a
        # caller's own jax.jit). The Wigner-d recursion above never
        # depends on a traced value, but this converted array is only
        # valid within the current trace: caching it would let a later,
        # unrelated trace read back a leaked tracer. Return it directly
        # for immediate use in this trace only; do not persist it.
        return array

    with _KERNEL_CACHE_LOCK:
        _KERNEL_CACHE[key] = array
        _KERNEL_CACHE.move_to_end(key)
        while len(_KERNEL_CACHE) > _KERNEL_CACHE_MAXSIZE:
            _KERNEL_CACHE.popitem(last=False)
        return _KERNEL_CACHE[key]


def clear_kernel_cache():
    """Release all cached kernels held by croissant."""
    with _KERNEL_CACHE_LOCK:
        _KERNEL_CACHE.clear()


def _spatial_ndim(sampling):
    """Number of trailing axes that hold the field's spatial samples."""
    return 1 if sampling == "healpix" else 2


def kernel_compute_alm(
    data,
    lmax,
    sampling,
    nside=None,
    niter=0,
    spin=0,
    reality=True,
):
    """
    Compute alm by contracting a cached Wigner-d kernel.

    Every axis before the spatial axes is treated as a batch axis, and
    the returned layout matches :func:`croissant.sphere.compute_alm`.

    Parameters
    ----------
    data : array_like
        Field samples, with spatial axes trailing.
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    niter : int
        Number of iterative refinement steps. Refinement is run by
        croissant, not by s2fft; see the module docstring.
    spin : int
        Spin weight of the field.
    reality : bool
        Whether the field is real. Forced False for nonzero spin, which
        s2fft's precompute path requires.

    Returns
    -------
    jax.Array
        Coefficients of shape ``batch + (lmax + 1, 2 * lmax + 1)``.

    """
    if niter < 0:
        raise ValueError(f"niter must be non-negative, got {niter}.")
    floor = transform_lmax(lmax, sampling, nside=nside)
    if floor != lmax:
        raise ValueError(
            f"The kernel engine needs lmax >= {floor} for "
            f"nside={nside} (s2fft's HEALPix FFT requires "
            "L >= 2 * nside), but lmax="
            f"{lmax} was requested. Use engine='dense', which builds at "
            "the required band-limit and keeps only the low-ell rows, or "
            "engine='s2fft'."
        )
    # Croissant's engines share a dtype contract, owned and documented by
    # sphere._dense_dtypes: they reproduce s2fft.forward, which returns
    # complex128 on an x64 runtime even for float32 maps. s2fft's
    # PRECOMPUTE path instead inherits the input dtype, so a float32 map
    # would come back complex64 with ~1e-7 relative error. Promote the
    # input rather than casting the result: casting the result would keep
    # that error. Imported lazily because sphere imports this module.
    from .sphere import _dense_dtypes

    real_dtype, _ = _dense_dtypes()
    data = jnp.asarray(data)
    if data.dtype.kind == "c":
        data = data.astype(jnp.result_type(real_dtype, 1j))
    else:
        data = data.astype(real_dtype)
    spatial_ndim = _spatial_ndim(sampling)
    spatial_shape = data.shape[-spatial_ndim:]
    batch_shape = data.shape[:-spatial_ndim]
    flat = data.reshape((-1,) + spatial_shape)

    reality = bool(reality) and spin == 0
    L = lmax + 1
    forward_kernel = precompute_kernel(
        lmax, sampling, nside=nside, spin=spin, reality=reality, forward=True
    )
    analyse = partial(
        s2fft.precompute_transforms.spherical.forward,
        L=L,
        spin=spin,
        kernel=forward_kernel,
        sampling=sampling,
        reality=reality,
        method="jax",
        nside=nside,
        iter=0,  # never delegate refinement; see the module docstring
    )
    if niter == 0:
        flat_alm = jax.vmap(analyse)(flat)
        return flat_alm.reshape(batch_shape + (lmax + 1, 2 * lmax + 1))

    # Iterative refinement, run here rather than delegated to s2fft:
    # its precompute refinement branch builds an inverse kernel with the
    # broken jax builder and diverges for spin != 0. The iteration is
    # flm <- flm + F(f - I(flm)), the same one sphere.py applies to the
    # scalar dense matrix in gram form.
    inverse_kernel = precompute_kernel(
        lmax, sampling, nside=nside, spin=spin, reality=reality, forward=False
    )
    synthesise = partial(
        s2fft.precompute_transforms.spherical.inverse,
        L=L,
        spin=spin,
        kernel=inverse_kernel,
        sampling=sampling,
        reality=reality,
        method="jax",
        nside=nside,
    )

    def refine(field):
        alm = analyse(field)
        for _ in range(niter):
            alm = alm + analyse(field - synthesise(alm))
        return alm

    flat_alm = jax.vmap(refine)(flat)
    return flat_alm.reshape(batch_shape + (lmax + 1, 2 * lmax + 1))
