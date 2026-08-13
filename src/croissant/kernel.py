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
from threading import Lock

import jax.numpy as jnp
import s2fft
import s2fft.precompute_transforms

from .footprints import kernel_nbytes, transform_lmax

__all__ = [
    "clear_kernel_cache",
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
