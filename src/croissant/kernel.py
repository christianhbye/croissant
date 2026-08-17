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
import numpy as np
import s2fft
import s2fft.precompute_transforms

from . import utils
from .footprints import kernel_nbytes, transform_lmax

__all__ = [
    "clear_kernel_cache",
    "kernel_compute_alm",
    "kernel_nbytes",
    "precompute_kernel",
    "transform_lmax",
]

#: Cache size is a kernel COUNT, not a byte budget: cached kernels at,
#: say, nside=128 retain roughly 511 MiB each while engine_select.py's
#: "auto" policy advertises a single 512 MiB cap for one choice. That cap
#: governs only which engine a single call to resolve_engine picks, not
#: how much this module retains across many calls, so the two numbers are
#: not in tension by design -- but they can add up. Deliberately not made
#: byte-based: doing so would need to rank cached kernels by reuse
#: likelihood, not merely size, to decide what to evict. If total
#: retention becomes a problem, call clear_kernel_cache() to release
#: everything; that is the release valve this module offers instead of an
#: eviction policy.
#:
#: The FLOOR on this number is one polarized simulation's working set. A
#: PairStokesBeam and a PolarizedSky at niter > 0 need a forward and an
#: inverse kernel per transformed block, and a cache smaller than their
#: sum makes the two objects evict each other on every construction: a
#: parameter sweep or an MCMC step that rebuilds fields then pays the
#: full build cost every iteration while appearing to share a cache.
_KERNEL_CACHE_MAXSIZE = 32
_KERNEL_CACHE = OrderedDict()
_KERNEL_CACHE_LOCK = Lock()


def _kernel_dtype(sampling):
    """
    dtype ``jnp.asarray(built)`` will actually produce for a kernel.

    s2fft's numpy kernel builder always returns float64 (equiangular
    samplings) or complex128 (``"healpix"``, which carries per-ring
    phase shifts); ``jnp.asarray`` then silently downcasts to
    float32/complex64 whenever ``jax_enable_x64`` is off, like any
    other JAX array. Included in the cache key for the same reason
    ``dense._dense_matrix_key`` includes ``complex_dtype``: a kernel
    built before ``jax.config.update("jax_enable_x64", True)`` must not
    be silently reused afterwards at the earlier, reduced precision.
    """
    is_complex = sampling == "healpix"
    if jax.config.x64_enabled:
        return jnp.complex128 if is_complex else jnp.float64
    return jnp.complex64 if is_complex else jnp.float32


def precompute_kernel(
    lmax, sampling, nside=None, spin=0, reality=False, forward=True
):
    """
    Build and cache the Wigner-d kernel for one transform configuration.

    Must be called outside ``jax.jit``. Converting the numpy-built
    kernel to a ``jax.Array`` while a trace is active would return a
    tracer bound to that trace; caching it in the module-level,
    trace-independent cache would let a later, unrelated call read
    back a leaked tracer. Callers reaching this from inside a trace
    (:func:`kernel_compute_alm`, :class:`croissant.sphere.SphBase`)
    raise ``RuntimeError`` instead of calling this function, and expect
    the kernel to have been built here first, outside any trace.

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
        same value here and at apply time. Defaults to False to match
        :func:`kernel_compute_alm`, :func:`croissant.sphere.compute_alm`
        and :func:`croissant.footprints.kernel_nbytes`: this function is
        the documented way to warm a kernel for a jitted call, and a
        default that disagreed with the apply path would make the
        documented recipe raise. Forced to False for nonzero spin, the
        same ``reality and spin == 0`` rule the other three apply,
        because s2fft's real precompute path is only valid at spin 0.
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
    # Forced here rather than left to callers, so the key, the built
    # shape and the value the apply path passes cannot disagree.
    reality = bool(reality) and spin == 0
    # Keyed on the band-limit the kernel is BUILT at, not the one
    # requested: every sub-floor lmax at one nside builds the identical
    # kernel, so keying on the request would fill a cache whose whole
    # purpose is to hold a working set with byte-identical duplicates.
    build_lmax = transform_lmax(lmax, sampling, nside=nside)
    key = (
        int(build_lmax),
        str(sampling),
        None if nside is None else int(nside),
        int(spin),
        reality,
        bool(forward),
        np.dtype(_kernel_dtype(sampling)).str,
        jax.default_backend(),
    )
    with _KERNEL_CACHE_LOCK:
        if key in _KERNEL_CACHE:
            _KERNEL_CACHE.move_to_end(key)
            return _KERNEL_CACHE[key]

    # NOTE: the numpy builder, deliberately. See the module docstring.
    built = s2fft.precompute_transforms.construct.spin_spherical_kernel(
        L=build_lmax + 1,
        spin=int(spin),
        reality=reality,
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


def kernel_compute_alm(
    data,
    lmax,
    sampling,
    nside=None,
    niter=0,
    spin=0,
    reality=False,
    *,
    kernel=None,
    inverse_kernel=None,
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
        Whether the field is real. Defaults to False, matching
        :func:`croissant.sphere.compute_alm`: only a caller that knows
        its own data is real may claim the packed transform. Forced
        False for nonzero spin, which s2fft's precompute path requires.
    kernel : jax.Array or None
        Precomputed forward (analysis) kernel, as returned by
        :func:`precompute_kernel` with ``forward=True``. This is
        primarily used internally by :class:`croissant.sphere.SphBase`
        so its jitted ``compute_alm`` methods never build a kernel
        while being traced. If None and ``data`` is not a jax tracer,
        it is built (and cached) here via :func:`precompute_kernel`.
    inverse_kernel : jax.Array or None
        Precomputed synthesis (inverse) kernel, as returned by
        :func:`precompute_kernel` with ``forward=False``. Only
        consulted when ``niter > 0``; same fallback as ``kernel``.

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
    # utils.engine_dtypes: they reproduce s2fft.forward, which returns
    # complex128 on an x64 runtime regardless of the input map dtype.
    real_dtype, _ = utils.engine_dtypes()
    data = jnp.asarray(data)
    if data.dtype.kind == "c":
        data = data.astype(jnp.result_type(real_dtype, 1j))
    else:
        data = data.astype(real_dtype)
    spatial_ndim = utils.spatial_ndim(sampling)
    spatial_shape = data.shape[-spatial_ndim:]
    batch_shape = data.shape[:-spatial_ndim]
    flat = data.reshape((-1,) + spatial_shape)

    reality = bool(reality) and spin == 0
    L = lmax + 1
    forward_kernel = kernel
    if forward_kernel is None:
        if isinstance(data, jax.core.Tracer):
            raise RuntimeError(
                "The kernel must be precomputed before "
                "kernel_compute_alm is called inside jax.jit. Call "
                "precompute_kernel(...) once outside jax.jit."
            )
        forward_kernel = precompute_kernel(
            lmax,
            sampling,
            nside=nside,
            spin=spin,
            reality=reality,
            forward=True,
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
    if inverse_kernel is None:
        if isinstance(data, jax.core.Tracer):
            raise RuntimeError(
                "The inverse kernel must be precomputed before "
                "kernel_compute_alm is called inside jax.jit with "
                "niter > 0. Call precompute_kernel(..., forward=False) "
                "once outside jax.jit."
            )
        inverse_kernel = precompute_kernel(
            lmax,
            sampling,
            nside=nside,
            spin=spin,
            reality=reality,
            forward=False,
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
