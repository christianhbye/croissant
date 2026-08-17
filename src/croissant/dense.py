"""Cached dense spherical harmonic analysis for scalar and spin fields."""

from functools import partial
from threading import RLock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import s2fft

from . import utils
from .footprints import spatial_shape as _spatial_shape
from .footprints import transform_lmax

_DENSE_MATRIX_CACHE = {}
_DENSE_MATRIX_CACHE_LOCK = RLock()


def _dense_matrix_key(
    spatial_shape, lmax, sampling, nside, spin, packed, niter, complex_dtype
):
    """Return a hashable key for a cached dense SHT analysis matrix.

    ``packed`` separates the m >= 0 real operator from the full complex
    one. Both flavours exist at spin 0 and identical geometry, so
    without it a caller asking for one could be handed the other.

    Keys on ``jax.default_backend()`` rather than a device string, which
    is what sphere's half and ``kernel.precompute_kernel`` already do.
    Two devices on one backend therefore share an entry, costing a
    transfer rather than correctness.
    """
    return (
        tuple(spatial_shape),
        int(lmax),
        str(sampling),
        None if nside is None else int(nside),
        int(spin),
        bool(packed),
        int(niter),
        np.dtype(complex_dtype).str,
        jax.default_backend(),
    )


def _positive_lm_indices(lmax):
    """Indices for healpy-ordered, independent m >= 0 coefficients."""
    ell = np.concatenate(
        [np.arange(m, lmax + 1, dtype=np.int32) for m in range(lmax + 1)]
    )
    emm = np.concatenate(
        [np.full(lmax - m + 1, m, dtype=np.int32) for m in range(lmax + 1)]
    )
    return ell, emm


def _build_dense_matrix_healpix(
    spatial_shape,
    lmax,
    nside,
    niter,
    chunk_size,
):
    """Build a HEALPix matrix by evaluating spherical harmonics directly."""
    from scipy import special

    npix = int(np.prod(spatial_shape))
    if spatial_shape != (npix,):
        raise ValueError("HEALPix maps must have one spatial pixel axis")

    ell, emm = _positive_lm_indices(lmax)
    nalm = len(ell)
    _, complex_dtype = utils.engine_dtypes()
    if chunk_size is None:
        # Limit each host-side spherical-harmonic block to roughly 64 MiB.
        complex_itemsize = np.dtype(complex_dtype).itemsize
        chunk_size = min(256, max(1, (64 << 20) // (npix * complex_itemsize)))
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = min(int(chunk_size), nalm)

    theta = np.asarray(
        utils.generate_theta(lmax=None, sampling="healpix", nside=nside)
    )
    phi = np.asarray(
        utils.generate_phi(lmax=None, sampling="healpix", nside=nside)
    )
    blocks = []
    for start in range(0, nalm, chunk_size):
        stop = min(start + chunk_size, nalm)
        ell_chunk = ell[start:stop, None]
        emm_chunk = emm[start:stop, None]
        if hasattr(special, "sph_harm_y"):
            block = special.sph_harm_y(
                ell_chunk,
                emm_chunk,
                theta[None, :],
                phi[None, :],
            )
        else:  # pragma: no cover - SciPy < 1.15 compatibility
            block = special.sph_harm(
                emm_chunk,
                ell_chunk,
                phi[None, :],
                theta[None, :],
            )
        blocks.append(block.astype(np.dtype(complex_dtype)))
    harmonics = jnp.asarray(np.concatenate(blocks, axis=0))

    pixel_area = jnp.asarray(4 * np.pi / npix, dtype=harmonics.real.dtype)
    if niter == 0:
        return pixel_area * jnp.conj(harmonics)

    # For iterative refinement, express synthesis and analysis in the L**2
    # independent real harmonic degrees of freedom: m=0 contributes one real
    # value, while m>0 contributes real and imaginary parts.
    packed_real = []
    packed_imag = []
    row = 0
    for m in range(lmax + 1):
        for _ in range(m, lmax + 1):
            packed_real.append(row)
            row += 1
            if m == 0:
                packed_imag.append(-1)
            else:
                packed_imag.append(row)
                row += 1

    packed_real = np.asarray(packed_real, dtype=np.int32)
    packed_imag = np.asarray(packed_imag, dtype=np.int32)
    ndof = (lmax + 1) ** 2
    synthesis = jnp.zeros((ndof, npix), dtype=harmonics.real.dtype)
    base = jnp.zeros_like(synthesis)

    zero = emm == 0
    synthesis = synthesis.at[packed_real].set(
        jnp.where(zero[:, None], harmonics.real, 2 * harmonics.real)
    )
    base = base.at[packed_real].set(pixel_area * harmonics.real)

    positive = ~zero
    synthesis = synthesis.at[packed_imag[positive]].set(
        -2 * harmonics.imag[positive]
    )
    base = base.at[packed_imag[positive]].set(
        -pixel_area * harmonics.imag[positive]
    )

    analysis = base
    gram = base @ synthesis.T
    for _ in range(niter):
        analysis = analysis + base - gram @ analysis

    matrix = analysis[packed_real].astype(complex_dtype)
    positive = packed_imag >= 0
    matrix = matrix.at[positive].add(1j * analysis[packed_imag[positive]])
    return matrix


@partial(eqx.filter_jit, inline=True)
def _forward_real_chunk(basis, lmax, sampling, nside, niter):
    """Forward-transform a batch of real maps with the s2fft engine.

    A local copy of sphere._compute_alm_s2fft's inner call, specialised
    to what the builder needs: one leading batch axis, spin 0, real
    input. Duplicated rather than imported because dense.py must not
    import sphere.py -- sphere dispatches to this module, and a
    module-level import back would be a cycle.
    """
    m2alm = partial(
        s2fft.forward,
        L=lmax + 1,
        spin=0,
        nside=nside,
        sampling=sampling,
        method="jax",
        # Valid because the one-hot basis maps are real: s2fft then
        # computes only the m >= 0 half, which is the half the caller
        # keeps. This saves work and allocation; it does not change the
        # coefficients, which the general transform returns too.
        reality=True,
        precomps=None,
        spmd=False,
        L_lower=0,
        iter=niter,
    )
    return jax.vmap(m2alm)(basis)


def _build_dense_matrix_from_pixels(
    spatial_shape,
    lmax,
    sampling,
    nside,
    niter,
    chunk_size=None,
):
    """Materialize a general s2fft analysis operator from pixel bases."""
    npix = int(np.prod(spatial_shape))
    real_dtype, _ = utils.engine_dtypes()
    itemsize = np.dtype(real_dtype).itemsize
    if chunk_size is None:
        # Keep each identity-map chunk below 64 MiB. A ceiling of 256 gives
        # enough batch parallelism on a GPU without making s2fft's vmapped
        # intermediate arrays uncomfortably large.
        chunk_size = min(256, max(1, (64 << 20) // (npix * itemsize)))
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = min(int(chunk_size), npix)

    ell, emm = _positive_lm_indices(lmax)
    blocks = []
    for start in range(0, npix, chunk_size):
        stop = min(start + chunk_size, npix)
        indices = jnp.arange(start, stop)
        basis = jax.nn.one_hot(indices, npix, dtype=real_dtype)
        basis = basis.reshape((stop - start,) + tuple(spatial_shape))
        dense = _forward_real_chunk(basis, lmax, sampling, nside, niter)
        packed = dense[:, ell, lmax + emm].T
        # Bound peak memory by ensuring that s2fft's much larger dense-layout
        # result can be released before the next chunk is submitted.
        blocks.append(packed.block_until_ready())

    return jnp.concatenate(blocks, axis=1)


def _build_dense_matrix(
    spatial_shape,
    lmax,
    sampling,
    nside,
    niter,
    chunk_size=None,
):
    """Materialize the exact s2fft analysis operator in bounded chunks."""
    if sampling.lower() == "healpix":
        return _build_dense_matrix_healpix(
            tuple(spatial_shape),
            lmax,
            nside,
            niter,
            chunk_size,
        )
    return _build_dense_matrix_from_pixels(
        spatial_shape,
        lmax,
        sampling,
        nside,
        niter,
        chunk_size=chunk_size,
    )


def precompute_dense_matrix(
    spatial_shape,
    lmax,
    sampling,
    nside=None,
    niter=0,
    chunk_size=None,
):
    """
    Build and cache a dense spherical harmonic analysis matrix.

    The returned matrix stores only the independent ``m >= 0`` coefficients
    and has shape ``((lmax + 1) * (lmax + 2) // 2, prod(spatial_shape))``.
    It exactly represents Croissant's standard ``s2fft`` transform, including
    the requested iterative-refinement count. Its precision follows JAX's
    x64 setting, matching the alm dtype that ``s2fft`` produces: complex128
    when x64 is enabled and complex64 otherwise, independently of the dtype
    of the maps it is applied to.

    Parameters
    ----------
    spatial_shape : tuple of int
        Shape of one input map, excluding its frequency axis.
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Spherical sampling scheme understood by s2fft.
    nside : int or None
        HEALPix nside, required for HEALPix sampling.
    niter : int
        Number of s2fft iterative-refinement steps to fold into the matrix.
    chunk_size : int or None
        Number of basis rows generated together while building the matrix.
        The default bounds each input chunk to roughly 64 MiB, with a ceiling
        of 256.

    Returns
    -------
    matrix : jax.Array
        Cached dense analysis matrix on the current default JAX device.
    """
    _, complex_dtype = utils.engine_dtypes()
    key = _dense_matrix_key(
        spatial_shape,
        lmax,
        sampling,
        nside,
        spin=0,
        packed=True,
        niter=niter,
        complex_dtype=complex_dtype,
    )
    with _DENSE_MATRIX_CACHE_LOCK:
        matrix = _DENSE_MATRIX_CACHE.get(key)
        if matrix is None:
            matrix = _build_dense_matrix(
                tuple(spatial_shape),
                lmax,
                sampling,
                nside,
                niter,
                chunk_size=chunk_size,
            )
            _DENSE_MATRIX_CACHE[key] = matrix
    return matrix


def dense_matrix_for(
    spatial_shape,
    lmax,
    sampling,
    nside=None,
    niter=0,
    *,
    tracing,
    explicit,
):
    """
    Fetch the packed-real dense operator for one field configuration.

    Outside a trace this is an ordinary cached build. Inside one, an
    AUTOMATIC choice still builds: the matrix depends only on static
    geometry, so ``jax.ensure_compile_time_eval`` yields a concrete
    array rather than a tracer, exactly as
    :class:`DenseSphericalTransform` already builds its own operator
    mid-trace. Refusing here would leave an automatic dense choice with
    nowhere to go, because dense is selected precisely when the
    band-limit is below the HEALPix floor and the matrix-free engine
    cannot serve it at all.

    An EXPLICIT ``engine="dense"`` keeps the documented contract: warm
    the cache with :func:`precompute_dense_matrix` outside ``jax.jit``,
    or get a ``RuntimeError``. A caller who pinned the engine is never
    silently charged for a build inside their own jit.

    Parameters
    ----------
    spatial_shape : tuple of int
        Shape of one input map, excluding all batch axes.
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Spherical sampling scheme understood by s2fft.
    nside : int or None
        HEALPix nside, required for HEALPix sampling.
    niter : int
        Number of s2fft iterative-refinement steps folded into the matrix.
    tracing : bool
        Whether a jax trace is active.
    explicit : bool
        Whether the caller named ``"dense"`` rather than ``"auto"``.

    Returns
    -------
    matrix : jax.Array
        Cached dense analysis matrix.

    Raises
    ------
    RuntimeError
        If an explicit dense request is made inside a trace and no
        matching matrix has been precomputed.

    """
    spatial_shape = tuple(spatial_shape)
    if not tracing:
        return precompute_dense_matrix(
            spatial_shape, lmax, sampling, nside=nside, niter=niter
        )
    _, complex_dtype = utils.engine_dtypes()
    key = _dense_matrix_key(
        spatial_shape,
        lmax,
        sampling,
        nside,
        spin=0,
        packed=True,
        niter=niter,
        complex_dtype=complex_dtype,
    )
    with _DENSE_MATRIX_CACHE_LOCK:
        matrix = _DENSE_MATRIX_CACHE.get(key)
    if matrix is not None:
        return matrix
    if explicit:
        raise RuntimeError(
            "The dense SHT matrix must be precomputed before an explicit "
            "dense transform runs inside jax.jit. Call "
            "precompute_dense_matrix(...) once outside jax.jit."
        )
    with jax.ensure_compile_time_eval():
        return precompute_dense_matrix(
            spatial_shape, lmax, sampling, nside=nside, niter=niter
        )


def clear_dense_matrix_cache():
    """Remove all in-process dense SHT matrices from Croissant's cache."""
    with _DENSE_MATRIX_CACHE_LOCK:
        _DENSE_MATRIX_CACHE.clear()


@partial(eqx.filter_jit, inline=True)
def apply_packed_matrix(data, matrix, lmax, spatial_ndim=None):
    """
    Apply a packed dense analysis matrix and restore s2fft's layout.

    The matrix carries only the independent ``m >= 0`` coefficients, so
    the negative-m half is rebuilt from the Hermitian symmetry of a real
    field's coefficients. The result therefore has the same full layout
    ``s2fft.forward`` returns, and the route is only valid for the real
    scalar fields the packed operator was built for.

    Parameters
    ----------
    data : jax.Array
        Field data with its spatial axes trailing. Every axis before
        them is treated as a batch axis.
    matrix : jax.Array
        Packed analysis matrix from :func:`precompute_dense_matrix`,
        built for this field's spatial shape, band-limit and sampling.
    lmax : int
        Maximum spherical harmonic degree, matching ``matrix``.
    spatial_ndim : int or None
        Number of trailing spatial axes. The default ``None`` assumes a
        single leading batch axis, as ``(N_freqs, npix)`` HEALPix data
        has.

    Returns
    -------
    alm : jax.Array
        Spherical harmonic coefficients in s2fft's layout. Shape is
        ``batch_shape + (lmax + 1, 2 * lmax + 1)``.
    """
    if spatial_ndim is None:
        batch_shape = data.shape[:1]
    else:
        batch_shape = data.shape[:-spatial_ndim]
    flat_data = data.reshape((int(np.prod(batch_shape, dtype=int)), -1))
    packed = flat_data @ matrix.T

    ell, emm = _positive_lm_indices(lmax)
    alm = jnp.zeros(
        (flat_data.shape[0], lmax + 1, 2 * lmax + 1),
        dtype=packed.dtype,
    )
    alm = alm.at[:, ell, lmax + emm].set(packed)

    positive = emm > 0
    ell_neg = ell[positive]
    emm_neg = emm[positive]
    negative = ((-1) ** emm_neg)[None, :] * jnp.conj(packed[:, positive])
    alm = alm.at[:, ell_neg, lmax - emm_neg].set(negative)
    return alm.reshape(batch_shape + (lmax + 1, 2 * lmax + 1))


def _valid_lm_indices(lmax, spin):
    ell_indices = []
    m_indices = []
    offset = lmax
    for ell in range(abs(spin), lmax + 1):
        for emm in range(-ell, ell + 1):
            ell_indices.append(ell)
            m_indices.append(emm + offset)
    return np.asarray(ell_indices), np.asarray(m_indices)


def _build_analysis_matrix(
    lmax,
    sampling,
    nside,
    spin,
    niter,
    complex_dtype_name,
):
    """Materialize selected rows of corrected s2fft's linear operator."""
    complex_dtype = np.dtype(complex_dtype_name)
    spatial_shape = _spatial_shape(lmax, sampling, nside)
    ell_indices, m_indices = _valid_lm_indices(lmax, spin)
    ncoeff = ell_indices.size

    # s2fft's HEALPix FFT requires L >= 2*nside even when only lower modes
    # are retained. Build that supported operator and select the requested
    # low-l rows so low-lmax, high-nside dense analysis remains available.
    build_lmax = transform_lmax(lmax, sampling, nside=nside)
    transform_L = build_lmax + 1
    selected_m = np.asarray(
        [
            emm + build_lmax
            for ell in range(abs(spin), lmax + 1)
            for emm in range(-ell, ell + 1)
        ]
    )

    def selected_forward(data):
        full = s2fft.forward(
            data,
            L=transform_L,
            spin=spin,
            nside=nside,
            sampling=sampling,
            method="jax",
            reality=False,
            precomps=None,
            spmd=False,
            L_lower=0,
            iter=niter,
        )
        return full[ell_indices, selected_m]

    zero_map = jnp.zeros(spatial_shape, dtype=complex_dtype)
    coefficients, pullback = jax.vjp(selected_forward, zero_map)
    # s2fft may transform at a wider precision than the map it was given,
    # and a VJP only accepts cotangents in the dtype its primal output
    # actually has. Seed the basis from that dtype rather than from the
    # requested one, which is a statement about the stored matrix.
    cotangent_dtype = coefficients.dtype
    matrix = jnp.empty(
        (ncoeff, int(np.prod(spatial_shape))),
        dtype=complex_dtype,
    )
    chunk_size = 32
    for start in range(0, ncoeff, chunk_size):
        stop = min(start + chunk_size, ncoeff)
        coefficient_basis = jax.nn.one_hot(
            jnp.arange(start, stop),
            ncoeff,
            dtype=cotangent_dtype,
        )
        rows = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
            coefficient_basis
        ).reshape(stop - start, -1)
        matrix = matrix.at[start:stop].set(rows.astype(complex_dtype))
    # JAX's holomorphic VJP uses the complex transpose convention, so each
    # pulled-back coefficient basis vector is already one analysis row.
    return matrix


def _full_matrix_for(lmax, sampling, nside, spin, niter, complex_dtype):
    """Fetch the full complex operator for one configuration.

    The build runs under ``jax.ensure_compile_time_eval`` here rather
    than at the call sites, so that "nothing traced ever enters this
    cache" is a property of the cache itself. The matrix depends only
    on static geometry, so the context yields a concrete array even
    when a trace is active; without it a caller inside ``jax.jit``
    would store a tracer, and since retention is unbounded that entry
    would poison its key for the life of the process. Nesting inside a
    caller's own such context is harmless.

    The build happens while the cache lock is held, deliberately: it
    makes two threads racing on one configuration wait for a single
    build rather than each starting their own, which for the full
    operator is minutes of work apiece. Releasing the lock around the
    build would turn this into a double-checked pattern that admits
    exactly those duplicate builds.
    """
    shape = _spatial_shape(lmax, sampling, nside)
    key = _dense_matrix_key(
        shape,
        lmax,
        sampling,
        nside,
        spin=spin,
        packed=False,
        niter=niter,
        complex_dtype=complex_dtype,
    )
    with _DENSE_MATRIX_CACHE_LOCK:
        matrix = _DENSE_MATRIX_CACHE.get(key)
        if matrix is None:
            with jax.ensure_compile_time_eval():
                matrix = _build_analysis_matrix(
                    lmax,
                    sampling,
                    nside,
                    spin,
                    niter,
                    np.dtype(complex_dtype).name,
                )
            _DENSE_MATRIX_CACHE[key] = matrix
    return matrix


class DenseSphericalTransform(eqx.Module):
    """A cached dense analysis matrix differentiable with respect to maps."""

    matrix: jax.Array
    ell_indices: jax.Array
    m_indices: jax.Array
    lmax: int = eqx.field(static=True)
    sampling: str = eqx.field(static=True)
    nside: int | None = eqx.field(static=True)
    spin: int = eqx.field(static=True)
    niter: int = eqx.field(static=True)
    spatial_shape: tuple = eqx.field(static=True)

    def __init__(
        self,
        lmax,
        sampling,
        nside=None,
        spin=0,
        niter=0,
        dtype=jnp.complex128,
    ):
        dtype = np.dtype(dtype)
        if dtype.kind != "c":
            raise ValueError("Dense transform dtype must be complex.")
        with jax.ensure_compile_time_eval():
            matrix = _full_matrix_for(
                int(lmax),
                str(sampling),
                None if nside is None else int(nside),
                int(spin),
                int(niter),
                dtype,
            )
        ell_indices, m_indices = _valid_lm_indices(int(lmax), int(spin))
        spatial_shape = _spatial_shape(
            int(lmax), str(sampling), None if nside is None else int(nside)
        )
        self.matrix = jnp.asarray(matrix)
        self.ell_indices = jnp.asarray(ell_indices, dtype=jnp.int32)
        self.m_indices = jnp.asarray(m_indices, dtype=jnp.int32)
        self.lmax = int(lmax)
        self.sampling = str(sampling)
        self.nside = None if nside is None else int(nside)
        self.spin = int(spin)
        self.niter = int(niter)
        self.spatial_shape = tuple(spatial_shape)

    @jax.jit
    def __call__(self, data):
        """Apply the cached analysis matrix to arbitrary leading batches."""
        data = jnp.asarray(data)
        if data.shape[-len(self.spatial_shape) :] != self.spatial_shape:
            raise ValueError(
                f"Expected trailing spatial shape {self.spatial_shape}; "
                f"got {data.shape}."
            )
        batch_shape = data.shape[: -len(self.spatial_shape)]
        flat = data.reshape((-1, int(np.prod(self.spatial_shape))))
        valid = jnp.einsum("kn,bn->bk", self.matrix, flat)
        shape = (flat.shape[0], self.lmax + 1, 2 * self.lmax + 1)
        full = jnp.zeros(shape, dtype=valid.dtype)
        full = full.at[:, self.ell_indices, self.m_indices].set(valid)
        return full.reshape(batch_shape + (self.lmax + 1, 2 * self.lmax + 1))


def dense_compute_alm(
    data,
    lmax,
    sampling,
    nside=None,
    spin=0,
    niter=0,
    dtype=None,
):
    """Convenience wrapper around :class:`DenseSphericalTransform`."""
    if dtype is None:
        input_dtype = np.dtype(jnp.asarray(data).dtype)
        use_128 = (input_dtype.kind == "f" and input_dtype.itemsize >= 8) or (
            input_dtype.kind == "c" and input_dtype.itemsize >= 16
        )
        dtype = jnp.complex128 if use_128 else jnp.complex64
    transform = DenseSphericalTransform(
        lmax,
        sampling,
        nside=nside,
        spin=spin,
        niter=niter,
        dtype=dtype,
    )
    return transform(data)
