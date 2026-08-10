from functools import partial
from threading import RLock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import s2fft

from . import utils

_DENSE_MATRIX_CACHE = {}
_DENSE_MATRIX_CACHE_LOCK = RLock()


def _dense_dtypes():
    """Return the real and complex dtypes of a dense SHT matrix.

    The dense engine reproduces ``s2fft.forward``, which always outputs
    complex128 alms on an x64-enabled runtime (float32 maps included) and
    complex64 otherwise. The matrix precision therefore follows JAX's x64
    setting rather than the dtype of the input maps.
    """
    if jax.config.x64_enabled:
        return jnp.float64, jnp.complex128
    return jnp.float32, jnp.complex64


def _dense_matrix_key(spatial_shape, lmax, sampling, nside, niter):
    """Return a hashable key for a cached dense SHT analysis matrix."""
    _, complex_dtype = _dense_dtypes()
    return (
        tuple(spatial_shape),
        int(lmax),
        str(sampling),
        None if nside is None else int(nside),
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


@eqx.filter_jit
def _compute_alm_s2fft(
    data, lmax, sampling, nside=None, niter=0, spin=0, reality=True
):
    """Compute alms with the standard matrix-free s2fft engine.

    Every axis before the spatial axes is treated as a batch axis. The
    scalar defaults are identical to the original Croissant API.
    """
    data = jnp.asarray(data)
    spatial_ndim = 1 if sampling == "healpix" else 2
    spatial_shape = data.shape[-spatial_ndim:]
    batch_shape = data.shape[:-spatial_ndim]
    flat_data = data.reshape((-1,) + spatial_shape)
    m2alm = partial(
        s2fft.forward,
        L=lmax + 1,
        spin=spin,
        nside=nside,
        sampling=sampling,
        method="jax",
        reality=reality,
        precomps=None,
        spmd=False,
        L_lower=0,
        iter=niter,
    )
    flat_alm = jax.vmap(m2alm)(flat_data)
    return flat_alm.reshape(batch_shape + (lmax + 1, 2 * lmax + 1))


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
    _, complex_dtype = _dense_dtypes()
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
    real_dtype, _ = _dense_dtypes()
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
        dense = _compute_alm_s2fft(
            basis,
            lmax,
            sampling,
            nside=nside,
            niter=niter,
        )
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
    key = _dense_matrix_key(spatial_shape, lmax, sampling, nside, niter)
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


def clear_dense_matrix_cache():
    """Remove all in-process dense SHT matrices from Croissant's cache."""
    with _DENSE_MATRIX_CACHE_LOCK:
        _DENSE_MATRIX_CACHE.clear()


@partial(eqx.filter_jit, inline=True)
def _apply_dense_matrix(data, matrix, lmax, spatial_ndim=None):
    """Apply a packed dense analysis matrix and restore s2fft's layout."""
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


def compute_alm(
    data,
    lmax,
    sampling,
    nside=None,
    niter=0,
    spin=0,
    reality=True,
    engine="s2fft",
    *,
    dense_matrix=None,
):
    """
    Compute the spherical harmonic coefficients of a scalar or spin field
    on the sphere. The default ``"s2fft"`` engine wraps ``s2fft.forward``.
    The ``"dense"`` engine materializes that same linear transform once and
    subsequently evaluates it as a native JAX matrix multiplication.

    Every axis before the spatial axes is treated as a batch axis. For
    nonzero spin (or complex scalar input) set ``reality=False``; the dense
    engine then dispatches to :mod:`croissant.dense`, which builds the
    spin-weighted operator in the full 2D harmonic layout.

    Parameters
    ----------
    data : array_like
        Field data. First axis is frequency, second axis is theta
        (colatitude), and third axis is phi (longitude). If `sampling`
        is "healpix", the data only has two dimensions: frequency and
        pixel index.
    lmax : int
        Maximum spherical harmonic degree to compute.
    sampling : str
        Sampling scheme of the field data. Supported schemes are determined
        by s2fft, currently they include {"mw", "mwss", "dh", "gl",
        "healpix"}.
    nside : int or None,
        Nside parameter for healpix sampling. Required if `sampling` is
        "healpix". Ignored otherwise.
    niter : int
        Number of iterations for the s2fft algorithm. Higher values can
        improve accuracy at the cost of increased computation time.
        Default is 0, which corresponds to the default behavior of
        s2fft.
    spin : int
        Spin weight of the input field. Default is 0.
    reality : bool
        Whether to use the real-valued scalar transform optimization.
        Set to False for complex inputs and all nonzero-spin transforms.
    engine : {"s2fft", "dense"}
        Spherical harmonic transform engine. ``"s2fft"`` is the existing
        matrix-free implementation. ``"dense"`` caches the exact transform
        matrix and is intended for low band-limits.
    dense_matrix : jax.Array or None
        Precomputed packed dense matrix. This is primarily used internally
        by :class:`SphBase` so its jitted methods never build a matrix while
        being traced.

    Returns
    -------
    alm : jax.Array
        Spherical harmonic coefficients of the field. Shape is
        (len(data), lmax+1, 2*lmax+1)

    """
    data = jnp.asarray(data)
    spatial_ndim = 1 if sampling == "healpix" else 2
    if data.ndim < spatial_ndim:
        raise ValueError(
            f"Data for {sampling!r} sampling must have at least "
            f"{spatial_ndim} spatial dimension(s)."
        )
    if spin != 0 and reality:
        raise ValueError("Nonzero-spin transforms require reality=False.")

    if engine == "s2fft":
        return _compute_alm_s2fft(
            data,
            lmax,
            sampling,
            nside=nside,
            niter=niter,
            spin=spin,
            reality=reality,
        )
    if engine != "dense":
        raise ValueError(
            f"Unsupported SHT engine {engine!r}. Supported engines are "
            "{'s2fft', 'dense'}."
        )

    if spin != 0 or not reality:
        # Complex and spin-weighted dense analysis uses the full-layout
        # operator from croissant.dense (no packed-real optimization).
        from . import dense as _dense

        return _dense.dense_compute_alm(
            data,
            lmax,
            sampling,
            nside=nside,
            spin=spin,
            niter=niter,
        )

    if dense_matrix is None:
        spatial_shape = tuple(data.shape[-spatial_ndim:])
        key = _dense_matrix_key(
            spatial_shape, lmax, sampling, nside, niter, data.dtype
        )
        if isinstance(data, jax.core.Tracer):
            with _DENSE_MATRIX_CACHE_LOCK:
                dense_matrix = _DENSE_MATRIX_CACHE.get(key)
            if dense_matrix is None:
                raise RuntimeError(
                    "The dense SHT matrix must be precomputed before "
                    "compute_alm is called inside jax.jit. Call "
                    "precompute_dense_matrix(...) once outside jax.jit."
                )
        else:
            dense_matrix = precompute_dense_matrix(
                spatial_shape,
                lmax,
                sampling,
                nside=nside,
                niter=niter,
            )
    return _apply_dense_matrix(data, dense_matrix, lmax, spatial_ndim)


class SphBase(eqx.Module):
    data: jax.Array
    freqs: jax.Array
    sampling: str = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    _L: int = eqx.field(static=True)  # L = lmax + 1 for s2fft
    _niter: int = eqx.field(static=True)  # niter for sht
    _engine: str = eqx.field(static=True)  # spherical harmonic engine
    _dense_matrix: jax.Array | None
    nside: int | None = eqx.field(static=True)
    theta: jax.Array  # in radians
    phi: jax.Array  # in radians

    @property
    def engine(self):
        """Name of the configured spherical harmonic transform engine."""
        return self._engine

    def __init__(
        self,
        data,
        freqs,
        sampling,
        niter=0,
        engine="s2fft",
        lmax=None,
    ):
        """
        Base class for scalar fields on the sphere. Holds the field
        data and associated metadata. The field must be defined on the
        grid specified by the `sampling` scheme.

        Parameters
        ----------
        data : array_like
            Field data. First axis is frequency, second axis is theta
            (colatitude), and third axis is phi (longitude). If
            `sampling` is "healpix", the data only has two dimensions:
            frequency and pixel index.
        freqs : array_like
            Frequencies corresponding to the field data.
        sampling : str
            Sampling scheme of the field data. Supported schemes are
            determined by s2fft, currently they include {"mw", "mwss",
            "dh", "gl", "healpix"}. The default is "mwss", which is a 1
            deg equiangular sampling in theta and phi and includes the
            poles.
        niter : int
            Number of iterations for the s2fft algorithm. Higher values
            can improve accuracy at the cost of increased computation
            time. Default is 0 for all sampling schemes. For healpix
            sampling, setting niter=3 improves accuracy but
            significantly increases JIT compile time.
        engine : {"s2fft", "dense"}
            Spherical harmonic transform engine. The default ``"s2fft"``
            preserves the existing matrix-free behavior. ``"dense"`` builds
            and caches an exact transform matrix for fast repeated low-lmax
            transforms.
        lmax : int or None
            Maximum spherical harmonic degree. For HEALPix data this may be
            set below the default ``2 * nside`` to reduce transform work and
            dense-matrix storage. Other samplings determine their band-limit
            from the input grid and do not support an override. Default is
            None.

        Raises
        ------
        ValueError
            If `sampling` is "healpix" and the number of pixels in
            `data` is not valid for healpix sampling.

        """
        if engine not in {"s2fft", "dense"}:
            raise ValueError(
                f"Unsupported SHT engine {engine!r}. Supported engines are "
                "{'s2fft', 'dense'}."
            )

        self.data = jnp.asarray(data)
        self.freqs = jnp.atleast_1d(freqs)

        if sampling == "healpix":
            npix = self.data.shape[1]
            if not utils.hp_valid_npix(npix):
                raise ValueError(
                    f"Invalid number of pixels {npix} for healpix sampling. "
                    "Number of pixels must be of the form 12 * nside^2."
                )

        self._niter = niter
        self._engine = engine

        self.sampling = sampling
        inferred_lmax = utils.lmax_from_ntheta(
            self.data.shape[1], self.sampling
        )
        if lmax is None:
            self.lmax = inferred_lmax
        else:
            lmax = int(lmax)
            if lmax < 0:
                raise ValueError("lmax must be non-negative")
            if self.sampling != "healpix" and lmax != inferred_lmax:
                raise ValueError(
                    "An explicit lmax different from the grid band-limit is "
                    "only supported for HEALPix sampling."
                )
            self.lmax = lmax
        self._L = self.lmax + 1  # for s2fft, L = lmax + 1

        if self.sampling == "healpix":
            self.nside = utils.hp_npix2nside(self.data.shape[1])
        else:
            self.nside = None

        if self._engine == "dense":
            if isinstance(self.data, jax.core.Tracer):
                key = _dense_matrix_key(
                    self.data.shape[1:],
                    self.lmax,
                    self.sampling,
                    self.nside,
                    self._niter,
                )
                with _DENSE_MATRIX_CACHE_LOCK:
                    self._dense_matrix = _DENSE_MATRIX_CACHE.get(key)
                if self._dense_matrix is None:
                    raise RuntimeError(
                        "The dense SHT matrix must be precomputed before a "
                        "dense SphBase object is constructed inside jax.jit. "
                        "Call precompute_dense_matrix(...) once outside "
                        "jax.jit."
                    )
            else:
                self._dense_matrix = precompute_dense_matrix(
                    self.data.shape[1:],
                    self.lmax,
                    self.sampling,
                    nside=self.nside,
                    niter=self._niter,
                )
        else:
            self._dense_matrix = None

        self.phi = utils.generate_phi(
            lmax=self.lmax, sampling=self.sampling, nside=self.nside
        )
        self.theta = utils.generate_theta(
            lmax=self.lmax, sampling=self.sampling, nside=self.nside
        )
