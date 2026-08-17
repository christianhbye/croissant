from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import s2fft

from . import dense, utils


@eqx.filter_jit
def _compute_alm_s2fft(
    data, lmax, sampling, nside=None, niter=0, spin=0, reality=False
):
    """Compute alms with the standard matrix-free s2fft engine.

    Every axis before the spatial axes is treated as a batch axis. The
    defaults are identical to s2fft's: only a caller that knows its own
    data is real may ask for the packed real transform.
    """
    data = jnp.asarray(data)
    spatial_ndim = utils.spatial_ndim(sampling)
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


def compute_alm(
    data,
    lmax,
    sampling,
    nside=None,
    niter=0,
    spin=0,
    reality=False,
    engine="auto",
    *,
    dense_matrix=None,
    kernel=None,
    inverse_kernel=None,
):
    """
    Compute the spherical harmonic coefficients of a scalar or spin field
    on the sphere. The ``"s2fft"`` engine wraps ``s2fft.forward``. The
    ``"dense"`` engine materializes that same linear transform once and
    subsequently evaluates it as a native JAX matrix multiplication.

    Every axis before the spatial axes is treated as a batch axis. The
    general complex transform is the default, as it is in s2fft; a
    caller that knows its own field is real may set ``reality=True`` to
    take the packed real optimization. For nonzero spin, or with
    ``reality=False``, the dense engine dispatches to
    :mod:`croissant.dense`, which builds the spin-weighted operator in
    the full 2D harmonic layout.

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
        Whether to use the real-valued scalar transform optimization,
        which exploits the Hermitian symmetry of a real field's
        coefficients and so costs no accuracy. This is an assertion
        about `data`, not a hint: it is rejected for complex input and
        for all nonzero-spin transforms. Default is False.
    engine : {"auto", "s2fft", "kernel", "dense"}
        Spherical harmonic transform engine. Default is ``"auto"``, which
        resolves to one of the others via
        :func:`croissant.engine_select.resolve_engine`. ``"s2fft"`` is the
        matrix-free implementation, recomputing the recursion every call.
        ``"kernel"`` caches the Wigner-d kernel and contracts it per call.
        ``"dense"`` caches the exact transform matrix and is the only
        engine able to serve a band-limit below the HEALPix floor. All
        three compute the same map to ~1e-13, so the choice is about
        memory and reuse rather than results.
    dense_matrix : jax.Array or None
        Precomputed packed dense matrix. This is primarily used internally
        by :class:`SphBase` so its jitted methods never build a matrix while
        being traced.
    kernel : jax.Array or None
        Precomputed forward Wigner-d kernel for ``engine="kernel"``, as
        returned by :func:`croissant.kernel.precompute_kernel` with
        ``forward=True``. Same jit-safety role as ``dense_matrix``, and
        passed straight through to
        :func:`croissant.kernel.kernel_compute_alm`.
    inverse_kernel : jax.Array or None
        Precomputed synthesis Wigner-d kernel for ``engine="kernel"``
        with ``niter > 0``, as returned by
        :func:`croissant.kernel.precompute_kernel` with
        ``forward=False``.

    Returns
    -------
    alm : jax.Array
        Spherical harmonic coefficients of the field. Shape is
        (len(data), lmax+1, 2*lmax+1)

    """
    data = jnp.asarray(data)
    spatial_ndim = utils.spatial_ndim(sampling)
    if data.ndim < spatial_ndim:
        raise ValueError(
            f"Data for {sampling!r} sampling must have at least "
            f"{spatial_ndim} spatial dimension(s)."
        )
    if spin != 0 and reality:
        raise ValueError("Nonzero-spin transforms require reality=False.")
    if reality and jnp.iscomplexobj(data):
        raise ValueError("Complex input requires reality=False.")

    explicit_engine = engine != "auto"
    tracing = isinstance(data, jax.core.Tracer)
    if engine == "auto":
        from .engine_select import degrade_for_trace, resolve_engine
        from .footprints import transform_lmax

        batch_size = int(np.prod(data.shape[:-spatial_ndim], dtype=int))
        engine, _ = resolve_engine(
            lmax,
            sampling,
            nside=nside,
            spin=spin,
            niter=niter,
            reality=reality,
            batch_size=batch_size,
        )
        if tracing:
            engine = degrade_for_trace(
                engine,
                has_kernel=kernel is not None,
                has_inverse_kernel=inverse_kernel is not None,
                niter=niter,
                sub_floor=(
                    transform_lmax(lmax, sampling, nside=nside) != int(lmax)
                ),
            )

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
    if engine == "kernel":
        from . import kernel as _kernel

        return _kernel.kernel_compute_alm(
            data,
            lmax,
            sampling,
            nside=nside,
            niter=niter,
            spin=spin,
            reality=reality,
            kernel=kernel,
            inverse_kernel=inverse_kernel,
        )
    if engine != "dense":
        raise ValueError(
            f"Unsupported SHT engine {engine!r}. Supported engines are "
            "{'s2fft', 'kernel', 'dense'}."
        )

    if spin != 0 or not reality:
        # Complex and spin-weighted dense analysis uses the full-layout
        # operator from croissant.dense (no packed-real optimization).
        return dense.dense_compute_alm(
            data,
            lmax,
            sampling,
            nside=nside,
            spin=spin,
            niter=niter,
        )

    if dense_matrix is None:
        dense_matrix = dense.dense_matrix_for(
            data.shape[-spatial_ndim:],
            lmax,
            sampling,
            nside=nside,
            niter=niter,
            tracing=tracing,
            explicit=explicit_engine,
        )
    return dense.apply_packed_matrix(data, dense_matrix, lmax, spatial_ndim)


class SphBase(eqx.Module):
    data: jax.Array
    freqs: jax.Array
    sampling: str = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    _L: int = eqx.field(static=True)  # L = lmax + 1 for s2fft
    _niter: int = eqx.field(static=True)  # niter for sht
    _engine: str = eqx.field(static=True)  # spherical harmonic engine
    _engine_reason: str = eqx.field(static=True)
    _dense_matrix: jax.Array | None
    _kernel: jax.Array | None
    _inverse_kernel: jax.Array | None
    nside: int | None = eqx.field(static=True)
    theta: jax.Array  # in radians
    phi: jax.Array  # in radians

    @property
    def engine(self):
        """Name of the resolved spherical harmonic transform engine.

        Reports the concrete engine that was chosen, never ``"auto"``.
        """
        return self._engine

    @property
    def engine_reason(self):
        """Why the configured engine was chosen (see engine_select)."""
        return self._engine_reason

    def __init__(
        self,
        data,
        freqs,
        sampling,
        niter=0,
        engine="auto",
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
        engine : {"auto", "s2fft", "kernel", "dense"}
            Spherical harmonic transform engine. The default ``"auto"``
            chooses from the band-limit, sampling, niter and batch size;
            the resolved choice is reported by the ``engine`` and
            ``engine_reason`` properties. ``"s2fft"`` is the matrix-free
            implementation, recomputing the recursion every call.
            ``"kernel"`` caches the Wigner-d kernel and contracts it per
            call. ``"dense"`` builds and caches an exact transform matrix,
            and is the only engine able to serve a band-limit below the
            HEALPix floor. Pin one explicitly to freeze behaviour.
        lmax : int or None
            Maximum spherical harmonic degree. For HEALPix data this may be
            set below the default ``2 * nside`` to reduce transform work and
            dense-matrix storage. Other samplings determine their band-limit
            from the input grid and do not support an override. Default is
            None.

        Raises
        ------
        ValueError
            If `engine` is not a recognized engine name (checked before
            any data or frequency processing), or if `sampling` is
            "healpix" and the number of pixels in `data` is not valid
            for healpix sampling.

        """
        from .engine_select import validate_engine

        validate_engine(engine)
        # Captured before resolve_engine overwrites `engine` below: only
        # an automatic choice may be degraded when a precompute turns out
        # to be impossible, and by then the caller's own request is gone.
        explicit_engine = engine != "auto"

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

        from .engine_select import resolve_engine

        engine, engine_reason = resolve_engine(
            self.lmax,
            self.sampling,
            nside=self.nside,
            niter=self._niter,
            # Beam and Sky are real fields and transform with
            # reality=True, so the engine must be chosen by sizing the
            # packed operator that will actually be built, not the full
            # one compute_alm assumes when it is told nothing.
            reality=True,
            batch_size=int(self.data.shape[0]),
            requested=engine,
        )
        self._engine = engine
        self._engine_reason = engine_reason

        tracing = isinstance(self.data, jax.core.Tracer)
        if self._engine == "kernel" and tracing:
            # A kernel cannot be built while a trace is active: converting
            # the numpy-built kernel to a jax.Array would yield a tracer
            # bound to this trace, which the module-level cache must never
            # retain (see kernel.precompute_kernel's docstring).
            if explicit_engine:
                raise RuntimeError(
                    "The kernel must be precomputed before a kernel "
                    "SphBase object is constructed inside jax.jit. Call "
                    "precompute_kernel(...) once outside jax.jit."
                )
            # Constructing a field inside a trace is how a caller
            # differentiates through the construction itself, so raising
            # here would cost them that for a choice they never made.
            # Only cost changes: the engines agree to ~1e-13.
            from .engine_select import degrade_for_trace
            from .footprints import transform_lmax

            self._engine = degrade_for_trace(
                self._engine,
                niter=self._niter,
                sub_floor=(
                    transform_lmax(self.lmax, self.sampling, nside=self.nside)
                    != self.lmax
                ),
            )
            self._engine_reason = (
                "kernels cannot be built inside a jax trace; degraded "
                "from the automatic choice"
            )

        if self._engine == "dense":
            self._dense_matrix = dense.dense_matrix_for(
                self.data.shape[1:],
                self.lmax,
                self.sampling,
                nside=self.nside,
                niter=self._niter,
                tracing=tracing,
                explicit=explicit_engine,
            )
            self._kernel = None
            self._inverse_kernel = None
        elif self._engine == "kernel":
            self._dense_matrix = None
            # Reached only outside a trace: the degrade-or-raise check
            # above has already dealt with the traced case, so the built
            # kernel is always a concrete array, never a leaked tracer
            # (see kernel.precompute_kernel's docstring). Unlike dense,
            # there is no per-key cache fallback for a traced build --
            # an explicit engine="kernel" must precompute first, exactly
            # as kernel_compute_alm requires when called inside a trace.
            from . import kernel as _kernel

            self._kernel = _kernel.precompute_kernel(
                self.lmax,
                self.sampling,
                nside=self.nside,
                spin=0,
                reality=True,
                forward=True,
            )
            if self._niter > 0:
                self._inverse_kernel = _kernel.precompute_kernel(
                    self.lmax,
                    self.sampling,
                    nside=self.nside,
                    spin=0,
                    reality=True,
                    forward=False,
                )
            else:
                self._inverse_kernel = None
        else:
            self._dense_matrix = None
            self._kernel = None
            self._inverse_kernel = None

        self.phi = utils.generate_phi(
            lmax=self.lmax, sampling=self.sampling, nside=self.nside
        )
        self.theta = utils.generate_theta(
            lmax=self.lmax, sampling=self.sampling, nside=self.nside
        )
