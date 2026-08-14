"""Full-Stokes transforms and component-aware convolution.

The public convention is IAU with Stokes order I, Q, U, V. Internally the
Q/U block is represented by the two harmonic dual pairs needed to preserve a
single diagonal harmonic contraction for arbitrary complex response beams.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import dense, rotations, sphere, utils

STOKES_IQUV = ("I", "Q", "U", "V")
POLARIZATION_COMPONENTS = ("I", "V", "P_MINUS", "P_PLUS")
POLARIZATION_SPINS = (0, 0, -2, 2)
POLARIZATION_CONVENTIONS = ("IAU", "COSMO")


def _normalize_convention(convention):
    normalized = str(convention).upper()
    if normalized not in POLARIZATION_CONVENTIONS:
        raise ValueError(
            f"Unsupported polarization convention {convention!r}; expected "
            f"one of {POLARIZATION_CONVENTIONS}."
        )
    return normalized


def _validate_stokes(stokes):
    value = tuple(stokes)
    if value != STOKES_IQUV:
        raise ValueError(
            f"Stokes order must be exactly {STOKES_IQUV}; got {value}."
        )
    return value


def _spatial_metadata(data, freqs, sampling, niter):
    data = jnp.asarray(data)
    freqs_np = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if freqs_np.size == 0 or not np.all(np.isfinite(freqs_np)):
        raise ValueError(
            "freqs must be a nonempty finite one-dimensional array."
        )
    if freqs_np.size > 1 and not np.all(np.diff(freqs_np) > 0):
        raise ValueError("freqs must be strictly increasing.")

    spatial_ndim = utils.spatial_ndim(sampling)
    if data.ndim < spatial_ndim + 2:
        raise ValueError(
            "Polarized data must include frequency, Stokes, and spatial axes."
        )
    ntheta_or_npix = data.shape[-spatial_ndim]
    if sampling == "healpix":
        if not utils.hp_valid_npix(ntheta_or_npix):
            raise ValueError(
                f"Invalid number of HEALPix pixels: {ntheta_or_npix}."
            )
        nside = utils.hp_npix2nside(ntheta_or_npix)
    else:
        nside = None
    if niter is None:
        niter = 0
    lmax = utils.lmax_from_ntheta(ntheta_or_npix, sampling)
    theta = utils.generate_theta(lmax=lmax, sampling=sampling, nside=nside)
    phi = utils.generate_phi(lmax=lmax, sampling=sampling, nside=nside)
    return (
        data,
        jnp.asarray(freqs_np),
        nside,
        int(niter),
        lmax,
        theta,
        phi,
    )


def convert_stokes_convention(data, source, target, stokes_axis=-1):
    """Convert Stokes values or response coefficients between IAU and COSMO.

    I and Q are unchanged and U changes sign. Applying the same diagonal
    transformation to a response vector is the required contragredient
    conversion, so the physical Stokes-response contraction is invariant.
    Stokes V is intentionally unchanged here because circular-polarization
    sign conventions are independent of the IAU/COSMO U convention.
    """
    source = _normalize_convention(source)
    target = _normalize_convention(target)
    array = jnp.asarray(data)
    axis = stokes_axis % array.ndim
    if array.shape[axis] != 4:
        raise ValueError(
            f"Stokes axis must have length 4; got shape {array.shape} and "
            f"axis {stokes_axis}."
        )
    if source == target:
        return array
    signs = jnp.asarray([1.0, 1.0, -1.0, 1.0], dtype=array.real.dtype)
    shape = [1] * array.ndim
    shape[axis] = 4
    return array * signs.reshape(shape)


def iau_to_cosmo(data, stokes_axis=-1):
    """Convert IAU-ordered IQUV data to the COSMO U-sign convention."""
    return convert_stokes_convention(
        data, "IAU", "COSMO", stokes_axis=stokes_axis
    )


def cosmo_to_iau(data, stokes_axis=-1):
    """Convert COSMO-ordered IQUV data to the IAU U-sign convention."""
    return convert_stokes_convention(
        data, "COSMO", "IAU", stokes_axis=stokes_axis
    )


#: The three distinct transform configurations a polarized field runs.
#: The spin-0 block carries I and V together; the two spin-weighted
#: blocks carry the Q/U duals. Resolved engines and the objects they
#: precompute are stored as tuples in this order.
_BLOCK_NAMES = ("IV", "P_MINUS", "P_PLUS")
_BLOCK_SPINS = (0, *POLARIZATION_SPINS[2:])
_IV, _P_MINUS, _P_PLUS = range(3)


def _block_reality(spin, spin0_reality):
    """Whether one block's transform is of a real field.

    Only the spin-0 block can be real, and only for a sky: a pair
    response is complex in every Stokes component. ``kernel_compute_alm``
    applies the same rule, and a kernel must be built with the same
    ``reality`` it is applied with, so the two must not drift apart.
    """
    return bool(spin0_reality) if spin == 0 else False


def _analysis_alm(
    data,
    target_lmax,
    native_lmax,
    sampling,
    nside,
    niter,
    *,
    spin,
    reality,
    engine,
    kernel=None,
    inverse_kernel=None,
    dense_matrix=None,
):
    """Analyse one polarized block with its own resolved engine.

    The precomputed objects are built once per field by
    :func:`_prepare_engines` and threaded in, because this function runs
    inside the jitted ``compute_alm`` methods and the kernel and dense
    engines both refuse to build while a trace is active.
    """
    if sampling == "healpix" and target_lmax < native_lmax:
        # Below the HEALPix L >= 2*nside - 1 floor no kernel can be
        # built, so this branch stays on croissant.dense regardless of
        # the engine resolved for the block: dense is the only engine
        # that can serve a band-limit under the floor, by building at
        # the floor and keeping the low-ell rows. Nothing threaded in
        # applies here; croissant.dense caches its own operator.
        return dense.dense_compute_alm(
            data,
            target_lmax,
            sampling,
            nside=nside,
            spin=spin,
            niter=niter,
        )
    result = sphere.compute_alm(
        data,
        native_lmax,
        sampling,
        nside=nside,
        niter=niter,
        spin=spin,
        reality=reality,
        engine=engine,
        dense_matrix=dense_matrix,
        kernel=kernel,
        inverse_kernel=inverse_kernel,
    )
    return utils.reduce_lmax(result, target_lmax)


def _prepare_engines(
    data,
    lmax,
    sampling,
    nside,
    niter,
    *,
    spatial_shape,
    batch_sizes,
    spin0_reality,
    requested,
):
    """
    Resolve one engine per transform block and precompute what it needs.

    A polarized field is not one transform but three, and they do not
    share a footprint: the spin-0 kernel of a real sky is packed to
    ``m >= 0`` and so is roughly half the size of a spin-weighted one,
    and the blocks are batched over different numbers of maps. Resolving
    once for the whole field would have to nominate one block as
    representative and would then be wrong for the others, so each block
    is resolved on its own terms. An explicit ``requested`` engine
    passes through :func:`~croissant.engine_select.resolve_engine`
    unchanged and therefore pins every block.

    Parameters
    ----------
    data : array_like
        The field data, consulted only to detect an active trace.
    lmax : int
        Native band-limit of the field; kernels are always built here,
        since :func:`_analysis_alm` transforms at the native band-limit
        and truncates afterwards.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    niter : int
        Number of iterative refinement steps; ``> 0`` additionally needs
        a synthesis kernel per kernel-engine block.
    spatial_shape : tuple of int
        Shape of one map, excluding all batch axes.
    batch_sizes : tuple
        Number of maps each block transforms, in ``_BLOCK_NAMES`` order.
        ``None`` marks a block the field never transforms, which is the
        P+ block on the samplings where conjugation supplies it.
    spin0_reality : bool
        Whether the spin-0 block is a real field: True for a sky, False
        for a complex pair response.
    requested : str
        Engine name from the caller, or ``"auto"``. Validated here with
        the same rule :class:`croissant.sphere.SphBase` applies.

    Returns
    -------
    tuple of tuple
        ``(engines, reasons, kernels, inverse_kernels, dense_matrices)``,
        each with one entry per block.

    Raises
    ------
    ValueError
        If ``requested`` is not a recognized engine name.
    RuntimeError
        If an explicitly requested precomputing engine cannot build
        because a trace is active.

    """
    from . import kernel as _kernel
    from .engine_select import (
        degrade_for_trace,
        resolve_engine,
        validate_engine,
    )
    from .footprints import transform_lmax

    validate_engine(requested)
    tracing = isinstance(data, jax.core.Tracer)
    explicit = requested != "auto"
    engines, reasons = [], []
    kernels, inverse_kernels, dense_matrices = [], [], []

    for spin, batch in zip(_BLOCK_SPINS, batch_sizes):
        if batch is None:
            engines.append(None)
            reasons.append("not transformed on this sampling")
            kernels.append(None)
            inverse_kernels.append(None)
            dense_matrices.append(None)
            continue

        reality = _block_reality(spin, spin0_reality)
        engine, reason = resolve_engine(
            lmax,
            sampling,
            nside=nside,
            spin=spin,
            niter=niter,
            reality=reality,
            batch_size=batch,
            requested=requested,
        )

        if engine == "kernel" and tracing:
            if explicit:
                raise RuntimeError(
                    "The kernel must be precomputed before a kernel "
                    "polarized field is constructed inside jax.jit. Call "
                    "precompute_kernel(...) once outside jax.jit."
                )
            # Constructing a polarized field inside a trace worked before
            # these fields became engine-selectable, so degrade rather
            # than break it; only cost changes, since the engines agree
            # to ~1e-13. An explicit request is never softened.
            engine = degrade_for_trace(
                engine,
                niter=niter,
                sub_floor=(
                    transform_lmax(lmax, sampling, nside=nside) != int(lmax)
                ),
            )
            reason = (
                "kernels cannot be built inside a jax trace; "
                "degraded from the automatic choice"
            )

        kernel = inverse_kernel = dense_matrix = None
        if engine == "kernel":
            kernel = _kernel.precompute_kernel(
                lmax,
                sampling,
                nside=nside,
                spin=spin,
                reality=reality,
                forward=True,
            )
            if niter > 0:
                inverse_kernel = _kernel.precompute_kernel(
                    lmax,
                    sampling,
                    nside=nside,
                    spin=spin,
                    reality=reality,
                    forward=False,
                )
        elif engine == "dense" and spin == 0 and reality:
            # The one block that takes sphere.py's packed-real dense
            # route, which needs its matrix threaded in. Every other
            # block is complex or spin-weighted, so sphere.compute_alm
            # routes it to croissant.dense, which builds under
            # jax.ensure_compile_time_eval and needs nothing from here.
            dense_matrix = sphere.dense_matrix_for(
                spatial_shape,
                lmax,
                sampling,
                nside=nside,
                niter=niter,
                tracing=tracing,
                explicit=explicit,
            )

        engines.append(engine)
        reasons.append(reason)
        kernels.append(kernel)
        inverse_kernels.append(inverse_kernel)
        dense_matrices.append(dense_matrix)

    return (
        tuple(engines),
        tuple(reasons),
        tuple(kernels),
        tuple(inverse_kernels),
        tuple(dense_matrices),
    )


def _block_kwargs(engines, kernels, inverse_kernels, dense_matrices):
    """Per-block keyword arguments for :func:`_analysis_alm`."""
    return tuple(
        {
            "engine": engine,
            "kernel": kernel,
            "inverse_kernel": inverse_kernel,
            "dense_matrix": dense_matrix,
        }
        for engine, kernel, inverse_kernel, dense_matrix in zip(
            engines, kernels, inverse_kernels, dense_matrices
        )
    )


# Samplings whose forward transform is a plain quadrature (real weights
# on a fixed pixel set), which commutes with complex conjugation of the
# input map. The mw/mwss sampling-theorem transforms alias out-of-band
# power asymmetrically between spins +2 and -2, so for generic pixel
# data the conjugation identity below fails there at order unity near
# the band limit and they must keep the explicit second transform.
_CONJUGATE_EXACT_SAMPLINGS = frozenset({"healpix", "dh", "gl"})


def _conjugate_spin_flip(alm, lmax):
    """Spin +2 analysis of conj(f) from the spin -2 analysis of f.

    For any field f with spin -2 coefficients a_lm, the spin +2
    analysis of its complex conjugate is b_lm = (-1)^m conj(a_l,-m),
    by the conjugation relation sY*_lm = (-1)^(s+m) -sY_l,-m. For a
    real-pixel IQUV sky this yields the P_PLUS dual directly from the
    P_MINUS one.
    """
    emms = jnp.arange(-lmax, lmax + 1)
    phase = jnp.where(emms % 2 == 0, 1.0, -1.0)
    return phase * jnp.conjugate(alm[..., ::-1])


def _qu_spin_combination(stokes_q, stokes_u, spin):
    """Return the complex Q/U combination that carries ``spin``.

    With croissant's internal IAU ``U``, the spin -2 object is
    ``Q + iU`` and the spin +2 object is ``Q - iU`` in s2fft's
    Goldberg basis (see docs/polarization.md). Analyzing each
    combination at its physical spin keeps the duals of band-limited
    E/B skies band-limited and makes Wigner-D frame rotation apply
    the physical transport phase (a mismatched label applies its
    complex conjugate). The same pairing applies to a Q/U response
    vector, so sky and response duals stay in lockstep by
    construction.
    """
    if spin == -2:
        return stokes_q + 1j * stokes_u
    if spin == 2:
        return stokes_q - 1j * stokes_u
    raise ValueError(f"No Q/U combination carries spin {spin}.")


def _compute_sky_dual_alm(
    data,
    target_lmax,
    native_lmax,
    sampling,
    nside,
    niter,
    blocks,
):
    """Transform IQUV sky maps to the harmonic contraction dual."""
    stokes_i = data[:, 0]
    stokes_q = data[:, 1]
    stokes_u = data[:, 2]
    stokes_v = data[:, 3]
    spin0 = _analysis_alm(
        jnp.stack((stokes_i, stokes_v), axis=1),
        target_lmax,
        native_lmax,
        sampling,
        nside,
        niter,
        spin=0,
        reality=True,
        **blocks[_IV],
    )
    # The P_MINUS and P_PLUS slots, in POLARIZATION_COMPONENTS order.
    p_minus_spin, p_plus_spin = POLARIZATION_SPINS[2:]
    p_minus = _analysis_alm(
        _qu_spin_combination(stokes_q, stokes_u, p_minus_spin),
        target_lmax,
        native_lmax,
        sampling,
        nside,
        niter,
        spin=p_minus_spin,
        reality=False,
        **blocks[_P_MINUS],
    )
    if sampling in _CONJUGATE_EXACT_SAMPLINGS:
        # Real sky pixels make the P_PLUS input the exact conjugate of
        # the P_MINUS input, and on quadrature samplings the discrete
        # analysis commutes with that conjugation, so the second spin-2
        # transform (and, on the dense HEALPix path, its whole cached
        # analysis matrix) is redundant.
        p_plus = _conjugate_spin_flip(p_minus, target_lmax)
    else:
        p_plus = _analysis_alm(
            _qu_spin_combination(stokes_q, stokes_u, p_plus_spin),
            target_lmax,
            native_lmax,
            sampling,
            nside,
            niter,
            spin=p_plus_spin,
            reality=False,
            **blocks[_P_PLUS],
        )
    return jnp.stack((spin0[:, 0], spin0[:, 1], p_minus, p_plus), axis=1)


def _compute_response_dual_alm(
    data,
    target_lmax,
    native_lmax,
    sampling,
    nside,
    niter,
    blocks,
):
    """Transform complex pair-IQUV maps to their harmonic response dual."""
    response_i = data[:, :, 0]
    response_q = data[:, :, 1]
    response_u = data[:, :, 2]
    response_v = data[:, :, 3]
    spin0 = _analysis_alm(
        jnp.stack((response_i, response_v), axis=2),
        target_lmax,
        native_lmax,
        sampling,
        nside,
        niter,
        spin=0,
        reality=False,
        **blocks[_IV],
    )
    # Each polarized slot carries half the response combination at its
    # spin, which the conjugated einsum contracts with the same-spin
    # sky analysis, reproducing the physical integral BQ*Q + BU*U.
    polarized = [
        _analysis_alm(
            0.5 * _qu_spin_combination(response_q, response_u, spin),
            target_lmax,
            native_lmax,
            sampling,
            nside,
            niter,
            spin=spin,
            reality=False,
            **block,
        )
        for spin, block in zip(
            POLARIZATION_SPINS[2:], blocks[_P_MINUS : _P_PLUS + 1]
        )
    ]
    return jnp.stack(
        (spin0[:, :, 0], spin0[:, :, 1], *polarized),
        axis=2,
    )


class PolarizedSky(eqx.Module):
    """A real IQUV sky on one supported spherical sampling grid."""

    data: jax.Array
    freqs: jax.Array
    sampling: str = eqx.field(static=True)
    coord: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    stokes: tuple = eqx.field(static=True)
    units: str = eqx.field(static=True)
    frame: str = eqx.field(static=True)
    tangent_basis: str = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    _L: int = eqx.field(static=True)
    _niter: int = eqx.field(static=True)
    _engines: tuple = eqx.field(static=True)
    _engine_reasons: tuple = eqx.field(static=True)
    _kernels: tuple
    _inverse_kernels: tuple
    _dense_matrices: tuple
    nside: int | None = eqx.field(static=True)
    theta: jax.Array
    phi: jax.Array

    @property
    def engine(self):
        """Resolved transform engine for each block, by block name.

        A polarized field resolves an engine per transform block rather
        than one for the whole object, so this is a mapping where the
        scalar fields' ``engine`` is a single string.
        """
        return dict(zip(_BLOCK_NAMES, self._engines))

    @property
    def engine_reason(self):
        """Why each block's engine was chosen (see engine_select)."""
        return dict(zip(_BLOCK_NAMES, self._engine_reasons))

    def __init__(
        self,
        data,
        freqs,
        sampling="healpix",
        coord="galactic",
        convention="IAU",
        stokes=STOKES_IQUV,
        units="K",
        frame=None,
        tangent_basis="theta-phi",
        niter=0,
        engine="auto",
    ):
        convention = _normalize_convention(convention)
        self.stokes = _validate_stokes(stokes)
        if coord not in {"galactic", "equatorial", "mepa", "topo"}:
            raise ValueError(f"Unsupported coordinate system: {coord}.")
        data = jnp.asarray(data)
        if not jnp.issubdtype(data.dtype, jnp.floating):
            raise ValueError("PolarizedSky pixel data must be real-valued.")
        if data.shape[1] != 4:
            raise ValueError(
                "PolarizedSky data must have shape (frequency, 4, spatial...)."
            )
        if convention == "COSMO":
            data = cosmo_to_iau(data, stokes_axis=1)
            convention = "IAU"
        (
            self.data,
            self.freqs,
            self.nside,
            self._niter,
            self.lmax,
            self.theta,
            self.phi,
        ) = _spatial_metadata(data, freqs, sampling, niter)
        if self.data.shape[0] != self.freqs.size:
            raise ValueError("Frequency axis length does not match freqs.")
        self.sampling = sampling
        self.coord = coord
        self.convention = convention
        self.units = str(units)
        self.frame = str(frame if frame is not None else coord)
        self.tangent_basis = str(tangent_basis)
        self._L = self.lmax + 1
        nfreq = int(self.data.shape[0])
        (
            self._engines,
            self._engine_reasons,
            self._kernels,
            self._inverse_kernels,
            self._dense_matrices,
        ) = _prepare_engines(
            self.data,
            self.lmax,
            self.sampling,
            self.nside,
            self._niter,
            spatial_shape=tuple(self.data.shape[2:]),
            batch_sizes=(
                2 * nfreq,  # I and V transform together
                nfreq,
                # Conjugation supplies P+ from P- on these samplings, so
                # no transform and no kernel for that block.
                None if sampling in _CONJUGATE_EXACT_SAMPLINGS else nfreq,
            ),
            spin0_reality=True,
            requested=engine,
        )

    @partial(jax.jit, static_argnames=("lmax",))
    def compute_alm(self, lmax=None):
        """Return the I, V, P-, P+ harmonic dual up to ``lmax``."""
        target_lmax = self.lmax if lmax is None else int(lmax)
        if target_lmax < 0 or target_lmax > self.lmax:
            raise ValueError(
                f"lmax must lie in [0, {self.lmax}]; got {target_lmax}."
            )
        return _compute_sky_dual_alm(
            self.data,
            target_lmax,
            self.lmax,
            self.sampling,
            self.nside,
            self._niter,
            _block_kwargs(
                self._engines,
                self._kernels,
                self._inverse_kernels,
                self._dense_matrices,
            ),
        )

    def compute_alm_eq(self, world="moon", et=None):
        """Return the contraction dual in the requested equatorial frame."""
        if world not in {"moon", "earth"}:
            raise ValueError("world must be either 'moon' or 'earth'.")
        if self.coord == "topo":
            raise ValueError(
                "A topocentric sky cannot be transported by compute_alm_eq() "
                "without a concrete observer location and reference epoch; "
                "use compute_alm() to keep it in the local frame."
            )
        alm = self.compute_alm()
        if self.coord != "galactic":
            expected = "mepa" if world == "moon" else "equatorial"
            if self.coord != expected:
                raise ValueError(
                    f"Unsupported coordinate transformation: "
                    f"{self.coord} to {world}."
                )
            return alm
        if world == "moon":
            euler, dl_array = rotations.generate_euler_dl(
                self.lmax, "galactic", "mepa", et=et
            )
        else:
            euler, dl_array = rotations.generate_euler_dl(
                self.lmax, "galactic", "fk5"
            )
        return rotations.rotate_alm(alm, euler, dl_array=dl_array)


class PairStokesBeam(eqx.Module):
    """Complex pair-response maps.

    Layout is pair, frequency, IQUV, spatial.
    """

    data: jax.Array
    freqs: jax.Array
    pairs: tuple = eqx.field(static=True)
    sampling: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    stokes: tuple = eqx.field(static=True)
    units: str = eqx.field(static=True)
    frame: str = eqx.field(static=True)
    tangent_basis: str = eqx.field(static=True)
    baseline_direction: str = eqx.field(static=True)
    visibility_definition: str = eqx.field(static=True)
    beam_rot: jax.Array
    horizon: jax.Array
    lmax: int = eqx.field(static=True)
    _L: int = eqx.field(static=True)
    _niter: int = eqx.field(static=True)
    _engines: tuple = eqx.field(static=True)
    _engine_reasons: tuple = eqx.field(static=True)
    _kernels: tuple
    _inverse_kernels: tuple
    _dense_matrices: tuple
    nside: int | None = eqx.field(static=True)
    theta: jax.Array
    phi: jax.Array

    @property
    def engine(self):
        """Resolved transform engine for each block, by block name."""
        return dict(zip(_BLOCK_NAMES, self._engines))

    @property
    def engine_reason(self):
        """Why each block's engine was chosen (see engine_select)."""
        return dict(zip(_BLOCK_NAMES, self._engine_reasons))

    def __init__(
        self,
        data,
        freqs,
        pairs,
        sampling="mwss",
        convention="IAU",
        stokes=STOKES_IQUV,
        units="m^2",
        frame="topo",
        tangent_basis="theta-phi",
        baseline_direction="a<=b",
        visibility_definition="<v_a v_b*>",
        horizon=None,
        beam_rot=0.0,
        niter=0,
        engine="auto",
    ):
        convention = _normalize_convention(convention)
        self.stokes = _validate_stokes(stokes)
        data = jnp.asarray(data)
        if data.ndim < 4 or data.shape[2] != 4:
            raise ValueError(
                "PairStokesBeam data must have shape "
                "(pair, frequency, 4, spatial...)."
            )
        pairs = tuple(tuple(int(index) for index in pair) for pair in pairs)
        if len(pairs) != data.shape[0]:
            raise ValueError("Pair metadata length does not match pair axis.")
        if convention == "COSMO":
            data = cosmo_to_iau(data, stokes_axis=2)
            convention = "IAU"
        (
            self.data,
            self.freqs,
            self.nside,
            self._niter,
            self.lmax,
            self.theta,
            self.phi,
        ) = _spatial_metadata(data, freqs, sampling, niter)
        if self.data.shape[1] != self.freqs.size:
            raise ValueError("Frequency axis length does not match freqs.")
        self.pairs = pairs
        self.sampling = sampling
        self.convention = convention
        self.units = str(units)
        self.frame = str(frame)
        self.tangent_basis = str(tangent_basis)
        self.baseline_direction = str(baseline_direction)
        self.visibility_definition = str(visibility_definition)
        self.beam_rot = jnp.asarray(beam_rot)
        self._L = self.lmax + 1
        if horizon is None:
            horizon = self.theta <= jnp.pi / 2
            if sampling != "healpix":
                horizon = horizon[:, None]
        self.horizon = jnp.asarray(horizon)
        nmap = int(self.data.shape[0]) * int(self.data.shape[1])
        (
            self._engines,
            self._engine_reasons,
            self._kernels,
            self._inverse_kernels,
            self._dense_matrices,
        ) = _prepare_engines(
            self.data,
            self.lmax,
            self.sampling,
            self.nside,
            self._niter,
            spatial_shape=tuple(self.data.shape[3:]),
            # A complex response has no real-pixel conjugation identity
            # to exploit, so both spin-weighted blocks are transformed
            # on every sampling.
            batch_sizes=(2 * nmap, nmap, nmap),
            spin0_reality=False,
            requested=engine,
        )

    @partial(jax.jit, static_argnames=("lmax",))
    def compute_alm(self, lmax=None):
        """Return calibrated pair-response alms up to ``lmax``."""
        target_lmax = self.lmax if lmax is None else int(lmax)
        if target_lmax < 0 or target_lmax > self.lmax:
            raise ValueError(
                f"lmax must lie in [0, {self.lmax}]; got {target_lmax}."
            )
        data = self.data * self.horizon
        alm = _compute_response_dual_alm(
            data,
            target_lmax,
            self.lmax,
            self.sampling,
            self.nside,
            self._niter,
            _block_kwargs(
                self._engines,
                self._kernels,
                self._inverse_kernels,
                self._dense_matrices,
            ),
        )
        emms = jnp.arange(-target_lmax, target_lmax + 1)
        phase = jnp.exp(1j * emms * jnp.radians(self.beam_rot))
        return alm * phase

    def compute_alm_in_frame(self, rotation, dl_array, lmax=None):
        """Rotate all pair/component alms with a shared spatial rotation."""
        return rotations.rotate_alm(
            self.compute_alm(lmax=lmax), rotation, dl_array=dl_array
        )


@jax.jit
def polarized_convolve(beam_alm, sky_alm, phases, normalization=None):
    """Convolve frequency-aligned full-Stokes sky and pair-response alms.

    Parameters
    ----------
    beam_alm : array_like
        Shape ``(pair, frequency, 4, ell, m)`` in the internal harmonic dual.
    sky_alm : array_like
        Shape ``(frequency, 4, ell, m)`` in the internal harmonic dual.
    phases : array_like
        Shape ``(time, m)`` using Croissant's ``exp(-i*m*phi)`` convention.
    normalization : array_like or None
        Optional scalar, pair, or pair-by-frequency normalization.

    Returns
    -------
    array
        Complex visibilities with shape ``(time, pair, frequency)``.
    """
    result = jnp.einsum(
        "fclm,tm,pfclm->tpf",
        sky_alm.conjugate(),
        phases,
        beam_alm,
    )
    if normalization is None:
        return result
    norm = jnp.asarray(normalization)
    if norm.ndim == 0:
        return result / norm
    if norm.ndim == 1:
        return result / norm[None, :, None]
    if norm.ndim == 2:
        return result / norm[None, :, :]
    raise ValueError(
        "normalization must be scalar, pair, or pair-by-frequency."
    )
