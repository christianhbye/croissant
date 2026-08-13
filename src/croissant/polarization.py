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

    spatial_ndim = 1 if sampling == "healpix" else 2
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
):
    if sampling == "healpix" and target_lmax < native_lmax:
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
    )
    return utils.reduce_lmax(result, target_lmax)


def _compute_sky_dual_alm(
    data,
    target_lmax,
    native_lmax,
    sampling,
    nside,
    niter,
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
    )
    # Internally U is IAU (= -U_COSMO), so Q + iU is the spin -2 object
    # and Q - iU the spin +2 object in s2fft's Goldberg basis. Each
    # combination must be analyzed at its physical spin: this keeps the
    # duals of band-limited E/B skies band-limited and makes Wigner-D
    # frame rotation apply the physical transport phase (a mismatched
    # label applies its complex conjugate). See docs/polarization.md.
    p_minus = _analysis_alm(
        stokes_q + 1j * stokes_u,
        target_lmax,
        native_lmax,
        sampling,
        nside,
        niter,
        spin=-2,
        reality=False,
    )
    p_plus = _analysis_alm(
        stokes_q - 1j * stokes_u,
        target_lmax,
        native_lmax,
        sampling,
        nside,
        niter,
        spin=2,
        reality=False,
    )
    return jnp.stack((spin0[:, 0], spin0[:, 1], p_minus, p_plus), axis=1)


def _compute_response_dual_alm(
    data,
    target_lmax,
    native_lmax,
    sampling,
    nside,
    niter,
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
    )
    # The spin -2 slot carries (BQ + iBU)/2, which the conjugated
    # einsum contracts with the spin -2 sky analysis of Q + iU (and
    # mirror for spin +2), reproducing the physical integral
    # BQ*Q + BU*U. Like the sky dual, each combination is analyzed at
    # its physical spin so frame rotation transports it correctly.
    q_plus_dual = _analysis_alm(
        0.5 * (response_q + 1j * response_u),
        target_lmax,
        native_lmax,
        sampling,
        nside,
        niter,
        spin=-2,
        reality=False,
    )
    q_minus_dual = _analysis_alm(
        0.5 * (response_q - 1j * response_u),
        target_lmax,
        native_lmax,
        sampling,
        nside,
        niter,
        spin=2,
        reality=False,
    )
    return jnp.stack(
        (
            spin0[:, :, 0],
            spin0[:, :, 1],
            q_plus_dual,
            q_minus_dual,
        ),
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
    nside: int | None = eqx.field(static=True)
    theta: jax.Array
    phi: jax.Array

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
    nside: int | None = eqx.field(static=True)
    theta: jax.Array
    phi: jax.Array

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
