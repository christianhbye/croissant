"""Tests for the Beam class."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from croissant import utils
from croissant.beam import Beam

rng = np.random.default_rng(seed=1)

# healpix beam parameters used in multiple tests
_NSIDE = 4
_LMAX_HP = 2 * _NSIDE  # = 8
_NPIX = 12 * _NSIDE**2
_FREQS = jnp.array([50.0, 100.0])


def _make_beam(sampling, lmax, N_freqs=2, value=1.0, horizon=None):
    """Create a beam with constant value for the given sampling scheme."""
    freqs = jnp.linspace(50.0, 100.0, N_freqs)
    if sampling == "healpix":
        nside = lmax // 2
        npix = 12 * nside**2
        data = value * jnp.ones((N_freqs, npix))
    else:
        L = lmax + 1
        ntheta = s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling)
        nphi = s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling)
        data = value * jnp.ones((N_freqs, ntheta, nphi))
    return Beam(data, freqs, sampling=sampling, horizon=horizon)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampling,lmax",
    [("mwss", 8), ("mw", 8), ("dh", 8), ("healpix", 8)],
)
def test_beam_init(sampling, lmax):
    """Beam should initialize with correct attributes."""
    beam = _make_beam(sampling, lmax)
    assert beam.sampling == sampling
    assert beam.lmax == lmax
    assert beam.data.shape[0] == 2  # N_freqs


def test_beam_tilt_raises():
    """Non-zero beam_tilt should raise NotImplementedError."""
    data = jnp.ones((1, _NPIX))
    with pytest.raises(NotImplementedError):
        Beam(data, _FREQS[:1], sampling="healpix", beam_tilt=5.0)


def test_beam_default_horizon_shape():
    """Default horizon should match the beam data spatial shape."""
    beam = _make_beam("healpix", _LMAX_HP)
    assert beam.horizon.shape == (_NPIX,)


def test_beam_custom_horizon():
    """Custom horizon mask should be stored as given."""
    data = jnp.ones((1, _NPIX))
    horizon = jnp.zeros(_NPIX, dtype=bool)  # no sky visible
    beam = Beam(data, _FREQS[:1], sampling="healpix", horizon=horizon)
    assert jnp.all(beam.horizon == 0)


# ---------------------------------------------------------------------------
# compute_norm – total beam integral over the full sphere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disable_jit", [True, False])
def test_beam_norm_isotropic(disable_jit):
    """Isotropic healpix beam should integrate to 4π."""
    beam = _make_beam("healpix", _LMAX_HP)
    with jax.disable_jit(disable_jit):
        norm = beam.compute_norm()
    assert norm.shape == (_FREQS.shape[0],)
    assert jnp.allclose(norm, 4 * jnp.pi, rtol=1e-3)


@pytest.mark.parametrize("disable_jit", [True, False])
def test_beam_norm_scaling(disable_jit):
    """Beam norm should scale linearly with beam amplitude."""
    beam_unit = _make_beam("healpix", _LMAX_HP, value=1.0)
    beam_double = _make_beam("healpix", _LMAX_HP, value=2.0)
    with jax.disable_jit(disable_jit):
        norm1 = beam_unit.compute_norm()
        norm2 = beam_double.compute_norm()
    assert jnp.allclose(norm2, 2.0 * norm1, rtol=1e-10)


# ---------------------------------------------------------------------------
# compute_fgnd – ground fraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disable_jit", [True, False])
def test_beam_fgnd_no_horizon(disable_jit):
    """All-sky horizon (no blocking) → fgnd ≈ 0."""
    data = jnp.ones((1, _NPIX))
    horizon = jnp.ones(_NPIX, dtype=bool)  # full sky visible
    beam = Beam(data, _FREQS[:1], sampling="healpix", horizon=horizon)
    with jax.disable_jit(disable_jit):
        fgnd = beam.compute_fgnd()
    assert jnp.allclose(fgnd, 0.0, atol=1e-10)


@pytest.mark.parametrize("disable_jit", [True, False])
def test_beam_fgnd_full_horizon(disable_jit):
    """All-blocking horizon → fgnd = 1."""
    data = jnp.ones((1, _NPIX))
    horizon = jnp.zeros(_NPIX, dtype=bool)  # nothing visible
    beam = Beam(data, _FREQS[:1], sampling="healpix", horizon=horizon)
    with jax.disable_jit(disable_jit):
        fgnd = beam.compute_fgnd()
    assert jnp.allclose(fgnd, 1.0, atol=1e-10)


@pytest.mark.parametrize("disable_jit", [True, False])
def test_beam_fgnd_default_horizon(disable_jit):
    """Default horizon (theta ≤ π/2) → fgnd ≈ 0.5 for an isotropic beam."""
    beam = _make_beam("healpix", _LMAX_HP)
    with jax.disable_jit(disable_jit):
        fgnd = beam.compute_fgnd()
    # fgnd should be close to 0.5 (within healpix discretization tolerance)
    assert jnp.allclose(fgnd, 0.5, atol=0.1)


def test_beam_fgnd_plus_fsky_equals_one():
    """fgnd + fsky = 1 by construction."""
    beam = _make_beam("healpix", _LMAX_HP)
    fgnd = beam.compute_fgnd()
    norm_above = beam._compute_norm(use_horizon=True)
    norm_total = beam.compute_norm()
    fsky = norm_above / norm_total
    assert jnp.allclose(fgnd + fsky, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_alm – spherical harmonic transform
# ---------------------------------------------------------------------------


def test_beam_alm_shape():
    """compute_alm should return shape (N_freqs, lmax+1, 2*lmax+1)."""
    beam = _make_beam("healpix", _LMAX_HP)
    alm = beam.compute_alm()
    lmax = beam.lmax
    assert alm.shape == (len(_FREQS), lmax + 1, 2 * lmax + 1)


def test_beam_alm_is_real():
    """Beam alm should correspond to a real signal (reality condition)."""
    beam = _make_beam("healpix", _LMAX_HP)
    alm = beam.compute_alm()
    for i in range(len(_FREQS)):
        assert utils.is_real(alm[i])


# ---------------------------------------------------------------------------
# beam_az_rot – azimuthal rotation only changes phases, not power
# ---------------------------------------------------------------------------


def test_beam_az_rot_preserves_power():
    """Rotating the beam azimuthally should not change the alm magnitudes."""
    # use a phi-dependent beam so that m != 0 modes are populated
    phi = utils.generate_phi(_LMAX_HP, "healpix", _NSIDE)
    data = (1.0 + 0.3 * jnp.cos(phi))[None]  # non-isotropic
    freqs = _FREQS[:1]
    beam0 = Beam(data, freqs, sampling="healpix")
    beam90 = Beam(data, freqs, sampling="healpix", beam_az_rot=90.0)

    alm0 = beam0.compute_alm()
    alm90 = beam90.compute_alm()

    # Magnitudes should be identical
    assert jnp.allclose(jnp.abs(alm0), jnp.abs(alm90), atol=1e-8)
    # But the alms themselves differ (phases change) for m ≠ 0
    assert not jnp.allclose(alm0, alm90)


def test_beam_az_rot_phase_formula():
    """beam_az_rot applies exp(-i*m*az_rot) phase to each m-mode."""
    phi = utils.generate_phi(_LMAX_HP, "healpix", _NSIDE)
    data = (1.0 + 0.3 * jnp.cos(phi))[None]
    freqs = _FREQS[:1]
    az_rot = 45.0  # degrees
    beam0 = Beam(data, freqs, sampling="healpix")
    beam_rot = Beam(data, freqs, sampling="healpix", beam_az_rot=az_rot)

    alm0 = beam0.compute_alm()
    alm_rot = beam_rot.compute_alm()
    lmax = beam0.lmax
    emms = jnp.arange(-lmax, lmax + 1)
    expected_phases = jnp.exp(-1j * emms * jnp.radians(az_rot))

    # Check that ratio alm_rot / alm0 matches expected phase for non-zero entries
    mask = jnp.abs(alm0) > 1e-12
    ratio = alm_rot / jnp.where(mask, alm0, 1.0 + 0j)
    assert jnp.allclose(
        ratio[mask], jnp.broadcast_to(expected_phases, alm0.shape)[mask]
    )
