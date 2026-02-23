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
_NSIDE = 16
_LMAX = 32
_NPIX = 12 * _NSIDE**2


def _make_beam(sampling, lmax, N_freqs=50, value=1.0, horizon=None):
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


@pytest.mark.parametrize("sampling", ["mwss", "mw", "gl", "dh", "healpix"])
@pytest.mark.parametrize("lmax", [8, 32, 64])
@pytest.mark.parametrize("N_freqs", [1, 10, 50])
def test_beam_init(sampling, lmax, N_freqs):
    """Beam should initialize with correct attributes."""
    beam = _make_beam(sampling, lmax, N_freqs=N_freqs)
    assert beam.sampling == sampling
    assert beam.lmax == lmax
    assert beam.data.shape[0] == N_freqs
    if sampling == "healpix":
        assert beam.data.shape[1] == 12 * (beam.nside**2)
    else:
        L = lmax + 1
        ntheta = s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling)
        nphi = s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling)
        assert beam.data.shape[1] == ntheta
        assert beam.data.shape[2] == nphi


@pytest.mark.parametrize("sampling", ["mwss", "mw", "gl", "dh", "healpix"])
def test_beam_tilt_raises(sampling):
    """Non-zero beam_tilt should raise NotImplementedError."""
    data = jnp.ones((1, _NPIX))
    with pytest.raises(NotImplementedError):
        Beam(data, 1.0, sampling=sampling, beam_tilt=5.0)


@pytest.mark.parametrize("sampling", ["mwss", "mw", "gl", "dh", "healpix"])
def test_beam_default_horizon_shape(sampling):
    """Default horizon should match the beam data spatial shape."""
    beam = _make_beam(sampling, _LMAX)
    assert beam.horizon.shape[0] == beam.data.shape[1]
    if sampling != "healpix":
        assert beam.horizon.ndim == 2  # extra broadcast dimension for phi
        assert beam.horizon.shape[1] == 1  # extra broadcast dimension for phi
    else:
        assert beam.horizon.ndim == 1  # no extra dimension for phi


@pytest.mark.parametrize("sampling", ["mwss", "mw", "gl", "dh", "healpix"])
def test_beam_custom_horizon(sampling):
    """Custom horizon mask should be stored as given."""
    horizon = jnp.zeros(_NPIX, dtype=bool)  # no sky visible
    beam = _make_beam(sampling, _LMAX, horizon=horizon)
    assert jnp.all(beam.horizon == 0)


# ---------------------------------------------------------------------------
# compute_norm – total beam integral over the full sphere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disable_jit", [True, False])
@pytest.mark.parametrize("N_freqs", [1, 10, 50])
@pytest.mark.parametrize("sampling", ["gl", "dh", "healpix"])
def test_beam_norm_isotropic(N_freqs, disable_jit, sampling):
    """Isotropic healpix beam should integrate to 4π."""
    beam = _make_beam(sampling, _LMAX, N_freqs=N_freqs)
    with jax.disable_jit(disable_jit):
        norm = beam.compute_norm()
    assert norm.shape == (N_freqs,)
    assert jnp.allclose(norm, 4 * jnp.pi, rtol=1e-3)


# can't disable jit for mw/mwss
@pytest.mark.parametrize("N_freqs", [1, 10, 50])
@pytest.mark.parametrize("sampling", ["mwss", "mw"])
def test_beam_norm_isotropic_mw(N_freqs, sampling):
    """Isotropic healpix beam should integrate to 4π."""
    beam = _make_beam(sampling, _LMAX, N_freqs=N_freqs)
    norm = beam.compute_norm()
    assert norm.shape == (N_freqs,)
    assert jnp.allclose(norm, 4 * jnp.pi, rtol=1e-3)


@pytest.mark.parametrize("disable_jit", [True, False])
@pytest.mark.parametrize("sampling", ["gl", "dh", "healpix"])
def test_beam_norm_scaling(disable_jit, sampling):
    """Beam norm should scale linearly with beam amplitude."""
    beam_unit = _make_beam(sampling, _LMAX, value=1.0)
    beam_double = _make_beam(sampling, _LMAX, value=2.0)
    with jax.disable_jit(disable_jit):
        norm1 = beam_unit.compute_norm()
        norm2 = beam_double.compute_norm()
    assert jnp.allclose(norm2, 2.0 * norm1, rtol=1e-10)


@pytest.mark.parametrize("sampling", ["mwss", "mw"])
def test_beam_norm_scaling_mw(sampling):
    """Beam norm should scale linearly with beam amplitude."""
    beam_unit = _make_beam(sampling, _LMAX, value=1.0)
    beam_double = _make_beam(sampling, _LMAX, value=2.0)
    norm1 = beam_unit.compute_norm()
    norm2 = beam_double.compute_norm()
    assert jnp.allclose(norm2, 2.0 * norm1, rtol=1e-10)


# ---------------------------------------------------------------------------
# compute_fgnd – ground fraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disable_jit", [True, False])
def test_beam_fgnd_no_horizon(disable_jit):
    """All-sky horizon (no blocking) → fgnd ≈ 0."""
    horizon = jnp.ones(_NPIX, dtype=bool)  # full sky visible
    beam = _make_beam("healpix", _LMAX, horizon=horizon)
    with jax.disable_jit(disable_jit):
        fgnd = beam.compute_fgnd()
    assert jnp.allclose(fgnd, 0.0, atol=1e-10)


@pytest.mark.parametrize("disable_jit", [True, False])
def test_beam_fgnd_full_horizon(disable_jit):
    """All-blocking horizon → fgnd = 1."""
    horizon = jnp.zeros(_NPIX, dtype=bool)  # no sky visible
    beam = _make_beam("healpix", _LMAX, horizon=horizon)
    with jax.disable_jit(disable_jit):
        fgnd = beam.compute_fgnd()
    assert jnp.allclose(fgnd, 1.0, atol=1e-10)


@pytest.mark.parametrize("disable_jit", [True, False])
@pytest.mark.parametrize("sampling", ["gl", "dh", "healpix"])
def test_beam_fgnd_default_horizon(disable_jit, sampling):
    """Default horizon (theta ≤ π/2) → fgnd ≈ 0.5 for an isotropic beam."""
    beam = _make_beam(sampling, _LMAX)
    with jax.disable_jit(disable_jit):
        fgnd = beam.compute_fgnd()
    # fgnd should be close to 0.5 (within pixel discretization tolerance)
    assert jnp.allclose(fgnd, 0.5, atol=0.1)


@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl", "dh", "healpix"])
def test_beam_fgnd_plus_fsky_equals_one(sampling):
    """fgnd + fsky = 1 by construction."""
    beam = _make_beam(sampling, _LMAX)
    fgnd = beam.compute_fgnd()
    norm_above = beam._compute_norm(use_horizon=True)
    norm_total = beam.compute_norm()
    fsky = norm_above / norm_total
    assert jnp.allclose(fgnd + fsky, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_alm – spherical harmonic transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N_freqs", [1, 10, 50])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl", "dh", "healpix"])
def test_beam_alm_shape(N_freqs, sampling):
    """compute_alm should return shape (N_freqs, lmax+1, 2*lmax+1)."""
    beam = _make_beam(sampling, _LMAX, N_freqs=N_freqs)
    alm = beam.compute_alm()
    lmax = beam.lmax
    assert alm.shape == (N_freqs, lmax + 1, 2 * lmax + 1)


@pytest.mark.parametrize("N_freqs", [1, 10, 50])
@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl", "dh", "healpix"])
def test_beam_alm_is_real(N_freqs, sampling):
    """Beam alm should correspond to a real signal (reality condition)."""
    beam = _make_beam(sampling, _LMAX, N_freqs=N_freqs)
    alm = beam.compute_alm()
    for i in range(len(alm)):
        assert utils.is_real(alm[i])


# ---------------------------------------------------------------------------
# beam_az_rot – azimuthal rotation only changes phases, not power
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl", "dh", "healpix"])
def test_beam_az_rot_preserves_power(sampling):
    """Rotating the beam azimuthally should not change the alm magnitudes."""
    # use a phi-dependent beam so that m != 0 modes are populated
    phi = utils.generate_phi(_LMAX, sampling, _NSIDE)
    data = 1.0 + 0.3 * jnp.cos(phi)  # non-isotropic
    # add freq dimension
    data = data[None]
    if sampling != "healpix":
        ntheta = s2fft.sampling.s2_samples.ntheta(
            L=_LMAX + 1, sampling=sampling
        )
        data = data[:, None, :]  # add extra dimension for theta
        data = jnp.repeat(data, ntheta, axis=1)  # broadcast to all theta
    freq = 1.0  # single frequency
    beam0 = Beam(data, freq, sampling=sampling)
    beam90 = Beam(data, freq, sampling=sampling, beam_az_rot=90.0)

    alm0 = beam0.compute_alm()
    alm90 = beam90.compute_alm()

    # Magnitudes should be identical
    assert jnp.allclose(jnp.abs(alm0), jnp.abs(alm90), atol=1e-8)
    # But the alms themselves differ (phases change) for m ≠ 0
    assert not jnp.allclose(alm0, alm90)


@pytest.mark.parametrize("sampling", ["mw", "mwss", "gl", "dh", "healpix"])
def test_beam_az_rot_phase_formula(sampling):
    """beam_az_rot applies exp(-i*m*az_rot) phase to each m-mode."""
    phi = utils.generate_phi(_LMAX, sampling, _NSIDE)
    data = 1.0 + 0.3 * jnp.cos(phi)
    # add freq dimension
    data = data[None]
    if sampling != "healpix":
        ntheta = s2fft.sampling.s2_samples.ntheta(
            L=_LMAX + 1, sampling=sampling
        )
        data = data[:, None, :]  # add extra dimension for theta
        data = jnp.repeat(data, ntheta, axis=1)  # broadcast to all theta
    freq = 1.0  # single frequency
    az_rot = 45.0  # degrees
    beam0 = Beam(data, freq, sampling=sampling)
    beam_rot = Beam(data, freq, sampling=sampling, beam_az_rot=az_rot)

    alm0 = beam0.compute_alm()
    alm_rot = beam_rot.compute_alm()
    lmax = beam0.lmax
    emms = jnp.arange(-lmax, lmax + 1)
    expected_phases = jnp.exp(-1j * emms * jnp.radians(az_rot))

    # check that alm_rot / alm0 matches expected phase for non-zero entries
    mask = jnp.abs(alm0) > 1e-12
    ratio = alm_rot / jnp.where(mask, alm0, 1.0 + 0j)
    assert jnp.allclose(
        ratio[mask], jnp.broadcast_to(expected_phases, alm0.shape)[mask]
    )
