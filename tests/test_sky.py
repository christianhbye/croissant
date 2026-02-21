"""Tests for the Sky class."""

import jax.numpy as jnp
import numpy as np
import pytest

from croissant import utils
from croissant.constants import Y00
from croissant.sky import Sky

rng = np.random.default_rng(seed=2)

_NSIDE = 4
_LMAX = 2 * _NSIDE  # = 8
_NPIX = 12 * _NSIDE**2
_FREQS = jnp.array([50.0, 100.0])
_T_SKY = 1000.0


def _uniform_sky(coord="galactic", N_freqs=1):
    """Create a uniform healpix sky map."""
    data = _T_SKY * jnp.ones((N_freqs, _NPIX))
    freqs = jnp.linspace(50.0, 100.0, N_freqs)
    return Sky(data, freqs, coord=coord)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_sky_init_defaults():
    """Sky should default to healpix sampling and galactic coordinates."""
    sky = _uniform_sky()
    assert sky.sampling == "healpix"
    assert sky.coord == "galactic"
    assert sky.lmax == _LMAX
    assert sky.data.shape == (1, _NPIX)


def test_sky_invalid_coord():
    """Unsupported coordinate system should raise ValueError."""
    data = jnp.ones((1, _NPIX))
    with pytest.raises(ValueError, match="Unsupported coordinate system"):
        Sky(data, _FREQS[:1], coord="ecliptic")


# ---------------------------------------------------------------------------
# compute_alm – spherical harmonic transform
# ---------------------------------------------------------------------------


def test_sky_alm_shape():
    """compute_alm should return shape (N_freqs, lmax+1, 2*lmax+1)."""
    sky = _uniform_sky(N_freqs=2)
    alm = sky.compute_alm()
    assert alm.shape == (2, _LMAX + 1, 2 * _LMAX + 1)


def test_sky_alm_monopole_uniform():
    """Uniform sky should have dominant monopole alm ≈ T / Y00."""
    sky = _uniform_sky()
    alm = sky.compute_alm()
    l_ix, m_ix = utils.getidx(_LMAX, 0, 0)
    assert jnp.isclose(alm[0, l_ix, m_ix].real, _T_SKY / Y00, rtol=1e-3)


# ---------------------------------------------------------------------------
# compute_alm_eq – rotation to equatorial / mcmf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "world,coord", [("moon", "mcmf"), ("earth", "equatorial")]
)
def test_sky_alm_eq_native_coords_no_rotation(world, coord):
    """Sky already in equatorial/mcmf should not be rotated."""
    sky = _uniform_sky(coord=coord)
    alm_direct = sky.compute_alm()
    alm_eq = sky.compute_alm_eq(world=world)
    assert jnp.allclose(alm_direct, alm_eq)


@pytest.mark.parametrize("world", ["moon", "earth"])
def test_sky_alm_eq_galactic_preserves_monopole(world):
    """Galactic → equatorial/mcmf rotation should preserve the monopole."""
    sky = _uniform_sky(coord="galactic")
    alm_eq = sky.compute_alm_eq(world=world)
    l_ix, m_ix = utils.getidx(_LMAX, 0, 0)
    assert jnp.isclose(alm_eq[0, l_ix, m_ix].real, _T_SKY / Y00, rtol=1e-3)


def test_sky_alm_eq_invalid_world():
    """Unsupported world keyword should raise ValueError."""
    sky = _uniform_sky(coord="galactic")
    with pytest.raises(ValueError, match="Unsupported world"):
        sky.compute_alm_eq(world="mars")


def test_sky_alm_eq_coord_world_mismatch_mcmf_earth():
    """mcmf sky + world='earth' should raise ValueError."""
    sky = _uniform_sky(coord="mcmf")
    with pytest.raises(
        ValueError, match="Unsupported coordinate transformation"
    ):
        sky.compute_alm_eq(world="earth")


def test_sky_alm_eq_coord_world_mismatch_equatorial_moon():
    """equatorial sky + world='moon' should raise ValueError."""
    sky = _uniform_sky(coord="equatorial")
    with pytest.raises(
        ValueError, match="Unsupported coordinate transformation"
    ):
        sky.compute_alm_eq(world="moon")


# ---------------------------------------------------------------------------
# Frequency axis
# ---------------------------------------------------------------------------


def test_sky_multifreq_alm_scales_correctly():
    """Alm monopole should scale with the sky temperature at each frequency."""
    N_freqs = 3
    freqs = jnp.linspace(50.0, 100.0, N_freqs)
    # sky temperature follows a power law: T ∝ freq^(-2.5)
    T = _T_SKY * (freqs / freqs[0]) ** (-2.5)
    data = T[:, None] * jnp.ones((N_freqs, _NPIX))
    sky = Sky(data, freqs, coord="mcmf")
    alm = sky.compute_alm()
    l_ix, m_ix = utils.getidx(_LMAX, 0, 0)
    for i, Ti in enumerate(T):
        assert jnp.isclose(alm[i, l_ix, m_ix].real, Ti / Y00, rtol=1e-3)
