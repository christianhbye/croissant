"""Tests for the Simulator class."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.time import Time as AstroTime
from lunarsky import Time as LunarTime

from croissant import Beam, Simulator, Sky
from croissant.constants import sidereal_day

rng = np.random.default_rng(seed=3)

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

_NSIDE = 4
_LMAX = 2 * _NSIDE  # = 8
_NPIX = 12 * _NSIDE**2
_FREQS = jnp.array([50.0, 100.0])
_N_FREQS = len(_FREQS)

# Julian-day arrays spanning one sidereal day, 12 steps
_N_TIMES = 12

_T0_MOON = LunarTime("2022-01-01 00:00:00")
_TIMES_JD_MOON = jnp.linspace(
    _T0_MOON.jd,
    _T0_MOON.jd + sidereal_day["moon"] / 86400.0,
    _N_TIMES,
)

_T0_EARTH = AstroTime("2022-01-01 00:00:00")
_TIMES_JD_EARTH = jnp.linspace(
    _T0_EARTH.jd,
    _T0_EARTH.jd + sidereal_day["earth"] / 86400.0,
    _N_TIMES,
)

_TIMES_JD = {"moon": _TIMES_JD_MOON, "earth": _TIMES_JD_EARTH}
_SKY_COORD = {"moon": "mcmf", "earth": "equatorial"}


def _make_sim(world="moon", Tgnd=0.0):
    """Create a Simulator with an isotropic beam and uniform monopole sky."""
    T_sky = jnp.array([1000.0, 500.0])
    sky_data = T_sky[:, None] * jnp.ones((_N_FREQS, _NPIX))
    sky = Sky(sky_data, _FREQS, coord=_SKY_COORD[world])
    beam_data = jnp.ones((_N_FREQS, _NPIX))
    beam = Beam(beam_data, _FREQS, sampling="healpix")
    times_jd = _TIMES_JD[world]
    return Simulator(
        beam, sky, times_jd, _FREQS, 0.0, 0.0, world=world, Tgnd=Tgnd
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world", ["moon", "earth"])
def test_simulator_init(world):
    """Simulator should store attributes correctly."""
    sim = _make_sim(world=world)
    assert sim.world == world
    assert sim.lmax == _LMAX
    assert jnp.allclose(sim.freqs, _FREQS)
    assert sim.phases.shape == (_N_TIMES, 2 * _LMAX + 1)


def test_simulator_freq_mismatch():
    """Mismatched frequencies between beam, sky and Simulator should raise."""
    npix = _NPIX
    freqs_beam = jnp.array([50.0, 100.0])
    freqs_sim = jnp.array([50.0, 200.0])  # different

    sky_data = jnp.ones((2, npix))
    sky = Sky(sky_data, freqs_beam, coord="mcmf")
    beam_data = jnp.ones((2, npix))
    beam = Beam(beam_data, freqs_beam, sampling="healpix")

    with pytest.raises(ValueError, match="frequencies"):
        Simulator(beam, sky, _TIMES_JD_MOON, freqs_sim, 0.0, 0.0, world="moon")


def test_simulator_lmax_too_large():
    """Requesting lmax larger than beam/sky lmax should raise."""
    sim_data = jnp.ones((_N_FREQS, _NPIX))
    sky = Sky(sim_data, _FREQS, coord="mcmf")
    beam = Beam(sim_data, _FREQS, sampling="healpix")

    with pytest.raises(ValueError, match="lmax"):
        Simulator(
            beam,
            sky,
            _TIMES_JD_MOON,
            _FREQS,
            0.0,
            0.0,
            world="moon",
            lmax=_LMAX + 10,
        )


def test_simulator_invalid_world():
    """Invalid world keyword should raise ValueError."""
    sim_data = jnp.ones((_N_FREQS, _NPIX))
    sky = Sky(sim_data, _FREQS, coord="galactic")
    beam = Beam(sim_data, _FREQS, sampling="healpix")
    with pytest.raises(ValueError, match="world"):
        Simulator(beam, sky, _TIMES_JD_MOON, _FREQS, 0.0, 0.0, world="saturn")


# ---------------------------------------------------------------------------
# sim() – output shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world", ["moon", "earth"])
def test_sim_output_shape(world):
    """sim() output should have shape (N_times, N_freqs) and be real."""
    sim = _make_sim(world=world)
    vis = sim.sim()
    assert vis.shape == (_N_TIMES, _N_FREQS)
    assert jnp.issubdtype(vis.dtype, jnp.floating)


# ---------------------------------------------------------------------------
# Physical invariant: monopole sky → constant visibility over time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world", ["moon", "earth"])
def test_sim_monopole_sky_is_constant(world):
    """
    For a monopole (uniform) sky, the visibility must be constant over time.

    Physical reasoning: convolve(...) sums over m-modes weighted by
    exp(-i*m*phi(t)). A monopole sky has alm = 0 for all l > 0, so only
    the m = 0 phase (which equals 1 for all t) contributes.
    """
    sim = _make_sim(world=world, Tgnd=0.0)
    vis = sim.sim()
    # all time steps should agree within healpix-discretization tolerance
    assert jnp.allclose(vis, vis[0:1], rtol=5e-3)


@pytest.mark.parametrize("world", ["moon", "earth"])
def test_sim_ground_contribution_increases_vis(world):
    """Adding a ground temperature should increase the simulated visibility."""
    vis_no_gnd = _make_sim(world=world, Tgnd=0.0).sim()
    vis_with_gnd = _make_sim(world=world, Tgnd=300.0).sim()
    assert jnp.all(vis_with_gnd > vis_no_gnd)


# ---------------------------------------------------------------------------
# compute_beam_eq – shape and reality condition
# ---------------------------------------------------------------------------


def test_compute_beam_eq_shape():
    """compute_beam_eq should return shape (N_freqs, lmax+1, 2*lmax+1)."""
    sim = _make_sim(world="moon")
    beam_eq = sim.compute_beam_eq()
    assert beam_eq.shape == (_N_FREQS, _LMAX + 1, 2 * _LMAX + 1)
