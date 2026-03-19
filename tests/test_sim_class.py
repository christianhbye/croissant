"""Tests for the Simulator class."""

import astropy.units as u
import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time as AstroTime
from lunarsky import Time as LunarTime

from croissant.beam import Beam
from croissant.constants import sidereal_day
from croissant.simulator import Simulator, correct_ground_loss
from croissant.sky import Sky

rng = np.random.default_rng(seed=3)

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

_NSIDE = 8
_LMAX = 2 * _NSIDE
_NPIX = 12 * _NSIDE**2
_FREQS = jnp.linspace(50.0, 100.0, num=50)
_N_FREQS = len(_FREQS)

_TSKY = 180 * (_FREQS / 180) ** (-2.5)  # power-law sky temperature in K

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
_SKY_COORD = {"moon": "mepa", "earth": "equatorial"}


def _make_sim(world="moon", Tgnd=0.0, nside=_NSIDE):
    """Create a Simulator with an isotropic beam and uniform monopole sky."""
    npix = 12 * nside**2
    sky_data = _TSKY[:, None] * jnp.ones((_N_FREQS, npix))
    sky = Sky(sky_data, _FREQS, coord=_SKY_COORD[world], niter=0)
    beam_data = jnp.ones((_N_FREQS, npix))
    beam = Beam(beam_data, _FREQS, sampling="healpix", niter=0)
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
    sky = Sky(sky_data, freqs_beam, coord="mepa", niter=0)
    beam_data = jnp.ones((2, npix))
    beam = Beam(beam_data, freqs_beam, sampling="healpix", niter=0)

    with pytest.raises(ValueError, match="frequencies"):
        Simulator(beam, sky, _TIMES_JD_MOON, freqs_sim, 0.0, 0.0, world="moon")


def test_simulator_lmax_too_large():
    """Requesting lmax larger than beam/sky lmax should raise."""
    sim_data = jnp.ones((_N_FREQS, _NPIX))
    sky = Sky(sim_data, _FREQS, coord="mepa", niter=0)
    beam = Beam(sim_data, _FREQS, sampling="healpix", niter=0)

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
    sky = Sky(sim_data, _FREQS, coord="galactic", niter=0)
    beam = Beam(sim_data, _FREQS, sampling="healpix", niter=0)
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
@pytest.mark.parametrize("Tgnd", [0.0, 100.0, 300.0])
def test_sim_ground_contribution_increases_vis(world, Tgnd):
    """Adding a ground temperature should increase the simulated visibility."""
    sim_no_gnd = _make_sim(world=world, Tgnd=0.0)
    vis_no_gnd = sim_no_gnd.sim()
    sim_with_gnd = _make_sim(world=world, Tgnd=Tgnd)
    vis_with_gnd = sim_with_gnd.sim()
    # vis should increase by fgnd * Tgnd
    fgnd = sim_with_gnd.beam.compute_fgnd()
    assert jnp.allclose(vis_with_gnd, vis_no_gnd + fgnd * Tgnd, rtol=1e-3)


# ---------------------------------------------------------------------------
# compute_beam_eq – shape and reality condition
# ---------------------------------------------------------------------------


def test_compute_beam_eq_shape():
    """compute_beam_eq should return shape (N_freqs, lmax+1, 2*lmax+1)."""
    sim = _make_sim(world="moon")
    beam_eq = sim.compute_beam_eq()
    assert beam_eq.shape == (_N_FREQS, _LMAX + 1, 2 * _LMAX + 1)


@pytest.fixture
def vis_highres():
    """
    High-resolution visibility for testing ground loss correction.
    This has Tgnd = 0 but we can add it back in tests.
    """
    sim = _make_sim(world="moon", Tgnd=0, nside=64)
    return sim.sim()[0], sim.beam.compute_fgnd()


# ------ test correct ground loss -------------
@pytest.mark.parametrize("Tgnd", [0.0, 100.0, 300.0])
def test_ground_loss(vis_highres, Tgnd):
    """Ground loss should be zero when Tgnd=0, and increase with Tgnd."""
    _vis, fgnd = vis_highres
    # add ground contribution to visibility
    vis = _vis + fgnd * Tgnd
    true_Tsky = correct_ground_loss(vis, fgnd, Tgnd)
    assert jnp.allclose(true_Tsky, _TSKY)


def test_ground_loss_biased(vis_highres):
    Tgnd = 0.0
    _vis, fgnd = vis_highres
    vis = _vis + fgnd * Tgnd
    true_fgnd = fgnd
    # if we overestimate fgnd, we get biased Tsky
    wrong_fgnd = true_fgnd * 1.5
    wrong_Tsky = correct_ground_loss(vis, wrong_fgnd, Tgnd)
    fsky_ratio = (1 - true_fgnd) / (1 - wrong_fgnd)
    assert jnp.allclose(wrong_Tsky, fsky_ratio * _TSKY)
    # if we get Tgnd wrong, we also get biased Tsky
    wrong_Tgnd = 100.0
    wrong_Tsky = correct_ground_loss(vis, true_fgnd, wrong_Tgnd)
    bias = wrong_Tsky - _TSKY
    true_fsky = 1 - true_fgnd
    expected_bias = -true_fgnd / true_fsky * wrong_Tgnd
    assert jnp.allclose(bias, expected_bias, rtol=1e-3)


# ---------------------------------------------------------------------------
# Moon simulation depends on start time (regression test for MEPA fix)
# ---------------------------------------------------------------------------


def test_moon_sim_depends_on_start_time():
    """
    For a non-uniform sky on the Moon, changing the observation start
    time should change the visibility. This verifies that the beam
    transform correctly captures the Moon's spin phase via the MEPA
    frame.
    """
    nside = 4
    npix = 12 * nside**2
    freqs = jnp.array([50.0])
    # non-uniform sky: point source at one pixel
    sky_data = jnp.zeros((1, npix))
    sky_data = sky_data.at[0, npix // 3].set(1.0)
    sky = Sky(sky_data, freqs, coord="galactic", niter=0)
    beam_data = jnp.ones((1, npix))
    beam = Beam(beam_data, freqs, sampling="healpix", niter=0)

    t0 = LunarTime("2022-01-01 00:00:00")
    times1 = jnp.linspace(t0.jd, t0.jd + sidereal_day["moon"] / 86400, 12)
    times2 = times1 + 7.0  # start 7 days later

    sim1 = Simulator(beam, sky, times1, freqs, 0.0, 5.0, world="moon")
    sim2 = Simulator(beam, sky, times2, freqs, 0.0, 5.0, world="moon")

    vis1 = sim1.sim()
    vis2 = sim2.sim()
    # visibilities must differ (sky has rotated ~92° in 7 days)
    assert not jnp.allclose(vis1, vis2, atol=1e-6)


# ---------------------------------------------------------------------------
# ENU beam orientation: East-peaked beam should see East sources stronger
# ---------------------------------------------------------------------------


def _point_source_sky(nside, freqs, ra_deg, dec_deg, flux=1.0):
    """Create a HEALPix sky with a single point source."""
    npix = 12 * nside**2
    pix = hp.ang2pix(nside, ra_deg, dec_deg, lonlat=True)
    sky_data = jnp.zeros((len(freqs), npix))
    sky_data = sky_data.at[:, pix].set(flux)
    return sky_data


def test_beam_enu_east_direction():
    """
    A beam with peak toward East (in ENU) should see sources at East
    more strongly than at North. This verifies the ENU-to-topo
    conversion in compute_alm.

    The beam pattern is ``1 + 0.5 * sin(theta) * cos(phi)`` in ENU,
    where phi=0 is East. For a source at altitude ``alt`` toward East,
    the beam value is ``1 + 0.5 * cos(alt)``. Toward North the
    cos(phi) term vanishes, giving beam = 1.
    """
    nside = 16
    npix = 12 * nside**2
    freqs = jnp.array([75.0])

    # Beam: 1 + 0.5*sin(theta)*cos(phi) in ENU convention
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    beam_data = 1.0 + 0.5 * np.sin(theta) * np.cos(phi)
    beam_data = jnp.array(beam_data[None, :])  # (1, npix)
    beam = Beam(beam_data, freqs, sampling="healpix", niter=3)

    # Observer on Earth
    lon, lat = 0.0, 45.0
    t0 = AstroTime("2022-06-21 00:00:00")
    loc = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    altaz_frame = AltAz(obstime=t0, location=loc)

    alt_deg = 45.0
    # East source (az=90 in AltAz)
    east_eq = SkyCoord(
        alt=alt_deg * u.deg, az=90 * u.deg, frame=altaz_frame
    ).transform_to("fk5")
    # North source (az=0 in AltAz)
    north_eq = SkyCoord(
        alt=alt_deg * u.deg, az=0 * u.deg, frame=altaz_frame
    ).transform_to("fk5")

    sky_east = Sky(
        _point_source_sky(nside, freqs, east_eq.ra.deg, east_eq.dec.deg),
        freqs,
        coord="equatorial",
        niter=3,
    )
    sky_north = Sky(
        _point_source_sky(nside, freqs, north_eq.ra.deg, north_eq.dec.deg),
        freqs,
        coord="equatorial",
        niter=3,
    )

    times_jd = jnp.array([t0.jd])
    sim_east = Simulator(
        beam,
        sky_east,
        times_jd,
        freqs,
        lon,
        lat,
        world="earth",
        Tgnd=0.0,
    )
    sim_north = Simulator(
        beam,
        sky_north,
        times_jd,
        freqs,
        lon,
        lat,
        world="earth",
        Tgnd=0.0,
    )

    vis_east = sim_east.sim()[0, 0]
    vis_north = sim_north.sim()[0, 0]

    alt_rad = np.radians(alt_deg)
    expected_ratio = (1 + 0.5 * np.cos(alt_rad)) / 1.0
    np.testing.assert_allclose(vis_east / vis_north, expected_ratio, rtol=0.15)
