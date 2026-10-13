"""Scalar Earth simulations against an independent Astropy pixel sum."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers

from croissant import Beam, Simulator, Sky, horizon_weights, utils
from croissant.simulator import correct_ground_loss

from .pixel_reference import simulate_earth

pytestmark = [pytest.mark.slow, pytest.mark.timeout(600)]

FREQS = np.array([45.0, 83.0, 137.0])
GROUND = np.array([240.0, 280.0, 310.0])
# Fixed budgets for the apparent-sky reference versus a rotation-only
# simulation; tests below check numerical convergence separately.
FRAME_RTOL = 2e-4
# The sharp terrain edge is not band-limited. At L=64 its discretization
# dominates the frame mismatch; doubling L must reduce that residual below.
TERRAIN_RTOL = 2e-3


def sky_model(theta, phi, freqs):
    """Positive, anisotropic low-degree Galactic sky with spectral shape."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta) + np.zeros_like(phi)
    f = np.asarray(freqs).reshape((-1,) + (1,) * x.ndim) / 75.0
    return 1800 * f**-2.5 * (
        1 + 0.35 * x - 0.23 * y + 0.17 * z + 0.2 * x * z
    ) + 220 * f**-2.1 * (1 + y * z + 0.3 * (x * x - y * y))


def beam_model(theta, phi, freqs):
    """Chromatic asymmetric beam, with power in both hemispheres."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    f = np.asarray(freqs).reshape((-1,) + (1,) * x.ndim) / 75.0
    return (1 + 0.2 * f) * z**2 * (1 + 0.5 * x - 0.3 * y / f + 0.15 * x * y)


def flat_horizon(az):
    return np.zeros_like(az)


def no_horizon(az):
    return np.full_like(az, -np.pi / 2)


def terrain_horizon(az):
    """A smooth asymmetric ridge, specified in compass azimuth."""
    return np.deg2rad(16 + 9 * np.cos(az) + 4 * np.sin(2 * az))


@pytest.fixture(scope="module", autouse=True)
def offline_earth_orientation():
    """Use bundled historical IERS-B, independent of the user's cache."""
    table = iers.IERS_B.open(iers.IERS_B_FILE)
    with iers.conf.set_temp("auto_download", False):
        with iers.earth_orientation_table.set(table):
            yield


def make_simulator(times, location, rotation, horizon, frame, L=32):
    # MWSS exactly represents the low-degree sky, so its HEALPix SHT
    # error cannot obscure rotation errors. Only the beam has a sharp edge.
    theta = utils.generate_theta(L - 1, "mwss")[:, None]
    phi = utils.generate_phi(L - 1, "mwss")[None, :]
    kwargs = {}
    if horizon is not flat_horizon:
        if frame == "topocentric":
            kwargs["horizon_theta"] = lambda az: np.pi / 2 - horizon(az)
            kwargs["horizon_subsamples"] = 8
        else:
            # For an attached mask the same profile is on the antenna's
            # unrotated compass grid, A = pi/2 - phi.
            kwargs["horizon"] = horizon_weights(
                theta[:, 0],
                phi[0],
                horizon_theta=lambda az: np.pi / 2 - horizon(az),
                horizon_subsamples=8,
            )
    beam = Beam(
        beam_model(theta, phi, FREQS),
        FREQS,
        sampling="mwss",
        beam_rot=rotation,
        horizon_frame=frame,
        engine="s2fft",
        **kwargs,
    )
    sky = Sky(
        sky_model(theta, phi, FREQS),
        FREQS,
        sampling="mwss",
        coord="galactic",
        engine="s2fft",
    )
    return Simulator(
        beam,
        sky,
        times.jd,
        FREQS,
        location.lon.deg,
        location.lat.deg,
        alt=location.height.to_value(u.m),
        world="earth",
        Tgnd=GROUND,
    )


def run_reference(times, location, rotation, horizon, frame, nside=128):
    return simulate_earth(
        sky_model,
        beam_model,
        horizon,
        FREQS,
        times,
        location,
        nside=nside,
        beam_rot=rotation,
        horizon_frame=frame,
        ground_temperature=GROUND,
    )


def assert_temperature(actual, expected, rtol, record_property, label):
    """Check every cell and retain worst residuals per frequency in JUnit."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(expected))
    residual = np.abs(actual - expected)
    for f, freq in enumerate(FREQS):
        record_property(
            f"{label}_{freq:g}MHz_max_abs_K", float(residual[:, f].max())
        )
        record_property(
            f"{label}_{freq:g}MHz_max_rel",
            float((residual[:, f] / np.abs(expected[:, f])).max()),
        )
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=0, err_msg=label
    )


def assert_simulation(sim, ref, rtol, record_property):
    actual = np.asarray(sim.sim())
    corrected = correct_ground_loss(actual, sim.beam.compute_fgnd(), GROUND)
    assert_temperature(
        actual, ref.temperature, rtol, record_property, "full_beam"
    )
    assert_temperature(
        corrected, ref.sky_temperature, rtol, record_property, "visible_sky"
    )
    np.testing.assert_allclose(
        np.broadcast_to(sim.beam.compute_fgnd(), ref.ground_fraction.shape),
        ref.ground_fraction,
        rtol=rtol,
        atol=1e-12,
    )


@pytest.mark.parametrize("rotation", [-90.0, 36.0])
@pytest.mark.parametrize("date", ["2014-03-21", "2022-09-17"])
@pytest.mark.parametrize(
    "lat,lon", [(37, -118), (-31, 117), (90, 23), (-90, -61)]
)
def test_earth_against_pixels(date, lat, lon, rotation, record_property):
    times = (
        Time(date, scale="utc")
        + np.array([0, 0.7, 4.1, 9.3, 17, 23.8]) * u.hour
    )
    location = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, 120 * u.m)
    sim = make_simulator(
        times, location, rotation, flat_horizon, "topocentric"
    )
    ref = run_reference(times, location, rotation, flat_horizon, "topocentric")
    assert_simulation(sim, ref, FRAME_RTOL, record_property)


@pytest.mark.parametrize("rotation", [-90.0, 45.0])
@pytest.mark.parametrize("frame", ["beam", "topocentric"])
def test_terrain_against_pixels(frame, rotation, record_property):
    times = (
        Time("2022-09-17", scale="utc") + np.array([0, 3.7, 11, 19.2]) * u.hour
    )
    location = EarthLocation.from_geodetic(117 * u.deg, -31 * u.deg)
    sim = make_simulator(
        times, location, rotation, terrain_horizon, frame, L=64
    )
    ref = run_reference(
        times, location, rotation, terrain_horizon, frame, nside=256
    )
    assert_simulation(sim, ref, TERRAIN_RTOL, record_property)


@pytest.mark.parametrize(
    "horizon,L,nside,pixel_rtol,harmonic_rtol",
    [
        pytest.param(flat_horizon, 32, 128, 1e-5, 1e-5, id="flat"),
        pytest.param(terrain_horizon, 64, 256, 5e-5, 6e-4, id="terrain"),
    ],
)
def test_resolution_convergence(
    horizon, L, nside, pixel_rtol, harmonic_rtol, record_property
):
    """Refine the two independent quadratures one at a time."""
    times = (
        Time("2022-09-17", scale="utc") + np.array([0, 3.7, 11, 19.2]) * u.hour
    )
    location = EarthLocation.from_geodetic(117 * u.deg, -31 * u.deg)
    args = times, location, 45.0, horizon, "topocentric"
    coarse = run_reference(*args, nside=nside)
    fine = run_reference(*args, nside=2 * nside)
    for field in ["temperature", "sky_temperature"]:
        assert_temperature(
            getattr(coarse, field),
            getattr(fine, field),
            pixel_rtol,
            record_property,
            f"pixel_convergence_{field}",
        )
    np.testing.assert_allclose(
        coarse.ground_fraction, fine.ground_fraction, rtol=pixel_rtol, atol=0
    )
    coarse_sim = make_simulator(*args, L=L)
    fine_sim = make_simulator(*args, L=2 * L)
    coarse_temperature = np.asarray(coarse_sim.sim())
    fine_temperature = np.asarray(fine_sim.sim())
    assert_temperature(
        coarse_temperature,
        fine_temperature,
        harmonic_rtol,
        record_property,
        "harmonic_convergence",
    )
    if horizon is terrain_horizon:
        # Refinement must improve the unresolved edge or reach the frame
        # mismatch budget. Once both grids are accurate, the apparent-sky
        # mismatch cannot be reduced by increasing L further.
        coarse_error = np.max(
            np.abs(coarse_temperature / fine.temperature - 1)
        )
        fine_error = np.max(np.abs(fine_temperature / fine.temperature - 1))
        record_property("coarse_max_rel_to_pixels", float(coarse_error))
        record_property("fine_max_rel_to_pixels", float(fine_error))
        assert fine_error <= max(FRAME_RTOL, 0.75 * coarse_error), (
            f"Terrain refinement: coarse={coarse_error:.6g}, "
            f"fine={fine_error:.6g}, frame budget={FRAME_RTOL:.6g}"
        )


def test_unmasked_sky_against_pixels(record_property):
    times = Time("2022-09-17") + np.array([0, 3.7, 11, 19.2]) * u.hour
    location = EarthLocation.from_geodetic(117 * u.deg, -31 * u.deg)
    args = times, location, 45.0, no_horizon, "topocentric"
    sim = make_simulator(*args)
    ref = run_reference(*args)
    assert_simulation(sim, ref, FRAME_RTOL, record_property)


def test_monopole_with_response_at_the_horizon(record_property):
    """A horizon row must not count as an entire extra strip of sky."""
    times = Time(["2014-03-21T03:00:00", "2014-03-21T16:00:00"])
    location = EarthLocation.from_geodetic(-118 * u.deg, 37 * u.deg)
    temperature = 1800 * (FREQS / 75) ** -2.5

    def monopole(theta, phi, freqs):
        return temperature[:, None] * np.ones_like(theta)

    def isotropic(theta, phi, freqs):
        return np.ones((len(freqs), theta.size))

    ref = simulate_earth(
        monopole,
        isotropic,
        flat_horizon,
        FREQS,
        times,
        location,
        nside=128,
        ground_temperature=GROUND,
    )
    L = 16  # Includes an equatorial row with nonzero beam response.
    beam = Beam(np.ones((FREQS.size, L + 1, 2 * L)), FREQS)
    sky = Sky(
        temperature[:, None, None] * np.ones_like(beam.data),
        FREQS,
        sampling="mwss",
    )
    sim = Simulator(
        beam, sky, times.jd, FREQS, -118, 37, world="earth", Tgnd=GROUND
    )
    # The analytic hemisphere limit also validates the pixel estimator.
    expected = np.broadcast_to((temperature + GROUND) / 2, (2, FREQS.size))
    assert_temperature(
        ref.temperature, expected, 5e-4, record_property, "pixel_monopole"
    )
    assert_temperature(
        sim.sim(), expected, 1e-12, record_property, "harmonic_monopole"
    )
    np.testing.assert_allclose(
        ref.sky_temperature,
        np.broadcast_to(temperature, ref.sky_temperature.shape),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        sim.beam.compute_fgnd(), 0.5, atol=1e-14, rtol=0
    )
