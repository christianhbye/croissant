"""Horizon weights built by the beam from a colatitude or a profile."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from croissant import Beam, PairStokesBeam, horizon_weights, utils

REGULAR = ["mw", "mwss", "dh", "gl"]


def _data(sampling):
    lmax, nside = (4, 2) if sampling == "healpix" else (7, None)
    theta = utils.generate_theta(lmax, sampling, nside)
    phi = utils.generate_phi(lmax, sampling, nside)
    if sampling == "healpix":
        data = 1 + 0.5 * np.sin(theta) * np.cos(phi)
    else:
        data = 1 + 0.5 * np.sin(theta[:, None]) * np.cos(phi)
    return jnp.asarray(data[None])


def _beam(sampling, **kwargs):
    return Beam(
        _data(sampling), [50.0], sampling=sampling, engine="s2fft", **kwargs
    )


def _profile(A):
    return jnp.pi / 2 + 0.3 * jnp.cos(A)


def _north_blocked(A):
    return jnp.where(jnp.cos(A) > 0.9, 0.0, jnp.pi)


@pytest.mark.parametrize("sampling", REGULAR)
def test_scalar_matches_horizon_weights(sampling):
    beam = _beam(sampling, horizon_theta=1.3)
    np.testing.assert_array_equal(
        beam.horizon, horizon_weights(beam.theta, theta_h=1.3)
    )


@pytest.mark.parametrize("sampling", REGULAR + ["healpix"])
def test_scalar_at_equator_reproduces_default(sampling):
    default = _beam(sampling)
    explicit = _beam(sampling, horizon_theta=jnp.pi / 2)
    assert explicit.horizon.shape == default.horizon.shape
    np.testing.assert_allclose(explicit.horizon, default.horizon, atol=1e-12)
    np.testing.assert_allclose(
        explicit.compute_fgnd(), default.compute_fgnd(), atol=1e-12
    )


@pytest.mark.parametrize("sampling", ["mwss", "healpix"])
def test_constant_callable_matches_scalar(sampling):
    scalar = _beam(sampling, horizon_theta=1.3, horizon_frame="topocentric")
    profile = _beam(
        sampling, horizon_theta=lambda A: 1.3, horizon_frame="topocentric"
    )
    np.testing.assert_array_equal(
        profile.horizon,
        jnp.broadcast_to(scalar.horizon, profile.horizon.shape),
    )


@pytest.mark.parametrize("sampling", ["mwss", "healpix"])
def test_callable_is_read_in_compass_azimuth(sampling):
    beam = _beam(
        sampling,
        horizon_theta=_north_blocked,
        horizon_frame="topocentric",
        beam_rot=-90,
    )
    phi = np.asarray(beam.phi)
    # Ground grid: A = pi/2 - phi. With beam_rot = -90 the beam's x axis
    # points North, so beam-frame phi looks at A = -phi.
    ground = np.cos(np.mod(np.pi / 2 - phi, 2 * np.pi)) > 0.9
    rotated = np.cos(np.mod(-phi, 2 * np.pi)) > 0.9
    assert ground.any() and not ground.all()
    shape = beam.horizon.shape
    np.testing.assert_array_equal(
        beam.horizon == 0, np.broadcast_to(ground, shape)
    )
    np.testing.assert_array_equal(
        beam.horizon_in_beam_frame == 0, np.broadcast_to(rotated, shape)
    )


def test_healpix_callable_uses_each_pixels_ring_band_and_azimuth():
    beam = _beam(
        "healpix", horizon_theta=_profile, horizon_frame="topocentric"
    )
    theta, phi = np.asarray(beam.theta), np.asarray(beam.phi)
    rings = np.unique(theta)
    mid = (rings[:-1] + rings[1:]) / 2
    ring = np.searchsorted(rings, theta)
    lower = np.concatenate(([0.0], mid))[ring]
    upper = np.concatenate((mid, [np.pi]))[ring]
    theta_h = np.pi / 2 + 0.3 * np.cos(np.mod(np.pi / 2 - phi, 2 * np.pi))
    expected = np.clip((theta_h - lower) / (upper - lower), 0, 1)
    assert np.any((expected > 0) & (expected < 1))
    np.testing.assert_allclose(beam.horizon, expected, atol=1e-12)


def test_terrain_profile_through_numpy_interp():
    az = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    elev = 0.15 + 0.1 * np.sin(3 * az)

    def profile(A):
        return np.pi / 2 - np.interp(A, az, elev, period=2 * np.pi)

    beam = _beam("mwss", horizon_theta=profile, horizon_frame="topocentric")
    ground_az = np.mod(np.pi / 2 - np.asarray(beam.phi), 2 * np.pi)
    expected = horizon_weights(beam.theta, beam.phi, profile(ground_az))
    np.testing.assert_allclose(beam.horizon, expected, atol=1e-12)


@pytest.mark.parametrize("sampling", ["mwss", "healpix"])
def test_callable_receives_azimuths_in_one_turn(sampling):
    seen = []

    def profile(A):
        seen.append(np.asarray(A))
        return jnp.full_like(A, 1.3)

    _beam(sampling, horizon_theta=profile, horizon_frame="topocentric")
    assert len(seen) == 1
    assert seen[0].min() >= 0 and seen[0].max() < 2 * np.pi


def test_scalar_is_traceable_and_differentiable():
    data = _data("mwss")

    def ground(theta_h):
        beam = Beam(
            data,
            [50.0],
            sampling="mwss",
            engine="s2fft",
            horizon_theta=theta_h,
        )
        return beam.compute_fgnd()[0]

    # 1.3 rad lies inside one row's band, where fgnd is linear in theta_h.
    grad = jax.jit(jax.grad(ground))(jnp.array(1.3))
    step = 1e-4
    numerical = (ground(1.3 + step) - ground(1.3 - step)) / (2 * step)
    assert grad != 0
    np.testing.assert_allclose(grad, numerical, rtol=1e-6)


def test_eighty_degree_horizon_on_one_degree_grid():
    # The #161 case: a 10 deg blockage on the 1 deg MWSS grid.
    lmax = 179
    theta = utils.generate_theta(lmax, "mwss")
    phi = utils.generate_phi(lmax, "mwss")
    assert (theta.size, phi.size) == (181, 360)
    data = jnp.asarray((1 + 0.5 * np.sin(theta[:, None]) * np.cos(phi))[None])
    theta_h = np.deg2rad(80.0)

    def beam(**kwargs):
        return Beam(data, [50.0], sampling="mwss", engine="s2fft", **kwargs)

    built = beam(horizon_theta=theta_h)
    by_hand = beam(horizon=horizon_weights(theta, theta_h=theta_h))
    boolean = beam(horizon=(theta <= theta_h)[:, None])
    np.testing.assert_array_equal(built.horizon, by_hand.horizon)
    np.testing.assert_array_equal(built.compute_fgnd(), by_hand.compute_fgnd())
    np.testing.assert_allclose(built.horizon[80, 0], 0.5, atol=1e-12)
    gap = built.compute_fgnd()[0] - boolean.compute_fgnd()[0]
    assert abs(float(gap)) > 1e-4


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"horizon": 1.0, "horizon_theta": 1.3}, "not both"),
        ({"horizon_theta": _profile}, "topocentric"),
        ({"horizon_theta": [1.3, 1.4]}, "callable of compass azimuth"),
        ({"horizon_theta": 80.0}, r"\[0, pi\]"),
        ({"horizon_theta": -0.1}, r"\[0, pi\]"),
        ({"horizon_theta": float("nan")}, r"\[0, pi\]"),
        (
            {
                "horizon_theta": lambda A: jnp.full_like(A, 80.0),
                "horizon_frame": "topocentric",
            },
            r"\[0, pi\]",
        ),
        (
            {
                "horizon_theta": lambda A: jnp.ones(3),
                "horizon_frame": "topocentric",
            },
            "one colatitude per azimuth",
        ),
    ],
)
def test_invalid_horizon_theta_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _beam("mwss", **kwargs)


def _pair(sampling, **kwargs):
    data = _data(sampling)
    response = jnp.broadcast_to(data, (1, 1, 4) + data.shape[1:]) * (1 + 0.2j)
    return PairStokesBeam(
        response,
        [50.0],
        [(0, 0)],
        sampling=sampling,
        engine="s2fft",
        **kwargs,
    )


@pytest.mark.parametrize("sampling", ["mwss", "healpix"])
@pytest.mark.parametrize("horizon_theta", [1.3, _profile])
def test_pair_beam_builds_the_same_horizon(sampling, horizon_theta):
    kwargs = dict(
        horizon_theta=horizon_theta,
        horizon_frame="topocentric",
        beam_rot=13,
    )
    pair = _pair(sampling, **kwargs)
    beam = _beam(sampling, **kwargs)
    np.testing.assert_array_equal(pair.horizon, beam.horizon)
    np.testing.assert_array_equal(
        pair.horizon_in_beam_frame, beam.horizon_in_beam_frame
    )


def test_pair_beam_rejects_both_horizons():
    with pytest.raises(ValueError, match="not both"):
        _pair("mwss", horizon=1.0, horizon_theta=1.3)


def test_azimuths_stay_below_one_turn_on_mwss_lmax_25():
    # On this grid the North column's phi rounds one ulp above pi/2, so a
    # bare mod gives A = 2 pi.
    lmax = 25
    theta = utils.generate_theta(lmax, "mwss")
    phi = utils.generate_phi(lmax, "mwss")
    data = jnp.ones((1, theta.size, phi.size))
    seen = []

    def profile(A):
        seen.append(np.asarray(A))
        return jnp.full_like(A, 1.3)

    Beam(
        data,
        [50.0],
        sampling="mwss",
        engine="s2fft",
        horizon_theta=profile,
        horizon_frame="topocentric",
    )
    assert seen[0].min() >= 0 and seen[0].max() < 2 * np.pi


def test_degrees_rejected_when_beam_built_under_jit():
    data = _data("mwss")

    def ground():
        beam = Beam(
            data, [50.0], sampling="mwss", engine="s2fft", horizon_theta=80.0
        )
        return beam.compute_fgnd()

    with pytest.raises(ValueError, match=r"\[0, pi\]"):
        jax.jit(ground)()


def test_numpy_profile_works_when_beam_built_under_jit():
    az = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    elev = 0.15 + 0.1 * np.sin(3 * az)

    def profile(A):
        return np.pi / 2 - np.interp(A, az, elev, period=2 * np.pi)

    def ground(data):
        beam = Beam(
            data,
            [50.0],
            sampling="mwss",
            engine="s2fft",
            horizon_theta=profile,
            horizon_frame="topocentric",
        )
        return beam.compute_fgnd()

    data = _data("mwss")
    np.testing.assert_allclose(jax.jit(ground)(data), ground(data), rtol=1e-12)
