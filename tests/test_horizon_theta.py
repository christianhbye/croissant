"""Horizon weights built by the beam from a colatitude or a profile."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from croissant import Beam, PairStokesBeam, horizon_weights, utils
from croissant.horizon import _horizon_from_theta

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
        beam.horizon, horizon_weights(beam.theta, horizon_theta=1.3)
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


@pytest.mark.parametrize("sampling", REGULAR)
def test_horizon_weights_reads_a_callable_as_the_beam_does(sampling):
    """The same profile handed to ``horizon_weights`` or to
    ``Beam(horizon_theta=...)`` gives the same weights: both read it in
    compass azimuth."""
    beam = _beam(sampling, horizon_theta=_profile, horizon_frame="topocentric")
    np.testing.assert_allclose(
        horizon_weights(beam.theta, beam.phi, _profile),
        beam.horizon,
        rtol=0,
        atol=1e-12,
    )


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
    by_hand = beam(horizon=horizon_weights(theta, horizon_theta=theta_h))
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


def test_healpix_rings_must_repeat_exactly():
    # Ring bands come from the distinct pixel colatitudes; a colatitude
    # computed per pixel that differs in the last bit would add a ring.
    theta = utils.generate_theta(4, "healpix", 2)
    phi = utils.generate_phi(4, "healpix", 2)
    theta = theta.copy()
    theta[5] = np.nextafter(theta[5], np.pi)
    with pytest.raises(ValueError, match="4 \\* nside - 1 rings"):
        _horizon_from_theta(1.3, theta, phi, "healpix")


# -------------------------------------------- azimuth sub-sampling (#167)
def _mwss_grid(lmax=7):
    theta = utils.generate_theta(lmax, "mwss")
    phi = utils.generate_phi(lmax, "mwss")
    return np.asarray(theta), np.asarray(phi)


@pytest.mark.parametrize("sampling", REGULAR + ["healpix"])
def test_one_subsample_is_bit_identical(sampling):
    kwargs = {"horizon_theta": _profile, "horizon_frame": "topocentric"}
    np.testing.assert_array_equal(
        _beam(sampling, horizon_subsamples=1, **kwargs).horizon,
        _beam(sampling, **kwargs).horizon,
    )


def test_subsamples_average_a_step_inside_a_column():
    """A cliff covering the middle half of one column's azimuth range is
    seen by half of 4 sub-samples (offsets +-1/8 and +-3/8 of the
    column), so that column gets the mean of the two weights."""
    theta, phi = _mwss_grid()
    width = 2 * np.pi / phi.size
    k = 3
    centre = np.mod(np.pi / 2 - phi[k], 2 * np.pi)
    low, high = 1.3, 1.7

    def cliff(A):
        distance = np.abs(np.angle(np.exp(1j * (np.asarray(A) - centre))))
        return np.where(distance < width / 4, high, low)

    weights = horizon_weights(theta, phi, cliff, horizon_subsamples=4)
    w_low = horizon_weights(theta, horizon_theta=low)[:, 0]
    w_high = horizon_weights(theta, horizon_theta=high)[:, 0]
    np.testing.assert_allclose(weights[:, k], (w_low + w_high) / 2, atol=1e-12)
    others = np.delete(np.asarray(weights), k, axis=1)
    np.testing.assert_allclose(
        others, np.broadcast_to(w_low[:, None], others.shape), atol=1e-12
    )
    # One sample per column reads the cliff at the centre for the whole cell.
    np.testing.assert_allclose(
        horizon_weights(theta, phi, cliff)[:, k], w_high, atol=1e-12
    )


def test_subsamples_converge_to_the_cell_average():
    theta, phi = _mwss_grid()

    def ridges(A):
        return jnp.pi / 2 - 0.2 - 0.15 * jnp.sin(23 * A)

    reference = horizon_weights(theta, phi, ridges, horizon_subsamples=2048)
    errors = [
        float(
            jnp.abs(
                horizon_weights(theta, phi, ridges, horizon_subsamples=s)
                - reference
            ).max()
        )
        for s in (1, 8, 64)
    ]
    assert errors[0] > 1e-2
    assert errors[1] < errors[0] / 4
    assert errors[2] < errors[1] / 4


@pytest.mark.parametrize("sampling", REGULAR)
def test_horizon_weights_and_beam_subsample_alike(sampling):
    beam = _beam(
        sampling,
        horizon_theta=_profile,
        horizon_frame="topocentric",
        horizon_subsamples=8,
    )
    np.testing.assert_allclose(
        horizon_weights(beam.theta, beam.phi, _profile, horizon_subsamples=8),
        beam.horizon,
        rtol=0,
        atol=1e-12,
    )


def test_healpix_subsamples_span_each_pixels_ring_width():
    seen = []

    def profile(A):
        seen.append(np.asarray(A))
        return jnp.full_like(A, 1.3)

    beam = _beam(
        "healpix",
        horizon_theta=profile,
        horizon_frame="topocentric",
        horizon_subsamples=3,
    )
    assert len(seen) == 1 and seen[0].shape == (3, beam.theta.size)
    _, ring, counts = np.unique(
        np.asarray(beam.theta), return_inverse=True, return_counts=True
    )
    width = 2 * np.pi / counts[ring]
    step = np.mod(seen[0][0] - seen[0][1], 2 * np.pi)
    np.testing.assert_allclose(step, width / 3, atol=1e-12)


def test_subsampled_profile_is_differentiable():
    data = _data("mwss")

    def ground(h0):
        beam = Beam(
            data,
            [50.0],
            sampling="mwss",
            engine="s2fft",
            horizon_frame="topocentric",
            horizon_theta=lambda A: h0 + 0.2 * jnp.cos(A),
            horizon_subsamples=4,
        )
        return beam.compute_fgnd()[0]

    grad = jax.jit(jax.grad(ground))(jnp.array(1.3))
    step = 1e-4
    numerical = (ground(1.3 + step) - ground(1.3 - step)) / (2 * step)
    assert grad != 0
    np.testing.assert_allclose(grad, numerical, rtol=1e-5)


@pytest.mark.parametrize("subsamples", [0, -1, 2.5, True])
def test_subsample_count_must_be_a_positive_integer(subsamples):
    with pytest.raises(ValueError, match="horizon_subsamples"):
        _beam(
            "mwss",
            horizon_theta=_profile,
            horizon_frame="topocentric",
            horizon_subsamples=subsamples,
        )


def test_subsamples_need_a_callable_horizon():
    with pytest.raises(ValueError, match="horizon_subsamples"):
        _beam("mwss", horizon_theta=1.3, horizon_subsamples=4)
    with pytest.raises(ValueError, match="horizon_subsamples"):
        _beam("mwss", horizon_subsamples=4)
    theta, _ = _mwss_grid()
    with pytest.raises(ValueError, match="horizon_subsamples"):
        horizon_weights(theta, horizon_theta=1.3, horizon_subsamples=4)


@pytest.mark.parametrize("sampling", ["mwss", "healpix"])
def test_pair_beam_subsamples_as_the_beam_does(sampling):
    kwargs = dict(
        horizon_theta=_profile,
        horizon_frame="topocentric",
        horizon_subsamples=5,
    )
    np.testing.assert_array_equal(
        _pair(sampling, **kwargs).horizon, _beam(sampling, **kwargs).horizon
    )
