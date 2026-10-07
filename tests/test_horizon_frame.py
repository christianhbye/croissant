"""Ground-fixed masks, compass handedness, and rotation regressions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from croissant import Beam, PairStokesBeam, horizon_weights, utils


def _fixture(sampling="mwss"):
    lmax, nside = (4, 2) if sampling == "healpix" else (7, None)
    theta = utils.generate_theta(lmax, sampling, nside)
    phi = utils.generate_phi(lmax, sampling, nside)
    if sampling == "healpix":
        data = 1 + 0.5 * np.sin(theta) * np.cos(phi)
        horizon = 0.5 + 0.4 * np.cos(phi)
    else:
        data = 1 + 0.5 * np.sin(theta[:, None]) * np.cos(phi)
        horizon = horizon_weights(
            theta, phi, theta_h=jnp.deg2rad(70 + 10 * np.cos(phi))
        )
    return jnp.asarray(data[None]), jnp.asarray(horizon), theta, phi


def _beam(data, horizon, sampling, **kwargs):
    return Beam(
        data,
        [50.0],
        sampling=sampling,
        horizon=horizon,
        engine="s2fft",
        **kwargs,
    )


@pytest.mark.parametrize("sampling", ["mwss", "healpix"])
def test_default_frame_preserves_attached_mask(sampling):
    data, mask, _, _ = _fixture(sampling)
    default = _beam(data, mask, sampling, beam_rot=90)
    explicit = _beam(data, mask, sampling, beam_rot=90, horizon_frame="beam")
    unrotated = _beam(data, mask, sampling)
    assert default.horizon_frame == "beam"
    np.testing.assert_array_equal(default.horizon_in_beam_frame, mask)
    np.testing.assert_array_equal(
        default.compute_alm(), explicit.compute_alm()
    )
    np.testing.assert_array_equal(
        default.compute_fgnd(), unrotated.compute_fgnd()
    )


@pytest.mark.parametrize("sampling", ["mwss", "healpix"])
@pytest.mark.parametrize("rotation", [90, -90, 180, 450])
def test_ground_fixed_mask_matches_rotating_beam_then_masking(
    sampling, rotation
):
    data, mask, theta, phi = _fixture(sampling)
    beam = _beam(
        data, mask, sampling, beam_rot=rotation, horizon_frame="topocentric"
    )
    # Independent physical reference: rotate the analytic beam on the
    # fixed ground grid, then apply the original terrain mask there.
    if sampling != "healpix":
        theta = theta[:, None]
    rotated = 1 + 0.5 * np.sin(theta) * np.cos(phi + np.deg2rad(rotation))
    reference = _beam(jnp.asarray(rotated[None]), mask, sampling)
    np.testing.assert_allclose(
        beam.compute_alm(), reference.compute_alm(), atol=1e-12
    )
    np.testing.assert_allclose(
        beam.compute_fgnd(), reference.compute_fgnd(), atol=1e-12
    )
    np.testing.assert_array_equal(beam.horizon, mask)


@pytest.mark.parametrize("sampling", ["mwss", "mw", "dh", "gl", "healpix"])
def test_fractional_rotation_interpolates_periodically_within_rows(sampling):
    data, mask, theta, phi = _fixture(sampling)
    beam = _beam(
        data, mask, sampling, beam_rot=13.0, horizon_frame="topocentric"
    )
    source = np.asarray(mask)
    if sampling != "healpix":
        expected = np.stack(
            [
                np.interp(
                    (phi - np.deg2rad(13)) % (2 * np.pi),
                    phi,
                    row,
                    period=2 * np.pi,
                )
                for row in source
            ]
        )
    else:
        expected = np.empty_like(source)
        for t in np.unique(theta):
            ring = theta == t
            expected[ring] = np.interp(
                (phi[ring] - np.deg2rad(13)) % (2 * np.pi),
                phi[ring],
                source[ring],
                period=2 * np.pi,
            )
    weights = beam.horizon_in_beam_frame
    np.testing.assert_allclose(weights, expected, atol=1e-13)
    assert jnp.all((weights >= 0) & (weights <= 1))


def test_compass_handedness_keeps_north_obstruction_fixed():
    data, _, theta, phi = _fixture()
    # A=0 is North; the blocked column is therefore phi=90 degrees.
    A = (np.pi / 2 - phi) % (2 * np.pi)
    mask = np.broadcast_to(~np.isclose(A, 0), data.shape[1:])
    beam = _beam(data, mask, "mwss", beam_rot=-90, horizon_frame="topocentric")
    # With the beam x axis pointing North, its phi=0 looks into blockage.
    np.testing.assert_array_equal(beam.horizon_in_beam_frame[:, 0], 0)
    np.testing.assert_array_equal(beam.horizon_in_beam_frame[:, 4], 1)


def test_ground_fraction_changes_with_orientation_and_traces_rotation():
    data, mask, _, _ = _fixture()
    beam = _beam(data, mask, "mwss", horizon_frame="topocentric")

    def ground(rot):
        return eqx.tree_at(lambda b: b.beam_rot, beam, rot).compute_fgnd()[0]

    assert abs(float(ground(0) - ground(180))) > 1e-3
    grad = jax.jit(jax.grad(ground))(jnp.array(13.0))
    step = 1e-3
    numerical = (ground(13 + step) - ground(13 - step)) / (2 * step)
    np.testing.assert_allclose(grad, numerical, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("sampling", ["mwss", "healpix"])
def test_default_flat_horizon_is_frame_invariant(sampling):
    data, _, _, _ = _fixture(sampling)
    beam = Beam(data, [50.0], sampling=sampling, beam_rot=13, engine="s2fft")
    terrain = Beam(
        data,
        [50.0],
        sampling=sampling,
        beam_rot=13,
        horizon_frame="topocentric",
        engine="s2fft",
    )
    np.testing.assert_array_equal(terrain.compute_alm(), beam.compute_alm())
    np.testing.assert_array_equal(terrain.compute_fgnd(), beam.compute_fgnd())


@pytest.mark.parametrize("sampling", ["mwss", "healpix"])
def test_pair_response_uses_same_ground_fixed_mask(sampling):
    data, mask, _, _ = _fixture(sampling)
    response = jnp.broadcast_to(data, (1, 1, 4) + data.shape[1:]) * (1 + 0.2j)
    beam = PairStokesBeam(
        response,
        [50.0],
        [(0, 0)],
        sampling=sampling,
        horizon=mask,
        beam_rot=13,
        horizon_frame="topocentric",
        engine="s2fft",
    )
    scalar = _beam(
        data, mask, sampling, beam_rot=13, horizon_frame="topocentric"
    )
    np.testing.assert_array_equal(
        beam.horizon_in_beam_frame, scalar.horizon_in_beam_frame
    )
    manual = PairStokesBeam(
        response,
        [50.0],
        [(0, 0)],
        sampling=sampling,
        horizon=scalar.horizon_in_beam_frame,
        beam_rot=13,
        engine="s2fft",
    )
    np.testing.assert_allclose(
        beam.compute_alm(), manual.compute_alm(), atol=1e-12
    )
    np.testing.assert_allclose(
        beam.compute_norm(), manual.compute_norm(), atol=1e-12
    )


@pytest.mark.parametrize("cls", [Beam, PairStokesBeam])
def test_invalid_frame_rejected(cls):
    with pytest.raises(ValueError, match="horizon_frame"):
        if cls is Beam:
            cls(None, None, horizon_frame="galactic")
        else:
            cls(None, None, None, horizon_frame="galactic")
