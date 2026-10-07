"""Cell-edge geometry and physical horizon regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from croissant import Beam, PairStokesBeam, horizon_weights


def test_boundary_row_and_subcell_motion():
    theta = jnp.linspace(0, jnp.pi, 181)
    weights = horizon_weights(theta, theta_h=jnp.deg2rad(80.0))[:, 0]
    np.testing.assert_allclose(weights[:80], 1)
    np.testing.assert_allclose(weights[80], 0.5, atol=1e-13)
    np.testing.assert_allclose(weights[81:], 0)
    shifted = horizon_weights(theta, theta_h=jnp.deg2rad(80.25))
    np.testing.assert_allclose(shifted[80, 0], 0.75, atol=1e-13)


def test_nonuniform_cells_and_full_visibility_limits():
    theta = jnp.array([0.0, 0.4, 1.0, 2.0, jnp.pi])
    # The cell at theta=1 spans 0.7..1.5 radians.
    np.testing.assert_allclose(
        horizon_weights(theta, theta_h=0.9)[:, 0],
        [1, 1, 0.25, 0, 0],
    )
    np.testing.assert_array_equal(horizon_weights(theta, theta_h=0), 0)
    np.testing.assert_array_equal(horizon_weights(theta, theta_h=jnp.pi), 1)


def test_azimuth_dependent_horizon_and_jax_gradient():
    theta = jnp.linspace(0, jnp.pi, 5)
    phi = jnp.array([0.0, jnp.pi])
    heights = jnp.pi / 2 + 0.1 * jnp.cos(phi)
    weights = horizon_weights(
        theta, phi, lambda p: jnp.pi / 2 + 0.1 * jnp.cos(p)
    )
    assert weights.shape == (5, 2)
    np.testing.assert_allclose(
        weights, horizon_weights(theta, theta_h=heights)
    )
    np.testing.assert_allclose(
        weights[2], 0.5 + jnp.array([0.1, -0.1]) / (jnp.pi / 4)
    )
    fn = jax.jit(lambda h: horizon_weights(theta, theta_h=h)[2, 0])
    np.testing.assert_allclose(fn(jnp.pi / 2), 0.5)
    np.testing.assert_allclose(jax.grad(fn)(jnp.pi / 2), 4 / jnp.pi)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theta": [0]},
        {"theta": [[0, 1]]},
        {"theta": [0, 0, 1]},
        {"theta": [1, 0]},
        {"theta": [-1, 1]},
        {"theta": [0, 180]},
        {"theta": [0, float("nan")]},
        {"theta": [0, 1], "theta_h": lambda p: p},
        {"theta": [0, 1], "phi": [[0]]},
        {"theta": [0, 1], "phi": [0, 1], "theta_h": [0.5]},
        {"theta": [0, 1], "theta_h": [[0.5]]},
    ],
)
def test_invalid_horizon_shapes(kwargs):
    with pytest.raises(ValueError):
        horizon_weights(**kwargs)


def _isotropic_beam(sampling, L):
    if sampling == "healpix":
        shape = (48,)
    else:
        shape = (
            s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling),
            s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling),
        )
    return Beam(
        jnp.ones((1,) + shape), [50.0], sampling=sampling, engine="s2fft"
    )


@pytest.mark.parametrize("sampling", ["mwss", "dh", "gl", "healpix"])
@pytest.mark.parametrize("L", [8, 9])
def test_default_horizon_bisects_isotropic_beam(sampling, L):
    beam = _isotropic_beam(sampling, L)
    np.testing.assert_allclose(beam.compute_fgnd(), 0.5, atol=1e-13)
    # Both classes must agree on the visibility of every spatial sample.
    pair = PairStokesBeam(
        jnp.ones((1, 1, 4) + beam.data.shape[1:]),
        [50.0],
        [(0, 0)],
        sampling=sampling,
        engine="s2fft",
    )
    np.testing.assert_array_equal(pair.horizon, beam.horizon)


def test_blocked_isotropic_cap_matches_solid_angle():
    beam = _isotropic_beam("mwss", 180)
    horizon = horizon_weights(beam.theta, theta_h=jnp.deg2rad(80.0))
    fractional = Beam(beam.data, beam.freqs, horizon=horizon, engine="s2fft")
    old = Beam(
        beam.data,
        beam.freqs,
        horizon=(beam.theta <= jnp.deg2rad(80.0))[:, None],
        engine="s2fft",
    )
    expected = (1 + np.cos(np.deg2rad(80.0))) / 2
    new_error = abs(float(fractional.compute_fgnd()[0]) - expected)
    old_error = abs(float(old.compute_fgnd()[0]) - expected)
    assert new_error < 1e-5
    assert new_error < old_error / 100


def test_pair_fractional_weights_match_explicitly_weighted_response():
    shape = (9, 16)
    data = jnp.ones((1, 1, 4) + shape, dtype=jnp.complex128)
    weights = jnp.linspace(0.0, 1.0, shape[0])[:, None]
    beam = PairStokesBeam(
        data, [50.0], [(0, 0)], horizon=weights, engine="s2fft"
    )
    reference = PairStokesBeam(
        data * weights,
        [50.0],
        [(0, 0)],
        horizon=jnp.ones(shape),
        engine="s2fft",
    )
    np.testing.assert_array_equal(beam.horizon, weights)
    np.testing.assert_allclose(
        beam.compute_alm(), reference.compute_alm(), atol=1e-12
    )
