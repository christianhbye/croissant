from copy import deepcopy
import pytest
import jax.numpy as jnp
from s2fft.sampling import s2_samples
from croissant.constants import Y00
from croissant.crojax import Beam

frequencies = jnp.linspace(1, 50, 50)
lmax = 32


def test_compute_total_power():
    # make a beam that is 1 everywhere so total power is 4pi:
    beam = Beam.zeros(lmax)
    beam[0, 0, 0] = 1 / Y00
    beam.compute_total_power()
    assert jnp.allclose(beam.total_power, 4 * jnp.pi)

    # beam(theta) = cos(theta)**2 * freq**2
    beam = Beam.zeros(lmax, frequencies=frequencies)
    beam[:, 0, 0] = 1 / (3 * Y00) * frequencies**2
    beam[:, 2, 0] = 4 * jnp.sqrt(jnp.pi / 5) * 1 / 3 * frequencies**2
    beam.compute_total_power()
    power = beam.total_power
    expected_power = 4 * jnp.pi / 3 * frequencies**2
    assert jnp.allclose(power, expected_power.ravel())


def test_horizon_cut():
    # make a beam that is 1 everywhere
    beam_base = Beam.zeros(lmax)
    beam_base[0, 0, 0] = 1 / Y00

    # default horizon (1 frequency)
    beam = deepcopy(beam_base)
    beam.horizon_cut()  # doesn't throw error

    # default horizon (multiple frequencies)
    beam_nf = Beam.zeros(lmax, frequencies=frequencies)
    beam[:, 0, 0] = 1 / Y00
    beam_nf.horizon_cut()  # doesn't throw error
    assert jnp.allclose(beam_nf.alm, beam.alm)

    # try custom horizon
    beam = deepcopy(beam_base)
    ntheta, nphi = s2_samples.f_shape(lmax + 1, sampling="mw")
    horizon = jnp.ones((1, ntheta, nphi))  # no horizon
    beam_map = beam.alm2map(sampling="mw")  # before horizon cut
    beam.horizon_cut(horizon=horizon, sampling="mw")
    # should be the same before and after since the horizon is all 1s
    assert jnp.allclose(beam_map, beam.alm2map(sampling="mw"))

    beam = deepcopy(beam_base)
    horizon = jnp.zeros((1, ntheta, nphi))  # full horizon
    beam.horizon_cut(horizon=horizon, sampling="mw")
    # should be all zeros since the horizon is all 0s
    assert jnp.allclose(beam.alm2map(sampling="mw"), 0)

    # try horizon with invalid values
    horizon = jnp.ones((1, ntheta, nphi))
    horizon = horizon.at[0, 0, 0].set(2.0)  # invalid value
    with pytest.raises(ValueError):
        beam.horizon_cut(horizon=horizon, sampling="mw")
    horizon = jnp.ones((1, ntheta, nphi))
    horizon = horizon.at[0, 0, 0].set(-1.0)  # invalid value
    with pytest.raises(ValueError):
        beam.horizon_cut(horizon=horizon, sampling="mw")
