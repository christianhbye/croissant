from copy import deepcopy
import healpy as hp
import numpy as np
import pytest

from croissant import Beam

phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
theta = np.linspace(0, np.pi, 181)
frequencies = np.linspace(1, 50, 50).reshape(-1, 1, 1)
lmax = 32


def test_compute_total_power():
    # make a beam that is 1 everywhere so total power is 4pi:
    data = np.ones((theta.size, phi.size))
    beam = Beam.from_grid(data, theta, phi, lmax)
    beam.compute_total_power()
    assert np.allclose(beam.total_power, 4 * np.pi)

    # beam(theta) = cos(theta)**2 * freq**2
    data = np.cos(theta.reshape(1, -1, 1)) ** 2 * frequencies**2
    data = np.repeat(data, phi.size, axis=2)
    beam = Beam.from_grid(data, theta, phi, lmax, frequencies=frequencies)
    beam.compute_total_power()
    power = beam.total_power
    expected_power = 4 * np.pi / 3 * frequencies**2
    assert np.allclose(power, expected_power.ravel())


def test_horizon_cut():
    # make a beam that is 1 everywhere
    data = np.ones((theta.size, phi.size))
    beam_base = Beam.from_grid(data, theta, phi, lmax)

    # default horizon (1 frequency)
    beam = deepcopy(beam_base)
    beam.horizon_cut()  # doesn't throw error

    # default horizon (multiple frequencies)
    data_nf = np.ones((frequencies.size, theta.size, phi.size))
    beam_nf = Beam.from_grid(
        data_nf, theta, phi, lmax, frequencies=frequencies
    )
    beam_nf.horizon_cut()  # doesn't throw error
    assert np.allclose(
        beam_nf.alm,
        np.repeat(np.expand_dims(beam.alm, axis=0), frequencies.size, axis=0),
    )

    # try custom horizon
    beam = deepcopy(beam_base)
    nside = 64
    npix = hp.nside2npix(nside)
    horizon = np.ones(npix)  # no horizon
    beam_map = beam.hp_map(nside=nside)  # before horizon cut
    beam.horizon_cut(horizon=horizon, nside=nside)
    # should be the same before and after since the horizon is all 1s
    assert np.allclose(beam_map, beam.hp_map(nside=nside))

    beam = deepcopy(beam_base)
    horizon = np.zeros(npix)  # full horizon
    beam.horizon_cut(horizon=horizon, nside=nside)
    # should be all zeros since the horizon is all 0s
    assert np.allclose(beam.hp_map(nside=nside), np.zeros(npix))

    # try horizon with invalid values
    horizon = np.ones(npix)
    horizon[0] = 2  # invalid value
    with pytest.raises(ValueError):
        beam.horizon_cut(horizon=horizon, nside=nside)
    horizon[0] = -1  # invalid value
    with pytest.raises(ValueError):
        beam.horizon_cut(horizon=horizon, nside=nside)
