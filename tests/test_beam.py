import numpy as np

from croissant.beam import Beam


def test_compute_total_power():
    phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    theta = np.linspace(0, np.pi, 181)
    # make a beam that is 1 everywhere so total power is 4pi:
    data = np.ones((theta.size, phi.size))
    beam = Beam(data, theta, phi)
    assert np.allclose(beam.compute_total_power(), 4 * np.pi)
    # dipole beam
    frequencies = np.linspace(1, 50, 50).reshape(-1, 1, 1)
    data = np.cos(theta.reshape(1, -1, 1)) ** 2 * frequencies**2
    data = np.repeat(data, phi.size, axis=2)
    beam = Beam(data, theta, phi, frequencies=frequencies)
    power = beam.compute_total_power()
    expected_power = 4 * np.pi / 3 * frequencies**2
    assert np.allclose(power, expected_power.ravel())


def test_horizon_cut():
    # generate mock data on grid
    phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    theta = np.linspace(0, np.pi, 181)
    frequencies = np.linspace(1, 50, 50).reshape(-1, 1, 1)
    # make a beam that's constant in space, but varies in frequency
    data = np.ones((1, theta.size, phi.size)) * frequencies**2
    beam = Beam(data, theta, phi, frequencies=frequencies)

    # default horizon, i.e. cuts everywhere below theta = 0
    beam.horizon_cut(horizon=None)
    below_horizon = theta < 0
    assert np.allclose(beam.data[:, below_horizon, :], 0)
    assert np.allclose(beam.data[:, ~below_horizon, :], data)

    # custom horizon
    beam = Beam(data, theta, phi, frequencies=frequencies)
    horizon = np.ones(data.shape)
    horizon[:, theta < np.pi / 8, :] = 0
    horizon[:, :, phi < np.pi] = 0
    beam.horizon_cut(horizon=horizon)
    assert np.allclose(beam.data[:, theta < np.pi / 8, :], 0)
    assert np.allclose(beam.data[:, :, phi < np.pi], 0)
