import numpy as np

from croissant.beam import Beam

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
