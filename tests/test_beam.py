from copy import deepcopy
import healpy as hp
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
    assert np.allclose(beam.compute_total_power(), 4 * np.pi)

    # beam(theta) = cos(theta)**2 * freq**2
    data = np.cos(theta.reshape(1, -1, 1)) ** 2 * frequencies**2
    data = np.repeat(data, phi.size, axis=2)
    beam = Beam.from_grid(data, theta, phi, lmax, frequencies=frequencies)
    beam.compute_total_power()
    power = beam.total_power
    expected_power = 4 * np.pi / 3 * frequencies**2
    assert np.allclose(power, expected_power.ravel())


def test_horizon_cut():
    # make a beam that's constant in space, but varies in frequency
    data = np.ones((1, theta.size, phi.size)) * frequencies**2
    beam = Beam.from_grid(data, theta, phi, lmax, frequencies=frequencies)
    beam_nocut = deepcopy(beam)  # beam that we don't cut the horizon on
    # default horizon, i.e. cuts everywhere at theta > np.pi/2
    beam.horizon_cut(horizon=None)
    nside = 32
    npix = hp.nside2npix(nside)
    beam_map = beam.hp_map(nside)
    beam_nc_map = beam_nocut.hp_map(nside)
    hp_theta = hp.pix2ang(nside, np.arange(npix))[0]
    # should be all zeros below the horizon
    assert np.allclose(beam_map[:, hp_theta > np.pi / 2], 0)
    # should be the same as the beam that we didn't cut above the horizon
    assert np.allclose(
        beam.map[:, hp_theta <= np.pi / 2],
        beam_nc_map[:, hp_theta <= np.pi / 2],
    )

    # custom horizon
    beam = Beam(data, theta, phi, frequencies=frequencies)
    beam_nocut = deepcopy(beam)
    horizon = np.ones((frequencies.size, npix))
    ipix = np.arange(npix)
    hp_theta, hp_phi = hp.pix2ang(nside, ipix)
    th_cut = hp_theta > np.pi / 2 - np.pi / 8
    ph_cut = hp_phi < np.pi
    below_pix = ipix[th_cut & ph_cut]
    above_pix = np.delete(ipix, below_pix)
    horizon[:, below_pix] = 0
    beam.horizon_cut(horizon=horizon)
    beam_map = beam.hp_map(nside)
    beam_nc_map = beam_nocut.hp_map(nside)
    # should be all zeros below the horizon
    assert np.allclose(beam_map[:, below_pix], 0)
    assert np.allclose(beam_map[:, above_pix], beam_nc_map[:, above_pix])
