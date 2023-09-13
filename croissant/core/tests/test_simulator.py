import numpy as np
from astropy import units
import healpy as hp
from croissant import Beam, Simulator, Sky
from croissant.simulatorbase import time_array

loc = (137.0, 40.0)
t_start = "2022-06-10 12:59:00"
N_times = 150
delta_t = 3600 * units.s
times = time_array(t_start=t_start, N_times=N_times, delta_t=delta_t)
kwargs = {"world": "moon", "location": loc, "times": times}


def test_run():
    # retrieve constant temperature sky
    freq = np.linspace(1, 50, 50)  # MHz
    lmax = 16
    kwargs["lmax"] = lmax
    sky_alm = np.zeros((freq.size, hp.Alm.getsize(lmax)), dtype=np.complex128)
    sky_alm[:, 0] = 10 * freq ** (-2.5)
    # sky is constant in space, varies like power law spectrally
    sky = Sky(sky_alm, lmax=lmax, frequencies=freq, coord="G")
    beam_alm = np.zeros_like(sky_alm)
    beam_alm[:, 0] = 1.0 * freq**2
    # make a constant beam with spectral power law
    beam = Beam(beam_alm, lmax=lmax, frequencies=freq, coord="T")
    # beam is no longer constant after horizon cut
    beam.horizon_cut()
    sim = Simulator(beam, sky, **kwargs)
    sim.run(dpss=False)
    beam_a00 = sim.beam[0, 0, 0]  # a00 @ freq = 1 MHz
    sky_a00 = sim.sky[0, 0, 0]  # a00 @ freq = 1 MHz
    # total spectrum should go like f ** (2 - 2.5)
    expected_vis = beam_a00 * sky_a00 * np.squeeze(freq) ** (-0.5)
    expected_vis /= sim.beam.total_power
    expected_vis.shape = (1, -1)  # add time axis
    assert np.allclose(sim.waterfall, np.repeat(expected_vis, N_times, axis=0))
    # with dpss
    sim.run(dpss=True, nterms=50)
    assert np.allclose(sim.waterfall, np.repeat(expected_vis, N_times, axis=0))

    # test with nonzero m-modes
    kwargs["times"] = None
    sky_alm = np.zeros_like(sky_alm[0])  # remove the frequency axis
    sky = Sky(sky_alm, lmax=lmax, coord="M")
    sky[0, 0] = 1e7
    sky[2, 0] = 1e4
    sky[3, 1] = -20.2 + 20.4j
    sky[6, 6] = 1.0 - 3.0j

    beam_alm = np.zeros_like(sky_alm)
    beam = Beam(beam_alm, lmax=lmax, coord="M")
    beam[0, 0] = 10
    beam[2, 0] = 5
    beam[3, 1] = 1 + 2j
    beam[6, 6] = -1 - 1.34j

    sim = Simulator(beam, sky, **kwargs)

    sim.run(dpss=False)
    expected_vis = (
        sky[0, 0] * beam[0, 0]
        + sky[2, 0] * beam[2, 0]
        + 2 * np.real(sky[3, 1] * np.conj(beam[3, 1]))
        + 2 * np.real(sky[6, 6] * np.conj(beam[6, 6]))
    )
    expected_vis /= sim.beam.total_power
    assert np.isclose(sim.waterfall, expected_vis)

    # test the einsum computation in dpss mode
    frequencies = np.linspace(1, 50, 50).reshape(-1, 1)
    beam_alm = beam.alm.reshape(1, -1) * frequencies**2
    beam = Beam(beam_alm, lmax=lmax, frequencies=frequencies, coord="M")
    sky_alm = sky.alm.reshape(1, -1) * frequencies ** (-2.5)
    sky = Sky(sky_alm, lmax=lmax, frequencies=frequencies, coord="M")
    sim = Simulator(beam, sky, **kwargs)
    sim.run(dpss=True, nterms=10)
    # expected output is dot product of alms in frequency space:
    sky_alm = sim.sky.alm
    beam_alm = sim.design_matrix @ sim.beam.coeffs
    temp_vector = np.empty(frequencies.size)
    for i in range(frequencies.size):
        t = sky_alm[i, : lmax + 1].real.dot(beam_alm[i, : lmax + 1].real)
        t += 2 * np.real(
            sky_alm[i, lmax + 1 :].dot(beam_alm[i, lmax + 1 :].conj())
        )
        temp_vector[i] = t
    # output of simulator
    wfall = sim.waterfall * sim.beam.total_power
    assert np.allclose(temp_vector, wfall)
