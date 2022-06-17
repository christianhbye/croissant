from astropy import units
from copy import deepcopy
import healpy
from lunarsky import Time
import numpy as np

from croissant.beam import Beam
from croissant import dpss
from croissant.constants import sidereal_day_earth
from croissant.rotations import radec2topo, rotate_alm
from croissant.healpix import Alm, alm2map, grid_interp, HealpixMap
from croissant.simulator import Simulator
from croissant.sky import Sky


# define default params for simulator

frequencies = np.linspace(1, 50, 50).reshape(-1, 1, 1)
theta = np.linspace(0, np.pi, 181).reshape(1, -1, 1)
phi = np.linspace(0, 2 * np.pi, 360, endpoint=False).reshape(1, 1, -1)
power = frequencies**2 * np.cos(theta) ** 2  # dipole
power = np.repeat(power, phi.size, axis=2)
beam = Beam(power, theta=theta, phi=phi, frequencies=frequencies)
sky = Sky.gsm(frequencies, power_law=True, gen_freq=25, spectral_index=-2.5)
loc = (40.0, 137.0, 0.0)
t_start = "2022-06-10 12:59:00"
N_times = 250
delta_t = 3600 * units.s
lmax = 32


def test_simulator_init():
    # check that the times are set conisstently regardless of
    # which parameters that specify it
    delta_t, step = np.linspace(0, sidereal_day_earth, N_times, retstep=True)
    step = step * units.s
    t_end = Time(t_start) + delta_t[-1] * units.s
    # specify end, ntimes:
    sim = Simulator(
        beam, sky, loc, t_start, t_end=t_end, N_times=N_times, lmax=lmax
    )
    assert np.allclose(delta_t, sim.dt)
    assert np.isclose(N_times, sim.N_times)
    # specify end, delta t
    sim = Simulator(
        beam, sky, loc, t_start, t_end=t_end, delta_t=step, lmax=lmax
    )
    assert np.allclose(delta_t, sim.dt)
    assert np.isclose(N_times, sim.N_times)
    # specify ntimes, delta t
    sim = Simulator(
        beam, sky, loc, t_start, N_times=N_times, delta_t=step, lmax=lmax
    )
    assert np.allclose(delta_t, sim.dt)
    assert np.isclose(N_times, sim.N_times)

    # check that power is computed correctly before horizon cut
    assert np.allclose(sim.beam.total_power, beam.total_power)
    # check that the init does the horizon cut
    beam_copy = deepcopy(beam)
    beam_copy.horizon_cut()
    assert np.allclose(sim.beam.data, beam_copy.data)

    # check that the simulation coords are set properly
    assert sim.sim_coords == "mcmf"
    # check sky is in the desired simulation coords
    assert sim.sky.coords == sim.sim_coords
    sky_alm = rotate_alm(sky.alm(lmax=lmax))
    assert np.allclose(sim.sky.alm, sky_alm)

    # check that init works correcttly on earth
    sim = Simulator(
        beam,
        sky,
        loc,
        t_start,
        moon=False,
        N_times=N_times,
        delta_t=step,
        lmax=lmax,
    )
    assert sim.sim_coords == "equatorial"


def test_beam_alm():
    sim = Simulator(
        beam,
        sky,
        loc,
        t_start,
        moon=False,
        N_times=N_times,
        delta_t=delta_t,
        lmax=lmax,
    )

    # check that input args are set correctly
    assert sim.beam.coords.lower() == "topocentric"
    assert sim.sim_coords == "equatorial"

    # test beam alm by inverting it
    nside = 128
    beam_map = alm2map(sim.beam.alm, nside=nside)  # in healpix ra/dec
    pix = np.arange(healpy.nside2npix(nside))
    ra, dec = healpy.pix2ang(nside, pix, nest=False, lonlat=True)
    inv_theta, inv_phi = radec2topo(ra, dec, t_start, loc)
    interp_sim_beam = grid_interp(
        sim.beam.data, theta, phi, inv_theta, inv_phi
    )
    diff = beam_map - interp_sim_beam
    rms = np.sqrt(np.mean(diff**2, axis=1)) / np.mean(
        interp_sim_beam, axis=1
    )
    assert rms.max() < 1e-3


def test_compute_dpss():
    sim = Simulator(
        beam, sky, loc, t_start, N_times=N_times, delta_t=delta_t, lmax=lmax
    )
    sim.compute_dpss(nterms=10)
    design_matrix = dpss.dpss_op(frequencies, nterms=10)
    assert np.allclose(design_matrix, sim.design_matrix)
    sky_alm = rotate_alm(sky.alm(lmax=lmax))
    sky_coeff = dpss.freq2dpss(
        sky_alm, frequencies, frequencies, design_matrix
    )
    assert np.allclose(sky_coeff, sim.sky.coeffs)


def test_run():
    # retrieve constant temperature (spatially) sky with dipole beam
    # sky has power law of index -2.5, beam of index 2
    sky = Sky()
    nside = 32
    npix = healpy.nside2npix(nside)
    frequencies = np.linspace(1, 50, 50)
    sky.power_law_map(
        frequencies,
        spectral_index=-2.5,
        ref_map=np.ones(npix) * 1e7,
        ref_freq=frequencies[0],
    )
    theta = np.linspace(0, np.pi, 181)
    phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    theta, frequencies, phi = np.meshgrid(theta, frequencies, phi, sparse=True)
    power = frequencies**2 * np.cos(theta) ** 2  # dipole
    power = np.repeat(power, phi.size, axis=2)
    beam = Beam(power, theta=theta, phi=phi, frequencies=frequencies)
    sim = Simulator(
        beam, sky, loc, t_start, N_times=N_times, delta_t=delta_t, lmax=lmax
    )
    sim.run(dpss=False)
    beam_a00 = sim.beam.alm[0, 0]  # a00 @ freq = 1 MHz
    sky_a00 = sim.sky.alm[0, 0]  # a00 @ freq = 1 MHz
    # resulting visibility spectrum should go as nu^-.5
    expected_vis = beam_a00 * sky_a00 * np.squeeze(frequencies) ** (-0.5)
    expected_vis /= sim.beam.total_power
    expected_vis.shape = (1, -1)  # add time axis
    assert np.allclose(sim.waterfall, np.repeat(expected_vis, N_times, axis=0))
    # with dpss
    sim.run(dpss=True, dpss_nterms=50)
    assert np.allclose(sim.waterfall, np.repeat(expected_vis, N_times, axis=0))

    # test with nonzero m-modes
    sky_alm = Alm(lmax=lmax, coords="mcmf")
    sky_alm[0, 0] = 1e7
    sky_alm[2, 0] = 1e4
    sky_alm[3, 1] = -20.2 + 20.4j
    sky_alm[6, 6] = 1.0 - 3.0j
    sky = HealpixMap.from_alm(sky_alm, nside=128)

    beam00 = 10  # a00
    beam20 = 5  # a20
    beam31 = 1 + 2j  # a31
    beam66 = -1 - 1.34j  # a66
    beam_alm = np.zeros((1, healpy.Alm.getsize(lmax)), dtype=np.complex128)
    beam_alm[0, 0] = beam00
    beam_alm[0, 2] = beam20
    ix31 = healpy.Alm.getidx(lmax, 3, 1)
    beam_alm[0, ix31] = beam31
    ix66 = healpy.Alm.getidx(lmax, 6, 6)
    beam_alm[0, ix66] = beam66
    beam = Beam(beam_alm, alm=True, coords="mcmf")

    sim = Simulator(
        beam,
        sky,
        loc,
        t_start,
        N_times=1,
        delta_t=delta_t,
        lmax=lmax,
    )

    sim.run(dpss=False)
    expected_vis = (
        sky_alm[0, 0] * beam00
        + sky_alm[2, 0] * beam20
        + 2 * np.real(sky_alm[3, 1] * np.conj(beam31))
        + 2 * np.real(sky_alm[6, 6] * np.conj(beam66))
    )
    expected_vis /= sim.beam.total_power
    assert np.isclose(sim.waterfall, expected_vis)
