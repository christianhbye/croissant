from astropy.time import Time
from astropy import units
import healpy
import numpy as np

from croissant.beam import Beam
from croissant import dpss
from croissant.constants import sidereal_day
from croissant.coordinates import radec2topo, rotate_alm
from croissant.healpix import alm2map, grid_interp
from croissant.simulator import Simulator
from croissant.sky import Sky

# define default params for simulator

theta = np.linspace(0, np.pi, 181)
phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
frequencies = np.linspace(1, 50, 50)
theta, frequencies, phi = np.meshgrid(theta, frequencies, phi, sparse=True)
power = frequencies**2 * np.cos(theta) ** 2  # dipole
power = np.repeat(power, phi.size, axis=2)
beam = Beam(power, theta, phi, frequencies=frequencies)
sky = Sky.gsm(25)
sky.power_law_map(frequencies)
loc = (40.0, 137.0, 0.0)
t_start = "2022-06-10 12:59:00"
N_times = 250
delta_t = 3600 * units.s
lmax = 32


def test_simulator_init():
    # check that the times are set conisstently regardless of
    # which parameters that specify it
    delta_t, step = np.linspace(0, sidereal_day, N_times, retstep=True)
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
    beam.horizon_cut()
    assert np.allclose(sim.beam.data, beam.data)

    # check sky is in the ra/dec coords
    assert sim.sky.coords == "equatorial"
    sky_alm = rotate_alm(sky.alm(lmax=lmax))
    assert np.allclose(sim.sky.alm, sky_alm)


def test_beam_alm():
    sim = Simulator(beam, sky, loc, t_start, N_times=N_times, delta_t=delta_t)

    # test beam alm by inverting it
    nside = 64
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
    sim = Simulator(beam, sky, loc, t_start, N_times=N_times, delta_t=delta_t)

    # 10 is the default nterms in Simulator.__init__
    design_matrix = dpss.dpss_op(frequencies, nterms=10)
    assert np.allclose(design_matrix, sim.design_matrix)
    sky_alm = rotate_alm(sky.alm(lmax=lmax))
    sky_coeff = dpss.freq2dpss(
        sky_alm, frequencies, frequencies, design_matrix
    )
    assert np.allclose(sky_coeff, sim.sky.coeffs)