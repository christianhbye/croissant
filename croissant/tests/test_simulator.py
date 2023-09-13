from astropy import units
from astropy.coordinates import EarthLocation
from copy import deepcopy
from lunarsky import MoonLocation, Time
import numpy as np
import pytest

from croissant import Beam, dpss, Rotator, Sky
from croissant.constants import sidereal_day_earth
from croissant.simulatorbase import SimulatorBase, time_array


# define default params for simulator
lmax = 32
frequencies = np.linspace(10, 50, 10)
theta = np.linspace(0, np.pi, 181)
phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
power = frequencies[:, None, None] ** 2 * np.cos(theta[None, :, None]) ** 2
power = np.repeat(power, phi.size, axis=2)
beam = Beam.from_grid(
    power, theta, phi, lmax, frequencies=frequencies, coord="T"
)
sky = Sky.gsm(frequencies, lmax=lmax)
loc = (137.0, 40.0)  # (lon, lat) in degrees
t_start = "2022-06-10 12:59:00"
N_times = 150
delta_t = 3600 * units.s
times = time_array(t_start=t_start, N_times=N_times, delta_t=delta_t)
args = (beam, sky)
kwargs = {"lmax": lmax, "world": "moon", "location": loc, "times": times}


def test_time_array():
    # check that the times are set consistently regardless of
    # which parameters that specify it
    delta_t, step = np.linspace(0, sidereal_day_earth, N_times, retstep=True)
    delta_t = delta_t * units.s
    step = step * units.s
    t_end = Time(t_start) + delta_t[-1]
    # specify end, ntimes:
    times = time_array(t_start, t_end=t_end, N_times=N_times)
    assert np.allclose(delta_t.value, (times - times[0]).sec)
    # specify end, delta t
    times = time_array(t_start, t_end=t_end, delta_t=step)
    assert np.allclose(delta_t.value, (times - times[0]).sec)
    # specify ntimes, delta t
    times = time_array(t_start, N_times=N_times, delta_t=step)
    assert np.allclose(delta_t.value, (times - times[0]).sec)
    times = time_array(N_times=N_times, delta_t=step)
    assert np.allclose(times, np.arange(N_times) * step)
    # check that we get a UserWarning if delta t does not have units
    delta_t = 2
    with pytest.warns(UserWarning):
        time_array(t_start, t_end=t_end, delta_t=delta_t)


def test_simulator_init():
    sim = SimulatorBase(*args, **kwargs)
    # check that the simulation attributes are set properly
    assert sim.sim_coord == "M"  # mcmf
    assert sim.location == MoonLocation(*loc)
    # check sky is in the desired simulation coords
    assert sim.sky.coord == sim.sim_coord
    rot = Rotator(coord="gm")
    sky_alm = rot.rotate_alm(sky.alm, lmax=sky.lmax)
    assert np.allclose(sim.sky.alm, sky_alm)

    # test lmax
    beam_lmax = 10  # smaller than sky lmax
    beam2 = deepcopy(beam)
    beam2.reduce_lmax(beam_lmax)
    sim = SimulatorBase(beam2, sky, **kwargs)
    assert sim.lmax == np.min([sky.lmax, beam2.lmax]) == beam_lmax
    assert sim.beam.lmax == sim.sky.lmax == sim.lmax
    kwargs["lmax"] = None
    sim = SimulatorBase(beam2, sky, **kwargs)
    assert sim.lmax == np.min([sky.lmax, beam2.lmax]) == beam_lmax
    assert sim.beam.lmax == sim.sky.lmax == sim.lmax
    kwargs["lmax"] = lmax

    # use a Location object instead of a tuple
    earth_loc = EarthLocation(*loc)
    kwargs["location"] = earth_loc
    with pytest.raises(TypeError):
        SimulatorBase(*args, **kwargs)  # loc is EarthLocation, world is moon
    moon_loc = MoonLocation(*loc)
    kwargs["location"] = moon_loc
    sim = SimulatorBase(*args, **kwargs)
    assert sim.location == moon_loc

    # check that init works correctly on earth
    kwargs["world"] = "earth"
    with pytest.raises(TypeError):
        SimulatorBase(*args, **kwargs)  # loc is MoonLocation, world is earth
    kwargs["location"] = earth_loc
    sim = SimulatorBase(*args, **kwargs)
    assert sim.sim_coord == "C"
    assert sim.location == earth_loc
    kwargs["location"] = loc

    # check that we get a KeyError if world is not "earth" or "moon"
    kwargs["world"] = "mars"
    with pytest.raises(KeyError):
        SimulatorBase(*args, **kwargs)

    kwargs["world"] = "moon"


def test_compute_dpss():
    sim = SimulatorBase(*args, **kwargs)
    sim.compute_dpss(nterms=10)
    design_matrix = dpss.dpss_op(frequencies, nterms=10)
    assert np.allclose(design_matrix, sim.design_matrix)
    beam_coeff = dpss.freq2dpss(
        sim.beam.alm, frequencies, frequencies, design_matrix
    )
    assert np.allclose(beam_coeff, sim.beam.coeffs)
