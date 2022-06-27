from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time as EarthTime
from copy import deepcopy
from lunarsky import MoonLocation, Time as LunarTime
import matplotlib.pyplot as plt
import numpy as np
import warnings

from . import dpss
from .rotations import rot_coords
from .healpix import Alm, grid2healpix, healpix2lonlat, map2alm


class Simulator:
    def __init__(
        self,
        beam,
        sky,
        obs_loc,
        t_start,
        moon=True,
        t_end=None,
        N_times=None,
        delta_t=None,
        frequencies=None,
        horizon=None,
        lmax=16,
    ):
        """
        Simulator class. Prepares and runs simulations.
        """
        self.lmax = lmax
        self.moon = moon
        # set up frequencies to run the simulation at
        if frequencies is None:
            frequencies = sky.frequencies
        self.frequencies = frequencies
        if moon:
            loc_object = MoonLocation
            self.sim_coords = "mcmf"  # simulation coordinate system
            Time = LunarTime  # define default time class
        else:
            loc_object = EarthLocation
            self.sim_coords = "equatorial"
            Time = EarthTime

        if isinstance(obs_loc, loc_object):
            self.loc = obs_loc
        else:
            lat, lon, alt = obs_loc
            self.loc = loc_object(
                lat=lat * units.deg,
                lon=lon * units.deg,
                height=alt * units.m,
            )

        self.t_start = Time(t_start, location=self.loc, scale="utc")
        if delta_t is not None:
            try:
                delta_t = delta_t.to_value(units.s)
            except AttributeError:
                warnings.warn(
                    "No units specified for delta_t, assuming seconds.",
                    UserWarning,
                )
                delta_t = delta_t
        if t_end is None:
            dt = np.arange(N_times) * delta_t
        else:
            t_end = Time(t_end, location=self.loc)
            total_time = (t_end - self.t_start).to_value(units.s)
            if delta_t is not None:
                dt = np.arange(0, total_time + delta_t, delta_t)
                N_times = len(dt)
            else:
                dt = np.linspace(0, total_time, N_times)
        self.dt = dt
        self.N_times = N_times
        beam = deepcopy(beam)
        if beam.alm is None:
            # apply horizon mask and initialize beam
            beam.horizon_cut(horizon=horizon)
            self.beam = beam
            self.beam_alm()  # compute alms in ra/dec
        else:
            self.beam = beam
        # initialize sky
        self.sky = Alm.from_healpix(sky, lmax=self.lmax)
        if self.sky.coords.lower() != self.sim_coords:
            self.sky.switch_coords(self.sim_coords)

    def beam_alm(self, nside=128):
        """
        Get the alm's of the beam in the equitorial coordinate system.
        """
        # get lon/lat in sim coordinates at healpix centers
        lon, lat = healpix2lonlat(nside)

        # get corresponding theta/phi in topocentric coords
        za = np.pi / 2 - np.deg2rad(lat)
        az = np.deg2rad(lon)
        theta, phi = rot_coords(
            za,
            az,
            self.sim_coords,
            self.beam.coords.lower(),
            time=self.t_start,
            loc=self.loc,
            lonlat=False,
        )

        pixel_centers = np.array([theta, phi]).T
        # get healpix map
        hp_maps = grid2healpix(
            self.beam.data,
            nside,
            self.beam.theta,
            self.beam.phi,
            pixel_centers=pixel_centers,
        )
        self.beam.alm = map2alm(hp_maps, self.lmax)

    def compute_dpss(self, **kwargs):
        # generate the set of target frequencies (subset of all freqs)
        x = np.unique(
            np.concatenate(
                (
                    self.beam.frequencies,
                    self.frequencies,
                ),
                axis=None,
            )
        )

        self.design_matrix = dpss.dpss_op(x, **kwargs)
        self.beam.coeffs = dpss.freq2dpss(
            self.beam.alm,
            self.beam.frequencies,
            self.frequencies,
            self.design_matrix,
        )

    def run(self, dpss=True, **dpss_kwargs):
        """
        Compute the convolution for a range of times.
        """
        if self.moon:
            world = "moon"
        else:
            world = "earth"
        # the rotation phases
        phases = self.sky.rotate_alm_time(self.dt, world=world)
        if dpss:
            self.compute_dpss(**dpss_kwargs)
            # get the sky coefficients at each time
            rot_sky_coeffs = np.expand_dims(self.sky.alm, axis=0) * phases
            # m = 0 modes
            res = (
                rot_sky_coeffs[:, :, : self.lmax + 1].real
                @ self.beam.coeffs[:, : self.lmax + 1].T.real
            )
            # for m != 0 we must account for +m and -m
            res += 2 * np.real(
                rot_sky_coeffs[:, :, self.lmax + 1 :]
                @ self.beam.coeffs[:, self.lmax + 1 :].T.conj()
            )

            waterfall = np.einsum("ijk, jk -> ij", res, self.design_matrix)

        else:
            # add time dimensions and rotate the sky alms
            rot_sky_coeffs = np.expand_dims(self.sky.alm, axis=0) * phases
            beam_coeffs = np.expand_dims(self.beam.alm, axis=0)
            # m = 0 modes (already real)
            waterfall = np.einsum(
                "ijk, ijk -> ij",
                rot_sky_coeffs[:, :, : self.lmax + 1].real,
                beam_coeffs[:, :, : self.lmax + 1].real,
            )
            # for m != 0 we must account for +m and -m
            waterfall += 2 * np.real(
                np.einsum(
                    "ijk, ijk -> ij",
                    rot_sky_coeffs[:, :, self.lmax + 1 :],
                    beam_coeffs.conj()[:, :, self.lmax + 1 :],
                )
            )

        norm = self.beam.total_power.reshape(1, -1)
        self.waterfall = waterfall / norm

    def plot(
        self,
        figsize=None,
        extent=None,
        interpolation="none",
        aspect="auto",
        power=0,
    ):
        """
        Plot the result of the simulation.
        """
        plt.figure(figsize=figsize)
        if extent is None:
            extent = [
                self.frequencies.min(),
                self.frequencies.max(),
                self.dt[-1] / 3600,
                0,
            ]
        weight = self.frequencies**power
        plt.imshow(
            self.waterfall * weight.reshape(1, -1),
            extent=extent,
            aspect=aspect,
            interpolation=interpolation,
        )
        plt.colorbar(label="Temperature [K]")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel(
            f"Hours since {self.t_start.to_value('iso', subfmt='date_hm')}"
        )
        plt.show()
