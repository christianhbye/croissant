from astropy import units
from astropy.coordinates import EarthLocation
from copy import deepcopy
from lunarsky import MoonLocation, Time
import matplotlib.pyplot as plt
import numpy as np
import warnings

from . import dpss


class Simulator:
    def __init__(
        self,
        beam,
        sky,
        obs_loc,
        t_start,
        world="moon",
        t_end=None,
        N_times=None,
        delta_t=None,
        frequencies=None,
        lmax=16,
    ):
        """
        Simulator class. Prepares and runs simulations.
        """
        self.lmax = lmax
        self.world = world.lower()
        # set up frequencies to run the simulation at
        if frequencies is None:
            frequencies = sky.frequencies
        self.frequencies = frequencies
        if self.world == "moon":
            Location = MoonLocation
            self.sim_coords = "mcmf"  # simulation coordinate system
        elif self.world == "earth":
            Location = EarthLocation
            self.sim_coords = "equatorial"
        else:
            raise KeyError('Keyword ``world\'\' must be "earth" or "moon".')

        if isinstance(obs_loc, Location):
            self.loc = obs_loc
        else:
            self.loc = Location(*obs_loc)

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

        # initialize beam
        self.beam = deepcopy(beam)
        if self.beam.coords != self.sim_coords:
            self.beam.switch_coords(self.sim_coords)

        # initialize sky
        self.sky = deepcopy(sky)
        if self.sky.coords != self.sim_coords:
            self.sky.switch_coords(self.sim_coords)

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
