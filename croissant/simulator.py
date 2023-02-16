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
        lmax=None,
        frequencies=None,
        world="moon",
        location=None,
        t_start=None,
        t_end=None,
        N_times=None,
        delta_t=None,
    ):
        """
        Simulator class. Prepares and runs simulations.
        """
        self.world = world.lower()
        # set up frequencies to run the simulation at
        if frequencies is None:
            frequencies = sky.frequencies
        self.frequencies = frequencies
        if self.world == "moon":
            Location = MoonLocation
            self.sim_coord = "M"  # mcmf
        elif self.world == "earth":
            Location = EarthLocation
            self.sim_coord = "C"  # equatorial
        else:
            raise KeyError('Keyword ``world\'\' must be "earth" or "moon".')

        try:
            self.location = Location(*location)
        except TypeError:  # location is None or already Location
            self.location = location

        if lmax is None:
            lmax = np.min([beam.lmax, sky.lmax])
        else:
            lmax = np.min([lmax, beam.lmax, sky.lmax])
        self.lmax = lmax

        if t_start is not None:
            t_start = Time(t_start, location=self.location, scale="utc")

        # initialize beam and sky
        self.beam = deepcopy(beam)
        if not hasattr(self.beam, "total_power"):
            self.beam.compute_total_power()
        if self.beam.coord != self.sim_coord:
            self.beam.switch_coords(
                self.sim_coord, loc=self.location, time=t_start
            )
        if self.beam.lmax > self.lmax:
            self.beam.reduce_lmax(self.lmax)
        self.sky = deepcopy(sky)
        if self.sky.coord != self.sim_coord:
            self.sky.switch_coords(
                self.sim_coord, loc=self.location, time=t_start
            )
        if self.sky.lmax > self.lmax:
            self.sky.reduce_lmax(self.lmax)

        # set up times to run the simulation at
        try:
            try:
                delta_t = delta_t.sec
            except AttributeError:
                warnings.warn(
                    "No units specified for delta_t, assuming seconds.",
                    UserWarning,
                )
            dt = np.arange(N_times) * delta_t  # seconds
        except TypeError:
            t_end = Time(t_end, location=self.location, scale="utc")
            total_time = (t_end - t_start).sec
            if N_times is None:
                dt = np.arange(0, total_time + delta_t, delta_t)
                N_times = len(dt)
            else:
                dt, delta_t = np.linspace(0, total_time, N_times, retstep=True)

        if t_start is None:
            self.times = dt
        else:
            self.times = t_start + dt * units.s

        self.dt = dt
        self.N_times = N_times

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
        # the rotation phases
        phases = self.sky.rot_alm_z(times=self.dt, world=self.world)
        phases.shape = (self.N_times, 1, -1)  # add freq axis
        if self.frequencies is None:
            dpss = False  # no need to compute dpss if no frequencies
            sky_alm = self.sky.alm.reshape(1, 1, -1)  # add time and freq axes
        else:
            sky_alm = np.expand_dims(self.sky.alm, axis=0)  # add time axis
        rot_sky_coeffs = sky_alm * phases

        if dpss:
            self.compute_dpss(**dpss_kwargs)
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
            beam_coeffs = np.expand_dims(self.beam.alm, axis=0)
            if self.beam.frequencies is None:  # add freq axis
                beam_coeffs = beam_coeffs.reshape(1, 1, -1)
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

        self.waterfall = np.squeeze(waterfall) / self.beam.total_power

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
        if self.times[0] == 0:
            time_label = "Time [hours]"
        else:
            t_start = self.times[0].to_value("iso", subfmt="date_hm")
            time_label = f"Hours since {t_start}"
        temp_label = "Temperature [K]"
        plt.figure(figsize=figsize)
        if self.waterfall.ndim == 1:  # no frequency axis
            plt.plot(self.dt / 3600, self.waterfall)
            plt.xlabel(time_label)
            plt.ylabel(temp_label)
        else:
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
            plt.colorbar(label=temp_label)
            plt.xlabel("Frequency [MHz]")
            plt.ylabel(time_label)
        plt.show()
