from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import warnings
from .healpix import Alm


class Simulator:
    def __init__(
        self,
        beam,
        sky,
        obs_lat,
        obs_lon,
        obs_alt,
        t_start,
        t_end=None,
        N_times=None,
        delta_t=None,
        frequencies=None,
    ):
        """
        Simulator class. Prepares and runs simulations.
        """
        if frequencies is None:
            frequencies = sky.frequencies
        self.frequencies = frequencies
        self.loc = EarthLocation(
            lat=obs_lat * units.deg,
            lon=obs_lon * units.deg,
            height=obs_alt * units.m,
        )
        t_start = Time(t_start, location=self.loc)
        if delta_t is not None:
            try:
                delta_t = delta_t.to(units.s)
            except AttributeError:
                warnings.warn(
                    "No units specified for delta_t, assuming seconds.",
                    UserWarning,
                )
                delta_t = delta_t * units.s
        if t_end is None:
            dt = np.arange(N_times) * delta_t
        else:
            t_end = Time(t_end, location=self.loc)
            total_time = (t_end - t_start).to_value(units.s)
            if delta_t is not None:
                dt = np.arange(0, total_time.value, delta_t.value)
                N_times = len(dt)
                dt *= units.s
            else:
                dt = np.linspace(0, total_time.value, N_times) * units.s
        self.times = t_start + dt

        self.beam = beam
        self.sky = Alm.from_healpix(sky, lmax=self.beam.lmax)

        # horizon cutoff
        # rotate beam and sky to correct coords

        nfreqs = len(self.frequencies)
        self.waterfall = np.empty((N_times, nfreqs))

    def _run_onetime(self, time, index=None):
        """
        Compute the convolution for one specfic time.
        """
        beam_alm = self.beam.alm
        sky_alm = self.sky.alm * self.sky.rotate_z_time(time)
        prod = beam_alm * sky_alm
        conv = prod.sum(axis=1)
        return index, conv.real

    def run(self, parallel=False, **kwargs):
        """
        Compute the convolution for a range of times.
        """
        if parallel:
            ncpu = kwargs.pop("ncpu", None)

            def get_res(result):
                i, conv = result
                self.waterfall[i] = conv

            with Pool(processes=ncpu) as pool:
                for i, t in enumerate(self.times):
                    pool.apply_async(
                        self._run_onetime,
                        args=t,
                        kwds={"index": i},
                        callback=get_res,
                    )

        for i, t in enumerate(self.times):
            conv = self._run_onetime(t)[1]
            self.waterfall[i] = conv

    def plot(self, **kwargs):
        """
        Plot the result of the simulation.
        """
        plt.figure()
        _extent = [
            self.frequencies.min(),
            self.frequencies.max(),
            self.times.max(),
            self.times.min(),
        ]
        extent = kwargs.pop("extent", _extent)
        interpolation = kwargs.pop("interpolation", "none")
        aspect = kwargs.pop("aspect", "auto")
        plt.imshow(
            self.waterfall,
            extent=extent,
            aspect=aspect,
            interpolation=interpolation,
        )
        plt.colorbar(label="Temperature [K]")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
