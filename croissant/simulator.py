import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from .healpix import Alm


class Simulator:
    def __init__(self, beam, sky, times):
        """
        Simulator class. Prepares and runs simulations.
        """
        assert np.allclose(
            beam.frequencies, sky.frequencies
        ), "Frequencies don't match."
        self.frequencies = beam.frequencies
        times = np.array(times)
        if times.ndim == 0:
            times = np.expand_dims(times, axis=0)
        self.times = times - times[0]
        self.beam = beam
        self.sky = Alm.from_healpix(sky, lmax=self.beam.lmax)
        ntimes = len(self.times)
        nfreqs = len(self.frequencies)
        self.waterfall = np.empty((ntimes, nfreqs))

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
