from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from scipy.interpolate import RectSphereBivariateSpline
import warnings

from . import dpss
from .coordinates import topo_to_radec
from .healpix import Alm, grid2alm


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
        horizon=None,
    ):
        """
        Simulator class. Prepares and runs simulations.
        """
        # set up frequencies to run the simulation at
        if frequencies is None:
            frequencies = sky.frequencies
        self.frequencies = frequencies
        # set up the location of the telescope as astropy.EarthLocation object
        self.loc = EarthLocation(
            lat=obs_lat * units.deg,
            lon=obs_lon * units.deg,
            height=obs_alt * units.m,
        )
        # set up observing time as astropy.Time object
        self.t_start = Time(t_start, location=self.loc)
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
                dt = np.arange(0, total_time, delta_t)
                N_times = len(dt)
            else:
                dt = np.linspace(0, total_time, N_times)
        self.dt = dt
        self.N_times = N_times
        # apply horizon mask to beam
        beam.horizon_cut(horizon=horizon)
        self.beam = beam
        # initialize sky
        self.sky = Alm.from_healpix(sky, lmax=self.beam.lmax)
        if self.sky.coords != "equitorial":
            self.sky.switch_coords("equitorial")

    def _prepare_beam(self):
        # rotate to ra/dec at first observing time given location
        ph, th = self.beam.phi, self.beam.theta
        ra, dec = topo_to_radec(ph, th, self.t_start, self.loc)
        # interpolate to ra/decs to get even sampling
        dec = np.pi / 2 - dec  # colatitude, [0, pi]
        ra -= np.pi  # move to [-pi, pi)
        smooth_dec = np.linspace(0, np.pi, 181)
        smooth_ra = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        interp = RectSphereBivariateSpline(dec, ra, self.beam.data)
        interp_beam = interp(smooth_dec, smooth_ra)
        # compute alms of beam from ra/dec
        smooth_ra += np.pi  # must be 0-2pi for scipy spherical harms
        alm = grid2alm(interp_beam, smooth_dec, smooth_ra, lmax=self.lmax)
        self.beam.alm = alm

    def _compute_dpss(self, nterms=10):
        # generate the set of target frequencies (subset of all freqs)
        x = np.unique(
            np.concatenate(
                (
                    self.sky.frequencies,
                    self.beam.frequencies,
                    self.frequencies,
                ),
                axis=None,
            )
        )

        self.design_matrix = dpss.dpss_op(x, nterms=nterms)
        self.sky.coeffs = dpss.freq2dpss(
            self.sky.alm,
            self.sky.frequencies,
            self.frequencies,
            self.design_matrix,
        )
        self.beam.coeffs = dpss.freq2dpss(
            self.beam.alm,
            self.beam.frequencies,
            self.frequencies,
            self.design_matrix,
        )

    def _run_onetime(self, time, index=None):
        """
        Compute the convolution for one specfic time.
        """
        sky_coeffs = self.sky.coeffs * self.sky.rotate_z_time(time)
        prod = self.beam.coeffs * sky_coeffs
        conv = prod.sum(axis=1)
        return index, conv.real

    def run(self, parallel=False, **kwargs):
        """
        Compute the convolution for a range of times.
        """
        self._prepare_beam()
        nterms = kwargs.pop("nterms", None)
        self._compute_dpss(nterms=nterms)
        if nterms is None:
            nterms = self.beam.coeffs.shape[0]
        waterfall_dpss = np.empty((self.N_times, nterms), dtype="complex")
        if parallel:
            ncpu = kwargs.pop("ncpu", None)

            def get_res(result):
                i, conv = result
                waterfall_dpss[i] = conv

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
            waterfall_dpss[i] = conv

        # convert back to frequency
        self.waterfall = self.design_matrix @ waterfall_dpss.T

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
