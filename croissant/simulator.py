from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from scipy.interpolate import RectSphereBivariateSpline
import warnings

from . import dpss
from .beam import interpolate
from .coordinates import healpix2lonlat, radec2topo
from .healpix import Alm, map2alm


class Simulator:
    def __init__(
        self,
        beam,
        sky,
        lmax,
        obs_loc,
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
        self.lmax = lmax
        # set up frequencies to run the simulation at
        if frequencies is None:
            frequencies = sky.frequencies
        self.frequencies = frequencies
        # set up the location of the telescope as astropy.EarthLocation object
        lat, lon, alt = obs_loc
        self.loc = EarthLocation(
            lat=lat * units.deg, lon=lon * units.deg, height=alt * units.m
        )
        # set up observing time as astropy.Time object
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
                dt = np.arange(0, total_time, delta_t)
                N_times = len(dt)
            else:
                dt = np.linspace(0, total_time, N_times)
        self.dt = dt
        self.N_times = N_times
        # apply horizon mask and initialize beam
        beam.horizon_cut(horizon=horizon)
        self.beam = beam
        self.beam_alm()
        # initialize sky
        self.sky = Alm.from_healpix(sky, lmax=self.lmax)
        if self.sky.coords != "equitorial":
            self.sky.switch_coords("equitorial")
        # compute dpss coeffs
        self.compute_dpss()

    def beam_alm(self, nside=64):
        """
        Get the alm's of the beam in the equitorial coordinate system.
        """
        # get ra/dec at healpix centers
        ra, dec = healpix2lonlat(nside)
        # get corresponding theta/phi
        theta, phi = radec2topo(ra, dec, self.t_start, self.loc)
        # interpolate beam to those theta/phi's
        interp_beam = interpolate(
            self.beam.data, self.beam.theta, self.beam.phi, theta, phi
        )
            
        self.beam.alm = map2alm(interp_beam, self.lmax)

    def compute_dpss(self, nterms=10):
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

    def run(self, parallel=False):
        """
        Compute the convolution for a range of times.
        """
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

        else:
            for i, t in enumerate(self.times):
                print(f"{t=}")
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
