from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import warnings

from . import dpss
from .constants import Y00
from .coordinates import radec2topo
from .healpix import Alm, grid2healpix, healpix2lonlat, map2alm


class Simulator:
    def __init__(
        self,
        beam,
        sky,
        obs_loc,
        t_start,
        t_end=None,
        N_times=None,
        delta_t=None,
        frequencies=None,
        horizon=None,
        lmax=32,
        dpss_nterms=10,
    ):
        """
        Simulator class. Prepares and runs simulations.
        """
        self.lmax = lmax
        self.nterms = dpss_nterms
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
        if self.sky.coords != "equatorial":
            self.sky.switch_coords("equatorial")
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

    def compute_dpss(self):
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

        self.design_matrix = dpss.dpss_op(x, nterms=self.nterms)
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

    @staticmethod
    def alm_dot(sky, beam):
        """
        Compute the alm dot product between sky and beam. This assumes healpix
        convention for alms. The sky and beam may have extra dimensions
        (like a frequency axis). These must come before the alm axis, that is,
        the sky can have shape (N_freq, N_alm) for example.

        Parameters
        ----------
        sky : array-like
            The alms of the sky.
        beam : array-like
            The alms of the beam.

        Returns
        -------
        conv : np.ndarray
            The dot product of the sky and beam alms. This is the convolution
            of the sky and the beam since the spherical harmonics are
            orthonormal.

        """
        # transpose data so alm axis is first
        sky = np.array(sky, copy=True).T
        beam = np.array(beam, copy=True).T
        lmax = hp.Alm.getlmax(beam.shape[0])
        # m = 0 modes are already real:
        al0 = sky[: lmax + 1].real * beam[: lmax + 1].real
        # m != 0 (see docs/math for derivation):
        alm = 2 * (sky[lmax + 1 :] * beam[lmax + 1 :].conj()).real
        conv = al0.sum(axis=0) + alm.sum(axis=0)
        return conv.T

    def _run_onetime(self, time):
        """
        Compute the convolution for one specfic time.
        """
        sky_coeffs = self.sky.coeffs * self.sky.rotate_z_time(time)
        conv = dpss.dpss2freq(
            self.alm_dot(sky_coeffs, self.beam.coeffs), self.design_matrix
        )
        # normalize by beam integral over sphere = a00 * Y00 * 4pi
        norm = dpss.dpss2freq(
            self.beam.coeffs[:, 0].real * Y00 * 4 * np.pi,
            self.design_matrix,
        )
        return conv / norm

    def run(self):
        """
        Compute the convolution for a range of times.
        """
        waterfall = np.empty((self.N_times, self.frequencies.size))
        for i, t in enumerate(self.dt):
            conv = self._run_onetime(t)
            waterfall[i] = conv

        self.waterfall = waterfall

    def plot(self, **kwargs):
        """
        Plot the result of the simulation.
        """
        figsize = kwargs.pop("figsize", None)
        plt.figure(figsize=figsize)
        _extent = [
            self.frequencies.min(),
            self.frequencies.max(),
            self.dt[-1],
            self.dt[0],
        ]
        extent = kwargs.pop("extent", _extent)
        interpolation = kwargs.pop("interpolation", "none")
        aspect = kwargs.pop("aspect", "auto")
        power = kwargs.pop("power", 0)
        weight = self.frequencies**power
        plt.imshow(
            self.waterfall * weight.reshape(1, -1),
            extent=extent,
            aspect=aspect,
            interpolation=interpolation,
        )
        plt.colorbar(label="Temperature [K]")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
