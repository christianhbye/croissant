from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import warnings

from . import dpss
from .rotations import radec2topo
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
        lmax=16,
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
        sim_beam = deepcopy(beam)
        if sim_beam.alm is None:
            # apply horizon mask and initialize beam
            sim_beam.horizon_cut(horizon=horizon)
            self.beam = sim_beam
            self.beam_alm()  # compute alms in ra/dec
        else:
            self.beam = sim_beam
        # initialize sky
        self.sky = Alm.from_healpix(sky, lmax=self.lmax)
        if self.sky.coords != "equatorial":
            self.sky.switch_coords("equatorial")

    def beam_alm(self, nside=128):
        """
        Get the alm's of the beam in the equitorial coordinate system.
        """
        # get ra/dec at healpix centers
        ra, dec = healpix2lonlat(nside)

        # get corresponding theta/phi
        if self.beam.coords.lower() == "topocentric":
            theta, phi = radec2topo(ra, dec, self.t_start, self.loc)
        elif self.beam.coords.lower() == "equatorial":
            theta = np.pi / 2 - np.deg2rad(dec)
            phi = np.deg2rad(ra)
        else:
            raise ValueError(
                f"Cannot convert from {self.beam.coordinates} to ra/dec."
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

    def run(self, dpss=True, dpss_nterms=30):
        """
        Compute the convolution for a range of times.
        """
        phases = self.sky.rotate_alm_time(self.dt)  # the rotation phases
        if dpss:
            self.nterms = dpss_nterms
            self.compute_dpss()
            # get the sky coefficients at each time
            rot_sky_coeffs = np.expand_dims(self.sky.coeffs, axis=0) * phases
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

            waterfall = np.einsum(
                "jk, ikj -> ij",
                self.design_matrix,
                res @ self.design_matrix.T,
            )

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
