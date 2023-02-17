from healpy import npix2nside, pix2ang
import numpy as np

from .constants import Y00
from .healpix import Alm
from .sphtransform import map2alm


class Beam(Alm):
    def compute_total_power(self):
        """
        Compute the total integrated power in the beam at each frequency. This
        is a necessary normalization constant for computing the visibilities.
        It should be computed before applying the horizon cut in order to
        account for ground loss.
        """
        if self.alm.ndim == 2:
            a00 = self[:, 0, 0]
        else:
            a00 = self[0, 0]
        power = a00.real * Y00 * 4 * np.pi
        self.total_power = power

    def horizon_cut(self, horizon=None, nside=128):
        """
        horizon : array-like
        nside : int
            The resolution of the healpix map for the intermediate step.
        """
        if horizon is not None:
            if horizon.min() < 0 or horizon.max() > 1:
                raise ValueError("Horizon elements must be in [0, 1].")
            npix = horizon.shape[-1]
            nside = npix2nside(npix)

        hp_beam = self.hp_map(nside=nside)
        if horizon is None:
            horizon = np.ones_like(hp_beam)
            npix = horizon.shape[-1]
            theta = pix2ang(nside, np.arange(npix))[0]
            horizon[:, theta > np.pi / 2] = 0

        hp_beam *= horizon
        self.alm = map2alm(hp_beam, lmax=self.lmax)
