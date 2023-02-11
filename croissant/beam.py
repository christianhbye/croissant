from healpy import nside2npix
import numpy as np
import warnings

from .constants import Y00
from .healpix import HealpixMap, grid2healpix



    def compute_total_power(self, nside=128):
        """
        Compute the total integrated power in the beam at each frequency. This
        is a necessary normalization constant for computing the visibilities.
        It should be computed before applying the horizon cut in order to
        account for ground loss.
        """
        if self.data is not None:  # from grid
            healpix_beam = grid2healpix(
                self.data, nside, theta=self.theta, phi=self.phi
            )
            npix = nside2npix(nside)
            power = healpix_beam.sum(axis=-1) * 4 * np.pi / npix
        else:  # from alm
            power = self.alm[0, 0] * Y00 * 4 * np.pi
        return power

    def horizon_cut(self, horizon=None):
        """
        horizon : array-like (optional)
            An array indicating if a given (frequency/)theta/phi combination is
            above the horizon or not. Must have shape (theta, phi) or
            (freqs, theta, phi) if frequency dependent. The array will be
            multiplied by the antenna beam, hence 0 is intepreted as below the
            horizon and 1 is interpreted as above it. The elements must be in
            the range [0, 1].
        """
        if self.data is None:

        if horizon is None:
            horizon = np.ones_like(self.data)
            horizon[:, self.theta > np.pi / 2] = 0
        else:
            horizon = np.array(horizon, copy=True)
            horizon.shape = (-1, self.theta.size, self.phi.size)
            if horizon.min() < 0 or horizon.max() > 1:
                raise ValueError("Horizon elements must be in [0, 1].")
        self.data = self.data * horizon

