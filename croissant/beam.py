from healpy import nside2npix
import numpy as np
from .healpix import grid2healpix


class Beam:
    def __init__(
        self,
        data,
        theta,
        phi,
        frequencies=None,
    ):
        """
        Initialize beam object. The beam must be specified at a rectangular
        grid, that is, theta and phi must be coordinate axis of a grid.

        Parameters
        ----------
        data : array-like
            The power beam. Must have shape ([freqs,] theta, phi).
        theta : array-like
            Zenith angle(s) in radians. Must be in [0, pi].
        phi : array-like
            Azimuth angle(s) in radians. Must be in [0, 2*pi).
        frequencies : array-like (optional)
            The frequencies in MHz of the beam. Necessary if the beam is
            specified at more than one frequency.

        """
        data = np.array(data, copy=True)
        self.frequencies = np.ravel(frequencies).copy()
        self.nfreqs = self.frequencies.size
        self.theta = np.ravel(theta).copy()
        self.phi = np.ravel(phi).copy()
        if self.theta.min() < 0 or self.theta.max() > np.pi:
            raise ValueError("Theta must be in the range [0, pi].")
        if self.phi.min() < 0 or self.phi.max() >= 2 * np.pi:
            raise ValueError("Phi must be in the range [0, 2pi).")
        if not (
            np.allclose(np.diff(self.theta), self.theta[1] - self.theta[0])
            and np.allclose(np.diff(self.phi), self.phi[1] - self.phi[0])
        ):
            raise ValueError("The data must be sampled on a rectangular grid.")
        data.shape = (self.nfreqs, theta.size, phi.size)
        self.data = data
        self.total_power = self.compute_total_power()  # before horizon cut

    def compute_total_power(self, nside=128):
        """
        Compute the total integrated power in the beam at each frequency. This
        is a necessary normalization constant for computing the visibilities.
        It should be computed before applying the horizon cut in order to
        account for ground loss.
        """
        healpix_beam = grid2healpix(
            self.data, nside, theta=self.theta, phi=self.phi
        )
        npix = nside2npix(nside)
        power = healpix_beam.sum(axis=-1) * 4 * np.pi / npix
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
        if horizon is None:
            horizon = np.ones_like(self.data)
            horizon[:, self.theta > np.pi / 2] = 0
        else:
            horizon = np.array(horizon, copy=True)
            horizon.shape = (-1, self.theta.size, self.phi.size)
            if horizon.min() < 0 or horizon.max() > 1:
                raise ValueError("Horizon elements must be in [0, 1].")
        self.data = self.data * horizon

    @classmethod
    def from_file(path):
        raise NotImplementedError

    def to_file(fname):
        raise NotImplementedError
