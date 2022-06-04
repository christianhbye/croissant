import numpy as np


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
        grid for most methods (including simulation methods) to work.

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
        data = np.array(data)
        self.frequencies = np.squeeze(frequencies).reshape(-1)
        self.nfreqs = self.frequencies.size
        self.theta = np.squeeze(theta).reshape(-1)
        self.phi = np.squeeze(phi).reshape(-1)
        data.shape = (self.nfreqs, theta.size, phi.size)
        self.data = data

    def horizon_cut(self, horizon=None):
        """
        horizon : array-like (optional)
            An array of 0s and 1s, specifying if a given phi/theta combination
            is above the horizon or not. Must have shape (theta, phi) or
            (freqs, theta, phi).
        """
        if horizon is None:
            horizon = np.ones_like(self.data)
            horizon[:, self.theta < 0] = 0
        elif horizon.ndim == 2:
            horizon = np.expand_dims(horizon, axis=0)
        self.data = self.data * horizon

    @classmethod
    def from_file(path):
        raise NotImplementedError

    def to_file(fname):
        raise NotImplementedError
