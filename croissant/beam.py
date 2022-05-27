import numpy as np
from .healpix import Alm


class Beam:
    def __init__(
        self, data, phi, theta, frequencies=None, horizon=None
    ):
        """
        Class that holds antenna beam objects. Thin wrapper over Alm.
        Data must have shape ([freqs,] theta, phi) if from_grid is True.
        Otherwise must have shape ([freqs,] alm).
        horizon_mask must have same shape as data and is assumed to be in the
        same space (either both real space or both alms).
        """
        data = np.array(data)
        self.frequencies = np.squeeze(frequencies).reshape(-1)
        self.theta = np.squeeze(theta).reshape(-1)
        self.phi = np.squeeze(phi).reshape(-1)
        data.shape = (frequencies.size, theta.size, phi.size)
        self.data = data 

    def horizon_cut(self, horizon=None):
        if horizon is None:
            horizon = np.ones_like(self.data)
            horizon[:, self.theta < 0] = 0
        self.data = self.data * horizon

    @classmethod
    def from_file(path):
        raise NotImplementedError

    def to_file(fname):
        raise NotImplementedError
