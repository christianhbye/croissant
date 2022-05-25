import numpy as np
from .healpix import Alm


class Beam(Alm):
    def __init__(
        self, data, frequencies=None, from_grid=False, horizon=None, **kwargs
    ):
        """
        Class that holds antenna beam objects. Thin wrapper over Alm.
        Data must have shape ([freqs,] theta, phi) if from_grid is True.
        Otherwise must have shape ([freqs,] alm).
        horizon_mask must have same shape as data and is assumed to be in the
        same space (either both real space or both alms).
        """
        data = np.array(data)
        frequencies = np.squeeze(frequencies).reshape(-1)
        if from_grid:
            theta = np.squeeze(kwargs["theta"]).reshape(-1)
            phi = np.squeeze(kwargs["phi"]).reshape(-1)
            data.shape = (frequencies.size, theta.size, phi.size)
            default_horizon = np.ones_like(data)
            default_horizon[:, theta < 0] = 0
        else:
            data.shape = (frequencies.size, -1)
            lmax = kwargs.get("lmax")
            default_horizon = 1  # XXX

        if isinstance(horizon, str):
            horizon = horizon.lower()

        horizon_dict = {None: 1, "none": 1, "default": default_horizon}
        horizon_mask = horizon_dict.get(horizon, horizon)
        data = data * horizon_mask

        if from_grid:
            super().from_grid(data, theta, phi, frequencies=frequencies)
        else:
            super().__init__(alm=data, lmax=lmax, frequencies=frequencies)

    @classmethod
    def from_file(path):
        raise NotImplementedError

    def to_file(fname):
        raise NotImplementedError
