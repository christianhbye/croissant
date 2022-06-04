import numpy as np

def interpolate(data, theta, phi, to_theta, to_phi):
    """
    Interpolate on a sphere from specfied theta and phi. The data must be
    on a rectangular grid.
    
    Parameters
    ----------
    data : array-like
        The data to interpolate. The last two dimensions must be (theta, phi).
        Can optionally have a 0th dimmension (e.g. a frequency dimension).
    theta : 1d-array
        The polar angles (colatitudes) in radians. Must be regularly sampled
        and strictly increasing.
    phi : 1d-array
        The azimuthal angles in radians. Must be regularly sampled and strictly
        increasing. Must be in the interval [0, 2*pi).
    to_theta : array-like
        The polar angles to interpolate to in radians.
    to_phi : array-like
        The azimuthal angles to interpolate to in radians.

    Returns
    -------
    interp_data : np.ndarray
        The interpolated data.

    """
    theta = np.ravel(theta).copy()
    phi = np.ravel(phi).copy()
    data = np.array(data, copy=True).reshape(-1, theta.size, phi.size)

    # remove poles before interpolating
    northpole = theta[0] == 0
    southpole = theta[-1] == np.pi
    if northpole:
        theta = theta[1:]
    if southpole:
        theta = theta[:-1]
    phi -= np.pi  # different conventions

    interp_data = np.empty((len(data), to_theta.size))
    for i in range(len(data)):
        # remove poles from data and assign to list
        pole_values = [None, None]
        if northpole:
            pole_values[0] = data[0]
            data = data[1:]
        if southpole:
            pole_values[1] = data[-1]
            data = data[:-1]

        interp = RectSphereBivariateSpline(
            theta, phi, data, pole_value=pole_values
        )
        interp_data[i] = interp(to_theta, to_phi, grid=False)
    return interp_data

class Beam:
    def __init__(
        self,
        data,
        theta,
        phi,
        frequencies=None,
    ):
        """
        Parameters
        ----------
        data : array-like
            The power beam. Must have shape ([freqs,] theta, phi).
        theta : array-like
            Zenith angle(s) in radians.
        phi : array-like
            Azimuth angle(s) in radians.
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
