import warnings

import numpy as np
import s2fft
from astropy import units
from lunarsky import Time

from . import rotations


def _future_warning(func):
    """
    Decorator to add a FutureWarning to a function.

    Parameters
    ----------
    func : callable
        The function to add the FutureWarning to.

    Returns
    -------
    wrapper : callable
        The wrapped function that raises a FutureWarning when called.

    """

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is has been moved to the rotations module "
            "and will be removed from the utils module in a future release.",
            FutureWarning,
        )
        return func(*args, **kwargs)

    return wrapper


def coord_rep(coord):
    """
    Shorthand notation for coordinate systems.

    Parameters
    ----------
    coord : str
        The name of the coordinate system.

    Returns
    -------
    rep : str
        The one-letter shorthand notation for the coordinate system.

    """
    coord = coord.upper()
    if coord[0] == "E" and coord[1] == "Q":
        rep = "C"
    else:
        rep = coord[0]
    return rep


@_future_warning
def get_rot_mat(from_frame, to_frame):
    """
    Get the rotation matrix that transforms from one frame to another.

    Parameters
    ----------
    from_frame : str or astropy frame
        The coordinate frame to transform from.
    to_frame : str or astropy frame
        The coordinate frame to transform to.

    Returns
    -------
    rmat : np.ndarray
        The rotation matrix.

    """
    return rotations.get_rot_mat(from_frame, to_frame)


@_future_warning
def rotmat_to_euler(mat, eulertype="ZYX"):
    """
    Convert a rotation matrix to Euler angles in the specified convention.

    Parameters
    ----------
    mat : np.ndarray
        The rotation matrix.
    eulertype : str, either ``ZYX'' or ``ZYZ''.
        The Euler angle convention to use.

    Returns
    -------
    eul : tup
        The Euler angles in the specified convention.

    Notes
    -----
    ``ZYX'' is the default healpy convention, what you would make ``rot''
    when you call healpy.Rotator(rot, euletype="ZYX"). Wikipedia refers
    to this as Tait-Bryan angles X1-Y2-Z3.

    ``ZYZ'' is the convention typically used for Wigner D matrices, which
    s2fft uses. Wkipidia calls it Euler angles Z1-Y2-Z3. This would be
    used in s2fft.utils.rotation.rotate_flms.


    """
    return rotations.rotmat_to_euler(mat, eulertype=eulertype)


@_future_warning
def rotmat_to_eulerZYX(mat):
    """
    Convert a rotation matrix to Euler angles in the ZYX convention. This is
    sometimes referred to as Tait-Bryan angles X1-Y2-Z3.

    Parameters
    ----------
    mat : np.ndarray
        The rotation matrix.

    Returns
    --------
    eul : tup
        The Euler angles in the order yaw, -pitch, roll. This is the input
        healpy.rotator.Rotator expects when ``eulertype'' is ZYX.

    """
    return rotations.rotmat_to_eulerZYX(mat)


@_future_warning
def rotmat_to_eulerZYZ(mat):
    """
    Convert a rotation matrix to Euler angles in the ZYZ convention. This is
    sometimes referred to as Euler angles Z1-Y2-Z3.

    Parameters
    ----------
    mat : np.ndarray
        The rotation matrix.

    Returns
    --------
    eul : tup
        The Euler angles in the order alpha, beta, gamma. This is the input
        s2fft.utils.rotation.rotate_flms expects.

    """
    return rotations.rotmat_to_eulerZYZ(mat)


def hp_npix2nside(npix):
    """
    Calculate the nside of a HEALPix map from the number of pixels.

    Parameters
    ----------
    npix : int
        The number of pixels in the map.

    Returns
    -------
    nside : int
        The nside of the map.

    """
    nside = int(np.sqrt(npix / 12))
    return nside


def time_array(t_start=None, t_end=None, N_times=None, delta_t=None):
    """
    Generate an array of evenly sampled times to run the simulation at.

    Parameters
    ----------
    t_start : str or astropy.time.Time or lunarsky.Time
        The start time of the simulation.
    t_end : str or astropy.time.Time or lunarsky.Time
        The end time of the simulation.
    N_times : int
        The number of times to run the simulation at.
    delta_t : float or astropy.units.Quantity
        The time step between each time in the simulation.

    Returns
    -------
    times : astropy.time.Time or lunarsky.Time or astropy.units.Quantity
        The evenly sampled times to run the simulation at.

    """

    if t_start is not None:
        t_start = Time(t_start, scale="utc")

    try:
        dt = np.arange(N_times) * delta_t
    except TypeError:
        t_end = Time(t_end, scale="utc")
        total_time = (t_end - t_start).sec
        if N_times is None:
            try:
                delta_t = delta_t.to_value("s")
            except AttributeError:
                warnings.warn(
                    "delta_t is not an astropy.units.Quantity. Assuming "
                    "units of seconds.",
                    UserWarning,
                )
            dt = np.arange(0, total_time + delta_t, delta_t)
        else:
            dt = np.linspace(0, total_time, N_times)
        dt = dt * units.s

    if t_start is None:
        times = dt
    else:
        times = t_start + dt

    return times


def lmax_range(sampling, data_shape):
    """
    Calculate the minimum and maximum supported lmax for a given
    sampling and data shape.

    Parameters
    ----------
    sampling : str
        The type of sampling. Supported schemes are from s2fft and
        include {"mw", "mwss", "dh", "gl", "healpix"}.
    data_shape : tuple
        The shape of the data. For s2fft sampling schemes, this should
        be (N_theta, N_phi), unless the sampling is "healpix", in which
        case it should be (N_pix,).

    Returns
    -------
    tup
        The minimum and maximum supported lmax for the given sampling and
        data shape.

    """
    raise NotImplementedError("This function is not yet implemented.")


def generate_phi(lmax=None, sampling="mw", nside=None):
    """
    Generate an array of phi values for a given lmax and sampling scheme.

    Parameters
    ----------
    lmax : int
        The maximum spherical harmonic degree to support. Required if
        `sampling`` is not "healpix".
    sampling : str
        The type of sampling. Supported schemes are from s2fft and
        include {"mw", "mwss", "dh", "gl", "healpix"}.
    nside : int
        The nside of the HEALPix map. Only required if sampling is
        "healpix".

    Returns
    -------
    phi : np.ndarray
        An array of phi values for the given lmax and sampling scheme.

    """
    if sampling != "healpix":
        if lmax is None:
            raise ValueError(
                "lmax must be provided if sampling is not healpix."
            )
        L = lmax + 1
        phi = s2fft.sampling.s2_samples.phis_equiang(L, sampling=sampling)
    else:
        raise NotImplementedError(
            "This function is not yet implemented for healpix sampling."
        )
    return phi
