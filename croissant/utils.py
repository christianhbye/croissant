from astropy import units
from lunarsky import SkyCoord, Time
import numpy as np
import warnings


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
    try:
        from_name = from_frame.name
    except AttributeError:
        from_name = from_frame
    # skycoord does not support galactic -> cartesian, do the inverse
    if from_name.lower() == "galactic":
        from_frame = to_frame
        to_frame = "galactic"
        return_inv = True
    else:
        return_inv = False
    x, y, z = np.eye(3)  # unit vectors
    sc = SkyCoord(
        x=x, y=y, z=z, frame=from_frame, representation_type="cartesian"
    )
    rmat = sc.transform_to(to_frame).cartesian.xyz.value
    if return_inv:
        rmat = rmat.T
    return rmat


def rotmat_to_euler(mat):
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
        The Euler angles.

    """
    beta = np.arcsin(mat[0, 2])
    alpha = np.arctan2(mat[1, 2] / np.cos(beta), mat[2, 2] / np.cos(beta))
    gamma = np.arctan2(mat[0, 1] / np.cos(beta), mat[0, 0] / np.cos(beta))
    eul = (gamma, beta, alpha)
    return eul


def time_array(t_start=None, t_end=None, N_times=None, delta_t=None):
    """
    Generate an array of evenly sampled times to run the simulation at.

    Parameters
    ----------
    t_start : str or astropy.time.Time
        The start time of the simulation.
    t_end : str or astropy.time.Time
        The end time of the simulation.
    N_times : int
        The number of times to run the simulation at.
    delta_t : float or astropy.units.Quantity
        The time step between each time in the simulation.

    Returns
    -------
    times : astropy.time.Time or astropy.units.Quantity
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
