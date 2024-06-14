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
    if eulertype == "ZYX":
        return rotmat_to_eulerZYX(mat)
    elif eulertype == "ZYZ":
        return rotmat_to_eulerZYZ(mat)
    else:
        raise ValueError("Invalid Euler angle convention.")


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
    beta = -np.arcsin(mat[0, 2])  # pitch
    cb = np.cos(beta)
    if np.abs(cb) > 1e-10:  # can divide by cos(beta)
        gamma = np.arctan2(mat[1, 2] / cb, mat[2, 2] / cb)  # roll
        alpha = np.arctan2(mat[0, 1] / cb, mat[0, 0] / cb)  # yaw
    # else: cos(beta) = 0, sensitive only to alpha+gamma or alpha-gamma;
    # this is called gimbal lock. We take gamma = 0.
    else:
        gamma = 0
        alpha = np.arctan2(-mat[1, 0], mat[1, 1])

    eul = (alpha, -beta, gamma)  # healpy convention for ZYX
    return eul


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
    alpha = np.arctan2(mat[1, 2], mat[0, 2])
    cos_beta = mat[2, 2]
    beta = np.arctan2(np.sqrt(1 - cos_beta**2), cos_beta)
    gamma = np.arctan2(mat[2, 1], -mat[2, 0])
    eul = (alpha, beta, gamma)
    return eul


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
