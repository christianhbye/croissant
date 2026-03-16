from functools import partial

import jax
import numpy as np
import s2fft
from astropy.coordinates import SkyCoord

from .utils import lmax_from_shape


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


def rotmat_to_euler(mat, eulertype="ZYZ"):
    """
    Convert a rotation matrix to Euler angles in the specified convention.

    Parameters
    ----------
    mat : np.ndarray
        The rotation matrix.
    eulertype : {"ZYX", "ZYZ"}
        The Euler angle convention to use.

    Returns
    -------
    eul : tup
        The Euler angles in the specified convention.

    Notes
    -----
    The preferred convention for croissant is ZYZ, since it uses s2fft
    for rotations and spherical harmonics.

    ``ZYZ'' is the convention typically used for Wigner D matrices, which
    s2fft uses. Wikipedia calls it Euler angles Z1-Y2-Z3. This would be
    used in s2fft.utils.rotation.rotate_flms.

    ``ZYX'' is the default healpy convention, what you would make ``rot''
    when you call healpy.Rotator(rot, euletype="ZYX"). Wikipedia refers
    to this as Tait-Bryan angles X1-Y2-Z3.

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


def generate_euler_dl(lmax, from_frame, to_frame):
    """
    Generate the Euler angles and reduced Wigner d-function values for a
    coordinate transformation.

    Parameters
    ----------
    lmax : int
        The maximum spherical harmonic degree.
    from_frame : str or astropy frame
        The coordinate system of the input alm.
    to_frame : str or astropy frame
        The coordinate system of the output alm.

    Returns
    -------
    euler : jnp.ndarray
        The Euler angles for the coordinate transformation.
    dl_array : jnp.ndarray
        The reduced Wigner d-function values for the coordinate transformation.

    """
    rmat = get_rot_mat(from_frame, to_frame)
    euler = rotmat_to_euler(rmat, eulertype="ZYZ")
    dl_array = s2fft.generate_rotate_dls(lmax + 1, euler[1])
    return euler, dl_array


def get_mepa_rotation_matrix():
    """
    Get the rotation matrix from J2000 to the Mean Earth / Polar Axis
    (MEPA) frame. MEPA is an inertial frame with Z-axis along the
    Moon's mean rotation axis and X-axis toward the mean Earth
    direction, frozen at the J2000 epoch. It is equivalent to
    MOON_ME evaluated at J2000.

    Returns
    -------
    R : np.ndarray
        The 3x3 rotation matrix from J2000 to MEPA.

    """
    import lunarsky  # noqa: F401 — triggers SPICE kernel loading
    import spiceypy as spice

    return np.array(spice.pxform("J2000", "MOON_ME", 0.0))


def generate_euler_dl_from_rotmat(lmax, rotmat):
    """
    Generate Euler angles and reduced Wigner d-function values from
    a rotation matrix.

    Parameters
    ----------
    lmax : int
        The maximum spherical harmonic degree.
    rotmat : np.ndarray
        A 3x3 rotation matrix.

    Returns
    -------
    euler : tuple
        The Euler angles (alpha, beta, gamma) in ZYZ convention.
    dl_array : np.ndarray
        The reduced Wigner d-function values.

    """
    euler = rotmat_to_eulerZYZ(rotmat)
    dl_array = s2fft.generate_rotate_dls(lmax + 1, euler[1])
    return euler, dl_array


def topo_to_mepa_euler_dl(lmax, topo_frame, obstime_jd):
    """
    Compute the Euler angles and reduced Wigner d-function values for
    rotating alm from a LunarTopo frame to the MEPA frame.

    This transform is time-dependent because MEPA is inertial while
    LunarTopo co-rotates with the Moon. The Moon's spin phase at the
    observation time enters through the MCMF-to-J2000 rotation.

    Parameters
    ----------
    lmax : int
        The maximum spherical harmonic degree.
    topo_frame : lunarsky.LunarTopo
        The topocentric frame on the Moon.
    obstime_jd : float
        The observation time as a Julian date.

    Returns
    -------
    euler : tuple
        The Euler angles (alpha, beta, gamma) in ZYZ convention.
    dl_array : np.ndarray
        The reduced Wigner d-function values.

    """
    import lunarsky  # noqa: F401 — triggers SPICE kernel loading
    import spiceypy as spice

    # topo → MCMF (time-independent, depends only on observer location)
    R_topo_mcmf = get_rot_mat(topo_frame, "mcmf")
    # MCMF → J2000 at observation time (time-dependent: Moon's spin)
    et = (obstime_jd - 2451545.0) * 86400
    R_mcmf_j2000 = np.array(spice.pxform("MOON_ME", "J2000", et))
    # J2000 → MEPA (fixed)
    R_j2000_mepa = get_mepa_rotation_matrix()
    # compose: topo → MEPA
    R_topo_mepa = R_j2000_mepa @ R_mcmf_j2000 @ R_topo_mcmf
    return generate_euler_dl_from_rotmat(lmax, R_topo_mepa)


def _gal_to_sim_frame(alm, eul=None, dl_array=None, world="moon"):
    """
    Rotate alm from galactic to the simulation frame.

    For Earth, the simulation frame is FK5 (equatorial).
    For the Moon, the simulation frame is MEPA (Mean Earth / Polar
    Axis), an inertial frame with Z-axis along the Moon's polar axis.

    Parameters
    ----------
    alm : array_like
        The input alm in galactic coordinates. Last two axes should be
        the (l, m) indices. Preceding are batch dimensions.
    eul : tuple
        The Euler angles for the rotation. If not provided, they will
        be computed on the fly.
    dl_array : jnp.ndarray
        Precomputed reduced Wigner d-function values. If not provided,
        they will be computed on the fly.
    world : {"moon", "earth"}
        Which simulation frame to use. This is ignored if eul and
        dl_array are provided.

    Returns
    -------
    alm_sim : jnp.ndarray
        The output alm in the simulation frame.

    """
    lmax = lmax_from_shape(alm.shape)
    if eul is None or dl_array is None:
        if world == "moon":
            R_gal_fk5 = get_rot_mat("galactic", "fk5")
            R_j2000_mepa = get_mepa_rotation_matrix()
            R_gal_mepa = R_j2000_mepa @ R_gal_fk5
            eul, dl_array = generate_euler_dl_from_rotmat(lmax, R_gal_mepa)
        elif world == "earth":
            eul, dl_array = generate_euler_dl(lmax, "galactic", "fk5")
        else:
            raise ValueError("Invalid world. Must be 'moon' or 'earth'.")
    ct = partial(
        s2fft.utils.rotation.rotate_flms,
        L=lmax + 1,
        rotation=eul,
        dl_array=dl_array,
    )
    alm_sim = jax.vmap(ct)(alm)
    return alm_sim


def gal2eq(alm, eul=None, dl_array=None):
    """
    Rotate alm from galactic to equatorial coordinates.

    Parameters
    ----------
    alm : array_like
        The input alm in galactic coordinates. Last two axes should be
        the (l, m) indices. Preceding are batch dimensions.
    eul : tuple
        The Euler angles for the galactic to equatorial transformation.
    dl_array : jnp.ndarray
        Precomputed reduced Wigner d-function values for the galactic
        to equatorial transformation. If not provided, it will be
        computed on the fly.

    Returns
    -------
    alm_eq : jnp.ndarray
        The output alm in equatorial coordinates.

    """
    return _gal_to_sim_frame(alm, eul=eul, dl_array=dl_array, world="earth")


def gal2mepa(alm, eul=None, dl_array=None):
    """
    Rotate alm from galactic to MEPA coordinates (Mean Earth / Polar
    Axis frame for the Moon).

    Parameters
    ----------
    alm : array_like
        The input alm in galactic coordinates. Last two axes should be
        the (l, m) indices. Preceding are batch dimensions.
    eul : tuple
        The Euler angles for the galactic to MEPA transformation.
    dl_array : jnp.ndarray
        Precomputed reduced Wigner d-function values for the galactic
        to MEPA transformation. If not provided, it will be
        computed on the fly.

    Returns
    -------
    alm_mepa : jnp.ndarray
        The output alm in MEPA coordinates.

    """
    return _gal_to_sim_frame(alm, eul=eul, dl_array=dl_array, world="moon")
