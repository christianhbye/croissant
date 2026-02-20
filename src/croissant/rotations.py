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


def _gal_to_eq_mcmf(alm, eul=None, dl_array=None, world="moon"):
    """
    Rotate alm from galactic to equatorial or mcmf coordinates.

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
    world : {"moon", "earth"}
        Whether to use the galactic to equatorial transformation for
        the moon or the earth (equatorial coordinates are different for
        the two worlds). This is ignored if eul and dl_array are
        provided.

    Returns
    -------
    alm_eq : jnp.ndarray
        The output alm in equatorial coordinates.

    """
    lmax = lmax_from_shape(alm.shape)
    if eul is None or dl_array is None:
        if world == "moon":
            frame = "mcmf"
        elif world == "earth":
            frame = "fk5"
        else:
            raise ValueError("Invalid world. Must be 'moon' or 'earth'.")
        eul, dl_array = generate_euler_dl(lmax, "galactic", frame)
    ct = jax.vmap(
        s2fft.utils.rotation.rotate_flms, in_axes=(0, None, None, None)
    )
    return ct(alm, lmax, eul, dl_array=dl_array)


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
    return _gal_to_eq_mcmf(alm, eul=eul, dl_array=dl_array, world="earth")


def gal2mcmf(alm, eul=None, dl_array=None):
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
    return _gal_to_eq_mcmf(alm, eul=eul, dl_array=dl_array, world="mcmf")
