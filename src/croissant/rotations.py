from s2fft import generate_rotate_dls
from ..utils import get_rot_mat, rotmat_to_euler


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
    dl_array = generate_rotate_dls(lmax + 1, euler[1])
    return euler, dl_array
