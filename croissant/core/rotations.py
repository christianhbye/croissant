from astropy.coordinates import AltAz, EarthLocation
import healpy as hp
from lunarsky import LunarTopo, MoonLocation, SkyCoord
import numpy as np

from .sphtransform import map2alm, alm2map


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
    # cannot instantiate a SkyCoord with a gaalctic frame from cartesian
    from_name = from_frame.name if hasattr(from_frame, "name") else from_frame
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


class Rotator(hp.Rotator):
    def __init__(
        self,
        rot=None,
        coord=None,
        inv=None,
        deg=True,
        eulertype="ZYX",
        loc=None,
        time=None,
    ):
        """
        Subclass of healpy Rotator that adds functionality to transform to
        topocentric and moon centric coordinate systems. In addition, it can
        rotate lists of maps or alms.

        The allowed coordinate transforms are:
        - ecliptic <--> equatorial <--> galactic <--> mcmf
        - equatorial <--> topocentric (on earth)
        - mcmf <--> topocentric (on moon)

        Parameters
        ----------
        rot : sequence of floats
            Euler angles in degrees (or radians if deg=False) describing the
            rotation. The order of the angles depends on the value of
            eulertype.
        coord : sequence of strings
            Coordinate systems to rotate between. Supported values are
            "G" (galactic), "C" (equatorial), "E" (ecliptic), "M" (MCMF),
            "T" (topocentric). The order of the strings determines the order
            of the rotation. For example, coord=['G', 'C'] will rotate from
            galactic to equatorial coordinates.
        inv : bool
            If True, the inverse rotation is performed.
        deg : bool
            If True, the Euler angles are in degrees.
        eulertype : str
            The order of the Euler angles. Supported values are "ZYX"
            (default), "X", and "Y".
        loc : tup, astropy.coordinates.EarthLocation, or lunarsky.MoonLocation
            The location of the observer. If a tuple is provided, it must be
            able to instantiate an astropy.coordinates.EarthLocation object
            (on Earth) or a lunarsky.MoonLocation object (on the Moon).
        time : str, astropy.time.Time, or lunarsky.Time
            The time of the coordinate transform. If a string is provided, it
            must be able to instantiate an astropy.time.Time object (on Earth)
            or a lunarsky.Time object (on the Moon).

        """
        EUL_TYPES = ["ZYX", "X", "Y"]  # types supported by healpy
        # healpy does not warn about this but silently defaults to "ZYX"
        if eulertype not in EUL_TYPES:
            raise ValueError(f"eulertype must be in {EUL_TYPES}")
        # astropy frames (consistent with healpy)
        FRAMES = {
            "G": "galactic",
            "C": "fk5",
            "E": "BarycentricMeanEcliptic",
            "M": "mcmf",
        }

        if coord is not None:
            coord = [c.upper() for c in coord]
            if len(coord) != 2:
                raise ValueError("coord must be a sequence of length 2")
            if "T" in coord:  # topocentric
                if loc is None or time is None:
                    raise ValueError(
                        "loc and time must be provided if coord contains 'T'"
                    )
                if "M" in coord:  # on moon
                    if isinstance(loc, tuple):
                        loc = MoonLocation(*loc)
                    from_frame = FRAMES["M"]
                    to_frame = LunarTopo(location=loc, obstime=time)
                elif "C" in coord:  # on earth
                    if isinstance(loc, tuple):
                        loc = EarthLocation(*loc)
                    from_frame = FRAMES["C"]
                    to_frame = AltAz(location=loc, obstime=time)
                else:
                    raise ValueError(
                        "Can only transform between topocentric and "
                        "equatorial/mcmf"
                    )

                if coord[0] == "T":  # transforming from T
                    inv = not inv
            else:
                from_frame = FRAMES[coord[0]]
                to_frame = FRAMES[coord[1]]
            convmat = get_rot_mat(from_frame, to_frame)
            if rot is None:
                rot = rotmat_to_euler(convmat)
            else:  # combine the coordinate transform with rotation
                rotmat = hp.rotator.get_rotation_matrix(
                    rot, deg=deg, eulertype=eulertype
                )[0]
                rot = rotmat_to_euler(rotmat @ convmat)
            eulertype = "ZYX"
            deg = False
            coord = None

        super().__init__(
            rot=rot, coord=coord, inv=inv, deg=deg, eulertype=eulertype
        )

    def rotate_alm(self, alm, lmax=None, inplace=False):
        """
        Rotate an alm or a list of alms.

        Parameters
        ----------
        alm : array_like
            The alm or list of alms to rotate.
        lmax : int
            The maximum ell value to rotate.
        inplace : bool
            If True, the alm is rotated in place. Otherwise, a copy is
            rotated and returned.

        Returns
        -------
        rotated_alm : array_like
            The rotated alm or list of alms. This is only returned if
            inplace=False.

        """
        if inplace:
            rotated_alm = alm
        else:
            rotated_alm = np.array(alm, copy=True, dtype=np.complex128)

        if rotated_alm.ndim == 1:
            super().rotate_alm(rotated_alm, lmax=lmax, inplace=True)
        elif rotated_alm.ndim == 2:
            # iterate over the list of alms
            for i in range(len(rotated_alm)):
                super().rotate_alm(rotated_alm[i], lmax=lmax, inplace=True)
        else:
            raise ValueError(
                f"alm must have 1 or 2 dimensions, not {alm.ndim}."
            )

        if not inplace:
            return rotated_alm

    def rotate_map_alms(self, m, lmax=None, inplace=False):
        """
        Rotate a map or a list of maps in spherical harmonics space.

        Parameters
        ----------
        m : array-like
            The map or list of maps to rotate.
        lmax : int
            The maximum ell value to rotate.
        inplace : bool
            If True, the map is rotated in place. Otherwise, a copy is
            rotated and returned.

        Returns
        -------
        rotated_m : np.ndarray
            The rotated map or list of maps. This is only returned if
            inplace=False.

        """
        npix = m.shape[-1]
        nside = hp.npix2nside(npix)
        alm = map2alm(m, lmax=lmax)
        self.rotate_alm(alm, lmax=lmax, inplace=True)
        rotated_m = alm2map(alm, nside, lmax=lmax)
        if inplace:
            m = rotated_m
        else:
            return rotated_m

    def rotate_map_pixel(self, m, inplace=False):
        """
        Rotate a map or a list of maps in pixel space.

        Parameters
        -----------
        m : array-like
            The map or list of maps to rotate.
        inplace : bool
            If True, the map is rotated in place. Otherwise, a copy is
            rotated and returned.

        Returns
        -------
        rotated_m : np.ndarray
            The rotated map or list of maps. This is only returned if
            inplace=False.

        """
        if m.ndim == 1:
            rotated_m = super().rotate_map_pixel(m)
        elif m.ndim == 2:
            rotated_m = np.empty_like(m)
            for i in range(len(m)):
                rotated_m[i] = super().rotate_map_pixel(m[i])
        else:
            raise ValueError(f"m must have 1 or 2 dimensions, not {m.ndim}.")
        if inplace:
            m = rotated_m
        else:
            return rotated_m
