from astropy.coordinates import AltAz, EarthLocation
import healpy as hp
from lunarsky import LunarTopo, MoonLocation
import numpy as np

from ..utils import get_rot_mat, rotmat_to_euler
from .sphtransform import map2alm, alm2map


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

        Parameters
        ----------
        rot : sequence of floats
            Euler angles in degrees (or radians if deg=False) describing the
            rotation. The order of the angles depends on the value of
            ``eulertype''. When ``eulertype'' is "ZYX", the angles are
            (yaw, -pitch, roll).
        coord : sequence of strings
            Coordinate systems to rotate between. Supported values are
            "G" (galactic), "C" (equatorial), "E" (ecliptic), "M" (MCMF),
            "L" (topocentric on the moon) and "T" (topocentric on Earth).
            The order of the strings determines the order of the rotation. For
            example, coord=['G', 'C'] will rotate from galactic to equatorial
            coordinates.
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
            "T": "topocentric",
            "L": "lunar",
        }

        if coord is not None:
            coord = [c.upper() for c in coord]
            if len(coord) != 2:
                raise ValueError("coord must be a sequence of length 2")
            from_frame = FRAMES[coord[0]]
            to_frame = FRAMES[coord[1]]
            if from_frame in ["lunar", "topocentric"]:
                from_frame = FRAMES[coord[1]]
                to_frame = FRAMES[coord[0]]
                inv = not inv
            if to_frame == "lunar":
                if isinstance(loc, tuple):
                    loc = MoonLocation(*loc)
                to_frame = LunarTopo(location=loc, obstime=time)
            elif to_frame == "topocentric":
                if isinstance(loc, tuple):
                    loc = EarthLocation(*loc)
                to_frame = AltAz(location=loc, obstime=time)
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

    def rotate_alm(self, alm, lmax=None, polarized=False):
        """
        Rotate an alm or a list of alms.

        Parameters
        ----------
        alm : array_like
            The alm or list of alms to rotate.
        lmax : int
            The maximum ell value to rotate.
        polarized : bool
            If true, the alm is assumed to be a sequence of TEB alms
            corresponding to I, Q, U maps, where I is spin-0 and Q, U are
            spin-2. In this case, ``alm'' has two dimensions where the first
            has size 3. Multiple frequency maps are not yet supported in this
            case.

        Returns
        -------
        rotated_alm : array_like
            The rotated alm or list of alms.

        """
        rotated_alm = np.array(alm, copy=True, dtype=np.complex128)

        if rotated_alm.ndim == 1 or polarized:
            super().rotate_alm(rotated_alm, lmax=lmax, inplace=True)
        elif rotated_alm.ndim == 2:
            # iterate over the list of alms
            for i in range(len(rotated_alm)):
                super().rotate_alm(rotated_alm[i], lmax=lmax, inplace=True)
        else:
            raise ValueError(
                f"alm must have 1 or 2 dimensions, not {alm.ndim}."
            )

        return rotated_alm

    def rotate_map_alms(self, m, lmax=None, polarized=False):
        """
        Rotate a map or a list of maps in spherical harmonics space.

        Parameters
        ----------
        m : array-like
            The map or list of maps to rotate.
        lmax : int
            The maximum ell value to rotate.
        polarized : bool
            If true, ``m'' is assumed to be a list of I, Q, U polarizations.
            I is spin-0 and Q, U are spin-2. In this case, ``m'' has two
            dimensions where the first has size 3. Multiple frequency maps
            are not yet supported in this case.

        Returns
        -------
        rotated_m : np.ndarray
            The rotated map or list of maps.

        """
        npix = m.shape[-1]
        nside = hp.npix2nside(npix)
        alm = map2alm(m, lmax=lmax, polarized=polarized)
        alm = self.rotate_alm(alm, lmax=lmax, polarized=polarized)
        rotated_m = alm2map(alm, nside, lmax=lmax, polarized=polarized)
        return rotated_m

    def rotate_map_pixel(self, m, polarized=False):
        """
        Rotate a map or a list of maps in pixel space.

        Parameters
        -----------
        m : array-like
            The map or list of maps to rotate.
        polarized : bool
            If true, ``m'' is assumed to be a list of I, Q, U polarizations.
            I is spin-0 and Q, U are spin-2. In this case, ``m'' has two
            dimensions where the first has size 3. Multiple frequency maps
            are not yet supported in this case.

        Returns
        -------
        rotated_m : np.ndarray
            The rotated map or list of maps.

        """
        if m.ndim == 1 or polarized:
            rotated_m = super().rotate_map_pixel(m)
        elif m.ndim == 2:
            rotated_m = np.empty_like(m)
            for i in range(len(m)):
                rotated_m[i] = super().rotate_map_pixel(m[i])
        else:
            raise ValueError(f"m must have 1 or 2 dimensions, not {m.ndim}.")
        return rotated_m
