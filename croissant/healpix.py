import healpy as hp
import numpy as np
from scipy.interpolate import RectSphereBivariateSpline

from . import constants
from .rotations import Rotator
from .sphtransform import alm2map, map2alm


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


def healpix2lonlat(nside, pix=None):
    """
    Compute the longtitudes and latitudes of the pixel centers of a healpix
    map.

    Parameters
    ----------
    nside : int
        The nside of the healpix map.
    pix : array-like (optional)
        Which pixels to get the longtitudes and latitudes of. Defaults to all
        pixels.

    Returns
    -------
    lon : scalar or np.ndarray
        The longtitude(s) in degrees. Range: [0, 360).
    lat : scalar or np.ndarray
        The latitude(s) in degrees. Range: [-90, 90].

    """
    if pix is None:
        pix = np.arange(hp.nside2npix(nside))
    lon, lat = hp.pix2ang(nside, pix, nest=False, lonlat=True)
    return lon, lat


def grid_interp(data, theta, phi, to_theta, to_phi):
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
    pole_values = np.full((len(data), 2), None)
    northpole = theta[0] == 0
    southpole = theta[-1] == np.pi
    if northpole:
        theta = theta[1:]
        pole_values[:, 0] = data[:, 0, 0]
        data = data[:, 1:]
    if southpole:
        theta = theta[:-1]
        pole_values[:, 1] = data[:, -1, 0]
        data = data[:, :-1]

    interp_data = np.empty((len(data), to_theta.size))
    for i in range(len(data)):
        interp = RectSphereBivariateSpline(
            theta, phi, data[i], pole_values=pole_values[i]
        )
        interp_data[i] = interp(to_theta, to_phi, grid=False)
    return interp_data


def grid2healpix(data, nside, theta=None, phi=None, pixel_centers=None):
    """
    Transform data defined on a rectangular grid on a sphere to healpix map(s).
    To compute a healpix map in a different coordinate system, compute the
    pixel centers of the target coordinate system and set the keyword argument
    pixel_centers.

    Parameters
    ----------
    data : array-like
        The data to transform. The last two dimensions must be (theta, phi).
        It may have an optional 0th dimension to generate multiple maps at
        the same time.
    nside : int
        The nside of the output healpix map.
    theta : 1d-array (optional)
        The polar angles in radians. Must be in [0, pi]. Defaults to 1-degree
        sampling.
    phi : 1d-array (optional)
        The azimuthal angles in radians. Must be in [0, 2pi). Defaults to
        1-degree sampling.
    pixel_centers: 2d-array (optional)
        The centers of the pixels in radians. Must have shape (npix, 2) and be
        ordered like healpix RING pixels. The 0th column corresponds to theta
        and the 1st to phi.

    Returns
    -------
    hp_map : np.ndarray
        The healpix map(s) in RING order with shape (n_maps, n_pixels).

    """
    npix = hp.nside2npix(nside)
    if pixel_centers is not None:
        if np.shape(pixel_centers) != (npix, 2):
            raise ValueError(f"Shape must be ({npix}, 2).")
        pix_theta = pixel_centers[:, 0]
        pix_phi = pixel_centers[:, 1]
    else:
        lon, lat = healpix2lonlat(nside)
        pix_theta = np.pi / 2 - np.deg2rad(lat)
        pix_phi = np.deg2rad(lon)

    if theta is None:
        theta = np.linspace(0, np.pi, num=181)
    else:
        theta = np.array(theta, copy=True)

    if phi is None:
        phi = np.linspace(0, 2 * np.pi, num=360, endpoint=False)
    else:
        phi = np.array(phi, copy=True)

    hp_map = grid_interp(data, theta, phi, pix_theta, pix_phi)
    return hp_map


class HealpixMap:
    def __init__(
        self,
        data,
        nest=False,
        frequencies=None,
        coord=None,
    ):
        """
        The base class for healpix maps. This is a wrapper that does a lot of
        healpy operations in parallel for a list of frequencies.
        It ensures that all maps have the right shapes and provdes an
        interpolation method.
        """
        if frequencies is None:
            self.frequencies = None
        else:
            self.frequencies = np.array(frequencies)
            nfreq = self.frequencies.size

        if coord is None:
            self.coord = None
        else:
            self.coord = coord_rep(coord)

        data = np.array(data, copy=True, dtype=np.float64)
        if frequencies is not None:
            data.shape = (nfreq, -1)
        npix = data.shape[-1]
        nside = hp.npix2nside(npix)
        hp.pixelfunc.check_nside(nside, nest=nest)
        self.nside = nside

        if nest:
            ix = hp.nest2ring(self.nside, np.arange(self.npix))
            data = data[..., ix]

        self.data = np.squeeze(data)

    @property
    def npix(self):
        """
        Get the number of pixels of the map.
        """
        return hp.nside2npix(self.nside)

    @classmethod
    def from_alm(cls, alm_obj, nside=None):
        """
        Construct a healpy map class from an Alm object (defined below).
        """
        if nside is None:
            nside = (alm_obj.lmax + 1) // 3
        hp_map = alm_obj.hp_map(nside=nside)
        obj = cls(
            data=hp_map,
            nest=False,
            frequencies=alm_obj.frequencies,
            coord=alm_obj.coord,
        )
        return obj

    @classmethod
    def from_grid(cls, data, nside, theta, phi, frequencies=None, coord=None):
        """
        Construct a HealpixMap instance from data defined on a grid of points.
        Parameters
        ----------
        data : array_like
            The data defined on the grid.
        nside : int
            The nside of the healpix map.
        theta : array_like
            The theta coordinates of the grid.
        phi : array_like
            The phi coordinates of the grid.
        frequencies : array_like
            The frequencies of the grid.
        coord : str
            The coordinate system of the grid.
        """
        theta = np.array(theta, copy=True)
        phi = np.array(phi, copy=True)
        hp_map = grid2healpix(data, nside, theta=theta, phi=phi)
        obj = cls(
            data=hp_map,
            nest=False,
            frequencies=frequencies,
            coord=coord,
        )
        return obj

    def ud_grade(self, nside_out, **kwargs):
        """
        Change the resolution of the healpy map to nside_out.

        Note: The nside in and out must be valid for nested maps since
        this conversion is being done under the hood.
        """
        hp.pixelfunc.check_nside(self.nside, nest=True)
        hp.pixelfunc.check_nside(nside_out, nest=True)

        new_map = hp.ud_grade(self.data, nside_out, **kwargs)
        self.data = new_map
        self.nside = nside_out

    def switch_coords(
        self,
        to_coord,
        lmax=None,
        rot_pixel=False,
        loc=None,
        time=None,
    ):
        """
        Switch the coordinate system of the map. This is done by rotating the
        map in spherical harmonic space (if rot_pixel is False) or by rotating
        the map in pixel space (if rot_pixel is True).

        Parameters
        ----------
        to_coord : str
            The coordinate system to switch to. Must be one of "G" (galactic),
            "E" (ecliptic), "C" (equatorial), "M" (mcmf), or T" (topocentric).
        lmax : int
            The maximum l value to use for the spherical harmonic transform.
        rot_pixel : bool
            If True, rotate the map in pixel space. If False, rotate the map
            in spherical harmonic space.
        loc : tup, astropy.coordinates.EarthLocation or lunarsky.MoonLocation
            The observation location for the topocentric coordinate system. If
            a tuple is given, it must be able to instantiate an EarthLocation
            or MoonLocation object.
        time : str or astropy.time.Time
            The observation time for the topocentric coordinate system. If a
            string is given, it must be able to instantiate a Time object.

        """
        to_coord = coord_rep(to_coord)
        rot = Rotator(coord=[self.coord, to_coord], loc=loc, time=time)
        if rot_pixel:
            self.data = rot.rotate_map_pixel(self.data)
        else:
            self.data = rot.rotate_map_alms(self.data, lmax=lmax)
        self.coord = to_coord

    def alm(self, lmax=None):
        """
        Compute the spherical harmonics coefficents of the map.
        """
        if lmax is None:
            lmax = 3 * self.nside - 1
        return map2alm(self.data, lmax=lmax)

    def plot(self, frequency=None, **kwargs):
        """
        Simple plotter of healpix maps. Can plot in several projections,
        including ``mollweide'', ``cartesian'' and ``polar''.
        """
        m = kwargs.pop("m", self.data)
        title = None
        if self.frequencies is not None:
            if frequency is None:
                raise ValueError("Must specify which frequency to plot.")
            else:
                f_idx = np.argmin(np.abs(self.frequencies - frequency))
                f_to_plot = self.frequencies[f_idx]
                title = kwargs.pop("title", f"Frequency = {f_to_plot:.0f} MHz")
                m = self.data[f_idx]
        _ = hp.projview(m=m, title=title, **kwargs)


class Alm(hp.Alm):
    def __init__(self, alm, lmax=None, frequencies=None, coord=None):
        """
        Base class for spherical harmonics coefficients.

        Alm can be indexed with [freq_index, ell, emm] to get the
        coeffiecient corresponding to the given frequency index, and values of
        ell and emm. The frequencies can be indexed in the usual numpy way and
        may be 0 if the alms are specified for only one frequency.

        """
        alm = np.array(alm, copy=True, dtype=np.complex128)
        if frequencies is None:
            self.frequencies = None
            self.alm = alm
        else:
            self.frequencies = np.array(frequencies)
            alm.reshape(self.frequencies.size, -1)
            self.alm = alm
        try:
            self.lmax = np.min([lmax, self.getlmax])
        except TypeError:  # lmax is None
            self.lmax = self.getlmax
        if coord is None:
            self.coord = None
        else:
            self.coord = coord_rep(coord)

    def __setitem__(self, key, value):
        """
        Set the value of the alm given the frequency index and the values of
        ell and emm.
        """
        if not len(key) == self.alm.ndim + 1:
            raise IndexError("Number of indices must be alm.ndim + 1.")
        ell, emm = key[-2:]
        ix = self.getidx(ell, emm)
        if self.alm.ndim == 1:
            self.alm[ix] = value
        else:
            freq_idx = key[0]
            self.alm[freq_idx, ix] = value

    def __getitem__(self, key):
        if not len(key) == self.alm.ndim + 1:
            raise IndexError("Number of indices must be alm.ndim + 1.")
        ell, emm = key[-2:]
        ix = self.getidx(ell, np.abs(emm))
        if self.alm.ndim == 1:
            coeff = self.alm[ix]
        else:
            freq_idx = key[0]
            coeff = self.alm[freq_idx, ix]
        if emm < 0:
            coeff = (-1) ** emm * coeff.conj()
        return coeff

    def reduce_lmax(self, new_lmax):
        """
        Reduce the maximum l value of the alm.
        """
        ells, emms = super().getlm(new_lmax)
        ix = self.getidx(ells, emms)
        self.alm = self.alm[..., ix]
        self.lmax = new_lmax

    @classmethod
    def from_healpix(cls, hp_obj, lmax=None):
        """
        Construct an Alm from a HealpixMap object.
        """
        alm = hp_obj.alm(lmax=lmax)
        if lmax is None:
            lmax = hp.Alm.getlmax(alm.size)
        obj = cls(
            alm=alm,
            lmax=lmax,
            frequencies=hp_obj.frequencies,
            coord=hp_obj.coord,
        )
        return obj

    @classmethod
    def from_grid(
        cls,
        data,
        theta,
        phi,
        lmax,
        nside=128,
        frequencies=None,
        coord=None,
    ):
        """
        Construct an Alm from a grid in theta and phi. This function first
        interpolates the data onto a Healpix grid, then converts the data to
        spherical harmonics using healpy.

        Parameters
        ----------
        data : array_like
            The data to be converted to spherical harmonics. Must have shape
            (Nfreq, Ntheta, Nphi) or (Ntheta, Nphi).
        theta : array_like
            The theta values of the data. Must have shape (Ntheta,).
        phi : array_like
            The phi values of the data. Must have shape (Nphi,).
        lmax : int
            The maximum value of ell to use in the spherical harmonics.
        nside : int
            The nside of the Healpix grid to use for the interpolation.
        frequencies : array_like
            The frequencies corresponding to the data. Must have shape
            (Nfreq,).
        coord : str
            The coordinate system of the data.

        """
        theta = np.array(theta, copy=True)
        phi = np.array(phi, copy=True)
        hp_map = grid2healpix(data, nside, theta=theta, phi=phi)
        alm = map2alm(hp_map, lmax=lmax)
        obj = cls(
            alm=alm,
            lmax=lmax,
            frequencies=frequencies,
            coord=coord,
        )
        return obj

    def switch_coords(self, to_coord, loc=None, time=None):
        to_coord = coord_rep(to_coord)
        rot = Rotator(coord=[self.coord, to_coord], loc=loc, time=time)
        rot.rotate_alm(self.alm, lmax=self.lmax, inplace=True)
        self.coord = to_coord

    def getlm(self, i=None):
        """
        Get the ell and emm corresponding to the index of the alm array.
        """
        return super().getlm(self.lmax, i=i)

    def getidx(self, ell, emm):
        """
        Get the index of the alm array for a given ell and emm.
        """
        if not ((0 <= emm) & (emm <= ell) & (ell <= self.lmax)).all():
            raise IndexError("Ell or emm are out of set by m <= l <= lmax.")
        return super().getidx(self.lmax, ell, emm)

    @property
    def size(self):
        """
        Get the size of the alm array.
        """
        return super().getsize(self.lmax)

    @property
    def getlmax(self):
        """
        Get the maxmium ell of the Alm object.
        """
        if not hasattr(self, "lmax") or self.lmax is None:
            return super().getlmax(self.alm.shape[-1])
        else:
            return self.lmax

    def hp_map(self, nside):
        """
        Construct a healpy map from the Alm.
        """
        return alm2map(self.alm, nside=nside, lmax=self.lmax)

    def rot_alm_z(self, phi=None, times=None, world="moon"):
        """
        Get the coefficients that rotate the alms around the z-axis by phi
        (measured counterclockwise) or in time.

        Parameters
        ----------
        phi : array-like
            The angle(s) to rotate the azimuth by in radians.
        times : array-like
            The times to rotate the azimuth by in seconds. If given, phi will
            be ignored and the rotation angle will be calculated from the
            times and the sidereal day of the world.
        world : str
            The world to use for the sidereal day. Must be 'moon' or 'earth'.

        Returns
        -------
        phase : np.ndarray
            The coefficients (shape = (phi.size, alm.size) that rotate the
            alms by phi.

        """
        if times is not None:
            if world.lower() == "moon":
                sidereal_day = constants.sidereal_day_moon
            elif world.lower() == "earth":
                sidereal_day = constants.sidereal_day_earth
            else:
                raise ValueError(
                    f"World must be 'moon' or 'earth', not {world}."
                )
            phi = 2 * np.pi * times / sidereal_day
            return self.rot_alm_z(phi=phi, times=None)

        phi = np.ravel(phi).reshape(-1, 1)
        emms = self.getlm()[1].reshape(1, -1)
        phase = np.exp(-1j * emms * phi)
        return np.squeeze(phase)
