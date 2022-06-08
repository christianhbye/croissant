import healpy as hp
import numpy as np
from scipy.interpolate import RectSphereBivariateSpline

from . import coordinates, constants


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
    phi -= np.pi  # different conventions

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


def map2alm(data, lmax):
    """
    Compute the spherical harmonics coefficents of a healpix map.
    """

    data = np.array(data)
    npix = data.shape[-1]
    nside = hp.npix2nside(npix)
    use_pix_weights = nside in constants.PIX_WEIGHTS_NSIDE
    use_ring_weights = not use_pix_weights
    kwargs = {
        "lmax": lmax,
        "mmax": lmax,
        "use_weights": use_ring_weights,
        "use_pixel_weights": use_pix_weights,
    }
    if data.ndim == 1:
        alm = hp.map2alm(data, **kwargs)
    else:
        alm = np.empty(
            (len(data), hp.Alm.getsize(lmax, mmax=lmax)), dtype=np.complex128
        )
        for i in range(len(data)):
            alm[i] = hp.map2alm(data[i], **kwargs)
    return alm


class HealpixMap:
    def __init__(
        self,
        nside,
        data=None,
        nested_input=False,
        frequencies=None,
        coords="galactic",
    ):
        """
        The base class for healpix maps. This is a wrapper that does a lot of
        healpy operations in parallel for a list of frequencies.
        It ensures that all maps have the right shapes and provdes an
        interpolation method.
        """
        hp.pixelfunc.check_nside(nside, nest=nested_input)
        self.nside = nside
        self.frequencies = np.ravel(frequencies).copy()

        if data is not None:
            data = np.array(data, copy=True, dtype=np.float64)
            data.shape = (self.frequencies.size, self.npix)
            if nested_input:
                ix = hp.nest2ring(self.nside, np.arange(self.npix))
                data = data[:, ix]

        self.data = data
        self.coords = coords

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
            nside,
            data=hp_map,
            nested_input=False,
            frequencies=alm_obj.frequencies,
            coords=alm_obj.coords,
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

    def switch_coords(self, to_coords):
        rotated_map = coordinates.rotate_map(
            self.data, from_coords=self.coords, to_coords=to_coords
        )
        self.data = rotated_map
        self.coords = to_coords

    def alm(self, lmax=None):
        """
        Compute the spherical harmonics coefficents of the map.
        """
        if lmax is None:
            lmax = 3 * self.nside - 1
        return map2alm(self.data, lmax)

    def plot(self, frequency=None, **kwargs):
        """
        Simple plotter of healpix maps. Can plot in several projections,
        including ``mollweide'', ``cartesian'' and ``polar''.
        """
        if self.data.ndim == 2 and self.frequencies is None:
            _m = self.data[0]
        else:
            _m = self.data
        m = kwargs.pop("m", _m)
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
    def __init__(
        self, alm=None, lmax=None, frequencies=None, coords="galactic"
    ):
        """
        Base class for spherical harmonics coefficients.

        Alm can be indexed with [freq_index, ell, emm] to get the
        coeffiecient corresponding to the given frequency index, and values of
        ell and emm. The frequencies can be index in the usual numpy way and
        may be 0 if the alms are specified for only one frequency.

        """
        if alm is None and lmax is None:
            raise ValueError("Specify at least one of lmax and alm.")

        self.frequencies = np.ravel(frequencies).copy()
        if alm is None:
            self.lmax = lmax
            self.all_zero()
        elif lmax is None:
            alm = np.array(alm, copy=True, dtype=np.complex128)
            self.alm = alm.reshape(self.frequencies.size, -1)
            self.lmax = super().getlmax(alm.shape[1])
        else:
            self.lmax = lmax
            alm = np.array(alm, copy=True, dtype=np.complex128)
            self.alm = alm.reshape(*self.shape)

        self.coords = coords

    def __setitem__(self, key, value):
        """
        Set the value of the alm given the frequency index and the values of
        ell and emm.
        """
        # if alm only has one frequency, it doesn't matter if the freq_idx is
        # not specified:
        if self.shape[0] == 1 and len(key) == 2:
            ell, emm = key
            key = [0, ell, emm]
        if len(key) != 3:
            raise IndexError(
                f"Key has length {len(key)}, but must have length 3 to specify"
                " frequency index, ell, and emm."
            )
        freq_idx, ell, emm = key
        ix = self.getidx(ell, emm)
        self.alm[freq_idx, ix] = value

    def __getitem__(self, key):
        # if alm only has one frequency, it doesn't matter if the freq_idx is
        # not specified:
        if self.alm.shape[0] == 1 and len(key) == 2:
            ell, emm = key
            key = [0, ell, emm]
        if len(key) != 3:
            raise IndexError(
                f"Key has length {len(key)}, but must have length 3 to specify"
                " frequency index, ell, and emm."
            )
        freq_idx, ell, emm = key
        ix = self.getidx(ell, np.abs(emm))
        coeff = self.alm[freq_idx, ix]
        if emm < 0:
            coeff = (-1) ** emm * coeff.conj()
        return coeff

    def all_zero(self):
        self.alm = np.zeros(self.shape, dtype=np.complex128)

    @property
    def shape(self):
        """
        Get the expected shape of the spherical harmonics.
        """
        Nfreq = self.frequencies.size
        shape = (Nfreq, self.size)
        return shape

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
            coords=hp_obj.coords,
        )
        return obj

    @classmethod
    def from_angles(
        cls,
        data,
        theta,
        phi,
        frequencies=None,
        lmax=None,
        coords="topographic",
    ):
        """
        Construct an Alm from a grid in theta and phi.
        """
        raise NotImplementedError

    def switch_coords(self, to_coords):
        rotated_alm = coordinates.rotate_alm(
            self.alm, from_coords=self.coords, to_coords=to_coords
        )
        self.alm = rotated_alm
        self.coords = to_coords

    def getlm(self, i=None):
        """
        Get the ell and emm corresponding to the numpy index of the alm
        array.
        """
        return super().getlm(self.lmax, i=i)

    def getidx(self, ell, emm):
        """
        Get the index of the alm array for a given ell and emm.
        """
        if not (0 <= emm <= ell <= self.lmax):
            raise ValueError("Ell or emm are out of bounds.")
        return super().getidx(self.lmax, ell, emm)

    @property
    def size(self):
        """
        Get the size of the alm array.
        """
        return super().getsize(self.lmax, mmax=self.lmax)

    @property
    def getlmax(self):
        """
        Get the maxmium ell of the Alm object.
        """
        return self.lmax

    def hp_map(self, nside=None):
        """
        Construct a healpy map from the Alm.
        """
        if nside is None:
            nside = (self.lmax + 1) // 3
        if self.frequencies is None:
            hp_map = hp.alm2map(
                self.alm.astype("complex"),
                nside,
                lmax=self.lmax,
                mmax=self.lmax,
            )
        else:
            hp_map = np.empty((len(self.frequencies), hp.nside2npix(nside)))
            for i, freq in enumerate(self.frequencies):
                map_i = hp.alm2map(
                    self.alm[i].astype("complex"),
                    nside,
                    lmax=self.lmax,
                    mmax=self.lmax,
                )
                hp_map[i] = map_i
        return hp_map

    def rotate_z_phi(self, phi):
        """
        Get the coefficients that rotate the alms around the z-axis by phi
        (measured counterclockwise).

        Parameters
        ----------
        phi : float
            The angle to rotate the azimuth by in radians.

        """
        emms = self.getlm()[1]
        phase = np.exp(1j * emms * phi)
        phase.shape = (1, -1)  # frequency axis
        return phase

    def rotate_z_time(self, delta_t, world="earth"):
        """
        Rotate alms in time counterclockwise around the z-axis.
        """
        if not world == "earth":
            raise NotImplementedError("Moon will be added shortly.")
        dphi = 2 * np.pi * delta_t / constants.sidereal_day
        phase = self.rotate_z_phi(dphi)
        return phase
