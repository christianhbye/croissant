import healpy as hp
import numpy as np

from . import coordinates
from .constants import sidereal_day


# nside's for which pixel weights exist
PIX_WEIGHTS_NSIDE = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def map2alm(data, lmax):
    """
    Compute the spherical harmonics coefficents of a healpix map.
    """
    data = np.array(data, copy=True)
    npix = data.shape[-1]
    nside = hp.npix2nside(npix)
    use_pix_weights = nside in PIX_WEIGHTS_NSIDE
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
        alm = np.empty((len(data), hp.Alm.getsize(lmax, mmax=lmax)))
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
            data = np.array(data, copy=True)
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
        """
        if alm is None and lmax is None:
            raise ValueError("Specify at least one of lmax and alm.")

        self.frequencies = np.ravel(frequencies).copy()
        if alm is None:
            self.lmax = lmax
            self.alm = np.zeros(self.alm_shape)
        elif lmax is None:
            self.alm = np.squeeze(alm).reshape(self.frequencies.size, -1)
            self.lmax = super().getlmax(alm.shape[1])
        else:
            self.lmax = lmax
            self.alm = np.array(alm).reshape(*self.alm_shape)

        self.coords = coords

    @property
    def alm_shape(self):
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
            lmax = hp.Alm().getlmax(alm.size)
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

    def set_coeff(self, value, ell, emm, freq_idx=None):
        """
        Set the value of an a_lm given the ell and emm.
        """
        ix = self.getidx(ell, emm)
        if self.alm.ndim == 1:
            self.alm[ix] = value
        else:
            if freq_idx is None:
                if self.alm.shape[0] > 1:
                    raise ValueError("No frequency index given.")
                else:
                    freq_idx = 0
            self.alm[freq_idx, ix] = value

    def get_coeff(self, ell, emm, freq_idx=None):
        """
        Get the value of an a_lm given the ell and emm.
        """
        ix = self.getidx(ell, emm)
        return self.alm[freq_idx, ix]

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
        dphi = 2 * np.pi * delta_t / sidereal_day
        phase = self.rotate_z_phi(dphi)
        return phase
