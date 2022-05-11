import healpy as hp
import numpy as np


def nside2npix(nside):
    npix = 12 * nside**2
    return npix


def check_shapes(npix, data, frequencies):
    if data is None:
        return

    if frequencies is None:
        expected_shape = (npix,)
    else:
        nfreq = len(frequencies)
        expected_shape = (nfreq, npix)

    check = np.shape(data) == expected_shape

    if not check:
        raise ValueError(
            f"Expected data shape is {expected_shape}, but data has"
            f"shape {np.shape(data)}."
        )


# nside's for which pixel weights exist
PIX_WEIGHTS_NSIDE = [32, 64, 128, 256, 512, 1024, 2048, 4096]


class HealpixBase:
    def __init__(self, nside, data=None, nested_input=False, frequencies=None):
        hp.check_nside(nside, nest=nested_input)
        self.nside = nside
        check_shapes(self.npix, data, frequencies)

        if data is None:
            nested_input = False

        if nested_input:
            ix = hp.nest2ring(self.nside, np.arange(self.npix))
            if frequencies is None:
                data = data[ix]
            else:
                data = data[:, ix]

        self.data = data
        self.frequencies = frequencies

    @property
    def npix(self):
        return nside2npix(self.nside)

    @classmethod
    def from_alm(cls, alm_obj, nside=None):
        lmax = alm_obj.lmax
        if nside is None:
            nside = (lmax + 1) // 3
        hp_map = hp.alm2map(alm_obj.alm, nside, lmax=lmax, mmax=lmax)
        return cls(nside, data=hp_map, frequencies=alm_obj.frequencies)

    @classmethod
    def from_grid(cls):
        raise NotImplementedError

    def ud_grade(self, nside_out, **kwargs):
        new_map = hp.ud_grade(self.data, nside_out, **kwargs)
        self.data = new_map
        self.nside = nside_out

    @property
    def alm(self, lmax=None):
        if self.data is None:
            raise ValueError("data is None, cannot compute alms.")
        if lmax is None:
            lmax = 3 * self.nside - 1
        use_pix_weights = self.nside in PIX_WEIGHTS_NSIDE
        use_ring_weights = not use_pix_weights
        alm = hp.map2alm(
            self.data,
            lmax=lmax,
            mmax=lmax,
            use_ring_weights=use_ring_weights,
            use_pixel_weights=use_pix_weights,
        )
        return alm


    def plot(self, frequency=None, **kwargs):
        m = kwargs.pop("m", self.data)
        if frequency is not None and self.frequencies is not None:
            f_idx = np.argmin(np.abs(self.frequencies - frequency))
            f_to_plot = self.frequencies[f_idx]
            title = kwargs.pop("title", f"Frequency = {f_to_plot:.0f} MHz")
            m = m[f_idx]
        _ = hp.projview(m=m, title=title, **kwargs)


class Alm(hp.Alm):
    def __init__(self, alm=None, lmax=None, frequencies=None):
        if alm is not None and frequencies is not None:
            if not np.shape(alm)[0] == len(frequencies):
                raise ValueError("Shapes don't match: alms and frequencies.")

        self.alm = alm
        self.lmax = lmax
        self.frequencies = frequencies

    @classmethod
    def from_healpix(cls, hp_obj, lmax=None):
        alm = hp_obj.alm(lmax=lmax)
        return cls(alm=alm, lmax=lmax, frequencies=hp_obj.frequencies)

    @classmethod
    def from_grid(cls):
        raise NotImplementedError

    def getlm(self, i=None):
        return super().getlm(self.lmax, i=i)

    def getidx(self, ell, emm):
        return super().getidx(self.lmax, ell, emm)

    @property
    def size(self):
        return super().getsize(self.lmax, mmax=self.lmax)

    @property
    def getlmax(self):
        return self.lmax
