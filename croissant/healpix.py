import healpy as hp
import numpy as np
from uvtools.dspec import dpss_operator


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


def dpss_interpolator(target_frequencies, input_freqs, **kwargs):
    """
    Compute linear interpolator in frequency space using the Discrete Prolate
    Spheroidal Sequences (DPSS) basis.
    """
    if input_freqs is None:
        raise ValueError("No input frequencies are provided.")
    input_freqs = np.copy(input_freqs) * 1e6  # convert to Hz
    target_frequencies = np.array(target_frequencies) * 1e6  # Hz
    if np.max(target_frequencies) > np.max(input_freqs) or np.min(
        target_frequencies
    ) < np.min(input_freqs):
        raise ValueError(
            "Some of the target frequencies are outside the range of the "
            "input frequencies."
        )
    if not input_freqs in target_frequencies:
        target_frequencies = np.append(target_frequencies, input_freqs)
        target_frequencies.sort()

    fc = kwargs.pop("filter_centers", [0])
    fhw = kwargs.pop("filter_half_widths", [20e-9])
    ev_cut = kwargs.pop("eigenval_cutoff", [1e-12])
    B = dpss_operator(
        target_frequencies,
        filter_centers=fc,
        filter_half_widths=fhw,
        eigenval_cutoff=ev_cut,
        **kwargs,
    )
    A = B[np.isin(target_frequencies, input_frequencies)]
    interp = B @ np.linalg.inv(A.T @ A) @ A.T
    return interp


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
        hp_map = alm_obj.hp_map(nside=nside)
        return cls(nside, data=hp_map, frequencies=alm_obj.frequencies)

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

    def interp_frequencies(
        self,
        target_frequencies,
        input_frequencies=None,
        input_map=None,
        return_map=False,
        **kwargs,
    ):
        """
        Raises ValueError in case of shape mismatch (matmul)
        """
        if input_map is None:
            input_map = self.data
            if input_map is None:
                raise ValueError("No inut map provided.")
        if input_frequencies is None:
            input_frequencies = self.frequencies

        interp = dpss_interpolator(
            target_frequencies, input_frequencies, **kwargs
        )

        interpolated = interp @ input_map

        if return_map:
            return interpolated, target_frequencies
        else:
            self.data = interpolated
            self.frequencies = target_frequencies

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

        self.lmax = lmax
        self.frequencies = frequencies
        expected_shape = self.alm_shape

        if alm is None:
            self.alm = np.zeros(expected_shape)
        elif not np.shape(alm) == expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape} for alm, got shape"
                f"{np.shape(alm)}."
            )
        else:
            self.alm = alm

    @property
    def alm_shape(self):
        if self.lmax is None:
            return None

        if self.frequencies is None:
            Nfreq = 1
        else:
            Nfreq = len(self.frequencies)
        shape = (Nfreq, self.size)
        return shape

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

    def set_coeff(self, value, ell, emm, freq_idx=None):
        if freq_idx is None:
            if self.alm_shape[0] > 1:
                raise ValueError("No frequency index given.")
            else:
                freq_idx = 0
        ix = self.getidx(ell, emm)
        self.alm[freq_idx, ix] = value

    def get_coeff(self, ell, emm, freq_idx=None):
        ix = self.getidx(ell, emm)
        return self.alm[freq_idx, ix]

    def hp_map(self, nside=None):
        if nside is None:
            nside = (self.lmax + 1) // 3
        hp_map = hp.alm2map(self.alm, nside, lmax=self.lmax, mmax=self.lmax)
        return hp_map

    def interp_frequencies(
        self,
        target_frequencies,
        input_frequencies=None,
        input_alm=None,
        return_alm=False,
        **kwargs,
    ):
        """
        Raises ValueError in case of shape mismatch (matmul)
        """
        if input_alm is None:
            input_alm = self.alm
            if input_alm is None:
                raise ValueError("No inut alm provided.")
        if input_frequencies is None:
            input_frequencies = self.frequencies

        interp = dpss_interpolator(
            target_frequencies, input_frequencies, **kwargs
        )

        interpolated = interp @ input_alm

        if return_alm:
            return interpolated, target_frequencies
        else:
            self.alm = interpolated
            self.frequencies = target_frequencies

    def rotate_z(self, phi):
        """
        Rotate the alms around the z-axis by phi (measured counterclockwise).

        Parameters
        ----------
        phi : float
            The angle to rotate the azimuth by in radians.

        """
        emms = self.getlm()[1]
        phase = np.exp(1j * emms * phi)
        phase.shape = (1, -1)  # frequency axis
        self.alm *= phase
