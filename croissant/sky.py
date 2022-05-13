from astropy.io import fits
import numpy as np
from pygdsm import GlobalSkyModel2016 as GSM
from .healpix import HealpixBase


def npix2nside(npix):
    err = f"Invalid value of npix: {npix}."
    ns_sq = npix/12
    if not np.isclose(ns_sq, int(ns_sq)):
        raise ValueError(err)
    else:
        ns_sq = int(ns_sq)
    nside = np.sqrt(ns_sq)
    if not np.isclose(nside, int(nside)):
        raise ValueError(err)
    else:
        nside = int(nside)
        return nside

def check_sky_shapes(sky_map, frequencies)
    sh = np.shape(sky_map)
    sh_err = f"Unexpected shape of sky map: {sh}."
    npix = sh[-1]
    if len(sh) > 2:
        raise ValueError(sh_err)
    elif frequencies is None:
        allowed_shapes = [(1, npix), (npix,)]
    else:
        nfreqs = len(frequencies)
        allowed_shapes = [(nfreqs, npix)]
    if not sh in allowed_shapes:
        raise ValueError(sh_err)

class Sky(HealpixBase):
    def __init__(self, sky_map, frequencies=None, nested_input=False):
        check_sky_shapes(sky_map, frequencies)
        super().__init__(
            self.nside,
            data=sky_map,
            nested_input=nested_input,
            frequencies=frequencies,
        )

    @property
    def nside(self):
        npix = np.shape(self.sky_map)[-1]
        return npix2nside(npix)

    @classmethod
    def gsm(cls, frequencies, res="hi"):
        gsm16 = GSM(freq_unit="MHz", data_unit="TRJ", resolution=res)
        sky_map = gsm16.generate(frequencies)
        return cls(sky_map, frequencies=frequencies, nested_input=False)
    
    def power_law_map(
        self,
        freq_out,
        spectral_index=-2.5,
        ref_map=None,
        ref_freq=None,
        return_map=False,
    ):
        """
        Extrapolate or interpolate a map at specified frequencies to other
        frequencies using a power law of specified spectral index.

        Parameters
        ----------
        freq_out : array-like
            The frequencies to extrapolate the map to.
        spectral_index : float
            The spectral index of the power law. Defaults to -2.5
            (theoretical synchrotron power law).
        ref_map : np.ndarray (optional)
            The reference map to extrapolate. If None, use self.sky_map.
        ref_freq : array-like (optional)
            The reference frequency that the map is given at. If None, use
            self.frequencies.
        return_map : bool
            Set to True to return the map and frequencies as np.ndarrays.
            Otherwise, the map and frequencies replace the attributes sky_map
            and the frequencies.

        Returns
        -------
        These will only be returned if the flag return_map is True.

        sky_map : np.ndarray
            The extrapolated healpix map at the given frequencies.
        freq_out : array-like
           The frequencies of the extrapolated map.

        Raises
        ------
        ValueError :
            If the reference maps or frequencies are None, their
            shapes don't match.
            If multiple maps are provided but they don't follow the power law
            specified.
        """
        if ref_freq is None:
            ref_freq = self.frequencies
            if ref_freq is None:
                raise ValueError("No reference frequency is provided.")
        if ref_map is None:
            ref_map = self.sky_map
            if ref_map is None:
                raise ValueError("No reference map is provided.")
        ref_freq = np.array(ref_freq)
        if len(ref_map.shape) == 1:
            ref_map.shape = (1, -1)
        if not ref_freq.shape[0] == ref_map.shape[0]:
            raise ValueError(
                "Shape mismatch between reference frequencies and maps."
            )

        def _pow_law(f):
            t0 = ref_map[0].reshape(1, -1)
            f0 = ref_freq[0]
            return t0 * (f / f0) ** spectral_index

        if not np.allclose(_pow_law(ref_freq), ref_map):
            raise ValueError(
                "The provided maps don't satsify the specified power law."
            )

        sky_map = _pow_law(freq_out)

        if return_map:
            return sky_map, freq_out

        else:
            self.sky_map = sky_map
            self.frequencies = self.freq_out
