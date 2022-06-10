import numpy as np
from pygdsm import GlobalSkyModel2016 as GSM
from .healpix import HealpixMap


class Sky(HealpixMap):
    def __init__(
        self,
        sky_map=None,
        frequencies=None,
        nested_input=False,
        coords="galactic",
    ):
        """
        Class that holds Sky objects. Thin wrapper for HealpixMap objects.
        """
        super().__init__(
            data=sky_map,
            nside=None,
            nested_input=nested_input,
            frequencies=frequencies,
            coords=coords,
        )

    @classmethod
    def gsm(cls, frequencies):
        """
        Construct a sky object with pygdsm
        """
        gsm16 = GSM(freq_unit="MHz", data_unit="TRJ", resolution="lo")
        sky_map = gsm16.generate(frequencies)
        obj = cls(
            sky_map,
            frequencies=frequencies,
            nested_input=False,
            coords="galactic",
        )
        return obj

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
            The reference map to extrapolate. If None, use self.data.
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
        else:
            ref_freq = np.ravel(ref_freq).copy()

        if ref_map is None:
            ref_map = self.data
        else:
            ref_map = np.array(ref_map, copy=True)
            if ref_map.ndim == 1:
                ref_map.shape = (1, -1)

        def _pow_law(f):
            t0 = ref_map[0].reshape(1, -1)
            f0 = ref_freq[0]
            f = np.array(f).reshape(-1, 1)
            return t0 * (f / f0) ** spectral_index

        if not np.allclose(_pow_law(ref_freq), ref_map):
            raise ValueError(
                "The provided maps don't satsify the specified power law."
            )

        sky_map = _pow_law(freq_out)

        if return_map:
            return sky_map, freq_out

        else:
            self.data = sky_map
            self.frequencies = np.ravel(freq_out)
