import numpy as np
from pygdsm import GlobalSkyModel2016 as GSM
from .healpix import Alm


class Sky(Alm):
    @classmethod
    def gsm(cls, freq):
        """
        Construct a sky object with pygdsm.

        Parameters
        ----------
        freq : array-like
            Frequencies to make map at in MHz.
        """
        freq = np.ravel(freq)
        gsm16 = GSM(freq_unit="MHz", data_unit="TRJ", resolution="lo")
        sky_map = gsm16.generate(freq)

        obj = cls(
            sky_map,
            frequencies=freq,
            nested_input=False,
            coords="galactic",
        )

        return obj
