import numpy as np
from pygdsm import GlobalSkyModel2016 as GSM16
from .healpix import Alm
from .sphtransform import map2alm


class Sky(Alm):
    @classmethod
    def gsm(cls, freq, lmax=None):
        """
        Construct a sky object with pygdsm.

        Parameters
        ----------
        freq : array-like
            Frequencies to make map at in MHz.
        """
        freq = np.array(freq)
        gsm = GSM16(freq_unit="MHz", data_unit="TRJ", resolution="lo")
        sky_map = gsm.generate(freq)
        sky_alm = map2alm(sky_map, lmax=lmax)
        obj = cls(
            sky_alm,
            lmax=lmax,
            frequencies=freq,
            coord="G",
        )

        return obj
