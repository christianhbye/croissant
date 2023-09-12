from functools import partial
import jax
import jax.numpy as jnp
import s2fft
from pygdsm import GlobalSkyModel2016 as GSM16
from .healpix import Alm


class Sky(Alm):
    @classmethod
    def gsm(cls, freq, lmax):
        """
        Construct a sky object with pygdsm.

        Parameters
        ----------
        freq : jnp.ndarray
            Frequencies to make map at in MHz.
        lmax : int
            Maximum multipole to compute alm up to.
        """
        gsm = GSM16(freq_unit="MHz", data_unit="TRJ", resolution="lo")
        sky_map = gsm.generate(freq)
        sky_map = jnp.atleast_2d(sky_map)
        forward = partial(
            s2fft.forward_jax,
            spin=0,
            nside=gsm.nside,
            sampling="healpix",
            reality=True,
            precomps=None,
            spmd=True,
            L_lower=0,
        )
        L = lmax + 1
        sky_alm = jax.vmap(forward, in_axes=[0, None])(sky_map, L)
        obj = cls(sky_alm, frequencies=freq, coord="G")
        return obj
