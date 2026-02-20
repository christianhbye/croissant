from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import s2fft

from . import utils


@partial(jax.jit, static_argnames=["lmax", "sampling", "nside"])
def compute_alm(data, lmax, sampling, nside=None):
    """
    Compute the spherical harmonic coefficients of a scalar field on
    the sphere. Wraps the s2fft.forward_jax in a convenient interface
    and uses vmap to compute the coefficients for multiple frequencies
    in parallel.

    Parameters
    ----------
    data : array_like
        Field data. First axis is frequency, second axis is theta
        (colatitude), and third axis is phi (longitude). If `sampling`
        is "healpix", the data only has two dimensions: frequency and
        pixel index.
    lmax : int
        Maximum spherical harmonic degree to compute.
    sampling : str
        Sampling scheme of the field data. Supported schemes are determined
        by s2fft, currently they include {"mw", "mwss", "dh", "gl",
        "healpix"}.
    nside : int or None,
        Nside parameter for healpix sampling. Required if `sampling` is
        "healpix". Ignored otherwise.

    Returns
    -------
    alm : jax.Array
        Spherical harmonic coefficients of the field. Shape is
        (len(data), lmax+1, 2*lmax+1)

    """
    m2alm = jax.vmap(s2fft.forward_jax, in_axes=(0, None))
    alm = m2alm(
        data, lmax + 1, spin=0, nside=nside, sampling=sampling, reality=True
    )
    return alm


class SphBase(eqx.Module):
    data: jax.Array
    freqs: jax.Array
    sampling: str = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    _L: int = eqx.field(static=True)  # L = lmax + 1 for s2fft
    nside: int | None = eqx.field(static=True)
    theta: jax.Array  # in radians
    phi: jax.Array  # in radians

    def __init__(self, data, freqs, sampling, lmax=None):
        """
        Base class for scalar fields on the sphere. Holds the field
        data and associated metadata. The field must be defined on the
        grid specified by the `sampling` scheme.

        Parameters
        ----------
        data : array_like
            Field data. First axis is frequency, second axis is theta
            (colatitude), and third axis is phi (longitude). If `sampling`
            is "healpix", the data only has two dimensions: frequency and
            pixel index.
        freqs : array_like
            Frequencies corresponding to the field data.
        sampling : str
            Sampling scheme of the field data. Supported schemes are
            determined by s2fft, currently they include {"mw", "mwss",
            "dh", "gl", "healpix"}. The default is "mwss", which is a 1
            deg equiangular sampling in theta and phi and includes the
            poles.
        lmax : int or None
            Maximum spherical harmonic degree to compute. If None, it is
            inferred from the data shape and sampling scheme. Note that
            this value cannot be greater than the natural lmax of the
            sampling scheme and the data shape.

        """
        self.data = jnp.asarray(data)
        self.freqs = jnp.atleast_1d(freqs)

        lmax_range = utils.lmax_range(sampling, self.data.shape[1:])
        if lmax is None:
            lmax = lmax_range[1]
        elif lmax < lmax_range[0] or lmax > lmax_range[1]:
            raise ValueError(
                f"Requested lmax {lmax} is not compatible with the data shape "
                f"{self.data.shape} and sampling scheme {sampling}."
            )

        self.sampling = sampling
        self.lmax = lmax
        self._L = self.lmax + 1  # for s2fft, L = lmax + 1

        if self.sampling == "healpix":
            self.nside = utils.hp_npix2nside(self.data.shape[1])
        else:
            self.nside = None

        self.theta = s2fft.sampling.s2_samples.thetas(
            L=self._L, sampling=self.sampling, nside=self.nside
        )
        self.phi = utils.generate_phi(
            lmax=self.lmax, sampling=self.sampling, nside=self.nside
        )
