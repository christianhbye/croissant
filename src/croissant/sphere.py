from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import s2fft

from . import utils


@eqx.filter_jit
def compute_alm(
    data,
    lmax,
    sampling,
    nside=None,
    niter=0,
    spin=0,
    reality=True,
):
    """
    Compute spherical harmonic coefficients for scalar or spin fields.

    Wraps ``s2fft.forward`` and treats every axis before the spatial axes as
    a batch axis. The scalar defaults are intentionally identical to the
    original Croissant API.

    Parameters
    ----------
    data : array_like
        Field data. The final two axes are theta (colatitude) and phi
        (longitude), except for ``healpix``, where the final axis is pixel.
        Any leading axes are transformed in parallel.
    lmax : int
        Maximum spherical harmonic degree to compute.
    sampling : str
        Sampling scheme of the field data. Supported schemes are determined
        by s2fft, currently they include {"mw", "mwss", "dh", "gl",
        "healpix"}.
    nside : int or None,
        Nside parameter for healpix sampling. Required if `sampling` is
        "healpix". Ignored otherwise.
    niter : int
        Number of iterations for the s2fft algorithm. Higher values can
        improve accuracy at the cost of increased computation time.
        Default is 0, which corresponds to the default behavior of
        s2fft.
    spin : int
        Spin weight of the input field. Default is 0.
    reality : bool
        Whether to use the real-valued scalar transform optimization.
        Set to False for complex inputs and all nonzero-spin transforms.

    Returns
    -------
    alm : jax.Array
        Spherical harmonic coefficients of the field. Shape is
        ``data.shape[:-spatial_ndim] + (lmax+1, 2*lmax+1)``.

    """
    data = jnp.asarray(data)
    spatial_ndim = 1 if sampling == "healpix" else 2
    if data.ndim < spatial_ndim:
        raise ValueError(
            f"Data for {sampling!r} sampling must have at least "
            f"{spatial_ndim} spatial dimension(s)."
        )
    if spin != 0 and reality:
        raise ValueError("Nonzero-spin transforms require reality=False.")

    spatial_shape = data.shape[-spatial_ndim:]
    batch_shape = data.shape[:-spatial_ndim]
    flat_data = data.reshape((-1,) + spatial_shape)
    m2alm = partial(
        s2fft.forward,
        L=lmax + 1,
        spin=spin,
        nside=nside,
        sampling=sampling,
        method="jax",
        reality=reality,
        precomps=None,
        spmd=False,
        L_lower=0,
        iter=niter,
    )
    flat_alm = jax.vmap(m2alm)(flat_data)
    return flat_alm.reshape(batch_shape + (lmax + 1, 2 * lmax + 1))


class SphBase(eqx.Module):
    data: jax.Array
    freqs: jax.Array
    sampling: str = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    _L: int = eqx.field(static=True)  # L = lmax + 1 for s2fft
    _niter: int = eqx.field(static=True)  # niter for sht
    nside: int | None = eqx.field(static=True)
    theta: jax.Array  # in radians
    phi: jax.Array  # in radians

    def __init__(self, data, freqs, sampling, niter=0):
        """
        Base class for scalar fields on the sphere. Holds the field
        data and associated metadata. The field must be defined on the
        grid specified by the `sampling` scheme.

        Parameters
        ----------
        data : array_like
            Field data. First axis is frequency, second axis is theta
            (colatitude), and third axis is phi (longitude). If
            `sampling` is "healpix", the data only has two dimensions:
            frequency and pixel index.
        freqs : array_like
            Frequencies corresponding to the field data.
        sampling : str
            Sampling scheme of the field data. Supported schemes are
            determined by s2fft, currently they include {"mw", "mwss",
            "dh", "gl", "healpix"}. The default is "mwss", which is a 1
            deg equiangular sampling in theta and phi and includes the
            poles.
        niter : int
            Number of iterations for the s2fft algorithm. Higher values
            can improve accuracy at the cost of increased computation
            time. Default is 0 for all sampling schemes. For healpix
            sampling, setting niter=3 improves accuracy but
            significantly increases JIT compile time.

        Raises
        ------
        ValueError
            If `sampling` is "healpix" and the number of pixels in
            `data` is not valid for healpix sampling.

        """
        self.data = jnp.asarray(data)
        self.freqs = jnp.atleast_1d(freqs)

        if sampling == "healpix":
            npix = self.data.shape[1]
            if not utils.hp_valid_npix(npix):
                raise ValueError(
                    f"Invalid number of pixels {npix} for healpix sampling. "
                    "Number of pixels must be of the form 12 * nside^2."
                )

        self._niter = niter

        self.sampling = sampling
        self.lmax = utils.lmax_from_ntheta(self.data.shape[1], self.sampling)
        self._L = self.lmax + 1  # for s2fft, L = lmax + 1

        if self.sampling == "healpix":
            self.nside = utils.hp_npix2nside(self.data.shape[1])
        else:
            self.nside = None

        self.phi = utils.generate_phi(
            lmax=self.lmax, sampling=self.sampling, nside=self.nside
        )
        self.theta = utils.generate_theta(
            lmax=self.lmax, sampling=self.sampling, nside=self.nside
        )
