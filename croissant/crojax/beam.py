import jax.numpy as jnp
from s2fft.sampling import s2_samples
from healpy import get_nside

from ..constants import Y00
from .healpix import Alm


class Beam(Alm):
    def compute_total_power(self):
        """
        Compute the total integrated power in the beam at each frequency. This
        is a necessary normalization constant for computing the visibilities.
        It should be computed before applying the horizon cut in order to
        account for ground loss.
        """
        a00 = self[:, 0, 0]
        power = a00.real * Y00 * 4 * jnp.pi
        self.total_power = power

    def horizon_cut(self, horizon=None, sampling="mw", nside=None):
        """
        horizon : jnp.ndarray
           A mask 0s and 1s indicating the horizon, with 1s corresponding to
           above the horizon. If None, the horizon is assumed to be flat at
           theta = pi/2. The shape must match the sampling scheme given by
           ``sampling'' and the lmax of the beam given in self.lmax. See
           s2fft.sampling.s2_samples.f_shape for details.
        sampling : str
            Sampling scheme of the horizon mask. Must be in
            {"mw", "mwss", "dh", "healpix"}. Gets passed to s2fft.forward.
        nside : int
            The nside of the horizon mask for the intermediate step. Required
            if sampling == "healpix" and horizon is None.

        Raises
        ------
        ValueError
            If horizon is not None and has elements outside of [0, 1].
        """
        if horizon is not None:
            if horizon.min() < 0 or horizon.max() > 1:
                raise ValueError("Horizon elements must be in [0, 1].")
            if sampling.lower() == "healpix":
                nside = get_nside(horizon)

        # invoke horizon mask in pixel space
        m = self.alm2map(sampling=sampling, nside=nside)
        if horizon is None:
            horizon = jnp.ones_like(m)
            theta = s2_samples.thetas(
                L=self.lmax + 1, sampling=sampling, nside=nside
            )
            horizon.at[..., theta > jnp.pi / 2].set(0.0)

        m = m * horizon
        self.alm = jax.vmap(
            partial(
                s2fft.forward_jax,
                L=self.lmax + 1,
                spin=0,
                nside=nside,
                reality=self.is_real,
                precomps=None,
                spmd=False,
                L_lower=0,
            )
        )(m)
