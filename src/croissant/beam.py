import jax
import jax.numpy as jnp
import s2fft

from . import sphere


class Beam(sphere.SphBase):
    horizon: jax.Array  # boolean mask for above/below horizon
    beam_az_rot: jax.Array  # in degrees
    beam_tilt: jax.Array  # in degrees

    def __init__(
        self,
        data,
        freqs,
        sampling="mwss",
        horizon=None,
        beam_az_rot=0.0,
        beam_tilt=0.0,
    ):
        """
        Beam pattern object. Holds the beam pattern in local antenna
        coordinates and associated metadata. The beam must be defined
        on the grid specified by the `sampling` scheme.

        Parameters
        ----------
        data : array_like
            Power beam pattern data. First axis is frequency, second
            axis is theta (colatitude), and third axis is phi (longitude).
            If `sampling` is "healpix", the data only has two dimensions:
            frequency and pixel index.
        freqs : array_like
            Frequencies corresponding to the beam pattern data.
        sampling : str
            Sampling scheme of the beam pattern data. Supported schemes
            are determined by s2fft, cuttently they include
            {"mw", "mwss", "dh", "gl", "healpix"}. The default is
            "mwss", which is a 1 deg equiangular sampling in theta and
            phi and includes the poles.
        horizon : array_like or None
            The horizon mask: a boolean array specified for each
            (theta, phi) direction (or pixel), with the same shape as
            the last two (one for healpix) axes of data. It is an array
            with True values for directions that are above the horizon
            and False for directions that are below the horizon.
            If None, it is assumed that the horizon is at
            theta = 90 degrees.
        beam_az_rot : float
            Angle between the X-axis of the beam (antenna local frame)
            and the local East direction, in degrees. The angle is
            measured counter-clockwise from the local East direction.
            For example, if the X-axis of the beam points towards the
            local North direction, the `beam_az_rot` would be +90 deg.
        beam_tilt : float
            The tilt angle of the beam in degrees. The tilt is the
            angle measured from the local zenith towards the antenna
            pointing direction.

        """
        super().__init__(data, freqs, sampling)

        if not jnp.isclose(beam_tilt, 0.0):
            raise NotImplementedError("Beam tilt is not yet implemented.")

        if horizon is None:
            horizon = self.theta <= jnp.pi / 2
            if self.sampling != "healpix":
                horizon = jnp.expand_dims(horizon, axis=-1)  # add phi axis
        self.horizon = jnp.asarray(horizon)

        self.beam_az_rot = jnp.asarray(beam_az_rot)
        self.beam_tilt = jnp.asarray(beam_tilt)

    def _compute_norm(self, use_horizon=True):
        """
        Compute the integral of the beam pattern over the sphere,
        optionally including only the part above the horizon.

        Parameters
        ----------
        use_horizon : bool
            Whether to include only the part of the beam above the
            horizon.
            If False, the entire beam pattern is integrated over.

        Returns
        -------
        norm : jax.Array
            Normalization factor for the beam pattern. One number per
            frequency.

        """
        if self.sampling == "healpix":
            npix = 12 * self.nside**2
            wgts = jnp.ones(npix) * (4 * jnp.pi / npix)
        else:
            wgts = s2fft.utils.quadrature_jax.quad_weights(
                L=self._L, sampling=self.sampling, nside=self.nside
            )

        if use_horizon:
            data = self.data * self.horizon[None]
        else:
            data = self.data

        norm = jnp.einsum("ft...,t->f", data, wgts)
        return norm

    @jax.jit
    def compute_norm(self):
        """
        Compute the normalization factor for the beam pattern. This is
        the integral of the beam pattern over the whole sphere.

        Returns
        -------
        norm : jax.Array
            Normalization factor for the beam pattern. One number per
            frequency.

        """
        return self._compute_norm(use_horizon=False)

    @jax.jit
    def compute_fgnd(self):
        """
        Compute the ground fraction for the beam pattern. This is the
        integral of the beam pattern over the part of the sphere below
        the horizon, divided by the integral over the whole sphere.

        Returns
        -------
        fgnd : jax.Array
            Ground fraction for the beam pattern. One number per frequency.

        """
        norm_total = self._compute_norm(use_horizon=False)
        norm_above_horizon = self._compute_norm(use_horizon=True)
        fgnd = 1.0 - norm_above_horizon / norm_total
        return fgnd

    @jax.jit
    def compute_alm(self):
        """
        Compute the spherical harmonic coefficients of the beam pattern.
        Only the part of the beam above the horizon is included
        in the spherical harmonic transform. We automatically apply the
        rotations to the beam pattern based on the `beam_az_rot` and
        `beam_tilt` angles.

        Returns
        -------
        alm : jax.Array
            Normalized spherical harmonic coefficients of the beam
            pattern.

        """
        data = self.data * self.horizon[None]  # mask out below-horizon part
        alm = sphere.compute_alm(
            data, self.lmax, self.sampling, nside=self.nside
        )
        emms = jnp.arange(-self.lmax, self.lmax + 1)
        phase = jnp.exp(-1j * emms * jnp.radians(self.beam_az_rot))
        alm = alm * phase[None, None, :]  # add freq/ell axes
        return alm
