import equinox as eqx
import jax
import jax.numpy as jnp
import s2fft

from . import sphere
from .horizon import _default_horizon, _horizon_in_beam_frame


class Beam(sphere.SphBase):
    horizon: jax.Array  # visible fraction in [0, 1] per spatial sample
    horizon_frame: str = eqx.field(static=True)
    beam_rot: jax.Array  # in degrees
    beam_tilt: jax.Array  # in degrees

    def __init__(
        self,
        data,
        freqs,
        sampling="mwss",
        horizon=None,
        beam_rot=0.0,
        beam_tilt=0.0,
        niter=0,
        engine="auto",
        lmax=None,
        horizon_frame="beam",
    ):
        """
        Beam pattern object. Holds the beam pattern in local antenna
        coordinates and associated metadata. The beam must be defined
        on the grid specified by the `sampling` scheme.

        Theta is colatitude from zenith. Phi is right-handed about the
        zenith, from the beam's x axis towards its y axis. At zero
        ``beam_rot`` the axes are East and North; positive ``beam_rot``
        rotates clockwise as seen from above. A beam-grid direction has
        compass azimuth ``A = 90 + beam_rot - degrees(phi)`` (mod 360).

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
            are determined by s2fft, currently they include
            {"mw", "mwss", "dh", "gl", "healpix"}. The default is
            "mwss", which is a 1 deg equiangular sampling in theta and
            phi and includes the poles.
        horizon : array_like or None
            Visible fractions in [0, 1] for each (theta, phi) direction
            (or pixel), broadcastable to the spatial axes of data.
            Zero blocks a sample, one keeps it, and fractional values
            weight partially visible cells. Boolean masks are accepted
            unchanged. The weights apply to both the harmonic transform
            and the above-horizon integral used for the ground fraction.
            For terrain, use ``horizon_frame="topocentric"`` so these
            weights stay fixed to the ground. The compatibility default
            keeps them in the beam frame, rotating with the antenna.
            If None, the horizon is at theta = 90 degrees with
            fractional boundary cells (half weight on an equatorial
            row or HEALPix pixel). See ``horizon_weights`` for custom
            horizons on regular grids.
        beam_rot : float
            Azimuthal rotation of the beam in degrees. The rotation
            follows the astronomical azimuth convention: it is
            measured from local North towards East. A value of 0
            leaves the beam unrotated (phi=0 axis aligned with
            local East in ENU). For example, ``beam_rot=90`` rotates
            the beam so that its phi=0 axis points South.
        beam_tilt : float
            The tilt angle of the beam in degrees. The tilt is the
            angle measured from the local zenith towards the antenna
            pointing direction.
        niter : int
            Number of iterations for the spherical harmonic transform
            when using iterative methods. Default is 0 for all sampling
            schemes. For healpix, setting niter=3 improves accuracy
            but significantly increases JIT compile time.
        engine : {"auto", "s2fft", "kernel", "dense"}
            Spherical harmonic transform engine. Default is ``"auto"``.
            ``"auto"`` lets croissant choose from the band-limit,
            sampling, niter and batch size; the choice is reported by the
            ``engine`` and ``engine_reason`` attributes.
        lmax : int or None
            Maximum spherical harmonic degree. For HEALPix data this may be
            lower than the default ``2 * nside``. Default is None.
        horizon_frame : {"beam", "topocentric"}
            Use ``"topocentric"`` for terrain. It keeps weights on the
            fixed ground grid, whose phi=0 is East and phi=pi/2 is North.
            Build terrain
            heights from compass azimuth ``A = pi/2 - phi`` on that
            grid, independent of ``beam_rot``. We counter-rotate the
            mask into the beam frame using periodic linear longitude
            interpolation (within latitude rings for HEALPix). This
            preserves weights in [0, 1] but can soften sharp edges for
            rotations between grid columns. ``horizon`` retains the
            input weights; ``horizon_in_beam_frame`` gives the weights
            actually applied. Tilt remains unsupported; a future tilt
            must transform a ground-fixed mask in both theta and phi.
            Default ``"beam"`` preserves existing behavior: blockage
            rotates with the antenna. Use it for antenna-attached
            obstructions or masks already counter-rotated by the caller.

        """
        if horizon_frame not in {"beam", "topocentric"}:
            raise ValueError("horizon_frame must be 'beam' or 'topocentric'.")
        self.horizon_frame = horizon_frame
        super().__init__(
            data,
            freqs,
            sampling,
            niter=niter,
            engine=engine,
            lmax=lmax,
        )

        if not jnp.isclose(beam_tilt, 0.0):
            raise NotImplementedError("Beam tilt is not yet implemented.")

        if horizon is None:
            horizon = _default_horizon(self.theta, self.sampling)
        self.horizon = jnp.asarray(horizon)
        if horizon_frame == "topocentric":
            jnp.broadcast_to(self.horizon, self.data.shape[1:])

        self.beam_rot = jnp.asarray(beam_rot)
        self.beam_tilt = jnp.asarray(beam_tilt)

    @property
    def horizon_in_beam_frame(self):
        """Visibility weights applied at the current ``beam_rot``."""
        return _horizon_in_beam_frame(
            self.horizon,
            self.horizon_frame,
            self.beam_rot,
            self.sampling,
            self.data.shape[1:],
            self.nside,
        )

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
            data = self.data * self.horizon_in_beam_frame[None]
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
        rotations to the beam pattern based on the `beam_rot` (azimuth) and
        `beam_tilt` angles.

        Returns
        -------
        alm : jax.Array
            Normalized spherical harmonic coefficients of the beam
            pattern.

        """
        data = self.data * self.horizon_in_beam_frame[None]
        alm = sphere.compute_alm(
            data,
            self.lmax,
            self.sampling,
            nside=self.nside,
            niter=self._niter,
            # A beam pattern is a real power response, so it can claim
            # the packed real transform that compute_alm will not
            # assume. The complex azimuthal phase is applied below, in
            # harmonic space, and so does not affect this.
            reality=True,
            engine=self._engine,
            dense_matrix=self._dense_matrix,
            kernel=self._kernel,
            inverse_kernel=self._inverse_kernel,
        )
        # apply azimuthal rotation, N→E convention (no-op when beam_rot == 0)
        emms = jnp.arange(-self.lmax, self.lmax + 1)
        phase = jnp.exp(1j * emms * jnp.radians(self.beam_rot))
        alm = alm * phase[None, None, :]  # add freq/ell axes
        return alm
