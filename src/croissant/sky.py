import equinox as eqx
import jax

from . import rotations, sphere


class Sky(sphere.SphBase):
    coord: str = eqx.field(static=True)

    def __init__(
        self, data, freqs, sampling="healpix", coord="galactic", niter=None
    ):
        """
        Object that holds the sky model.

        Parameters
        ----------
        data : array_like
            The sky model data. Should be of shape (N_freqs, N_pix) if
            sampling is "healpix" and (N_freqs, N_theta, N_phi) if sampling
            is something else.
        freqs : array_like
            The frequencies corresponding to the sky model data. Should
            have shape (N_freqs,).
        sampling : str
            The sampling scheme of the sky model data. Supported
            schemes are determined by s2fft and include
            {"mw", "mwss", "dh", "gl", "healpix"}. Default is
            "healpix".
        coord : str
            The coordinate system of the sky model data. Supported
            systems are {"galactic", "equatorial", "mcmf"}. Default is
            "galactic". The alm's will be computed in equatorial
            coordinates (mcmf on moon).
        niter : int or None
            The number of iterations to use for the spherical harmonic
            transform. If None, the default number of iterations will be
            used (3 if sampling is "healpix", 0 otherwise).

        """
        if coord not in {"galactic", "equatorial", "mepa"}:
            raise ValueError(
                f"Unsupported coordinate system: {coord}. Supported systems "
                "are {'galactic', 'equatorial', 'mepa'}."
            )
        super().__init__(data, freqs, sampling, niter=niter)
        self.coord = coord

    @jax.jit
    def compute_alm(self):
        """
        Compute the spherical harmonic coefficients (alm) of the sky
        model.

        """
        return sphere.compute_alm(
            self.data,
            self.lmax,
            self.sampling,
            nside=self.nside,
            niter=self._niter,
        )

    def compute_alm_eq(self, world="moon"):
        """
        Compute the spherical harmonic coefficients (alm) of the sky
        model in the simulation frame.

        Parameters
        ---------
        world : {"moon", "earth"}
            Which simulation frame to use. If ``world'' is "moon", the
            alm's will be computed in the MEPA (Mean Earth / Polar Axis)
            coordinate system. If "earth", the alm's will be computed
            in FK5 equatorial coordinates.

        Notes
        -----
        This method does not support mepa <-> equatorial
        transformations. If the sky model is in galactic coordinates,
        both "earth" and "moon" are possible. Otherwise, ``world'' must
        match the coordinate system of the sky model.

        """
        if world not in {"moon", "earth"}:
            raise ValueError(
                f"Unsupported world: {world}. Supported worlds are "
                "{'moon', 'earth'}."
            )
        if (
            self.coord == "mepa"
            and world == "earth"
            or self.coord == "equatorial"
            and world == "moon"
        ):
            raise ValueError(
                f"Unsupported coordinate transformation: {self.coord} to "
                f"{world}. "
            )
        alm = self.compute_alm()
        if self.coord != "galactic":
            return alm

        if world == "earth":
            alm = rotations.gal2eq(alm)
        else:
            alm = rotations.gal2mepa(alm)
        return alm
