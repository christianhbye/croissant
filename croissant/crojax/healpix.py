from functools import partial
import warnings
import jax
import jax.numpy as jnp
import s2fft
from .. import constants, utils


@jax.jit
def lmax_from_shape(shape):
    """
    Get the lmax from the shape of the alm array.
    """
    return shape[1] - 1


@jax.jit
def _getlm(ix, lmax):
    ell = ix[0]
    emm = ix[1] - lmax
    return ell, emm


@jax.jit
def _getidx(ell, emm, lmax):
    l_ix = ell
    m_ix = emm + lmax
    return l_ix, m_ix


def _is_real(alm):
    """
    Check if the alm coefficients correspond to a real-valued signal.

    Parameters
    ----------
    alm : jnp.ndarray
        The spherical harmonics coefficients. Must have shape
        (nfreq, lmax+1, 2*lmax+1) corresponding to the frequencies, ell, and
        emm indices.

    Returns
    -------
    is_real : bool
        True if the coefficients correspond to a real-valued signal.

    """
    lmax = lmax_from_shape(alm.shape)
    emm = jnp.arange(1, lmax + 1)[None, None, :]  # positive ms
    # get alms for negative m, in reverse order (i.e., increasing abs(m))
    neg_m = alm[:, :, :lmax][:, :, ::-1]
    # get alms for positive m
    pos_m = alm[:, :, lmax + 1 :]
    return jnp.all(neg_m == (-1) ** emm * jnp.conj(pos_m)).item()


def _rot_alm_z(lmax, phi):
    """
    Get the coefficients that rotate the alms around the z-axis by phi
    (measured counterclockwise).

    Parameters
    ----------
    lmax : int
        The maximum l value.
    phi : jnp.ndarray
        The angle(s) to rotate the azimuth by in radians. Must have shape
        (n, 1).

    Returns
    -------
    phase : np.ndarray
        The coefficients that rotate the alms by phi. Has shape (n, 2*lmax+1),
        where n is the number of phi values and 2*lmax+1 is the number of
        m values given lmax.
    """
    emms = jnp.arange(-lmax, lmax + 1)[None]
    phase = jnp.exp(-1j * emms * phi)
    return phase


class Alm:
    def __init__(self, alm, frequencies=None, coord=None):
        """
        Base class for spherical harmonics coefficients.

        Alm can be indexed with [freq_index, ell, emm] to get the
        coeffiecient corresponding to the given frequency index, and values of
        ell and emm. The frequencies can be indexed in the usual numpy way and
        may be 0 if the alms are specified for only one frequency.

        Parameters
        ----------
        alm : jnp.ndarray
            The spherical harmonics coefficients. Must have shape
            (nfreq, lmax+1, 2*lmax+1).
        frequencies : jnp.ndarray
            The frequencies corresponding to the coefficients. Must have shape
            (nfreq,). If None, then the coefficients are assumed to be for a
            single frequency and nfreq is set to 1.
        coord : str
            The coordinate system of the coefficients.


        """
        self.alm = alm
        self.frequencies = frequencies
        self.lmax = lmax_from_shape(alm.shape)
        if coord is None:
            self.coord = None
        else:
            self.coord = utils.coord_rep(coord)

    def __setitem__(self, key, value):
        """
        Set the value of the spherical harmonics coefficient. The frequency
        axis is indexed in the usual numpy way, while the other two indices
        correspond to the values of l and m.
        """
        lix, mix = self.getidx(*key[1:])
        new_key = (key[0], lix, mix)
        self.alm = self.alm.at[new_key].set(value)

    def __getitem__(self, key):
        lix, mix = self.getidx(*key[1:])
        new_key = (key[0], lix, mix)
        return self.alm[new_key]

    def getlm(self, ix):
        """
        Get the l and m corresponding to the index of the alm array.

        Parameters
        ----------
        ix : jnp.ndarray
            The indices of the alm array. The first row corresponds to the l
            index, and the second row corresponds to the m index. Multiple
            indices can be passed in as an array with shape (2, n).

        Returns
        -------
        ell : jnp.ndarray
            The value of l. Has shape (n,).
        emm : jnp.ndarray
            The value of m. Has shape (n,).
        """
        return _getlm(ix, self.lmax)

    def getidx(self, ell, emm):
        """
        Get the index of the alm array for a given l and m.

        Parameters
        ----------
        ell : int or jnp.ndarray
            The value of l.
        emm : int or jnp.ndarray
            The value of m.

        Returns
        -------
        l_ix : int or jnp.ndarray
           The l index (which is the same as the input ell).
        m_ix : int or jnp.ndarray
            The m index.

        Raises
        ------
        IndexError
            If l,m don't satisfy abs(m) <= l <= lmax.
        """
        if not ((jnp.abs(emm) <= ell) & (ell <= self.lmax)).all():
            raise IndexError("l,m must satsify abs(m) <= l <= lmax.")
        return _getidx(ell, emm, self.lmax)

    @classmethod
    def zeros(cls, lmax, frequencies=None, coord=None):
        """
        Construct an Alm object with all zero coefficients.
        """
        s1, s2 = s2fft.sampling.s2_samples.flm_shape(lmax + 1)
        shape = (jnp.size(frequencies), s1, s2)
        alm = jnp.zeros(shape, dtype=jnp.complex128)
        obj = cls(
            alm=alm,
            frequencies=frequencies,
            coord=coord,
        )
        return obj

    @property
    def is_real(self):
        """
        Check if the coefficients correspond to a real-valued signal.
        Mathematically, this means that alm(l, m) = (-1)^m * conj(alm(l, -m)).
        """
        return _is_real(self.alm)

    def reduce_lmax(self, new_lmax):
        """
        Reduce the maximum l value of the alm.

        Parameters
        ----------
        new_lmax : int
            The new maximum l value.

        Raises
        ------
        ValueError
            If new_lmax is greater than the current lmax.
        """
        d = self.lmax - new_lmax  # number of ell values to remove
        if d < 0:
            raise ValueError(
                "new_lmax must be less than or equal to the current lmax"
            )
        elif d > 0:
            self.alm = self.alm[:, :-d, d:-d]
            self.lmax = new_lmax

    def switch_coords(self, to_coord, loc=None, time=None):
        raise NotImplementedError

    def alm2map(self, sampling="healpix", nside=None, frequencies=None):
        """
        Construct a Healpix map from the Alm for the given frequencies.

        Parameters
        ----------
        sampling : str
            Sampling scheme on the sphere. Must be in
            {"mw", "mwss", "dh", "healpix"}. Gets passed to s2fft.inverse.
        nside : int
            The nside of the Healpix map to construct. Required if sampling
            is "healpix".
        frequencies : jnp.ndarray
            The frequencies to construct the map for. If None, the map will
            be constructed for all frequencies.

        Returns
        -------
        m : jnp.ndarray
            The map(s) corresponding to the alm.

        """
        if frequencies is None:
            alm = self.alm
        else:
            indices = jnp.isin(
                self.frequencies, frequencies, assume_unique=True
            ).nonzero()[0]
            if indices.size < jnp.size(frequencies):
                warnings.warn(
                    "Some of the frequencies specified are not in"
                    "alm.frequencies.",
                    UserWarning,
                )
            alm = self.alm[indices]
        m = jax.vmap(
            partial(
                s2fft.inverse_jax,
                L=self.lmax + 1,
                spin=0,
                nside=nside,
                sampling=sampling,
                reality=self.is_real,
                precomps=None,
                spmd=False,
                L_lower=0,
            )
        )(alm)
        return m

    def rot_alm_z(self, phi=None, times=None, world="moon"):
        """
        Get the coefficients that rotate the alms around the z-axis by phi
        (measured counterclockwise) or in time.

        Parameters
        ----------
        phi : jnp.ndarray
            The angle(s) to rotate the azimuth by in radians.
        times : jnp.ndarray
            The times to rotate the azimuth by in seconds. If given, phi will
            be ignored and the rotation angle will be calculated from the
            times and the sidereal day of the world.
        world : str
            The world to use for the sidereal day. Must be 'moon' or 'earth'.

        Returns
        -------
        phase : np.ndarray
            The coefficients (shape = (phi.size, alm.size) that rotate the
            alms by phi.

        """
        if times is not None:
            if world.lower() == "moon":
                sidereal_day = constants.sidereal_day_moon
            elif world.lower() == "earth":
                sidereal_day = constants.sidereal_day_earth
            else:
                raise ValueError(
                    f"World must be 'moon' or 'earth', not {world}."
                )
            phi = 2 * jnp.pi * times / sidereal_day
            return self.rot_alm_z(phi=phi, times=None)
        phi = phi[:, None]  # add axis for broadcasting
        return _rot_alm_z(self.lmax, phi)
