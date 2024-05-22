from functools import partial
import jax
import jax.numpy as jnp

from .. import constants


@partial(jax.jit, static_argnums=(0,))
def rot_alm_z(lmax, times, sidereal_day=constants.sidereal_day_moon):
    """
    Compute the complex phases that rotate the sky for a range of times. The
    first time is the reference time and the phases are computed relative to
    this time.

    Parameters
    ----------
    lmax : int
        The maximum ell value.
    times : jnp.ndarray
        The times for which to compute the phases.
    sidereal_day : str
        The length of a sidereal day in the same units as ``times''. Default
        is the sidereal day of the Moon, see constants.py for the sidereal
        day of the Earth.

    Returns
    -------
    phases : jnp.ndarray
        The phases that rotate the sky, of the form exp(-i*m*phi(t)).
        Shape (N_times, 2*lmax+1).

    """
    dt = times - times[0]  # time difference from reference
    phi = 2 * jnp.pi * dt / sidereal_day  # rotation angle
    emms = jnp.arange(-lmax, lmax + 1)  # m values
    phases = jnp.exp(-1j * emms[None] * phi[:, None])
    return phases


@jax.jit
def run(sky_alm, beam_alm, phases):
    """
    Compute the convolution for a range of times in jax. The convolution is
    a dot product in l,m space. Axes are in the order: time, freq, ell, emm.

    Parameters
    ----------
    sky_alm : jnp.ndarray
        The sky alms. Shape (N_freqs, lmax+1, 2*lmax+1).
    beam_alm : jnp.ndarray
        The beam alms. Shape (N_freqs, lmax+1, 2*lmax+1). The beam should be
        normalized to have total power of unity.
    phases : jnp.ndarray
        The phases that rotate the sky, of the form exp(-i*m*phi(t)).
        Shape (N_times, 2*lmax+1).

    Returns
    -------
    res : jnp.ndarray
        The convolution. Shape (N_times, N_freqs).
    """
    s = sky_alm[None, :, :, :]  # add time axis
    p = phases[:, None, None, :]  # add freq and ell axes
    b = beam_alm.conjugate()[None, :, :, :]  # add time axis and conjugate
    res = jnp.sum(s * p * b, axes=(2, 3))  # dot product in l,m space
    return res
