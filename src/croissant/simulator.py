import jax
import jax.numpy as jnp

from .. import constants


def rot_alm_z(lmax, N_times=None, delta_t=None, times=None, world="moon"):
    """
    Compute the complex phases that rotate the sky for a range of times. The
    first time is the reference time and the phases are computed relative to
    this time. Can either provide `N_times` and `delta_t` for uniform times,
    or arbitrary time sampling with the `times` argument.

    Parameters
    ----------
    lmax : int
        The maximum ell value.
    N_times : int
        The number of times to compute the convolution at.
    delta_t : float
        The time difference between the times.
    times : array_like
        Explicit time array in seconds. Times are interpreted as absolute values
        and will be converted to differences relative to the first time. If
        provided, `N_times` and `delta_t` are ignored.
    world : str
        ``earth'' or ``moon''. Default is ``moon''.

    Returns
    -------
    phases : jnp.ndarray
        The phases that rotate the sky, of the form exp(-i*m*phi(t)).
        Shape (N_times, 2*lmax+1).

    """
    if times is not None:
        # Ensure `times` is a JAX array and at least 1D for consistent handling
        times = jnp.atleast_1d(jnp.asarray(times))
        if times.size == 0:
            raise ValueError("`times` must be a non-empty array-like object.")
        dt = times - times[0]
    else:
        if N_times is None or delta_t is None:
            raise ValueError("Must specify `times` or both `N_times` and `delta_t`.")
        dt = jnp.arange(N_times) * delta_t
    day = constants.sidereal_day[world]
    phi = 2 * jnp.pi * dt / day  # rotation angle
    emms = jnp.arange(-lmax, lmax + 1)  # m values
    phases = jnp.exp(-1j * emms[None] * phi[:, None])
    return phases


@jax.jit
def convolve(beam_alm, sky_alm, phases):
    """
    Compute the convolution for a range of times in jax. The convolution is
    a dot product in l,m space. Axes are in the order: time, freq, ell, emm.

    Note that normalization is not included in this function. The usual
    normalization factor can be computed with croissant.jax.alm.total_power
    of the beam alm.

    Parameters
    ----------
    beam_alm : jnp.ndarray
        The beam alms. Shape (N_freqs, lmax+1, 2*lmax+1).
    sky_alm : jnp.ndarray
        The sky alms. Shape (N_freqs, lmax+1, 2*lmax+1).
    phases : jnp.ndarray
        The phases that rotate the sky, of the form exp(-i*m*phi(t)).
        Shape (N_times, 2*lmax+1). See the function ``rot_alm_z''.

    Returns
    -------
    res : jnp.ndarray
        The convolution. Shape (N_times, N_freqs).

    """
    s = sky_alm.conjugate()[None, :, :, :]  # add time axis and conjugate
    p = phases[:, None, None, :]  # add freq and ell axes
    b = beam_alm[None, :, :, :]  # add time axis
    res = jnp.sum(s * p * b, axis=(2, 3))  # dot product in l,m space
    return res
