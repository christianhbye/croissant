from functools import partial
import jax
import jax.numpy as jnp

from .. import constants
from ..utils import hp_npix2nside
from . import alm, rotations


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


def convolve(beam_alm, sky_alm, phases):
    """
    Compute the convolution for a range of times in jax. The convolution is
    a dot product in l,m space. Axes are in the order: time, freq, ell, emm.

    Parameters
    ----------
    beam_alm : jnp.ndarray
        The beam alms. Shape (N_freqs, lmax+1, 2*lmax+1). The beam should be
        normalized to have total power of unity.
    sky_alm : jnp.ndarray
        The sky alms. Shape (N_freqs, lmax+1, 2*lmax+1).
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


def _spht_wrapper(m, lmax, sampling):
    """
    Wrapper for the spherical harmonic transform. This function is called
    by ``run'' to compute the spherical harmonic transform of the beam and
    sky.

    Parameters
    ----------
    m : jnp.ndarray
        The maps on the sphere with a frequency axis.
    lmax : int
        The maximum ell value.
    sampling : str
        The sampling scheme. Supported sampling schemes are ``mw'', ``mwss'',
        ``dh'', ```gl'' and ``healpix''. See s2fft documentation for more
        information.

    Returns
    -------
    alm : jnp.ndarray
        The spherical harmonic coefficients.

    """
    if sampling == "healpix":
        npix = m.shape[-1]
        nside = hp_npix2nside(npix)
    else:
        nside = None
    # arguments for map2alm
    args = {
        "lmax": lmax,
        "spin": 0,
        "nside": nside,
        "sampling": sampling,
        "reality": True,
        "precomps": None,
        "spmd": True,
    }
    return jax.vmap(partial(alm.map2alm, **args))(m)


def run(
    beam,
    sky,
    lmax,
    beam_type="dh",
    beam_coords="topocentric",
    normalize_beam=True,
    sky_type="healpix",
    sky_coords="galactic",
    world="moon",
    location=None,
    times=None,
    nfreqs=1,
):
    """
    Run the simulation in jax. The beam and sky could each be maps on the
    sphere or spherical harmonic coefficients. This is specified by the
    ``beam_type'' and ``sky_type'' arguments; if maps on the sphere, this
    should be the sampling scheme used.

    The shapes of the arrays depend on if they are alms or maps on the sphere
    (in which caase the shape also depends on the sampling scheme). See
    the functions ``f_shape'' and ``flm_shape'' in
    ``s2fft.sampling.s2_samples''.

    The beam and sky could be specified at several frequencies (or any other
    batch dimension). This needs to be the axis 0 of the input arrays. In
    this case, the argument ``nfreqs'' must be set accordingly.

    Parameters
    ----------
    beam : jnp.ndarray
        The beam maps or alms.
    sky : jnp.ndarray
        The sky maps or alms.
    lmax : int
        The maximum ell value (inclusive).
    beam_type : str
        Must be ``alm'' or a sampling shceme. Supported sampling schemes are
        ``mw'', ``mwss'', ``dh'', ```gl'' and ``healpix''. Default is ``dh'',
        which is equiangular sampling. See s2fft documentation for more
        information.
    beam_coords : str
        The coordinate system of the beam. Default is ``topocentric''.
        Other options are ``equatorial'' (earth) and ``mcmf'' (moon).
    normalize_beam : bool
        Whether to normalize the beam to have total power of unity. Default is
        True.
    sky_type : str
        Must be ``alm'' or a sampling shceme. Supported sampling schemes are
        ``mw'', ``mwss'', ``dh'', ```gl'' and ``healpix''. Default is
        ``healpix''. See s2fft documentation for more information.
    sky_coords : str
        The coordinate system of the sky. Default is ``galactic''. Other
        options are ``equatorial'' (earth) and ``mcmf'' (moon).
    world : str
        ``earth'' or ``moon''. Default is ``moon''.
    location : astropy.coordinates.EarthLocation or lunrsky.MoonLocation
        The location of the observer. Required if beam_coords is
        ``topocentric''.
    times : astropy.time.Time or lunarsky.Time or list of these
        The times for which to compute the convolution. Required if
        beam_coords is ``topocentric''. See ``utils.time_array'' for a
        convenient way to generate evenly spaced times.
    nfreqs : int
        The number of frequencies. Default is 1.

    Returns
    -------
    res : jnp.ndarray
        The convolution. Shape (N_times, N_freqs).

    """
    # add frequency axis
    if nfreqs == 1:
        beam = beam[None]
        sky = sky[None]
    # beam spherical harmonic transform
    if beam_type == "alm":
        beam_alm = beam
    else:
        beam_alm = _spht_wrapper(beam, lmax, beam_type)

    # get the reference time
    try:
        t0 = times[0]  # times is a list
        ntimes = len(times)
    except IndexError:
        t0 = times  # times is a single time
        ntimes = 1
    except TypeError:
        ntimes = 0  # times is None

    # beam coordinate transformation if topocentric
    if beam_coords == "topocentric":
        args = {"loc": location, "time": t0, "dl_array": None}
        if world == "earth":
            func = rotations.topo2eq
        elif world == "moon":
            func = rotations.topo2mcmf
        else:
            raise ValueError("world must be 'earth' or 'moon'")
        beam_alm = jax.vmap(partial(func, **args))(beam_alm)

    # normalize beam
    if normalize_beam:
        norm = alm.total_power(beam_alm)
        beam_alm /= norm

    # sky spherical harmonic transform
    if sky_type == "alm":
        sky_alm = sky
    else:
        sky_alm = _spht_wrapper(sky, lmax, sky_type)

    # sky coordinate transformation if galactic
    if sky_coords == "galactic":
        sky_alm = jax.vmap(rotations.gal2eq)(sky_alm)

    # compute the phases that rotate the sky
    if ntimes < 2:
        phases = jnp.array([1.0])
    else:
        t_sec = jnp.array([t.to_value("unix") for t in times])
        if world == "earth":
            sidereal_day = constants.sidereal_day_earth
        elif world == "moon":
            sidereal_day = constants.sidereal_day_moon
        else:
            raise ValueError("world must be 'earth' or 'moon'")
        phases = rot_alm_z(lmax, t_sec, sidereal_day=sidereal_day)

    # compute the convolution
    res = convolve(beam_alm, sky_alm, phases)
    return res
