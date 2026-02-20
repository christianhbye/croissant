import equinox as eqx
import jax
import jax.numpy as jnp
import s2fft
from astropy.coordinates import AltAz, EarthLocation
from astropy.time import Time as EarthTime
from lunarsky import LunarTopo, MoonLocation
from lunarsky import Time as LunarTime

from . import Beam, Sky, constants, rotations


def rot_alm_z(lmax, N_times=None, delta_t=None, times=None, world="moon"):
    """
    Compute the complex phases that rotate the sky for a range of times.
    The first time is the reference time and the phases are computed
    relative to this time. Can either provide `N_times` and `delta_t`
    for uniform times, or arbitrary time sampling with the `times`
    argument.

    Parameters
    ----------
    lmax : int
        The maximum ell value.
    N_times : int
        The number of times to compute the convolution at.
    delta_t : float
        The time difference between the times.
    times : array_like
        Explicit time array in seconds. Times are interpreted as
        absolute values and will be converted to differences relative
        to the first time. If provided, `N_times` and `delta_t` are
        ignored.
    world : str
        ``earth'' or ``moon''. Default is ``moon''.

    Returns
    -------
    phases : jnp.ndarray
        The phases that rotate the sky, of the form exp(-i*m*phi(t)).
        Shape (N_times, 2*lmax+1).

    """
    if times is not None:
        times = jnp.atleast_1d(jnp.asarray(times))
        if times.size == 0:
            raise ValueError("`times` must be a non-empty array-like object.")
        dt = times - times[0]
    else:
        if N_times is None or delta_t is None:
            raise ValueError(
                "Must specify `times` or both `N_times` and `delta_t`."
            )
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
    res = jnp.einsum("flm,tm,flm->tf", sky_alm.conjugate(), phases, beam_alm)
    return res


@jax.jit
def correct_ground_loss(vis, fgnd, Tgnd):
    """
    Correct for ground loss in the simulated visibilities.

    Parameters
    ----------
    vis : jax.Array
       The simulated visibilities that include ground loss.
    fgnd : jax.Array
       The assumed ground fraction to use for the correction.
    Tgnd : jax.Array
       The assumed ground temperature to use for the correction

    Returns
    -------
    corrected_vis : jax.Array
       The simulated visibilities with the ground loss corrected.

    """
    fsky = 1 - fgnd
    corrected_vis = vis - fgnd * Tgnd
    corrected_vis /= fsky
    return corrected_vis


class Simulator(eqx.Module):
    beam: Beam
    sky: Sky
    times_jd: jax.Array  # times in Julian day
    freqs: jax.Array  # in MHz
    lon: jax.Array  # longitude of in degrees
    lat: jax.Array  # latitude in degrees
    alt: jax.Array  # altitude in meters
    lmax: int = eqx.field(static=True)
    _L: int = eqx.field(static=True)
    eul_topo: tuple = eqx.field(static=True)  # euler angles for topo to eq
    dl_topo: jax.Array  # dl array for topocentric to eq frame
    Tgnd: jax.Array  # ground temperature in K
    world: str = eqx.field(static=True)  # "earth" or "moon"
    beam_eq_alm: jax.Array  # precomputed beam alm in equatorial coordinates
    sky_eq_alm: jax.Array  # precomputed sky alm in equatorial coordinates
    phases: jax.Array  # precomputed phases for sky rotation

    def __init__(
        self,
        beam,
        sky,
        times_jd,
        freqs,
        lon,
        lat,
        alt=0,
        world="moon",
        Tgnd=300.0,
    ):
        """
        Configure a simulation. This class holds all the relevant
        parameters for a simulation and provides necessary methods
        for coordinate transforms and spherical harmonics transforms.

        Note that beam and sky models must be consistent in terms of
        frequencies and lmax values.

        Parameters
        ----------
        beam : Beam
            The beam model to use for the simulation.
        sky : Sky
            The sky model to use for the simulation.
        times_jd : jax.Array
            The times in Julian day at which to simulate the
            observations.
        freqs : jax.Array
            The frequencies in MHz for the simulation. Must be
            consistent with the frequencies used for the beam and the
            sky models.
        lon : float
            The longitude of the observer in degrees.
        lat : float
            The latitude of the observer in degrees.
        alt : float
            The altitude of the observer in meters.
        world : {"moon", "earth"}
            Run the simulations on the moon or the Earth
        Tgnd : float
            The ground temperature in Kelvin. Only a constant
            temperature is supported for now.

        """
        if not jnp.all(beam.freqs == freqs) or not jnp.all(sky.freqs == freqs):
            raise ValueError(
                "Beam, sky and simulation frequencies do not match. Check "
                "beam.freqs, sky.freqs and the freqs argument passed to the "
                "Simulator."
            )
        self.freqs = freqs
        self.beam = beam
        self.sky = sky
        self.times_jd = times_jd

        if beam.lmax != sky.lmax:
            raise ValueError("Beam and sky alm have different lmax values.")
        self.lmax = beam.lmax
        self._L = self.lmax + 1

        self.Tgnd = jnp.array(Tgnd)

        self.lon = jnp.array(lon)
        self.lat = jnp.array(lat)
        self.alt = jnp.array(alt)

        if world == "earth":
            loc = EarthLocation.from_geodetic(lon, lat, height=alt)
            t0 = EarthTime(times_jd[0], format="jd")
            topo = AltAz(location=loc, obstime=t0)
            sim_frame = "fk5"
        elif world == "moon":
            loc = MoonLocation.from_geodetic(lon, lat, height=alt)
            t0 = LunarTime(times_jd[0], format="jd")
            topo = LunarTopo(location=loc, obstime=t0)
            sim_frame = "mcmf"

        eul_topo, dl_topo = rotations.generate_euler_dl(
            self.lmax, topo, sim_frame
        )
        self.eul_topo = tuple(float(angle) for angle in eul_topo)
        self.dl_topo = jnp.array(dl_topo)

        # precompute beam and sky alms in equatorial coordinates
        self.beam_eq_alm = self.compute_beam_eq()
        if self.world == "earth":
            self.sky_eq_alm = self.sky.compute_alm_eq()
        else:
            self.sky_eq_alm = self.sky.compute_alm_mcmf()

        # precompute the phases
        dt_sec = (self.times_jd - self.times_jd[0]) * 24 * 3600
        self.phases = rot_alm_z(self.lmax, times=dt_sec, world=self.world)

    @jax.jit
    def compute_beam_eq(self):
        """
        Compute the beam alm in equatorial coordinates. This uses the
        pre-computed Euler angles and dl array for the topocentric to
        equatorial transformation.

        Returns
        -------
        beam_eq_alm : jax.Array
            The beam alm in equatorial coordinates. Shape is
            (Nfreqs, lmax+1, 2*lmax+1).

        """
        beam_alm = self.beam.compute_alm()
        eq2topo = jax.vmap(
            s2fft.utils.rotation.rotate_flms,
            in_axes=(0, None, None, None),
        )
        beam_eq_alm = eq2topo(
            beam_alm, self._L, self.eul_topo, dl_array=self.dl_topo
        )
        return beam_eq_alm

    @jax.jit
    def compute_ground_contribution(self):
        """
        Compute the ground contribution to the visibility. This is
        simply the beam response below the horizon multiplied by the ground
        temperature.

        Returns
        -------
        vis_gnd : jax.Array
            The ground contribution to the visibility.

        """
        return self.beam.compute_fgnd() * self.Tgnd

    @jax.jit
    def sim(self):
        # this is the sky contribution, with implict ground loss
        vis_sky = convolve(self.beam_eq_alm, self.sky_eq_alm, self.phases)
        vis_sky /= self.beam.compute_norm()
        # add the ground contribution
        vis_gnd = self.compute_ground_contribution()
        vis = vis_sky + vis_gnd
        return vis.real
