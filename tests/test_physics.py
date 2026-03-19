"""
Physical behavior tests for croissant.

These tests validate physical invariants that must hold regardless of
internal implementation details. They serve as ground truth that should
never need changing unless there's a major physics-level breaking change.
"""

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pygdsm
import pytest
import s2fft
from astropy.time import Time as AstroTime
from lunarsky import Time as LunarTime

from croissant.beam import Beam
from croissant.constants import Y00, sidereal_day
from croissant.multipair import (
    compute_normalization,
    compute_visibilities,
    multi_convolve,
    pair_normalization,
)
from croissant.simulator import (
    Simulator,
    convolve,
    correct_ground_loss,
    rot_alm_z,
)
from croissant.sky import Sky
from croissant.utils import getidx, shape_from_lmax, total_power

# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------

_NSIDE = 8
_LMAX = 2 * _NSIDE  # 16
_NPIX = 12 * _NSIDE**2
_N_TIMES = 24


@pytest.fixture
def freqs():
    return jnp.linspace(50.0, 250.0, 5)


@pytest.fixture
def times_jd_earth():
    t0 = AstroTime("2022-01-01 00:00:00")
    return jnp.linspace(
        t0.jd,
        t0.jd + sidereal_day["earth"] / 86400.0,
        _N_TIMES,
        endpoint=False,
    )


@pytest.fixture
def times_jd_moon():
    t0 = LunarTime("2022-01-01 00:00:00")
    return jnp.linspace(
        t0.jd,
        t0.jd + sidereal_day["moon"] / 86400.0,
        _N_TIMES,
        endpoint=False,
    )


@pytest.fixture
def nside():
    return _NSIDE


@pytest.fixture
def lmax():
    return _LMAX


@pytest.fixture
def isotropic_beam(freqs):
    data = jnp.ones((len(freqs), _NPIX))
    return Beam(data, freqs, sampling="healpix", niter=0)


@pytest.fixture
def monopole_sky(freqs):
    tsky = 1e4 * (freqs / 150.0) ** (-2.5)
    data = tsky[:, None] * jnp.ones((_NPIX,))
    return Sky(data, freqs, coord="galactic", niter=0)


@pytest.fixture(scope="module")
def gsm_sky(freqs):
    gsm = pygdsm.GlobalSkyModel16(freq_unit="MHz")
    freqs_np = np.asarray(freqs)
    # Generate and downgrade one map at a time to avoid holding
    # all full-resolution maps in memory simultaneously.
    maps_low = np.empty((len(freqs_np), _NPIX))
    for i, f in enumerate(freqs_np):
        m = gsm.generate(f)
        maps_low[i] = hp.ud_grade(m, nside_out=_NSIDE)
    return Sky(
        jnp.array(maps_low),
        freqs,
        sampling="healpix",
        coord="galactic",
        niter=0,
    )


def _make_sim(beam, sky, times_jd, freqs, world, Tgnd=0.0):
    return Simulator(
        beam,
        sky,
        times_jd,
        freqs,
        0.0,
        0.0,
        world=world,
        Tgnd=Tgnd,
    )


# -----------------------------------------------------------------------
# A. Linearity and Superposition
# -----------------------------------------------------------------------


class TestLinearitySuperposition:
    def test_linearity_sky_scaling(
        self,
        freqs,
        times_jd_earth,
        isotropic_beam,
        gsm_sky,
    ):
        """Double the sky temperature -> visibility doubles."""
        sky1 = gsm_sky
        sky2_data = sky1.data * 2.0
        sky2 = Sky(
            sky2_data,
            freqs,
            sampling="healpix",
            coord="galactic",
            niter=0,
        )
        sim1 = _make_sim(
            isotropic_beam,
            sky1,
            times_jd_earth,
            freqs,
            "earth",
        )
        sim2 = _make_sim(
            isotropic_beam,
            sky2,
            times_jd_earth,
            freqs,
            "earth",
        )
        vis1 = sim1.sim()
        vis2 = sim2.sim()
        np.testing.assert_allclose(vis2, 2.0 * vis1, rtol=1e-10)

    def test_superposition(
        self,
        freqs,
        times_jd_earth,
        isotropic_beam,
        gsm_sky,
    ):
        """V(sky1 + sky2) = V(sky1) + V(sky2)."""
        # sky1 = GSM, sky2 = power-law monopole
        sky1 = gsm_sky
        tsky = 1e4 * (freqs / 150.0) ** (-2.5)
        sky2_data = tsky[:, None] * jnp.ones((_NPIX,))
        sky2 = Sky(
            sky2_data,
            freqs,
            sampling="healpix",
            coord="galactic",
            niter=0,
        )
        sky_sum = Sky(
            sky1.data + sky2_data,
            freqs,
            sampling="healpix",
            coord="galactic",
            niter=0,
        )

        sim1 = _make_sim(
            isotropic_beam,
            sky1,
            times_jd_earth,
            freqs,
            "earth",
        )
        sim2 = _make_sim(
            isotropic_beam,
            sky2,
            times_jd_earth,
            freqs,
            "earth",
        )
        sim_sum = _make_sim(
            isotropic_beam,
            sky_sum,
            times_jd_earth,
            freqs,
            "earth",
        )
        vis1 = sim1.sim()
        vis2 = sim2.sim()
        vis_sum = sim_sum.sim()
        np.testing.assert_allclose(vis_sum, vis1 + vis2, rtol=1e-10)

    def test_linearity_ground_temperature(
        self,
        freqs,
        times_jd_earth,
        isotropic_beam,
        gsm_sky,
    ):
        """Ground contribution scales linearly with Tgnd."""
        sim1 = _make_sim(
            isotropic_beam,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
            Tgnd=100.0,
        )
        sim2 = _make_sim(
            isotropic_beam,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
            Tgnd=200.0,
        )
        vis1 = sim1.sim()
        vis2 = sim2.sim()
        fgnd = isotropic_beam.compute_fgnd()
        diff = vis2 - vis1
        expected = jnp.broadcast_to(fgnd * 100.0, diff.shape)
        np.testing.assert_allclose(diff, expected, rtol=5e-3)


# -----------------------------------------------------------------------
# B. Time Domain Behavior
# -----------------------------------------------------------------------


class TestTimeDomain:
    @pytest.mark.parametrize("world", ["earth", "moon"])
    def test_sidereal_periodicity(
        self,
        freqs,
        times_jd_earth,
        times_jd_moon,
        isotropic_beam,
        gsm_sky,
        world,
    ):
        """Visibility repeats after one sidereal day."""
        times_1day = times_jd_earth if world == "earth" else times_jd_moon
        day_sec = sidereal_day[world]
        # second sidereal day
        times_2day = jnp.concatenate(
            [
                times_1day,
                times_1day + day_sec / 86400.0,
            ]
        )
        sim = _make_sim(
            isotropic_beam,
            gsm_sky,
            times_2day,
            freqs,
            world,
        )
        vis = sim.sim()
        n = len(times_1day)
        np.testing.assert_allclose(vis[:n], vis[n:], rtol=5e-3)

    @pytest.mark.parametrize("world", ["earth", "moon"])
    def test_sidereal_time_offset(
        self,
        freqs,
        times_jd_earth,
        times_jd_moon,
        gsm_sky,
        world,
    ):
        """
        For a structured beam and realistic sky:
        1) Visibility repeats after one sidereal day.
        2) A sim starting one sidereal day later gives identical
           output.
        3) A sim starting 1/4 sidereal day later gives output
           shifted by N_times/4 steps.
        """
        times_1day = times_jd_earth if world == "earth" else times_jd_moon
        day_sec = sidereal_day[world]
        day_jd = day_sec / 86400.0
        n = len(times_1day)

        # Structured beam: cos^2(theta) * (1 + 0.5*cos(phi))
        dummy = Beam(
            jnp.ones((1, _NPIX)),
            jnp.array([100.0]),
            sampling="healpix",
            niter=0,
        )
        theta = jnp.array(dummy.theta)
        phi = jnp.array(dummy.phi)
        pattern = jnp.cos(theta) ** 2 * (1.0 + 0.5 * jnp.cos(phi))
        pattern = jnp.where(
            theta <= jnp.pi / 2,
            pattern,
            0.0,
        )
        beam_data = jnp.broadcast_to(
            pattern[None, :],
            (len(freqs), _NPIX),
        )
        beam = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            niter=0,
        )

        # 1) Periodicity: simulate 2 sidereal days, check repeat
        times_2day = jnp.concatenate(
            [times_1day, times_1day + day_jd],
        )
        sim_2day = _make_sim(
            beam,
            gsm_sky,
            times_2day,
            freqs,
            world,
        )
        vis_2day = sim_2day.sim()
        np.testing.assert_allclose(
            vis_2day[:n],
            vis_2day[n:],
            rtol=5e-3,
        )

        # 2) Separate sim starting 1 sidereal day later
        sim_day1 = _make_sim(
            beam,
            gsm_sky,
            times_1day,
            freqs,
            world,
        )
        sim_day2 = _make_sim(
            beam,
            gsm_sky,
            times_1day + day_jd,
            freqs,
            world,
        )
        vis_day1 = sim_day1.sim()
        vis_day2 = sim_day2.sim()
        np.testing.assert_allclose(
            vis_day2,
            vis_day1,
            rtol=5e-3,
        )

        # 3) Sim starting 1/4 sidereal day later: output shifted
        quarter_jd = day_jd / 4.0
        shift = n // 4  # 6 steps for n=24
        sim_quarter = _make_sim(
            beam,
            gsm_sky,
            times_1day + quarter_jd,
            freqs,
            world,
        )
        vis_quarter = sim_quarter.sim()
        vis_day1_shifted = jnp.roll(vis_day1, -shift, axis=0)
        np.testing.assert_allclose(
            vis_quarter,
            vis_day1_shifted,
            rtol=5e-3,
        )

    def test_dipole_sinusoidal_variation(self, freqs, lmax):
        """
        A sky dipole (l=1, |m|=1 in equatorial coords) produces
        sinusoidal time variation with period = sidereal day.
        """
        world = "earth"
        n_times = 48
        day_sec = sidereal_day[world]
        dt = day_sec / n_times
        phases = rot_alm_z(
            lmax,
            N_times=n_times,
            delta_t=dt,
            world=world,
        )

        # Construct sky alm with only l=1, m=+1 and m=-1
        shape = (len(freqs), *shape_from_lmax(lmax))
        sky_alm = jnp.zeros(shape, dtype=jnp.complex128)
        li, mi_pos = getidx(lmax, 1, 1)
        li, mi_neg = getidx(lmax, 1, -1)
        sky_alm = sky_alm.at[:, li, mi_pos].set(1.0)
        sky_alm = sky_alm.at[:, li, mi_neg].set(-1.0)

        # Beam with l=0 and l=1 content to pick up the signal
        li0, mi0 = getidx(lmax, 0, 0)
        beam_alm = jnp.zeros(shape, dtype=jnp.complex128)
        beam_alm = beam_alm.at[:, li0, mi0].set(Y00)
        beam_alm = beam_alm.at[:, li, mi_pos].set(0.5)
        beam_alm = beam_alm.at[:, li, mi_neg].set(-0.5)

        vis = convolve(beam_alm, sky_alm, phases)
        # Extract a single frequency for analysis
        vis_f = vis[:, 0].real

        # FFT to find dominant frequency
        spectrum = jnp.abs(jnp.fft.rfft(vis_f))
        # DC component is index 0; the dominant oscillation should
        # be at index 1 (one cycle per sidereal day)
        peak = jnp.argmax(spectrum[1:]) + 1
        assert peak == 1, f"Expected peak at 1 cycle/day, got {peak}"
        # Verify it's a clean sinusoid: all power at the fundamental
        ac_spectrum = spectrum[1:]  # exclude DC
        power_fundamental = ac_spectrum[0]
        power_rest = jnp.sqrt(jnp.sum(ac_spectrum[1:] ** 2))
        assert power_rest / power_fundamental < 1e-10

    def test_axial_mode_constant_in_time(self, freqs, lmax):
        """Sky modes with m=0 give time-independent visibility."""
        world = "earth"
        n_times = 24
        day_sec = sidereal_day[world]
        dt = day_sec / n_times
        phases = rot_alm_z(
            lmax,
            N_times=n_times,
            delta_t=dt,
            world=world,
        )

        shape = (len(freqs), *shape_from_lmax(lmax))
        sky_alm = jnp.zeros(shape, dtype=jnp.complex128)
        # Set m=0 modes for several l values
        for ell in range(0, lmax + 1):
            li, mi = getidx(lmax, ell, 0)
            sky_alm = sky_alm.at[:, li, mi].set(
                1.0 / (ell + 1),
            )

        # Beam with all m-modes to verify beam m!=0 doesn't matter
        rng = np.random.default_rng(42)
        beam_alm = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        beam_alm = jnp.broadcast_to(
            beam_alm[None, :, :],
            shape,
        )

        vis = convolve(beam_alm, sky_alm, phases)
        # Imaginary part should be negligible (real sky, real beam)
        np.testing.assert_allclose(
            jnp.abs(vis.imag),
            0.0,
            atol=1e-10,
        )
        # All time steps should be identical
        vis_real = vis.real
        expected = jnp.broadcast_to(vis_real[0:1], vis_real.shape)
        np.testing.assert_allclose(
            vis_real,
            expected,
            rtol=1e-10,
        )

    def test_higher_m_oscillation_frequency(self, freqs, lmax):
        """An m=2 sky mode oscillates twice per sidereal day."""
        world = "earth"
        n_times = 48
        day_sec = sidereal_day[world]
        dt = day_sec / n_times
        phases = rot_alm_z(
            lmax,
            N_times=n_times,
            delta_t=dt,
            world=world,
        )

        shape = (len(freqs), *shape_from_lmax(lmax))
        # Sky with only l=2, m=+/-2
        sky_alm = jnp.zeros(shape, dtype=jnp.complex128)
        li, mi_pos = getidx(lmax, 2, 2)
        li, mi_neg = getidx(lmax, 2, -2)
        sky_alm = sky_alm.at[:, li, mi_pos].set(1.0)
        sky_alm = sky_alm.at[:, li, mi_neg].set(1.0)

        # Beam with l=2, m=+/-2 content to pick up the signal
        beam_alm = jnp.zeros(shape, dtype=jnp.complex128)
        li0, mi0 = getidx(lmax, 0, 0)
        beam_alm = beam_alm.at[:, li0, mi0].set(Y00)
        beam_alm = beam_alm.at[:, li, mi_pos].set(0.5)
        beam_alm = beam_alm.at[:, li, mi_neg].set(0.5)

        vis = convolve(beam_alm, sky_alm, phases)
        vis_f = vis[:, 0].real

        # FFT: dominant oscillation at 2 cycles per sidereal day
        spectrum = jnp.abs(jnp.fft.rfft(vis_f))
        peak = jnp.argmax(spectrum[1:]) + 1
        assert peak == 2, f"Expected peak at 2 cycles/day, got {peak}"
        # Verify it's a clean sinusoid at 2 cycles
        ac_spectrum = spectrum[1:]  # exclude DC
        power_fundamental = ac_spectrum[1]  # index 1 = 2 cycles
        power_rest = jnp.sqrt(
            jnp.sum(ac_spectrum[:1] ** 2) + jnp.sum(ac_spectrum[2:] ** 2),
        )
        assert power_rest / power_fundamental < 1e-10


# -----------------------------------------------------------------------
# C. Spectral Behavior
# -----------------------------------------------------------------------


class TestSpectralBehavior:
    def test_frequency_scaling_power_law(self, lmax):
        """
        For a power-law sky with achromatic beam, visibility ratio
        between frequencies follows the power law.
        """
        freqs = jnp.linspace(50.0, 250.0, 21)
        beta = -2.5
        T = 1e4 * (freqs / 150.0) ** beta

        shape = (len(freqs), *shape_from_lmax(lmax))

        # Sky with non-trivial structure, scaled by T(f)
        rng = np.random.default_rng(42)
        sky_unit = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        sky_alm = T[:, None, None] * sky_unit[None, :, :]

        # Achromatic beam (frequency-independent)
        beam_unit = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        beam_alm = jnp.broadcast_to(
            beam_unit[None, :, :],
            shape,
        )

        n_times = 24
        phases = rot_alm_z(
            lmax,
            N_times=n_times,
            delta_t=3600,
            world="earth",
        )
        vis = convolve(beam_alm, sky_alm, phases).real
        # ratio vis[t, f1] / vis[t, f2] = T(f1) / T(f2)
        f_ref = 10  # 150 MHz
        ratio = vis / vis[:, f_ref : f_ref + 1]
        expected = jnp.broadcast_to(
            (freqs / freqs[f_ref]) ** beta,
            ratio.shape,
        )
        np.testing.assert_allclose(ratio, expected, rtol=1e-10)


# -----------------------------------------------------------------------
# D. Beam Properties
# -----------------------------------------------------------------------


class TestBeamProperties:
    def test_isotropic_beam_recovers_sky_monopole(
        self,
        freqs,
        times_jd_earth,
        isotropic_beam,
        gsm_sky,
    ):
        """
        An isotropic beam recovers the sky's monopole temperature
        (time-averaged visibility = mean sky temperature * fsky).
        """
        sim = _make_sim(
            isotropic_beam,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
        )
        vis = sim.sim()
        # sim() divides by full-sphere beam integral but integrates
        # only over upper hemisphere, so vis = T_monopole * fsky
        mean_vis = vis.mean(axis=0)
        fsky = 1.0 - isotropic_beam.compute_fgnd()
        expected_monopole = gsm_sky.data.mean(axis=-1)
        np.testing.assert_allclose(
            mean_vis,
            expected_monopole * fsky,
            rtol=5e-3,
        )

    def test_isotropic_beam_full_sky_recovers_monopole(
        self,
        freqs,
        times_jd_earth,
    ):
        """
        An isotropic beam with no horizon cut on the MWSS grid
        recovers the sky monopole at machine precision.
        """
        lmax = _LMAX
        L = lmax + 1

        # Generate band-limited sky via random alm + inverse SHT
        rng = np.random.default_rng(42)
        sky_alm_unit = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                L,
                reality=True,
            ),
        )
        tsky = 1e4 * (freqs / 150.0) ** (-2.5)
        sky_alm = tsky[:, None, None] * sky_alm_unit[None, :, :]

        # Expected monopole from the (0,0) alm coefficient
        li0, mi0 = getidx(lmax, 0, 0)
        expected = jnp.real(sky_alm[:, li0, mi0]) * Y00

        # Inverse SHT to get sky data on MWSS grid
        sky_data = jnp.stack(
            [
                s2fft.inverse(
                    sky_alm[i],
                    L=L,
                    spin=0,
                    sampling="mwss",
                    method="jax",
                    reality=True,
                )
                for i in range(len(freqs))
            ]
        )
        sky = Sky(
            sky_data,
            freqs,
            sampling="mwss",
            coord="equatorial",
        )

        # Isotropic beam on MWSS grid, full sky
        ntheta, nphi = sky_data.shape[1], sky_data.shape[2]
        beam_data = jnp.ones((len(freqs), ntheta, nphi))
        horizon = jnp.ones((ntheta, 1), dtype=bool)
        beam = Beam(
            beam_data,
            freqs,
            sampling="mwss",
            horizon=horizon,
        )

        sim = _make_sim(
            beam,
            sky,
            times_jd_earth,
            freqs,
            "earth",
        )
        vis = sim.sim()
        mean_vis = vis.mean(axis=0)
        np.testing.assert_allclose(
            mean_vis,
            expected,
            rtol=1e-10,
        )

    def test_azimuthally_symmetric_beam_constant_visibility(
        self,
        freqs,
        lmax,
    ):
        """
        If the beam is azimuthally symmetric (only m=0 modes),
        visibility is constant in time.
        """
        world = "earth"
        n_times = 24
        day_sec = sidereal_day[world]
        dt = day_sec / n_times
        phases = rot_alm_z(
            lmax,
            N_times=n_times,
            delta_t=dt,
            world=world,
        )

        shape = (len(freqs), *shape_from_lmax(lmax))
        # Beam with only m=0 modes (azimuthally symmetric)
        beam_alm = jnp.zeros(shape, dtype=jnp.complex128)
        for ell in range(0, lmax + 1):
            li, mi = getidx(lmax, ell, 0)
            beam_alm = beam_alm.at[:, li, mi].set(
                1.0 / (ell + 1),
            )

        # General sky with all modes
        rng = np.random.default_rng(42)
        sky_alm = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        sky_alm = jnp.broadcast_to(
            sky_alm[None, :, :],
            shape,
        )

        vis = convolve(beam_alm, sky_alm, phases).real
        # Beam m=0 kills all sky m!=0 in the einsum, so only
        # sky m=0 modes contribute => time-independent.
        expected = jnp.broadcast_to(vis[0:1], vis.shape)
        np.testing.assert_allclose(vis, expected, rtol=1e-10)

    def test_beam_360_rotation_identity(
        self,
        freqs,
        times_jd_earth,
        gsm_sky,
    ):
        """Rotating a structured beam by 360° gives same visibility."""
        # Structured beam: cos^2(theta) * (1 + 0.5*cos(phi))
        # This has m=0 and m=±1 content, so rotation matters.
        dummy = Beam(
            jnp.ones((1, _NPIX)),
            jnp.array([100.0]),
            sampling="healpix",
            niter=0,
        )
        theta = jnp.array(dummy.theta)
        phi = jnp.array(dummy.phi)
        pattern = jnp.cos(theta) ** 2 * (1.0 + 0.5 * jnp.cos(phi))
        pattern = jnp.where(theta <= jnp.pi / 2, pattern, 0.0)
        beam_data = jnp.broadcast_to(
            pattern[None, :],
            (len(freqs), _NPIX),
        )
        beam0 = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            beam_az_rot=0.0,
            niter=0,
        )
        beam360 = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            beam_az_rot=360.0,
            niter=0,
        )
        sim0 = _make_sim(
            beam0,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
        )
        sim360 = _make_sim(
            beam360,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
        )
        vis0 = sim0.sim()
        vis360 = sim360.sim()
        np.testing.assert_allclose(vis360, vis0, rtol=1e-10)

    def test_beam_180_symmetric_rotation(
        self,
        freqs,
        times_jd_earth,
        gsm_sky,
    ):
        """
        A beam with 180° symmetry (only even-m modes) is invariant
        under 180° rotation.
        """
        # cos(2*phi) has only m=±2 modes (even m)
        dummy = Beam(
            jnp.ones((1, _NPIX)),
            jnp.array([100.0]),
            sampling="healpix",
            niter=0,
        )
        theta = jnp.array(dummy.theta)
        phi = jnp.array(dummy.phi)
        pattern = jnp.cos(theta) ** 2 * (1.0 + 0.5 * jnp.cos(2.0 * phi))
        pattern = jnp.where(theta <= jnp.pi / 2, pattern, 0.0)
        beam_data = jnp.broadcast_to(
            pattern[None, :],
            (len(freqs), _NPIX),
        )
        beam0 = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            beam_az_rot=0.0,
            niter=0,
        )
        beam180 = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            beam_az_rot=180.0,
            niter=0,
        )
        sim0 = _make_sim(
            beam0,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
        )
        sim180 = _make_sim(
            beam180,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
        )
        vis0 = sim0.sim()
        vis180 = sim180.sim()
        np.testing.assert_allclose(vis180, vis0, rtol=1e-10)

    def test_beam_rotation_changes_visibility(
        self,
        freqs,
        times_jd_earth,
        gsm_sky,
    ):
        """
        A non-symmetric beam rotated by a non-trivial angle
        produces different visibilities.
        """
        dummy = Beam(
            jnp.ones((1, _NPIX)),
            jnp.array([100.0]),
            sampling="healpix",
            niter=0,
        )
        theta = jnp.array(dummy.theta)
        phi = jnp.array(dummy.phi)
        # cos(phi) has m=±1: no rotational symmetry
        pattern = jnp.cos(theta) ** 2 * (1.0 + 0.5 * jnp.cos(phi))
        pattern = jnp.where(
            theta <= jnp.pi / 2,
            pattern,
            0.0,
        )
        beam_data = jnp.broadcast_to(
            pattern[None, :],
            (len(freqs), _NPIX),
        )
        beam0 = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            beam_az_rot=0.0,
            niter=0,
        )
        beam90 = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            beam_az_rot=90.0,
            niter=0,
        )
        sim0 = _make_sim(
            beam0,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
        )
        sim90 = _make_sim(
            beam90,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
        )
        vis0 = sim0.sim()
        vis90 = sim90.sim()
        assert not jnp.allclose(vis90, vis0, atol=1e-6)


# -----------------------------------------------------------------------
# E. Ground Loss
# -----------------------------------------------------------------------


class TestGroundLoss:
    def test_ground_loss_round_trip(
        self,
        freqs,
        times_jd_earth,
        isotropic_beam,
        gsm_sky,
    ):
        """
        correct_ground_loss applied to sims with different Tgnd
        recovers the same true sky temperature.
        """
        sim_gnd100 = _make_sim(
            isotropic_beam,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
            Tgnd=100.0,
        )
        sim_gnd300 = _make_sim(
            isotropic_beam,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
            Tgnd=300.0,
        )
        vis100 = sim_gnd100.sim()
        vis300 = sim_gnd300.sim()
        fgnd = isotropic_beam.compute_fgnd()
        recovered100 = correct_ground_loss(vis100, fgnd, 100.0)
        recovered300 = correct_ground_loss(vis300, fgnd, 300.0)
        # Both corrections should recover the same true sky temp
        np.testing.assert_allclose(
            recovered300,
            recovered100,
            rtol=5e-3,
        )

    def test_no_ground_above_horizon(
        self,
        freqs,
        times_jd_earth,
        gsm_sky,
    ):
        """
        When the beam horizon includes all sky (no ground), fgnd ~ 0
        and Tgnd has no effect.
        """
        beam_data = jnp.ones((len(freqs), _NPIX))
        # Set horizon to True everywhere => entire sphere is "sky"
        horizon = jnp.ones(_NPIX, dtype=bool)
        beam = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            horizon=horizon,
            niter=0,
        )
        fgnd = beam.compute_fgnd()
        np.testing.assert_allclose(fgnd, 0.0, atol=1e-12)

        sim0 = _make_sim(
            beam,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
            Tgnd=0.0,
        )
        sim300 = _make_sim(
            beam,
            gsm_sky,
            times_jd_earth,
            freqs,
            "earth",
            Tgnd=300.0,
        )
        vis0 = sim0.sim()
        vis300 = sim300.sim()
        np.testing.assert_allclose(vis300, vis0, rtol=1e-10)

        # correct_ground_loss is a no-op when fgnd=0
        recovered = correct_ground_loss(vis300, fgnd, 300.0)
        np.testing.assert_allclose(recovered, vis0, rtol=1e-10)


# -----------------------------------------------------------------------
# F. Multi-pair / Cross-correlation
# -----------------------------------------------------------------------


class TestMultipair:
    def test_auto_correlation_matches_convolve(self, freqs, lmax):
        """
        Multipair auto-correlation gives the same result as a
        direct convolve call, both before and after normalization.
        """
        world = "earth"
        n_times = 12
        day_sec = sidereal_day[world]
        dt = day_sec / n_times
        phases = rot_alm_z(
            lmax,
            N_times=n_times,
            delta_t=dt,
            world=world,
        )

        shape = (len(freqs), *shape_from_lmax(lmax))
        rng = np.random.default_rng(99)
        beam_single = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        beam_single = jnp.broadcast_to(
            beam_single[None, :, :],
            shape,
        ).copy()

        sky_alm = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        sky_alm = jnp.broadcast_to(
            sky_alm[None, :, :],
            shape,
        ).copy()

        # Multipair auto-correlation (unnormalized)
        auto_beam = beam_single[None, :, :, :]  # (1, F, L, M)
        vis_multi = multi_convolve(auto_beam, sky_alm, phases)
        # vis_multi shape: (1, N_times, N_freqs)

        # Direct convolve (unnormalized)
        vis_direct = convolve(beam_single, sky_alm, phases)
        # vis_direct shape: (N_times, N_freqs)

        np.testing.assert_allclose(
            vis_multi[0],
            vis_direct,
            rtol=1e-10,
        )

        # Normalized: multipair vs direct
        auto_powers = compute_normalization(auto_beam)
        pairs = [(0, 0)]
        norm = pair_normalization(auto_powers, pairs)
        vis_multi_norm = compute_visibilities(
            auto_beam,
            sky_alm,
            phases,
            norm,
        )
        # vis_multi_norm shape: (N_times, 1, N_freqs)

        tp = total_power(beam_single, lmax)
        vis_direct_norm = vis_direct / tp[None, :]

        np.testing.assert_allclose(
            vis_multi_norm[:, 0, :],
            vis_direct_norm,
            rtol=1e-10,
        )

    def test_identical_antennas_cross_equals_auto(
        self,
        freqs,
        lmax,
    ):
        """
        For identical antennas, cross-correlation equals
        auto-correlation.
        """
        world = "earth"
        n_times = 12
        day_sec = sidereal_day[world]
        dt = day_sec / n_times
        phases = rot_alm_z(
            lmax,
            N_times=n_times,
            delta_t=dt,
            world=world,
        )

        shape = (len(freqs), *shape_from_lmax(lmax))
        rng = np.random.default_rng(77)
        beam_single = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        beam_single = jnp.broadcast_to(
            beam_single[None, :, :],
            shape,
        ).copy()

        sky_alm = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        sky_alm = jnp.broadcast_to(
            sky_alm[None, :, :],
            shape,
        ).copy()

        # Two identical antennas
        beam_pair = jnp.stack(
            [beam_single, beam_single],
            axis=0,
        )  # (2, F, L, M)
        auto_powers = compute_normalization(beam_pair)
        # Auto-correlation pair (0,0) and cross pair (0,1)
        pairs_auto = [(0, 0)]
        pairs_cross = [(0, 1)]
        norm_auto = pair_normalization(
            auto_powers,
            pairs_auto,
        )
        norm_cross = pair_normalization(
            auto_powers,
            pairs_cross,
        )
        vis_auto = compute_visibilities(
            beam_pair,
            sky_alm,
            phases,
            norm_auto,
        )
        vis_cross = compute_visibilities(
            beam_pair,
            sky_alm,
            phases,
            norm_cross,
        )
        np.testing.assert_allclose(
            vis_cross,
            vis_auto,
            rtol=1e-10,
        )

    def test_dipole_azimuth_rotation_time_shift(self, freqs, lmax):
        """
        A beam rotated 90° in azimuth produces a visibility that
        is time-shifted by a quarter sidereal day, when compared
        to the visibility from the unrotated beam.
        """
        world = "earth"
        n_times = 24  # must be divisible by 4
        day_sec = sidereal_day[world]
        dt = day_sec / n_times
        phases = rot_alm_z(
            lmax,
            N_times=n_times,
            delta_t=dt,
            world=world,
        )

        shape = (len(freqs), *shape_from_lmax(lmax))
        rng = np.random.default_rng(55)
        beam_0 = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        beam_0 = jnp.broadcast_to(
            beam_0[None, :, :],
            shape,
        ).copy()

        # Rotate by 90°: multiply alm by exp(-i*m*pi/2)
        emms = jnp.arange(-lmax, lmax + 1)
        rot90 = jnp.exp(-1j * emms * jnp.pi / 2)
        beam_90 = beam_0 * rot90[None, None, :]

        sky_alm = jnp.array(
            s2fft.utils.signal_generator.generate_flm(
                rng,
                lmax + 1,
                reality=True,
            ),
        )
        sky_alm = jnp.broadcast_to(
            sky_alm[None, :, :],
            shape,
        ).copy()

        # Compute auto-visibilities for each beam orientation via multipair
        beams = jnp.stack([beam_0, beam_90], axis=0)
        vis = multi_convolve(beams, sky_alm, phases)
        # vis shape: (2, N_times, N_freqs); index 0/1 are not cross-correlations
        vis_beam_0 = vis[0]
        vis_beam_90 = vis[1]

        # vis_beam_90(t) should equal vis_beam_0(t + T_sid/4)
        # i.e., vis_beam_90 is vis_beam_0 shifted by n_times/4 = 6 steps
        shift = n_times // 4
        vis_beam_0_shifted = jnp.roll(vis_beam_0, -shift, axis=0)
        np.testing.assert_allclose(
            vis_beam_90,
            vis_beam_0_shifted,
            rtol=1e-10,
        )
