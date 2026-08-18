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

from croissant import rotations
from croissant.beam import Beam
from croissant.constants import Y00, sidereal_day
from croissant.multipair import (
    compute_normalization,
    compute_visibilities,
    multi_convolve,
    pair_normalization,
)
from croissant.polarization import (
    PairStokesBeam,
    PolarizedSky,
    iau_to_cosmo,
    polarized_convolve,
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


@pytest.fixture(scope="module")
def freqs():
    return jnp.linspace(50.0, 250.0, 5)


@pytest.fixture(scope="module")
def times_jd_earth():
    t0 = AstroTime("2022-01-01 00:00:00")
    return jnp.linspace(
        t0.jd,
        t0.jd + sidereal_day["earth"] / 86400.0,
        _N_TIMES,
        endpoint=False,
    )


@pytest.fixture(scope="module")
def times_jd_moon():
    t0 = LunarTime("2022-01-01 00:00:00")
    return jnp.linspace(
        t0.jd,
        t0.jd + sidereal_day["moon"] / 86400.0,
        _N_TIMES,
        endpoint=False,
    )


@pytest.fixture(scope="module")
def nside():
    return _NSIDE


@pytest.fixture(scope="module")
def lmax():
    return _LMAX


@pytest.fixture(scope="module")
def isotropic_beam(freqs):
    data = jnp.ones((len(freqs), _NPIX))
    return Beam(data, freqs, sampling="healpix", niter=0)


@pytest.fixture(scope="module")
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
            beam_rot=0.0,
            niter=0,
        )
        beam360 = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            beam_rot=360.0,
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
            beam_rot=0.0,
            niter=0,
        )
        beam180 = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            beam_rot=180.0,
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
            beam_rot=0.0,
            niter=0,
        )
        beam90 = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            beam_rot=90.0,
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
        # vis shape: (2, N_times, N_freqs); index 0/1 are not cross
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


# -----------------------------------------------------------------------
# G. Polarization
# -----------------------------------------------------------------------
#
# These tests operate at the ``polarized_convolve`` level on a full-sky
# MWSS grid, mirroring how the scalar sections use ``convolve``. MWSS is
# the sampling on which both spin +/-2 transforms actually run (other
# samplings derive the P+ sky block by conjugation), so the invariants
# below constrain the entire dual construction.

_POL_LMAX = 7
_POL_FREQS = np.array([50.0, 150.0])
# Per-frequency scale factors applied to synthesized maps so the
# frequency axis stays nontrivial without a second synthesis.
_POL_FREQ_SCALE = np.array([1.0, 0.7])
# ``PairStokesBeam`` defines a visibility as <v_a v_b*>, so the pair
# label constrains the response. An autocorrelation is |v_a|^2: real,
# with a non-negative intensity row. A cross pair is complex in general
# and may have no intensity response at all, as an orthogonal feed pair
# does. Tests label the pair to match the response they build.
_AUTO_PAIR = (0, 0)
_CROSS_PAIR = (0, 1)


def _pol_shape():
    """(ntheta, nphi) of the MWSS grid with band-limit ``_POL_LMAX``."""
    L = _POL_LMAX + 1
    return (
        s2fft.sampling.s2_samples.ntheta(L=L, sampling="mwss"),
        s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling="mwss"),
    )


def _pol_vis(beam_maps, sky_maps, phases, pair=_CROSS_PAIR):
    """Polarized visibilities of one pair on the full-sky MWSS grid."""
    horizon = np.ones(_pol_shape(), dtype=bool)
    beam = PairStokesBeam(
        beam_maps,
        _POL_FREQS,
        [pair],
        sampling="mwss",
        horizon=horizon,
    )
    sky = PolarizedSky(sky_maps, _POL_FREQS, sampling="mwss", coord="mepa")
    return polarized_convolve(beam.compute_alm(), sky.compute_alm(), phases)


def _pol_phases(n_times):
    """Phases for ``n_times`` uniform steps over one sidereal day."""
    return rot_alm_z(
        _POL_LMAX,
        N_times=n_times,
        delta_t=sidereal_day["earth"] / n_times,
        world="earth",
    )


def _assert_physical_stokes(maps, stokes_axis):
    """Assert IQUV maps lie in the physical Stokes cone."""
    maps = np.moveaxis(np.asarray(maps), stokes_axis, 0)
    intensity = maps[0]
    pol = np.sqrt((maps[1:] ** 2).sum(axis=0))
    assert intensity.min() > 0.0
    assert (pol <= intensity).all()


def _physical_stokes(maps, stokes_axis, pol_frac=0.8):
    """Project raw IQUV maps into the physical Stokes cone.

    A brightness distribution must satisfy ``I > 0`` and a polarization
    fraction ``sqrt(Q^2 + U^2 + V^2) / I <= 1`` at every pixel; white
    noise, or a zero-mean band-limited field, satisfies neither. The
    intensity is lifted by a constant offset and the polarized
    components are scaled by a single global factor -- a monopole shift
    and a uniform rescale, both of which leave a band-limited map
    band-limited and leave the angular structure of every component
    untouched.

    This constraint is on the *sky* alone. A pair response is a row of a
    Mueller matrix, not a Stokes vector: a differencing polarimeter
    legitimately has zero intensity response, so responses in this
    section are deliberately left unconstrained.
    """
    maps = np.moveaxis(np.asarray(maps, dtype=float), stokes_axis, 0)
    intensity, pol = maps[0], maps[1:]
    intensity = intensity - intensity.min() + np.ptp(intensity)
    pol_amp = np.sqrt((pol**2).sum(axis=0)).max()
    if pol_amp > 0.0:
        pol = pol * (pol_frac * intensity.min() / pol_amp)
    out = np.concatenate((intensity[None], pol), axis=0)
    _assert_physical_stokes(out, 0)
    return np.moveaxis(out, 0, stokes_axis)


def _random_stokes_sky(rng):
    """White-noise IQUV sky maps, shape (nfreq, 4, ntheta, nphi).

    Physical: positive intensity everywhere, polarization fraction
    below one everywhere.
    """
    maps = rng.normal(size=(len(_POL_FREQS), 4) + _pol_shape())
    return _physical_stokes(maps, stokes_axis=1)


def _random_pair_response(rng):
    """White-noise complex pair response, shape (1, nfreq, 4, ...).

    Complex, so it describes a cross pair rather than an
    autocorrelation.
    """
    shape = (1, len(_POL_FREQS), 4) + _pol_shape()
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _band_limited_scalar(rng):
    """A random real scalar map that is band-limited on the grid."""
    L = _POL_LMAX + 1
    flm = s2fft.utils.signal_generator.generate_flm(rng, L, reality=True)
    return np.asarray(
        s2fft.inverse(
            jnp.asarray(flm),
            L=L,
            spin=0,
            sampling="mwss",
            method="jax",
            reality=True,
        )
    )


def _half_band_limited_complex(rng):
    """A complex map band-limited to half the grid's band limit.

    Effective-length products are what a pair response is built from,
    and a product doubles the band limit. Synthesizing the factors at
    half the limit keeps the product exactly representable on the grid,
    so the quadrature that follows is exact rather than aliased.
    """
    L = _POL_LMAX + 1
    half = L // 2
    parts = []
    for _ in range(2):
        flm = s2fft.utils.signal_generator.generate_flm(
            rng, half, reality=True
        )
        padded = np.zeros((L, 2 * L - 1), dtype=complex)
        padded[:half, L - half : L + half - 1] = flm
        parts.append(
            np.asarray(
                s2fft.inverse(
                    jnp.asarray(padded),
                    L=L,
                    spin=0,
                    sampling="mwss",
                    method="jax",
                    reality=True,
                )
            )
        )
    return parts[0] + 1j * parts[1]


def _effective_length_pair_beam(rng, identical):
    """A pair beam built the way an instrument's really is.

    Intensity rows follow the multiport receive model: for ports a and
    b, M_I = H_theta_a H_theta_b* + H_phi_a H_phi_b*. Pairs are ordered
    (0,0), (1,1), (0,1) so the cross pair's autocorrelations are both
    present and the coherence is defined.
    """
    shape = _pol_shape()
    theta_a = _half_band_limited_complex(rng)
    phi_a = _half_band_limited_complex(rng)
    if identical:
        theta_b, phi_b = theta_a, phi_a
    else:
        theta_b = _half_band_limited_complex(rng)
        phi_b = _half_band_limited_complex(rng)

    def intensity(first, second):
        return (
            first[0] * second[0].conjugate() + first[1] * second[1].conjugate()
        )

    port_a = (theta_a, phi_a)
    port_b = (theta_b, phi_b)
    rows = [
        intensity(port_a, port_a),
        intensity(port_b, port_b),
        intensity(port_a, port_b),
    ]
    data = np.zeros((3, len(_POL_FREQS), 4) + shape, dtype=np.complex128)
    for index, row in enumerate(rows):
        data[index, :, 0] = _POL_FREQ_SCALE[:, None, None] * row[None]
    return PairStokesBeam(
        data,
        _POL_FREQS,
        [(0, 0), (1, 1), (0, 1)],
        sampling="mwss",
        horizon=np.ones(shape, dtype=bool),
    )


def _band_limited_positive(rng):
    """A band-limited scalar map that is positive everywhere.

    The offset is a monopole, so the map stays band-limited.
    """
    scalar = _band_limited_scalar(rng)
    return scalar - scalar.min() + np.ptp(scalar)


def _band_limited_qu_pair(rng):
    """A random real (Q, U) pair band-limited as a spin-2 field.

    Q + iU is synthesized from spin -2 coefficients, so both spin
    combinations Q -/+ iU are exactly band-limited and the discrete
    MWSS analysis of either one is exact.
    """
    L = _POL_LMAX + 1
    flm = s2fft.utils.signal_generator.generate_flm(
        rng, L, spin=2, reality=False
    )
    f_minus = np.asarray(
        s2fft.inverse(
            jnp.asarray(flm),
            L=L,
            spin=-2,
            sampling="mwss",
            method="jax",
        )
    )
    return f_minus.real, f_minus.imag


class TestPolarization:
    def test_polarized_linearity_superposition(self):
        """V(2*sky1 + sky2) = 2*V(sky1) + V(sky2), all Stokes at once.

        The physical Stokes states form a convex cone, so the
        superposed sky is itself a sky that could exist.
        """
        rng = np.random.default_rng(201)
        sky1 = _random_stokes_sky(rng)
        sky2 = _random_stokes_sky(rng)
        _assert_physical_stokes(2.0 * sky1 + sky2, stokes_axis=1)
        beam = _random_pair_response(rng)
        phases = _pol_phases(8)
        vis1 = _pol_vis(beam, sky1, phases)
        vis2 = _pol_vis(beam, sky2, phases)
        vis_comb = _pol_vis(beam, 2.0 * sky1 + sky2, phases)
        scale = np.abs(vis_comb).max()
        np.testing.assert_allclose(
            vis_comb,
            2.0 * vis1 + vis2,
            rtol=1e-10,
            atol=1e-10 * scale,
        )

    def test_unpolarized_sky_reduces_to_scalar_pipeline(self):
        """An (I, 0, 0, 0) sky through an intensity-only pair response
        reproduces the scalar pipeline over a sidereal day.

        Band-limited maps, because the two pipelines may pick different
        (equally valid) discrete representatives of an aliased map; the
        physical statement is about resolved fields.

        The scalar ``Beam`` is an autocorrelation power pattern, so this
        is the one test in the section on an auto pair, and its
        intensity response is non-negative accordingly.
        """
        rng = np.random.default_rng(202)
        shape = _pol_shape()
        nfreq = len(_POL_FREQS)
        intensity = (
            _POL_FREQ_SCALE[:, None, None] * _band_limited_positive(rng)[None]
        )
        response = (
            _POL_FREQ_SCALE[::-1, None, None]
            * _band_limited_positive(rng)[None]
        )
        horizon = np.ones(shape, dtype=bool)

        sky_maps = np.zeros((nfreq, 4) + shape)
        sky_maps[:, 0] = intensity
        _assert_physical_stokes(sky_maps, stokes_axis=1)
        beam_maps = np.zeros((1, nfreq, 4) + shape, dtype=np.complex128)
        beam_maps[0, :, 0] = response

        phases = _pol_phases(12)
        pol_vis = _pol_vis(beam_maps, sky_maps, phases, pair=_AUTO_PAIR)

        scalar_sky = Sky(
            jnp.asarray(intensity),
            jnp.asarray(_POL_FREQS),
            sampling="mwss",
            coord="mepa",
        )
        scalar_beam = Beam(
            jnp.asarray(response),
            jnp.asarray(_POL_FREQS),
            sampling="mwss",
            horizon=jnp.asarray(horizon),
        )
        scalar_vis = convolve(
            scalar_beam.compute_alm(),
            scalar_sky.compute_alm(),
            phases,
        )
        scale = np.abs(scalar_vis).max()
        np.testing.assert_allclose(
            pol_vis[:, 0, :],
            scalar_vis,
            rtol=1e-10,
            atol=1e-10 * scale,
        )
        # An auto pair of positive maps measures a real, positive power.
        scalar_vis = np.asarray(scalar_vis)
        np.testing.assert_allclose(scalar_vis.imag, 0.0, atol=1e-10 * scale)
        assert (scalar_vis.real > 0).all()
        # Non-vacuity: the two pipelines are matched on a signal that
        # actually turns with the sky, not just on a shared monopole.
        assert np.abs(scalar_vis - scalar_vis[0]).max() > 1e-3 * scale

    def test_stokes_convention_invariance(self):
        """Expressing sky and response in COSMO instead of IAU leaves
        the visibilities unchanged; mislabeling one side does not."""
        rng = np.random.default_rng(203)
        sky_maps = _random_stokes_sky(rng)
        beam_maps = _random_pair_response(rng)
        phases = _pol_phases(6)
        vis_iau = _pol_vis(beam_maps, sky_maps, phases)

        sky_cosmo = np.asarray(iau_to_cosmo(sky_maps, stokes_axis=1))
        beam_cosmo = np.asarray(iau_to_cosmo(beam_maps, stokes_axis=2))
        horizon = np.ones(_pol_shape(), dtype=bool)
        beam = PairStokesBeam(
            beam_cosmo,
            _POL_FREQS,
            [_CROSS_PAIR],
            sampling="mwss",
            convention="COSMO",
            horizon=horizon,
        )
        sky = PolarizedSky(
            sky_cosmo,
            _POL_FREQS,
            sampling="mwss",
            coord="mepa",
            convention="COSMO",
        )
        vis_cosmo = polarized_convolve(
            beam.compute_alm(), sky.compute_alm(), phases
        )
        scale = np.abs(vis_iau).max()
        np.testing.assert_allclose(
            vis_cosmo,
            vis_iau,
            rtol=1e-12,
            atol=1e-12 * scale,
        )

        # Non-vacuity: the U sign flip matters. COSMO-valued sky data
        # mislabeled as IAU produces different visibilities.
        vis_mislabeled = _pol_vis(beam_maps, sky_cosmo, phases)
        assert np.abs(vis_mislabeled - vis_iau).max() > 1e-6 * scale

    def test_polarized_sidereal_periodicity(self):
        """Full-Stokes visibilities repeat after one sidereal day."""
        rng = np.random.default_rng(204)
        sky_maps = _random_stokes_sky(rng)
        beam_maps = _random_pair_response(rng)
        n = 12
        phases = rot_alm_z(
            _POL_LMAX,
            N_times=2 * n,
            delta_t=sidereal_day["earth"] / n,
            world="earth",
        )
        vis = _pol_vis(beam_maps, sky_maps, phases)
        scale = np.abs(vis).max()
        np.testing.assert_allclose(
            vis[:n],
            vis[n:],
            rtol=1e-10,
            atol=1e-10 * scale,
        )
        # Non-vacuity: the visibility actually varies over the day.
        assert np.abs(vis[:n] - vis[0]).max() > 1e-3 * scale

    def test_z_rotated_polarized_sky_matches_time_shift(self):
        """Rigidly rotating the sky about the z axis equals advancing
        sidereal time, for every Stokes component at once.

        On the equiangular grid a rotation by a whole number of phi
        steps is an exact roll of the pixel axis, with Q and U carried
        unchanged since the theta-phi basis rotates with the sky. The
        phase machinery must therefore map the rolled sky at the
        reference time onto the unrolled sky at the matching time.
        """
        rng = np.random.default_rng(205)
        sky_maps = _random_stokes_sky(rng)
        beam_maps = _random_pair_response(rng)
        nphi = _pol_shape()[1]
        phases = _pol_phases(nphi)
        vis = _pol_vis(beam_maps, sky_maps, phases)
        shift = 5
        rolled = np.roll(sky_maps, -shift, axis=-1)
        vis_rolled = _pol_vis(beam_maps, rolled, phases)
        scale = np.abs(vis).max()
        np.testing.assert_allclose(
            vis_rolled[0],
            vis[shift],
            rtol=1e-10,
            atol=1e-10 * scale,
        )

    def test_joint_rotation_leaves_visibility_invariant(self):
        """Rotating sky and beam by the same rotation is unobservable.

        The Q/U blocks must transport as spin-2 under the shared
        Wigner-D action for the contraction to stay put.
        """
        rng = np.random.default_rng(206)
        sky_maps = _random_stokes_sky(rng)
        beam_maps = _random_pair_response(rng)
        horizon = np.ones(_pol_shape(), dtype=bool)
        sky = PolarizedSky(
            sky_maps,
            _POL_FREQS,
            sampling="mwss",
            coord="mepa",
        )
        beam = PairStokesBeam(
            beam_maps,
            _POL_FREQS,
            [_CROSS_PAIR],
            sampling="mwss",
            horizon=horizon,
        )
        sky_alm = sky.compute_alm()
        # Identity phases: a single time at the reference epoch.
        phases = rot_alm_z(_POL_LMAX, times=jnp.array([0.0]))
        vis0 = polarized_convolve(beam.compute_alm(), sky_alm, phases)

        rotation = (0.9, 0.6, -0.4)
        dl_array = s2fft.generate_rotate_dls(_POL_LMAX + 1, rotation[1])
        sky_alm_rot = rotations.rotate_alm(
            sky_alm, rotation, dl_array=dl_array
        )
        beam_alm_rot = beam.compute_alm_in_frame(rotation, dl_array)
        vis_rot = polarized_convolve(beam_alm_rot, sky_alm_rot, phases)
        scale = np.abs(vis0).max()
        np.testing.assert_allclose(
            vis_rot,
            vis0,
            rtol=1e-10,
            atol=1e-10 * scale,
        )
        # Non-vacuity: rotating only the sky does change the answer.
        vis_half = polarized_convolve(beam.compute_alm(), sky_alm_rot, phases)
        assert np.abs(vis_half - vis0).max() > 1e-6 * scale

    def test_stokes_selection_rules(self):
        """Blind responses see nothing: an intensity-only response
        rejects polarization, and I/V do not mix even though they share
        the spin-0 transform.

        Phrased as differences between physical skies, because a sky
        carrying Q/U or V but no intensity cannot exist. The
        convolution is linear in the sky, so the difference between two
        skies that agree except in one Stokes component is exactly that
        component's contribution to the visibility.
        """
        rng = np.random.default_rng(207)
        nfreq = len(_POL_FREQS)
        shape = _pol_shape()

        def beam_with(*names):
            maps = np.zeros((1, nfreq, 4) + shape, dtype=np.complex128)
            for name in names:
                maps[:, :, "IQUV".index(name)] = rng.normal(
                    size=(1, nfreq) + shape
                ) + 1j * rng.normal(size=(1, nfreq) + shape)
            return maps

        i_beam = beam_with("I")
        qu_beam = beam_with("Q", "U")
        v_beam = beam_with("V")

        # Physical skies differing in exactly one Stokes component.
        # Zeroing polarization or raising the intensity both keep the
        # sky inside the cone.
        sky = _random_stokes_sky(rng)
        no_linear = sky.copy()
        no_linear[:, 1:3] = 0.0
        no_circular = sky.copy()
        no_circular[:, 3] = 0.0
        brighter = sky.copy()
        brighter[:, 0] *= 2.0
        for variant in (no_linear, no_circular, brighter):
            _assert_physical_stokes(variant, stokes_axis=1)

        phases = _pol_phases(6)

        def response_to_change(beam_maps, variant):
            """What a response measures of the changed component."""
            return np.asarray(
                _pol_vis(beam_maps, sky, phases)
                - _pol_vis(beam_maps, variant, phases)
            )

        # Non-vacuity anchors: each response does see its own component.
        matched = [
            (i_beam, brighter),  # intensity response sees intensity
            (qu_beam, no_linear),  # linear-pol response sees Q, U
            (v_beam, no_circular),  # circular response sees V
        ]
        scale = min(np.abs(response_to_change(b, v)).max() for b, v in matched)
        assert scale > 0

        blind_cases = [
            (i_beam, no_linear),  # intensity response rejects Q, U
            (i_beam, no_circular),  # intensity response rejects V
            (qu_beam, brighter),  # linear-pol response rejects I
            (v_beam, brighter),  # circular response rejects I
        ]
        for beam_maps, variant in blind_cases:
            np.testing.assert_allclose(
                response_to_change(beam_maps, variant),
                0.0,
                atol=1e-12 * scale,
            )

    def test_polarization_angle_rotation_by_45_degrees(self):
        """A 45 degree polarization-angle rotation maps Q onto U.

        Rotating the polarization frame of sky and response together is
        unobservable; a response matched to the original sky is exactly
        orthogonal to the rotated sky at the reference time, since
        pointwise Q*Q' + U*U' = Q(-U) + UQ = 0.

        The sky carries the intensity that its partial polarization
        requires. The response is a pure polarization-differencing row
        with no intensity response -- what an orthogonal feed pair has,
        hence the cross-pair label -- which is what keeps that
        intensity out of the visibility.
        """
        rng = np.random.default_rng(208)
        q_raw, u_raw = _band_limited_qu_pair(rng)
        stokes = np.stack(
            (_band_limited_scalar(rng), q_raw, u_raw, np.zeros_like(q_raw))
        )
        stokes = _physical_stokes(stokes, stokes_axis=0)
        q_map, u_map = stokes[1], stokes[2]
        # Polarization angle rotated by 45 degrees: (Q, U) -> (-U, Q).
        # Intensity is unchanged by a rotation of the polarization
        # frame, so the rotated sky is physical too.
        rotated = stokes.copy()
        rotated[1] = -u_map
        rotated[2] = q_map
        _assert_physical_stokes(rotated, stokes_axis=0)
        # Responses see only the linear polarization.
        response = np.zeros_like(stokes)
        response[1], response[2] = q_map, u_map
        response_rot = np.zeros_like(stokes)
        response_rot[1], response_rot[2] = -u_map, q_map

        freq_scale = _POL_FREQ_SCALE[:, None, None, None]
        sky1 = freq_scale * stokes[None]
        sky2 = freq_scale * rotated[None]
        beam1 = (freq_scale * response[None])[None]
        beam2 = (freq_scale * response_rot[None])[None]

        phases = _pol_phases(8)
        vis11 = _pol_vis(beam1, sky1, phases)
        vis22 = _pol_vis(beam2, sky2, phases)
        scale = np.abs(vis11).max()
        assert scale > 0
        np.testing.assert_allclose(
            vis22,
            vis11,
            rtol=1e-10,
            atol=1e-10 * scale,
        )

        # Matched response at the reference time: full polarized power,
        # real and positive.
        vis_matched = np.asarray(vis11[0, 0])
        np.testing.assert_allclose(vis_matched.imag, 0.0, atol=1e-10 * scale)
        assert (vis_matched.real > 1e-6 * scale).all()
        # Orthogonal at the reference time after the 45 degree turn.
        vis_cross = _pol_vis(beam1, sky2, phases)
        np.testing.assert_allclose(vis_cross[0], 0.0, atol=1e-10 * scale)

    def test_real_responses_and_sky_give_real_visibilities(self):
        """A real Stokes response observing a real Stokes sky measures
        a real visibility at every time.

        Time evolution is a rigid rotation of a real sky, so the full
        Stokes integrand stays real; any imaginary residual would be a
        broken Hermitian pairing between the spin +/-2 blocks.
        """
        rng = np.random.default_rng(209)

        def real_stokes(rng):
            """A real band-limited IQUV block, shape (4, ntheta, nphi)."""
            i_map = _band_limited_scalar(rng)
            v_map = 0.1 * _band_limited_scalar(rng)
            q_map, u_map = _band_limited_qu_pair(rng)
            return np.stack((i_map, q_map, u_map, v_map))

        def add_freq_axis(stokes):
            return _POL_FREQ_SCALE[:, None, None, None] * stokes[None]

        # The sky must be a physical brightness distribution; the
        # response need only be real.
        sky_maps = add_freq_axis(_physical_stokes(real_stokes(rng), 0))
        beam_maps = add_freq_axis(real_stokes(rng))[None]  # add pair axis
        phases = _pol_phases(16)
        vis = np.asarray(_pol_vis(beam_maps, sky_maps, phases))
        scale = np.abs(vis).max()
        assert scale > 0
        np.testing.assert_allclose(vis.imag, 0.0, atol=1e-10 * scale)

    def test_isotropic_sky_returns_its_own_temperature(self):
        """An isotropic unpolarized sky at T reads back as T.

        This is Eq. 18 of the LuSEE antenna analysis, the definition
        that makes the normalized visibility a temperature:

            T_eff = int T(n) M_I(n) dOmega / int M_I(n) dOmega,

        which collapses to T for a sky with no structure, whatever the
        response pattern is. It holds at every time and frequency
        because an isotropic sky has no orientation for the phases to
        act on.
        """
        rng = np.random.default_rng(220)
        shape = _pol_shape()
        temperature = _POL_FREQ_SCALE * 300.0
        sky_maps = np.zeros((len(_POL_FREQS), 4) + shape)
        sky_maps[:, 0] = temperature[:, None, None]
        _assert_physical_stokes(sky_maps, stokes_axis=1)

        # An arbitrary non-negative intensity row: the invariant must
        # not depend on the shape of the response.
        beam_maps = np.zeros((1, len(_POL_FREQS), 4) + shape)
        beam_maps[0, :, 0] = (
            _POL_FREQ_SCALE[::-1, None, None]
            * _band_limited_positive(rng)[None]
        )
        horizon = np.ones(shape, dtype=bool)
        beam = PairStokesBeam(
            beam_maps,
            _POL_FREQS,
            [_AUTO_PAIR],
            sampling="mwss",
            horizon=horizon,
        )
        sky = PolarizedSky(sky_maps, _POL_FREQS, sampling="mwss", coord="mepa")
        vis = polarized_convolve(
            beam.compute_alm(),
            sky.compute_alm(),
            _pol_phases(8),
            normalization="auto-I",
        )
        expected = np.broadcast_to(
            temperature[None, None, :], np.asarray(vis).shape
        )
        np.testing.assert_allclose(np.asarray(vis).real, expected, rtol=1e-10)
        np.testing.assert_allclose(
            np.asarray(vis).imag, 0.0, atol=1e-10 * temperature.max()
        )

    def test_identical_antennas_make_the_two_normalizations_agree(self):
        """auto-I and multipair's sqrt(P_a * P_b) coincide exactly when
        the two antennas are identical.

        The pair response is then its own autocorrelation, so the
        pair's own intensity integral and the geometric mean of the two
        autos are the same number. This is the regime the scalar
        multipair section already exercises, and it is why the two
        conventions are not competing answers.
        """
        rng = np.random.default_rng(221)
        beam = _effective_length_pair_beam(rng, identical=True)
        norm = np.asarray(beam.compute_norm())
        auto_powers = jnp.stack([norm[0].real, norm[1].real])
        multipair_norm = np.asarray(pair_normalization(auto_powers, [(0, 1)]))
        np.testing.assert_allclose(
            np.abs(norm[2]), multipair_norm[0], rtol=1e-10
        )
        np.testing.assert_allclose(
            beam.response_diagnostics()["coherence"][2], 1.0, rtol=1e-10
        )

    def test_coherence_is_the_factor_between_the_two_normalizations(self):
        """For different antennas the two conventions differ, and the
        coherence is exactly the ratio between them.

        Cauchy-Schwarz bounds the pair's own intensity integral by the
        geometric mean of the autos, so the coherence lies in (0, 1]
        and converts one normalization into the other.
        """
        rng = np.random.default_rng(222)
        beam = _effective_length_pair_beam(rng, identical=False)
        norm = np.asarray(beam.compute_norm())
        auto_powers = jnp.stack([norm[0].real, norm[1].real])
        multipair_norm = np.asarray(pair_normalization(auto_powers, [(0, 1)]))[
            0
        ]
        coherence = np.asarray(beam.response_diagnostics()["coherence"][2])
        assert (coherence < 1.0).all()
        assert (coherence > 0.0).all()
        np.testing.assert_allclose(
            np.abs(norm[2]), coherence * multipair_norm, rtol=1e-10
        )
