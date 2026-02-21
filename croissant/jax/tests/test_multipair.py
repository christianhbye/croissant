"""
Tests for multi-pair visibility simulation.

Test suite for croissant.jax.multipair module, covering:
1. Regression against existing single-pair convolve
2. JAX gradient differentiability through vmap
3. Normalization helper functions
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from croissant.jax import alm, simulator, multipair

rng = np.random.default_rng(42)


def make_real_beam_alm(lmax, n_freqs=1, seed=None):
    """Generate a random real-valued beam in alm format.

    Always returns shape (n_freqs, lmax+1, 2*lmax+1).
    """
    if seed is not None:
        _rng = np.random.default_rng(seed)
    else:
        _rng = rng
    beam = s2fft.utils.signal_generator.generate_flm(_rng, lmax + 1, reality=True)
    # Always add frequency axis
    beam = jnp.repeat(beam[None, :, :], n_freqs, axis=0)
    return beam


class TestRegression:
    """Single auto-correlation pair matches existing convolve."""

    @pytest.mark.parametrize("lmax", [16, 32])
    @pytest.mark.parametrize("n_freqs", [1, 10])
    @pytest.mark.parametrize("n_times", [1, 50])
    def test_single_pair_matches_convolve(self, lmax, n_freqs, n_times):
        """Pack single auto-correlation into multi-pair and verify match."""
        delta_t = 3600.0  # 1 hour
        world = "moon"

        # Create a real beam (auto-correlation)
        beam = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=123)

        # Create sky
        sky = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=456)

        # Phases
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Normalization
        norm_single = alm.total_power(beam[0], lmax)

        # Single-pair convolve (existing function)
        vis_single = simulator.convolve(beam, sky, phases) / norm_single

        # Multi-pair with single pair
        beam_multi = beam[None, ...]  # Add pair axis
        norm_multi = jnp.array([norm_single])

        vis_multi = multipair.compute_visibilities(beam_multi, sky, phases, norm_multi)

        # Compare
        vis_multi_squeezed = vis_multi[:, 0, :]
        rel_err = jnp.abs(vis_multi_squeezed - vis_single) / (
            jnp.abs(vis_single) + 1e-30
        )
        assert jnp.all(rel_err < 1e-12), f"Max relative error: {rel_err.max()}"

        # Check that imaginary parts are at numerical-noise level
        max_imag_single = jnp.max(jnp.abs(jnp.imag(vis_single)))
        max_imag_multi = jnp.max(jnp.abs(jnp.imag(vis_multi_squeezed)))
        imag_tol = 1e-12
        assert max_imag_single < imag_tol, (
            f"Single-pair visibility has non-negligible imaginary part: {max_imag_single}"
        )
        assert max_imag_multi < imag_tol, (
            f"Multi-pair visibility has non-negligible imaginary part: {max_imag_multi}"
        )
class TestJAXGradient:
    """JAX gradient through vmap."""

    @pytest.mark.parametrize("lmax", [8, 16])
    def test_gradient_finite_nonzero(self, lmax):
        """Gradient of sum of visibility magnitudes squared is finite and nonzero."""
        n_freqs = 2
        n_times = 5
        n_pairs = 3
        delta_t = 3600.0
        world = "moon"

        # Create beams
        beams = []
        for i in range(n_pairs):
            beam = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=2000 + i)
            beams.append(beam)
        beam_alm = jnp.stack(beams, axis=0)

        # Nontrivial sky
        sky = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=3000)

        # Phases
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Norm
        norm = jnp.ones(n_pairs)

        def loss_fn(beam_alm):
            vis = multipair.compute_visibilities(beam_alm, sky, phases, norm)
            return jnp.sum(jnp.abs(vis) ** 2)

        # Compute gradient
        grad = jax.grad(loss_fn)(beam_alm)

        # Check finite
        assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"

        # Check nonzero
        assert jnp.any(grad != 0), "Gradient is all zeros"

        # Check shape preserved
        assert grad.shape == beam_alm.shape, "Gradient shape mismatch"


class TestComputeNormalization:
    """Tests for normalization helper functions."""

    def test_compute_normalization(self):
        """Test compute_normalization from auto beams."""
        lmax = 16
        n_antennas = 3
        n_freqs = 2

        # Create auto beams
        auto_beams = []
        for i in range(n_antennas):
            beam = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=7000 + i)
            auto_beams.append(beam)
        auto_beam_alm = jnp.stack(auto_beams, axis=0)

        powers = multipair.compute_normalization(auto_beam_alm)
        assert powers.shape == (n_antennas, n_freqs)

        # Verify against direct computation
        for i in range(n_antennas):
            for f in range(n_freqs):
                expected = alm.total_power(auto_beam_alm[i, f], lmax)
                assert jnp.isclose(powers[i, f], expected)

    def test_pair_normalization(self):
        """Test pair_normalization from antenna powers."""
        n_antennas = 4
        antenna_powers = jnp.array([1.0, 2.0, 3.0, 4.0])

        pairs = [(0, 0), (0, 1), (1, 2), (2, 3)]
        norm = multipair.pair_normalization(antenna_powers, pairs)

        expected = jnp.array(
            [
                jnp.sqrt(1.0 * 1.0),  # (0,0)
                jnp.sqrt(1.0 * 2.0),  # (0,1)
                jnp.sqrt(2.0 * 3.0),  # (1,2)
                jnp.sqrt(3.0 * 4.0),  # (2,3)
            ]
        )
        assert jnp.allclose(norm, expected)

    def test_pair_normalization_freq_dependent(self):
        """Test pair_normalization with frequency-dependent antenna powers."""
        n_antennas = 4
        n_freqs = 3

        # Shape (n_antennas, n_freqs); choose simple values for easy verification.
        antenna_powers = jnp.array(
            [
                [1.0, 2.0, 3.0],   # antenna 0
                [4.0, 5.0, 6.0],   # antenna 1
                [7.0, 8.0, 9.0],   # antenna 2
                [10.0, 11.0, 12.0] # antenna 3
            ]
        )

        pairs = [(0, 0), (0, 1), (1, 2), (2, 3)]
        norm = multipair.pair_normalization(antenna_powers, pairs)

        # Expect shape (n_pairs, n_freqs)
        assert norm.shape == (len(pairs), n_freqs)

        # Compute expected normalization per pair and frequency:
        # sqrt(power_i(f) * power_j(f)) for each (i, j) in pairs.
        expected = []
        for i, j in pairs:
            expected.append(jnp.sqrt(antenna_powers[i] * antenna_powers[j]))
        expected = jnp.stack(expected, axis=0)

        assert jnp.allclose(norm, expected)
