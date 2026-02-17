"""
Tests for multi-pair visibility simulation.

Test suite for croissant.jax.multipair module, covering:
1. Regression against existing single-pair convolve
2. Hermitian symmetry V_pq = conj(V_qp)
3. Uniform sky (monopole only)
4. Point source
5. Auto-correlation imaginary part at noise level
6. Identical beams cross = auto
7. Azimuthally symmetric sky (m=0 only)
8. JAX gradient differentiability
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from croissant.constants import Y00
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


def make_complex_beam_alm(lmax, n_freqs=1, seed=None):
    """Generate a random complex-valued beam in alm format.
    
    Always returns shape (n_freqs, lmax+1, 2*lmax+1).
    """
    if seed is not None:
        _rng = np.random.default_rng(seed)
    else:
        _rng = rng
    # Generate a fully complex beam (no reality constraint)
    beam = s2fft.utils.signal_generator.generate_flm(_rng, lmax + 1, reality=False)
    # Always add frequency axis
    beam = jnp.repeat(beam[None, :, :], n_freqs, axis=0)
    return beam


def make_monopole_sky(lmax, n_freqs, T_sky):
    """Create a sky with only monopole component."""
    shape = (n_freqs, *alm.shape_from_lmax(lmax))
    sky = jnp.zeros(shape, dtype=jnp.complex128)
    l_idx, m_idx = alm.getidx(lmax, 0, 0)
    # T_sky can be array or scalar
    sky_monopole = jnp.asarray(T_sky) / Y00
    sky = sky.at[:, l_idx, m_idx].set(sky_monopole)
    return sky


class TestRegression:
    """Test 1: Single auto-correlation pair matches existing convolve."""

    @pytest.mark.parametrize("lmax", [16, 32])
    @pytest.mark.parametrize("n_freqs", [1, 10])
    @pytest.mark.parametrize("n_times", [1, 50])
    def test_single_pair_matches_convolve(self, lmax, n_freqs, n_times):
        """Pack single auto-correlation into multi-pair and verify match."""
        delta_t = 3600.0  # 1 hour
        world = "moon"

        # Create a real beam (auto-correlation)
        # Shape: (n_freqs, lmax+1, 2*lmax+1)
        beam = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=123)

        # Create sky
        T_sky = 1e4 * jnp.ones(n_freqs)
        sky = make_monopole_sky(lmax, n_freqs, T_sky)

        # Phases
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Normalization for single-pair (per frequency)
        norm_single = alm.total_power(beam[0], lmax)  # Use first freq for now

        # Single-pair convolve (existing function)
        vis_single = simulator.convolve(beam, sky, phases) / norm_single
        # Shape: (n_times, n_freqs)

        # Multi-pair with single pair
        beam_multi = beam[None, ...]  # Add pair axis: (1, n_freqs, lmax+1, 2*lmax+1)
        norm_multi = jnp.array([norm_single])

        vis_multi = multipair.compute_visibilities(beam_multi, sky, phases, norm_multi)
        # Shape: (n_times, 1, n_freqs)

        # Compare real parts
        vis_multi_squeezed = vis_multi[:, 0, :]
        rel_err = jnp.abs(vis_multi_squeezed.real - vis_single.real) / jnp.abs(
            vis_single.real
        )
        assert jnp.all(rel_err < 1e-12), f"Max relative error: {rel_err.max()}"

        # Imaginary part should be at noise level for real beam
        imag_rel = jnp.abs(vis_multi_squeezed.imag) / jnp.abs(vis_single.real)
        assert jnp.all(imag_rel < 1e-12), f"Max imag/real ratio: {imag_rel.max()}"


def conj_alm(beam_alm, lmax):
    """Compute alm of conj(f) from alm of f.

    If f has coefficients a_{l,m}, then conj(f) has coefficients
    (-1)^m * conj(a_{l,-m}).

    In the s2fft convention where column j corresponds to m = j - lmax:
    conj_alm[..., l, j] = (-1)^(j-lmax) * conj(alm[..., l, 2*lmax - j])
    """
    # Flip m axis: a[..., l, j] -> a[..., l, 2*lmax - j]
    flipped = jnp.flip(beam_alm, axis=-1)
    # Apply (-1)^m sign pattern
    m_values = jnp.arange(2 * lmax + 1) - lmax
    signs = jnp.where(jnp.abs(m_values) % 2 == 0, 1.0, -1.0)
    return signs * jnp.conj(flipped)


class TestHermitianSymmetry:
    """Test 2: V_pq equals conjugate of V_qp."""

    @pytest.mark.parametrize("lmax", [16, 32])
    def test_conjugate_symmetry(self, lmax):
        """Use complex beam B_pq, set B_qp = conj(B_pq) on the sphere,
        and verify V_pq = conj(V_qp) at all times and frequencies.

        If f has alm a_{l,m}, then conj(f) has alm (-1)^m * conj(a_{l,-m}).
        """
        n_freqs = 5
        n_times = 20
        delta_t = 3600.0
        world = "moon"

        # Generate a COMPLEX pair beam B_pq (no reality constraint)
        beam_pq = make_complex_beam_alm(lmax, n_freqs=n_freqs, seed=456)

        # Construct B_qp = conj(B_pq) on the sphere via alm relation
        beam_qp = conj_alm(beam_pq, lmax)

        # Stack: pair 0 is (p,q), pair 1 is (q,p)
        beam_alm = jnp.stack([beam_pq, beam_qp], axis=0)
        pairs = [(0, 1), (1, 0)]

        # Create a REAL sky (ensures the test is nontrivial)
        sky = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=789)

        # Phases
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Same normalization for both pairs
        norm = jnp.ones(2)

        # Compute visibilities
        vis = multipair.compute_visibilities(beam_alm, sky, phases, norm)

        vis_pq = vis[:, 0, :]  # V_pq
        vis_qp = vis[:, 1, :]  # V_qp

        # Key assertion: V_pq = conj(V_qp)
        diff = vis_pq - jnp.conj(vis_qp)
        rel_err = jnp.abs(diff) / (jnp.abs(vis_pq) + 1e-30)
        assert jnp.all(rel_err < 1e-12), f"Max relative error: {rel_err.max()}"

        # Both should have nonzero imaginary parts (complex beam)
        assert jnp.any(jnp.abs(vis_pq.imag) > 1e-10), (
            "V_pq imaginary part is zero -- test is degenerate"
        )


class TestUniformSky:
    """Test 3: Uniform sky (only a00 nonzero)."""

    @pytest.mark.parametrize("lmax", [16, 32])
    def test_monopole_sky_auto(self, lmax):
        """Auto-correlations should yield T0 (real)."""
        n_freqs = 3
        n_times = 10
        delta_t = 3600.0
        world = "moon"
        T0 = 1000.0

        # Create real beams for two antennas
        beam_0 = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=100)
        beam_1 = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=200)

        # Auto-correlation pair beams are the beams themselves
        beam_00 = beam_0
        beam_11 = beam_1

        # Stack
        beam_alm = jnp.stack([beam_00, beam_11], axis=0)
        pairs = [(0, 0), (1, 1)]

        # Normalizations: total_power for each auto beam
        norm_0 = alm.total_power(beam_0[0], lmax)  # Use first freq
        norm_1 = alm.total_power(beam_1[0], lmax)
        norm = jnp.array([norm_0, norm_1])

        # Uniform sky
        sky = make_monopole_sky(lmax, n_freqs, jnp.full(n_freqs, T0))

        # Phases
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Compute visibilities
        vis = multipair.compute_visibilities(beam_alm, sky, phases, norm)

        # Auto-correlations should give T0 (real)
        for i in range(2):
            vis_auto = vis[:, i, :]
            assert jnp.allclose(vis_auto.real, T0, rtol=1e-10)
            # Imaginary part should be noise
            imag_rel = jnp.abs(vis_auto.imag) / T0
            assert jnp.all(imag_rel < 1e-12)

    @pytest.mark.parametrize("lmax", [16, 32])
    def test_monopole_sky_cross(self, lmax):
        """Cross-correlations analytical verification with different beams.

        For a monopole-only sky with temperature T0, the convolution picks up
        only the a_{0,0} coefficient. The normalized visibility for pair (p,q) is:
        V_pq = T0 * cross_beam_a00 / sqrt(auto_0_a00 * auto_1_a00)
        """
        n_freqs = 1
        n_times = 5
        delta_t = 3600.0
        world = "moon"
        T0 = 500.0

        l0, m0 = alm.getidx(lmax, 0, 0)
        shape = alm.shape_from_lmax(lmax)

        # Create monopole-only auto beams with distinct a00 values
        auto_0 = jnp.zeros((n_freqs, *shape), dtype=jnp.complex128)
        auto_0 = auto_0.at[:, l0, m0].set(3.0)
        auto_1 = jnp.zeros((n_freqs, *shape), dtype=jnp.complex128)
        auto_1 = auto_1.at[:, l0, m0].set(5.0)

        # Create cross beam with a different (complex) a00 value
        cross_01 = jnp.zeros((n_freqs, *shape), dtype=jnp.complex128)
        cross_a00 = 2.0 + 1.0j
        cross_01 = cross_01.at[:, l0, m0].set(cross_a00)

        # Stack: auto(0,0), auto(1,1), cross(0,1)
        beam_alm = jnp.stack([auto_0, auto_1, cross_01], axis=0)
        pairs = [(0, 0), (1, 1), (0, 1)]

        # Normalizations: sqrt(tp_p * tp_q)
        tp_0 = alm.total_power(auto_0[0], lmax)  # 3.0 / Y00
        tp_1 = alm.total_power(auto_1[0], lmax)  # 5.0 / Y00
        norm = jnp.array([
            tp_0,                        # auto(0,0): tp_0
            tp_1,                        # auto(1,1): tp_1
            jnp.sqrt(tp_0 * tp_1),       # cross(0,1): sqrt(tp_0*tp_1)
        ])

        # Uniform sky
        sky = make_monopole_sky(lmax, n_freqs, T0)

        # Phases
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Compute visibilities
        vis = multipair.compute_visibilities(beam_alm, sky, phases, norm)

        # Auto-correlations should give T0 (real)
        vis_00 = vis[:, 0, 0]
        vis_11 = vis[:, 1, 0]
        assert jnp.allclose(vis_00.real, T0, rtol=1e-10)
        assert jnp.allclose(vis_11.real, T0, rtol=1e-10)

        # Cross-correlation analytical prediction:
        # V_cross = T0 * cross_a00 / sqrt(auto_0_a00 * auto_1_a00)
        expected_cross = T0 * cross_a00 / jnp.sqrt(3.0 * 5.0)
        vis_01 = vis[:, 2, 0]
        assert jnp.allclose(vis_01, expected_cross, rtol=1e-10), (
            f"Cross vis={vis_01[0]}, expected={expected_cross}"
        )


class TestPointSource:
    """Test 4: Point source visibility."""

    @pytest.mark.parametrize("lmax", [32, 64])
    def test_point_source(self, lmax):
        """Delta function source at known position.
        
        For a point source with amplitude S at the North pole, observed with
        a uniform beam (B=1 everywhere), the antenna temperature is:
        T_ant = S * B(source) / Omega = S * 1 / 4*pi
        
        where Omega = 4*pi is the beam solid angle for a uniform beam.
        """
        n_freqs = 1
        n_times = 1
        delta_t = 0.0
        world = "moon"

        # Create a uniform beam (B=1 everywhere) using only monopole
        shape = alm.shape_from_lmax(lmax)
        beam = jnp.zeros(shape, dtype=jnp.complex128)
        l_idx, m_idx = alm.getidx(lmax, 0, 0)
        beam = beam.at[l_idx, m_idx].set(1.0 / Y00)  # Uniform beam = 1

        # Create point source at North pole (theta=0)
        # The alm for a point source of amplitude S at theta=0:
        # T_{l,m} = S * Y_{l,m}^*(0, 0) = S * sqrt((2l+1)/(4*pi)) * delta_{m,0}
        sky = jnp.zeros((n_freqs, *shape), dtype=jnp.complex128)
        source_amplitude = 100.0
        for ell in range(lmax + 1):
            l_idx_ell, m_idx_ell = alm.getidx(lmax, ell, 0)
            ylm = jnp.sqrt((2 * ell + 1) / (4 * jnp.pi))
            sky = sky.at[0, l_idx_ell, m_idx_ell].set(source_amplitude * ylm)

        # Phases (no rotation for single time)
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Single pair
        beam_3d = beam[None, :, :]  # (1, lmax+1, 2*lmax+1)
        beam_alm = beam_3d[None, ...]  # (1, 1, lmax+1, 2*lmax+1)
        norm = jnp.array([alm.total_power(beam, lmax)])

        vis = multipair.compute_visibilities(beam_alm, sky, phases, norm)

        # Expected: S / (4*pi) for uniform beam with solid angle 4*pi
        # T_ant = S * beam_at_source / beam_solid_angle = S * 1 / 4*pi
        expected = source_amplitude / (4 * jnp.pi)

        # Beam is monopole-only so the result is exact (no lmax truncation)
        rel_err = jnp.abs(vis[0, 0, 0].real - expected) / expected
        assert rel_err < 1e-10, f"Relative error: {rel_err}, vis={vis[0,0,0]}, expected={expected}"


class TestAutoImaginaryPart:
    """Test 5: Auto-correlation imaginary part at noise level."""

    @pytest.mark.parametrize("lmax", [16, 32])
    @pytest.mark.parametrize("n_pairs", [1, 4])
    def test_auto_imag_noise(self, lmax, n_pairs):
        """For auto-correlations (p=q), imaginary part should be numerical noise."""
        n_freqs = 5
        n_times = 20
        delta_t = 3600.0
        world = "moon"

        # Create real beams (auto-correlations)
        beams = []
        for i in range(n_pairs):
            beam = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=1000 + i)
            beams.append(beam)

        beam_alm = jnp.stack(beams, axis=0)
        pairs = [(i, i) for i in range(n_pairs)]

        # Normalizations
        norm_list = []
        for i in range(n_pairs):
            norm_list.append(alm.total_power(beams[i][0], lmax))
        norm = jnp.array(norm_list)

        # Create a general real sky
        sky = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=999)

        # Phases
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Compute
        vis = multipair.compute_visibilities(beam_alm, sky, phases, norm)

        # Check imaginary part for all auto-correlations
        for i, (p, q) in enumerate(pairs):
            if p == q:
                vis_auto = vis[:, i, :]
                imag_rel = jnp.abs(vis_auto.imag) / (jnp.abs(vis_auto.real) + 1e-30)
                assert jnp.all(
                    imag_rel < 1e-12
                ), f"Pair {(p,q)}: max imag/real = {imag_rel.max()}"


class TestIdenticalBeams:
    """Test 6: If cross beam equals auto beam, visibilities should match."""

    @pytest.mark.parametrize("lmax", [16, 32])
    def test_identical_beams(self, lmax):
        """Cross visibility equals auto when beams are identical."""
        n_freqs = 3
        n_times = 15
        delta_t = 3600.0
        world = "moon"

        # Create one real beam
        beam = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=555)

        # Use same beam for auto and cross
        beam_alm = jnp.stack([beam, beam], axis=0)  # (2, n_freqs, lmax+1, 2*lmax+1)
        pairs = [(0, 0), (0, 1)]

        # Same normalization
        norm_val = alm.total_power(beam[0], lmax)
        norm = jnp.array([norm_val, norm_val])

        # General sky
        sky = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=666)

        # Phases
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Compute
        vis = multipair.compute_visibilities(beam_alm, sky, phases, norm)

        vis_auto = vis[:, 0, :]
        vis_cross = vis[:, 1, :]

        # Should match
        rel_err = jnp.abs(vis_cross - vis_auto) / (jnp.abs(vis_auto) + 1e-30)
        assert jnp.all(rel_err < 1e-12), f"Max relative error: {rel_err.max()}"


class TestAzimuthallySymmetricSky:
    """Test 7: Sky with only m=0 modes gives constant-in-time visibilities."""

    @pytest.mark.parametrize("lmax", [16, 32])
    def test_m0_only_constant(self, lmax):
        """m=0 only sky should give time-invariant visibilities."""
        n_freqs = 2
        n_times = 24
        delta_t = 3600.0
        world = "moon"

        # Create beams
        beam = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=777)

        # Create sky with only m=0 modes
        shape = (n_freqs, *alm.shape_from_lmax(lmax))
        sky = jnp.zeros(shape, dtype=jnp.complex128)
        for ell in range(lmax + 1):
            l_idx, m_idx = alm.getidx(lmax, ell, 0)
            # Random real value for each ell and freq
            val = rng.random((n_freqs,)) * 100
            sky = sky.at[:, l_idx, m_idx].set(val)

        # Single pair
        beam_alm = beam[None, ...]
        norm = jnp.array([alm.total_power(beam[0], lmax)])

        # Phases
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)

        # Compute
        vis = multipair.compute_visibilities(beam_alm, sky, phases, norm)
        # Shape: (n_times, 1, n_freqs)

        # Check that visibility is constant in time
        vis_squeezed = vis[:, 0, :]  # (n_times, n_freqs)
        for f in range(n_freqs):
            vis_f = vis_squeezed[:, f]
            # All times should have the same value
            std = jnp.std(vis_f)
            mean = jnp.abs(jnp.mean(vis_f))
            rel_std = std / (mean + 1e-30)
            assert rel_std < 1e-12, f"Freq {f}: relative std = {rel_std}"


class TestJAXGradient:
    """Test 8: JAX gradient through vmap."""

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


class TestMultiPairSimulator:
    """Tests for the MultiPairSimulator class."""

    def test_simulator_basic(self):
        """Basic simulator functionality."""
        lmax = 16
        n_freqs = 3
        n_times = 10
        n_pairs = 2
        delta_t = 3600.0
        world = "moon"

        # Create beams
        beams = []
        for i in range(n_pairs):
            beam = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=4000 + i)
            beams.append(beam)
        beam_alm = jnp.stack(beams, axis=0)

        pairs = [(0, 0), (1, 1)]
        norm = jnp.array([1.0, 1.0])

        sim = multipair.MultiPairSimulator(beam_alm, norm, pairs)

        assert sim.n_pairs == n_pairs
        assert sim.lmax == lmax
        assert sim.pairs == tuple(pairs)

        # Test get_pair_index
        assert sim.get_pair_index(0, 0) == 0
        assert sim.get_pair_index(1, 1) == 1
        with pytest.raises(ValueError):
            sim.get_pair_index(0, 1)

        # Test simulate
        sky = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=5000)
        phases = simulator.rot_alm_z(lmax, n_times, delta_t, world=world)
        vis = sim.simulate(sky, phases)

        assert vis.shape == (n_times, n_pairs, n_freqs)

    def test_pytree_roundtrip(self):
        """Test JAX pytree registration."""
        lmax = 8
        n_freqs = 2
        n_pairs = 2

        beams = []
        for i in range(n_pairs):
            beam = make_real_beam_alm(lmax, n_freqs=n_freqs, seed=6000 + i)
            beams.append(beam)
        beam_alm = jnp.stack(beams, axis=0)

        pairs = [(0, 0), (0, 1)]
        norm = jnp.array([1.0, 2.0])

        sim = multipair.MultiPairSimulator(beam_alm, norm, pairs)

        # Flatten and unflatten
        leaves, treedef = jax.tree_util.tree_flatten(sim)
        sim_restored = jax.tree_util.tree_unflatten(treedef, leaves)

        assert jnp.allclose(sim_restored.beam_alm, sim.beam_alm)
        assert jnp.allclose(sim_restored.norm, sim.norm)
        assert sim_restored.pairs == sim.pairs


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
