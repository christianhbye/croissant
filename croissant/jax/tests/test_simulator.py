import jax.numpy as jnp
import numpy as np
import s2fft
import pytest
from croissant.constants import sidereal_day, Y00
from croissant.jax import alm, simulator

rng = np.random.default_rng(0)


@pytest.mark.parametrize("lmax", [8, 32])
@pytest.mark.parametrize("world", ["earth", "moon"])
@pytest.mark.parametrize("N_times", [1, 24])
def test_rot_alm_z(lmax, world, N_times):

    # do one sidereal day (2pi rotation), split into N_times
    delta_t = sidereal_day[world] / N_times
    phases = simulator.rot_alm_z(lmax, N_times, delta_t, world=world)
    # expected phases
    dphi = jnp.linspace(0, 2 * jnp.pi, N_times, endpoint=False)
    # the m-modes range from -lmax to lmax (inclusive)
    for m_index in range(phases.shape[1]):
        emm = m_index - lmax  # m = -lmax, -lmax+1, ..., lmax
        expected = jnp.exp(-1j * emm * dphi)
        assert jnp.allclose(phases[:, m_index], expected)

    # check that these phases really rotate the alm
    alm_arr = s2fft.utils.signal_generator.generate_flm(rng, lmax + 1)
    for i in range(N_times):
        phi = dphi[i].item()
        phase = phases[i]
        alm_rot = alm_arr * phase[None, :]
        euler = (phi, 0, 0)  # rotation about z-axis
        expected = s2fft.utils.rotation.rotate_flms(alm_arr, lmax + 1, euler)
        assert jnp.allclose(alm_rot, expected)


def test_convolve():
    lmax = 32
    freq = jnp.arange(50, 251)  # 50 to 250 MHz
    Ntimes = 100
    delta_t = 3600  # 1 hour cadence
    world = "earth"
    # check that we recover sky temperature for a monopole sky
    T_sky = 1e4 * (freq / 150) ** (-2.5)
    sky_monopole = T_sky / Y00  # monpole component
    shape = (freq.size, *alm.shape_from_lmax(lmax))
    sky = jnp.zeros(shape, dtype=jnp.complex128)
    l_indx, m_indx = alm.getidx(lmax, 0, 0)
    sky = sky.at[:, l_indx, m_indx].set(sky_monopole)
    # the beam is achromatic, but the details don't matter
    beam = s2fft.utils.signal_generator.generate_flm(
        rng, lmax + 1, reality=True
    )
    # normalization factor
    norm = alm.total_power(beam, lmax)
    # add frequency axis
    beam = jnp.repeat(beam[None, :], freq.size, axis=0)
    # get the phases that rotate the sky
    phases = simulator.rot_alm_z(lmax, Ntimes, delta_t, world=world)
    ant_temp = simulator.convolve(beam, sky, phases) / norm
    assert jnp.allclose(ant_temp, T_sky)

    # for a general sky, the telescope is sensitive to the multipole moments
    # that are in the beam. We consider a beam with 5 non-zero multipoles
    sky = s2fft.utils.signal_generator.generate_flm(
        rng, lmax + 1, reality=True
    )
    shape = alm.shape_from_lmax(lmax)
    beam = jnp.zeros(shape, dtype=jnp.complex128)
    beam = beam.at[l_indx, m_indx].set(1.0)  # monopole component
    # randomly, we choose 5 (l, m) pairs
    ells = rng.integers(1, lmax, size=5, endpoint=True)  # random l
    emms = [rng.integers(0, ell, endpoint=True) for ell in ells]  # random m
    # give the (l, m) mode a weight of 1 + 1j
    val = 1.0 + 1j
    for ell, emm in zip(ells, emms):
        l_indx, m_indx = alm.getidx(lmax, ell, emm)
        beam = beam.at[l_indx, m_indx].set(val)
        # we need to set -m to the conjugate of m since the beam is real
        neg_m_indx = alm.getidx(lmax, ell, -emm)[1]
        neg_val = (-1) ** emm * val.conjugate()
        beam = beam.at[l_indx, neg_m_indx].set(neg_val)
    # add frequency axis, but only one frequency
    ant_temp = simulator.convolve(beam[None], sky[None], phases)
    # the antenna temperature is the sum of the sky temperature * beam
    # over the multipoles that are in the beam
    for i in range(Ntimes):
        phase = phases[i]
        expected = jnp.sum(sky.conj() * beam * phase[None])
        assert jnp.isclose(ant_temp[i], expected)


@pytest.mark.parametrize("lmax", [8, 32])
@pytest.mark.parametrize("world", ["earth", "moon"])
def test_rot_alm_z_with_times_parameter(lmax, world):
    """Test rot_alm_z with explicit times parameter."""
    
    # Test 1: Uniform time array should match N_times/delta_t behavior
    N_times = 24
    delta_t = sidereal_day[world] / N_times
    
    # Using N_times and delta_t
    phases_uniform = simulator.rot_alm_z(lmax, N_times, delta_t, world=world)
    
    # Using times parameter with uniform spacing
    times_uniform = jnp.arange(N_times) * delta_t
    phases_times = simulator.rot_alm_z(lmax, times=times_uniform, world=world)
    
    assert jnp.allclose(phases_uniform, phases_times)
    
    # Test 2: Non-uniformly spaced times
    # Create a non-uniform time array (e.g., logarithmic spacing)
    times_nonuniform = jnp.logspace(0, 3, 10) * 60  # 1 to 1000 minutes in seconds
    phases_nonuniform = simulator.rot_alm_z(lmax, times=times_nonuniform, world=world)
    
    # Verify shape is correct
    assert phases_nonuniform.shape == (10, 2 * lmax + 1)
    
    # Verify that phases are computed correctly for non-uniform times
    dt = times_nonuniform - times_nonuniform[0]
    day = sidereal_day[world]
    expected_phi = 2 * jnp.pi * dt / day
    
    for i, phi in enumerate(expected_phi):
        for m_index in range(phases_nonuniform.shape[1]):
            emm = m_index - lmax
            expected_phase = jnp.exp(-1j * emm * phi)
            assert jnp.isclose(phases_nonuniform[i, m_index], expected_phase)
    
    # Test 3: Times are correctly converted to relative differences
    # Offset all times by a constant - should give same phases
    offset = 10000.0  # 10000 seconds offset
    times_offset = times_uniform + offset
    phases_offset = simulator.rot_alm_z(lmax, times=times_offset, world=world)
    
    assert jnp.allclose(phases_times, phases_offset)
    
    # Test 4: Single time (edge case)
    times_single = jnp.array([0.0])
    phases_single = simulator.rot_alm_z(lmax, times=times_single, world=world)
    
    assert phases_single.shape == (1, 2 * lmax + 1)
    # At t=0, all phases should be 1 (exp(-i*m*0) = 1)
    assert jnp.allclose(phases_single, jnp.ones_like(phases_single))
    
    # Test 5: Two times with specific spacing
    delta_specific = 3600.0  # 1 hour
    times_two = jnp.array([100.0, 100.0 + delta_specific])
    phases_two = simulator.rot_alm_z(lmax, times=times_two, world=world)
    
    assert phases_two.shape == (2, 2 * lmax + 1)
    # First time should have phase 1 (relative to itself)
    assert jnp.allclose(phases_two[0], jnp.ones_like(phases_two[0]))
    # Second time phases should match expected rotation
    day = sidereal_day[world]
    phi = 2 * jnp.pi * delta_specific / day
    for m_index in range(phases_two.shape[1]):
        emm = m_index - lmax
        expected = jnp.exp(-1j * emm * phi)
        assert jnp.isclose(phases_two[1, m_index], expected)
