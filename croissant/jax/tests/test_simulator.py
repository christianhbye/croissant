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
        expected = jnp.sum(sky * beam.conj() * phase[None])
        assert jnp.isclose(ant_temp[i], expected)
