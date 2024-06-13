import jax.numpy as jnp
import numpy as np
import pytest
import s2fft
from croissant.constants import sidereal_day
from croissant.jax import simulator

pytestmark = pytest.mark.parametrize("lmax", [8, 32])
rng = np.random.default_rng(0)

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
    alm = s2fft.utils.signal_generator.generate_flm(rng, lmax+1)
    for i in range(N_times):
        phi = dphi[i].item()
        phase = phases[i]
        alm_rot = alm * phase[None, :]
        euler = (phi, 0, 0)  # rotation about z-axis
        expected = s2fft.utils.rotation.rotate_flms(alm, lmax+1, euler)
        assert jnp.allclose(alm_rot, expected)
