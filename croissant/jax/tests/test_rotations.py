import healpy as hp
import numpy as np
import pytest
from s2fft.sampling.reindex import flm_2d_to_hp_fast
from s2fft.utils.signal_generator import generate_flm
from croissant.jax import rotations

rng = np.random.default_rng(seed=0)
pytestmark = pytest.mark.parametrize("lmax", [8, 16, 64, 128])


def test_rotate_alm(lmax):
    alm = generate_flm(rng, lmax + 1, reality=True)

    # galactic -> equatorial
    alm_rot = rotations.rotate_alm(alm, "galactic", "fk5")
    # need to convert to healpy ordering
    alm_hp = np.array(flm_2d_to_hp_fast(alm, lmax + 1))
    alm_rot_hp = np.array(flm_2d_to_hp_fast(alm_rot, lmax + 1))
    rot = hp.Rotator(coord=["G", "C"])
    assert np.allclose(alm_rot_hp, rot.rotate_alm(alm_hp))

    # equatorial -> galactic
    alm_rot = rotations.rotate_alm(alm, "fk5", "galactic")
    alm_rot_hp = np.array(flm_2d_to_hp_fast(alm_rot, lmax + 1))
    rot = hp.Rotator(coord=["C", "G"])
    assert np.allclose(alm_rot_hp, rot.rotate_alm(alm_hp))

    # galactic to mcmf
    # alm_rot = rotations.rotate_alm(alm, "galactic", "mcmf")
    # XXX this is not implemented in healpy
    # assert np.allclose(alm_rot, expected)  # XXX

    # topo to equatorial XXX
    # topo to mcmf XXX

    # check that inverse works
    alm_rot = rotations.rotate_alm(alm, "galactic", "fk5")
    assert np.allclose(alm, rotations.rotate_alm(alm_rot, "fk5", "galactic"))
    alm_rot = rotations.rotate_alm(alm, "galactic", "mcmf")
    assert np.allclose(alm, rotations.rotate_alm(alm_rot, "mcmf", "galactic"))
