import healpy as hp
import numpy as np
import pytest
from croissant.constants import Y00
from croissant.core.sphtransform import alm2map, map2alm

pytestmark = pytest.mark.parametrize("lmax", [16, 32, 64])
rng = np.random.default_rng(seed=1234)


def test_alm2map(lmax):
    # make constant map
    a00 = 3
    size = hp.Alm.getsize(lmax)
    alm = np.zeros(size, dtype=np.complex128)
    alm[0] = a00
    nside = 32
    hp_map = alm2map(alm, nside, lmax=lmax)
    npix = hp_map.shape[-1]
    assert nside == hp.npix2nside(npix)
    expected_map = np.full(npix, a00 * Y00)
    assert np.allclose(hp_map, expected_map)

    # make many maps
    frequencies = np.linspace(1, 50, 50)
    size = hp.Alm.getsize(lmax)
    alm = np.zeros((frequencies.size, size), dtype=np.complex128)
    alm[:, 0] = a00 * frequencies**2.5
    hp_map = alm2map(alm, nside, lmax=lmax)
    expected_maps = np.full((frequencies.size, npix), a00 * Y00)
    expected_maps *= frequencies.reshape(-1, 1) ** 2.5
    assert np.allclose(hp_map, expected_maps)

    # inverting map2alm
    size = hp.Alm.getsize(lmax)
    alm = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    # m = 0 modes are real
    alm[: lmax + 1] = alm[: lmax + 1].real
    nside = 64
    m = alm2map(alm, nside, lmax=lmax)
    alm_ = map2alm(m, lmax=lmax)
    assert np.allclose(alm, alm_)
    m_ = alm2map(alm_, nside, lmax=lmax)
    assert np.allclose(m, m_)

    # polarized case
    t_alm = alm.copy()
    e_alm = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    b_alm = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    alm = np.array([t_alm, e_alm, b_alm])
    # m = 0 modes are real
    alm[:, : lmax + 1] = alm[:, : lmax + 1].real
    m = alm2map(alm, nside, lmax=lmax, polarized=True)
    hp_map = hp.alm2map(alm, nside, lmax=lmax, pol=True)
    assert np.allclose(m, hp_map)


def test_map2alm(lmax):
    nside = 32
    npix = hp.nside2npix(nside)
    data = np.ones(npix)
    alm = map2alm(data, lmax=lmax)
    assert np.isclose(alm[0], Y00 * 4 * np.pi)  # a00 = Y00 * 4pi
    assert np.allclose(alm[1:], 0)  # rest of alms should be 0

    # multiple maps at once
    c = 10  # constant extra value
    d0 = np.arange(npix)
    d1 = d0 + c
    alm0 = map2alm(d0, lmax=lmax)
    alm1 = map2alm(d1, lmax=lmax)
    data = np.array([d0, d1])
    alm = map2alm(data, lmax=lmax)
    # check that computing jointly is the same as individually
    assert np.allclose(alm[0], alm0)
    assert np.allclose(alm[1], alm1)
    # d1 is d0 + constant value so they should only differ at a00
    assert np.allclose(alm0[1:], alm1[1:])
    assert np.isclose(alm0[0] + c * Y00 * 4 * np.pi, alm1[0])

    # polarized case
    nside = 64
    npix = hp.nside2npix(nside)
    iqu_map = rng.standard_normal((3, npix))
    alm = map2alm(iqu_map, lmax=lmax, polarized=True)
    hp_alm = hp.map2alm(iqu_map, lmax=lmax, pol=True, use_pixel_weights=True)
    assert np.allclose(alm, hp_alm)
