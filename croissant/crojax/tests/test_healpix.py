from copy import deepcopy
import pytest
from numpy.random import default_rng
import jax.numpy as jnp
import s2fft
from croissant.crojax import healpix as hp
from croissant.constants import sidereal_day_earth, sidereal_day_moon, Y00

pytestmark = pytest.mark.parametrize("lmax", [8, 16, 64, 128])
rng = default_rng(1913)
freqs = jnp.linspace(1, 50, 50)
nfreqs = freqs.size


def test_lmax_from_shape(lmax):
    s1, s2 = s2fft.sampling.s2_samples.flm_shape(lmax + 1)
    shape = (1, s1, s2)  # add frequency axis
    _lmax = hp.lmax_from_shape(shape)
    assert _lmax == lmax


def test_alm_indexing(lmax):
    # initialize all alms to 0
    alm = hp.Alm.zeros(lmax=lmax, frequencies=freqs)
    # set a00 = 1 for first half of frequencies
    alm[: nfreqs // 2, 0, 0] = 1.0
    # check __setitem__ acted correctly on alm.alm
    l_ix, m_ix = alm.getidx(0, 0)
    mask = jnp.zeros_like(alm.alm, dtype=bool)
    mask = mask.at[:nfreqs//2, l_ix, m_ix].set(True)
    # first half frequencies of a00, which should be 1
    assert jnp.allclose(alm.alm[mask], 1)
    # all other alm should be 0
    assert jnp.allclose(alm.alm[~mask], 0)
    # check that __getitem__ agrees:
    assert jnp.allclose(alm[: nfreqs // 2, 0, 0], 1)
    assert jnp.allclose(alm[nfreqs // 2 :, 0, 0], 0)
    # __getitem__ can't get multiple l-modes or m-modes at once...
    for ell in range(1, lmax + 1):
        for emm in range(-ell, ell + 1):
            assert jnp.allclose(alm[:, ell, emm], 0)

    # set everything back to 0
    alm = hp.Alm.zeros(lmax=lmax, frequencies=freqs)
    # negative indexing
    val = 3.0 + 2.3j
    alm[-1, 6, 3] = val
    assert alm[-1, 6, 3] == val
    l_ix, m_ix = alm.getidx(6, 3)
    assert alm[-1, 6, 3] == alm.alm[-1, l_ix, m_ix]

    # frequency index not specified
    with pytest.raises(TypeError):
        alm[3, 2] = 5
        alm[7, -1]


def test_zeros(lmax):
    alm = hp.Alm.zeros(lmax=lmax, frequencies=freqs)
    assert alm.lmax == lmax
    assert alm.frequencies is freqs
    s1, s2 = s2fft.sampling.s2_samples.flm_shape(lmax + 1)  
    assert alm.alm.shape == (nfreqs, s1, s2)
    assert jnp.allclose(alm.alm, 0)


def test_is_real(lmax):
    alm = hp.Alm.zeros(lmax=lmax)
    assert alm.is_real
    val = 1.0 + 2.0j
    alm[0, 2, 1] = val  # set l=2, m=1 mode but not m=-1 mode
    assert not alm.is_real
    alm[0, 2, -1] = -1 * val.conjugate()  # set m=-1 mode to complex conjugate
    assert alm.is_real

    # generate a real signal and check that alm.is_real is True
    alm = hp.Alm(
        s2fft.utils.signal_generator.generate_flm(rng, lmax, reality=True)
    )[None]  # add freq axis
    assert alm.is_real
    # complex
    alm = hp.Alm(
        s2fft.utils.signal_generator.generate_flm(rng, lmax, reality=False)
    )[None]
    assert not alm.is_real


def test_reduce_lmax(lmax):
    alm = hp.Alm(s2fft.utils.signal_generator.generate_flm(rng, lmax))
    old_alm = deepcopy(alm)
    # reduce to same lmax, should do nothing
    alm.reduce_lmax(lmax)
    assert alm.lmax == lmax
    assert jnp.allclose(alm.alm, old_alm.alm)
    # reduce to new lmax
    new_lmax = 5
    alm.reduce_lmax(new_lmax)
    assert alm.lmax == new_lmax
    assert alm.alm.shape == s2fft.sampling.s2_samples.flm_shape(new_lmax + 1)
    for ell in range(new_lmax + 1):
        for emm in range(-ell, ell + 1):
            assert alm[:, ell, emm] == old_alm[:, ell, emm]
    with pytest.raises(IndexError):
        alm[:, 7, 0]  # asking for ell > new_lmax should raise error
    # try to reduce to greater lmax
    new_lmax = 200
    with pytest.raises(ValueError):
        alm.reduce_lmax(new_lmax)


@pytest.mark.skip(reason="not implemented")
def test_getidx(lmax):
    alm = hp.Alm.zeros(lmax=lmax)
    ell = 3
    emm = 2
    bad_ell = 2 * lmax  # bigger than lmax
    bad_emm = 4  # bigger than ell
    with pytest.raises(IndexError):
        alm.getidx(bad_ell, emm)
        alm.getidx(ell, bad_emm)
        alm.getidx(-ell, emm)  # should fail since l < 0

    # try convert back and forth ell, emm <-> index
    ix = alm.getidx(ell, emm)
    ell_, emm_ = alm.getlm(i=ix)
    assert ell == ell_
    assert emm == emm_


@pytest.mark.skip(reason="not implemented")
def test_alm2map(lmax):
    # make constant map
    alm = hp.Alm.zeros(lmax=lmax)
    a00 = 5
    alm[0, 0, 0] = a00
    hp_map = alm.alm2map()  # use different samplings i guess ...
    assert jnp.allclose(hp_map, a00 * Y00)

    # make many maps
    frequencies = jnp.linspace(1, 50, 50)
    alm = hp.Alm.zeros(lmax=lmax, frequencies=frequencies)
    alm[:, 0, 0] = a00 * frequencies
    hp_map = alm.alm2map()  # XXX
    assert jnp.allclose(hp_map, a00 * Y00)

    # use subset of frequencies and compare to full set
    alm = hp.Alm.zeros(lmax=lmax, frequencies=frequencies)
    # some random map
    alm[:, 0, 0] = a00 * frequencies
    alm[:, 1, 1] = 2 * a00 * frequencies
    alm[::2, 8, 3] = -3 * a00 * frequencies[::2]
    hp_map = alm.alm2map()  # XXX
    freq_indices = [10, 20, 35]  # indices of frequencies to use
    freqs = frequencies[freq_indices]  # frequencies to use
    hp_map_select = alm.alm2map(frequencies=freqs)  # XXX
    assert jnp.allclose(hp_map_select, hp_map[freq_indices])

    # use some frequencies that are not in alm.frequencies
    with pytest.warns(UserWarning):
        alm.alm2map(frequencies=[0, 30, 100])  # XXX


@pytest.mark.skip(reason="not implemented")
def test_rot_alm_z(lmax):
    alm = hp.Alm.zeros(lmax=lmax)

    # rotate a single angle
    phi = jnp.pi / 2
    phase = alm.rot_alm_z(phi=phi)
    for ell in range(lmax + 1):
        for emm in range(ell + 1):
            ix = alm.getidx(ell, emm)
            assert jnp.isclose(phase[ix], jnp.exp(-1j * emm * phi))

    # rotate a set of angles
    phi = jnp.linspace(0, 2 * jnp.pi, num=361)  # 1 deg spacing
    phase = alm.rot_alm_z(phi=phi)
    for ell in range(lmax + 1):
        for emm in range(ell + 1):
            ix = alm.getidx(ell, emm)
            assert jnp.allclose(phase[:, ix], jnp.exp(-1j * emm * phi))

    # check that phi = 0 and phi = 2pi give the same answer
    assert jnp.allclose(phase[0], phase[-1])

    # rotate in time
    alm = hp.Alm.zeros(lmax=lmax)
    div = [1, 2, 4, 8]
    for d in div:
        dphi = 2 * jnp.pi / d
        # earth
        dt = sidereal_day_earth / d
        assert jnp.allclose(
            alm.rot_alm_z(times=dt, world="earth"), alm.rot_alm_z(phi=dphi)
        )
        # moon
        dt = sidereal_day_moon / d
        assert jnp.allclose(
            alm.rot_alm_z(times=dt, world="moon"), alm.rot_alm_z(phi=dphi)
        )
