import jax.numpy as jnp
import numpy as np
import pytest
import s2fft
from croissant.constants import Y00
import croissant.jax as crojax

pytestmark = pytest.mark.parametrize("lmax", [8, 16, 64, 128])
rng = np.random.default_rng(seed=0)


@pytest.mark.parametrize("sampling", ["dh", "mw", "healpix"])
def test_alm2map(lmax, sampling):
    if sampling == "healpix":
        nside = lmax // 2
    else:
        nside = None
    # make constant map
    shape = crojax.alm.shape_from_lmax(lmax)
    alm = jnp.zeros(shape, dtype=jnp.complex128)
    a00 = 5
    alm = alm.at[crojax.alm.getidx(lmax, 0, 0)].set(a00)
    m = crojax.alm.alm2map(alm, sampling=sampling, nside=nside)
    assert jnp.allclose(m, a00 * Y00)

    # XXX compare to healpy with more complex alm


@pytest.mark.parametrize("sampling", ["dh", "mw", "healpix"])
def test_map2alm(lmax, sampling):
    if sampling == "healpix":
        nside = lmax // 2
    else:
        nside = None
    # make constant map
    shape = s2fft.sampling.s2_samples.f_shape(
        lmax + 1, sampling=sampling, nside=nside
    )
    const = 10  # constant map with value 10
    m = jnp.ones(shape, dtype=jnp.float64) * const
    alm = crojax.alm.map2alm(m, sampling=sampling, nside=nside)
    a00_idx = crojax.alm.getidx(lmax, 0, 0)
    a00 = alm[a00_idx]
    assert jnp.allclose(a00, 4 * jnp.pi * Y00 * const)

    # XXX compare to healpy with more complex map

    # XXX test that map2alm(alm2map(alm)) == alm


def test_total_power(lmax):
    # make a map that is 1 everywhere so total power is 4pi:
    shape = crojax.alm.shape_from_lmax(lmax)
    alm = jnp.zeros(shape, dtype=jnp.complex128)
    a00_idx = crojax.alm.getidx(lmax, 0, 0)
    alm = alm.at[a00_idx].set(1 / Y00)
    power = crojax.alm.compute_power(alm)
    assert jnp.isclose(power, 4 * jnp.pi)

    # m(theta) = cos(theta)**2
    alm = jnp.zeros(shape, dtype=jnp.complex128)
    alm = alm.at[a00_idx].set(1 / (3 * Y00))
    a20_idx = crojax.alm.getidx(lmax, 2, 0)
    alm = alm.at[a20_idx].set(4 * jnp.sqrt(jnp.pi / 5) * 1 / 3)
    power = crojax.alm.compute_power(alm)
    expected_power = 4 * jnp.pi / 3
    assert jnp.isclose(power, expected_power)


def test_getidx(lmax):
    # using ints
    ell = 3
    emm = 2
    ix = crojax.alm.getidx(ell, emm, lmax)
    ell_, emm_ = crojax.alm.getlm(lmax, ix)
    assert ell == ell_
    assert emm == emm_

    # using arrays
    ls = lmax // jnp.arange(1, 10)
    ms = jnp.arange(-lmax, lmax + 1)
    ixs = crojax.alm.getidx(ls, ms, lmax)
    ls_, ms_ = crojax.alm.getlm(lmax, ixs)
    assert jnp.allclose(ls, ls_)
    assert jnp.allclose(ms, ms_)


def test_getlm(lmax):
    alm = jnp.zeros(crojax.alm.shape_from_lmax(lmax), dtype=jnp.complex128)
    nrows, ncols = alm.shape
    # l correspond to rows, m correspond to columns
    ls = jnp.arange(nrows)
    ms = jnp.arange(ncols) - lmax
    for i in range(nrows):
        for j in range(ncols):
            ix = (i, j)
            ell, emm = crojax.alm.getlm(lmax, ix)
            assert ell == ls[i]
            assert emm == ms[j]


def test_lmax_from_shape(lmax):
    shape = s2fft.sampling.s2_samples.flm_shape(lmax + 1)
    _lmax = crojax.alm.lmax_from_shape(shape)
    assert _lmax == lmax


def test_is_real(lmax):
    alm = jnp.zeros(crojax.alm.shape_from_lmax(lmax), dtype=jnp.complex128)
    assert crojax.alm.is_real(alm)
    val = 1.0 + 2.0j
    ix_21 = crojax.alm.getidx(2, 1, lmax)  # get index for l=2, m=1
    alm = alm.at[ix_21].set(val)  # set l=2, m=1 mode but not m=-1 mode
    assert not crojax.alm.is_real(alm)
    ix_2m1 = crojax.alm.getidx(2, -1, lmax)  # get index for l=2, m=-1
    # set m=-1 mode to complex conjugate
    alm = alm.at[ix_2m1].set(-1 * val.conjugate())  
    assert crojax.alm.is_real(alm)

    # generate a real signal and check that alm.is_real is True
    alm = s2fft.utils.signal_generator.generate_flm(
        rng, lmax + 1, reality=True
    )
    assert crojax.alm.is_real(alm)
    # complex
    alm = s2fft.utils.signal_generator.generate_flm(
        rng, lmax + 1, reality=False
    )
    assert not crojax.alm.is_real(alm)


def test_reduce_lmax(lmax):
    signal1 = s2fft.utils.signal_generator.generate_flm(rng, lmax + 1)
    # reduce to same lmax, should do nothing
    signal2 = crojax.alm.reduce_lmax(signal1, lmax)
    assert crojax.alm.lmax_from_shape(signal2.shape) == lmax
    assert jnp.allclose(signal1, signal2)
    # reduce lmax of signal 2 to new_lmax
    new_lmax = 5
    signal2 = crojax.alm.reduce_lmax(signal1, lmax)
    assert crojax.alm.lmax_from_shape(signal2.shape) == new_lmax
    # confirm that signal 2 has the expected shape
    expected_shape = crojax.alm.shape_from_lmax(new_lmax)
    assert signal2.shape == expected_shape
    # check that the signals are the same for all ell, emm
    for ell in range(new_lmax + 1):
        for emm in range(-ell, ell + 1):
            # indexing differes since lmax differs
            ix1 = crojax.alm.getidx(ell, emm, lmax)
            ix2 = crojax.alm.getidx(ell, emm, new_lmax)
            assert signal1[ix1] == signal2[ix2]


def test_shape_from_lmax(lmax):
    shape = crojax.alm.shape_from_lmax(lmax)
    expected_shape = s2fft.sampling.s2_samples.flm_shape(lmax + 1)
    assert shape == expected_shape
