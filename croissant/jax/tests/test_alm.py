import jax.numpy as jnp
import numpy as np
import pytest
import s2fft
from croissant.constants import Y00
import croissant.jax as crojax

pytestmark = pytest.mark.parametrize("lmax", [8, 16, 64, 128])
rng = np.random.default_rng(seed=0)

def test_total_power(lmax):
    # make a map that is 1 everywhere so total power is 4pi:
    shape = crojax.alm.shape_from_lmax(lmax)
    alm = jnp.zeros(shape, dtype=jnp.complex128)
    a00_idx = crojax.alm.getidx(lmax, 0, 0)
    alm = alm.at[a00_idx].set(1 / Y00)
    power = crojax.alm.total_power(alm)
    assert jnp.isclose(power, 4 * jnp.pi)

    # m(theta) = cos(theta)**2
    alm = jnp.zeros(shape, dtype=jnp.complex128)
    alm = alm.at[a00_idx].set(1 / (3 * Y00))
    a20_idx = crojax.alm.getidx(lmax, 2, 0)
    alm = alm.at[a20_idx].set(4 * jnp.sqrt(jnp.pi / 5) * 1 / 3)
    power = crojax.alm.total_power(alm)
    expected_power = 4 * jnp.pi / 3
    assert jnp.isclose(power, expected_power)


def test_getidx(lmax):
    # using ints
    ell = 3
    emm = 2
    ix = crojax.alm.getidx(lmax, ell, emm)
    ell_, emm_ = crojax.alm.getlm(lmax, ix)
    assert ell == ell_
    assert emm == emm_

    # using arrays
    ls = lmax // jnp.arange(1, 10)
    ms = jnp.arange(-lmax, lmax + 1)
    ixs = crojax.alm.getidx(lmax, ls, ms)
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
    ix_21 = crojax.alm.getidx(lmax, 2, 1)  # get index for l=2, m=1
    alm = alm.at[ix_21].set(val)  # set l=2, m=1 mode but not m=-1 mode
    assert not crojax.alm.is_real(alm)
    ix_2m1 = crojax.alm.getidx(lmax, 2, -1)  # get index for l=2, m=-1
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
    # reduce lmax to new_lmax
    new_lmax = 5
    signal2 = crojax.alm.reduce_lmax(signal1, new_lmax)
    assert crojax.alm.lmax_from_shape(signal2.shape) == new_lmax
    # confirm that signal 2 has the expected shape
    expected_shape = crojax.alm.shape_from_lmax(new_lmax)
    assert signal2.shape == expected_shape
    # check that the signals are the same for all ell, emm
    for ell in range(new_lmax + 1):
        for emm in range(-ell, ell + 1):
            # indexing differes since lmax differs
            ix1 = crojax.alm.getidx(lmax, ell, emm)
            ix2 = crojax.alm.getidx(new_lmax, ell, emm)
            assert signal1[ix1] == signal2[ix2]


def test_shape_from_lmax(lmax):
    shape = crojax.alm.shape_from_lmax(lmax)
    expected_shape = s2fft.sampling.s2_samples.flm_shape(lmax + 1)
    assert shape == expected_shape
