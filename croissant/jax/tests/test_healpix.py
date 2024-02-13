from copy import deepcopy
import pytest
from numpy.random import default_rng
import jax.numpy as jnp
import s2fft
from croissant.jax import healpix as hp
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
    mask = mask.at[: nfreqs // 2, l_ix, m_ix].set(True)
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


def test_getlm(lmax):
    alm = hp.Alm.zeros(lmax=lmax)
    nrows, ncols = alm.alm.shape[1:]
    # l correspond to rows, m correspond to columns
    ls = jnp.arange(nrows)
    ms = jnp.arange(ncols) - lmax
    for i in range(nrows):
        for j in range(ncols):
            ix = (i, j)
            ell, emm = alm.getlm(ix)
            assert ell == ls[i]
            assert emm == ms[j]


def test_getidx(lmax):
    # using ints
    ell = 3
    emm = 2
    ix = hp._getidx(ell, emm, lmax)
    ell_, emm_ = hp._getlm(ix, lmax)
    assert ell == ell_
    assert emm == emm_

    # using arrays
    ls = lmax // jnp.arange(1, 10)
    ms = jnp.arange(-lmax, lmax + 1)
    ixs = hp._getidx(ls, ms, lmax)
    ls_, ms_ = hp._getlm(ixs, lmax)
    assert jnp.allclose(ls, ls_)
    assert jnp.allclose(ms, ms_)

    # using ell > lmax should raise error in class method
    alm = hp.Alm.zeros(lmax=lmax)
    ell = 3
    emm = 2
    bad_ell = 2 * lmax  # bigger than lmax
    bad_emm = 4  # bigger than ell
    with pytest.raises(IndexError):
        alm.getidx(bad_ell, emm)
        alm.getidx(ell, bad_emm)
        alm.getidx(-ell, emm)  # should fail since l < 0

    # check that error is raised if array contains bad ell
    bad_ells = lmax + jnp.arange(-2, 2)
    with pytest.raises(IndexError):
        alm.getidx(bad_ells, emm)


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
    alm = s2fft.utils.signal_generator.generate_flm(
        rng, lmax + 1, reality=True
    )
    alm = alm[None]  # add frequency dimension
    assert hp._is_real(alm)
    assert hp.Alm(alm).is_real
    # complex
    alm = s2fft.utils.signal_generator.generate_flm(
        rng, lmax + 1, reality=False
    )
    alm = alm[None]  # add frequency dimension
    assert not hp._is_real(alm)
    assert not hp.Alm(alm).is_real


def test_reduce_lmax(lmax):
    sig = s2fft.utils.signal_generator.generate_flm(rng, lmax + 1)
    alm = hp.Alm(sig[None])
    old_alm = deepcopy(alm)
    # reduce to same lmax, should do nothing
    alm.reduce_lmax(lmax)
    assert alm.lmax == lmax
    assert jnp.allclose(alm.alm, old_alm.alm)
    # reduce to new lmax
    new_lmax = 5
    alm.reduce_lmax(new_lmax)
    assert alm.lmax == new_lmax
    s1, s2 = s2fft.sampling.s2_samples.flm_shape(new_lmax + 1)
    assert alm.alm.shape == (1, s1, s2)
    for ell in range(new_lmax + 1):
        for emm in range(-ell, ell + 1):
            assert alm[:, ell, emm] == old_alm[:, ell, emm]
    with pytest.raises(IndexError):
        alm[:, 7, 0]  # asking for ell > new_lmax should raise error
    # try to reduce to greater lmax
    new_lmax = 200
    with pytest.raises(ValueError):
        alm.reduce_lmax(new_lmax)


@pytest.mark.parametrize("sampling", ["mw", "healpix"])
def test_alm2map(lmax, sampling):
    if sampling == "healpix":
        nside = lmax // 2
    else:
        nside = None
    # make constant map
    alm = hp.Alm.zeros(lmax=lmax)
    a00 = 5
    alm[0, 0, 0] = a00
    m = alm.alm2map(sampling=sampling, nside=nside)
    assert jnp.allclose(m, a00 * Y00)

    # make many maps
    frequencies = jnp.linspace(1, 50, 50)
    alm = hp.Alm.zeros(lmax=lmax, frequencies=frequencies)
    alm[:, 0, 0] = a00 * frequencies
    m = alm.alm2map(sampling=sampling, nside=nside, frequencies=frequencies)
    m_ = a00 * frequencies * Y00
    for i in range(m.ndim - 1):
        m_ = m_[:, None]  # match dimensions of m
    assert jnp.allclose(m, m_)

    # use subset of frequencies and compare to full set
    alm = hp.Alm.zeros(lmax=lmax, frequencies=frequencies)
    # some random map
    alm[:, 0, 0] = a00 * frequencies
    alm[:, 1, 1] = 2 * a00 * frequencies
    alm[::2, 8, 3] = -3 * a00 * frequencies[::2]
    m = alm.alm2map(sampling=sampling, nside=nside, frequencies=frequencies)
    freq_indices = jnp.array([10, 20, 35])  # indices of frequencies to use
    freqs = frequencies[freq_indices]  # frequencies to use
    m_select = alm.alm2map(sampling=sampling, nside=nside, frequencies=freqs)
    assert jnp.allclose(m_select, m[freq_indices])

    # use some frequencies that are not in alm.frequencies
    f = jnp.array([0, 30, 100])
    with pytest.warns(UserWarning):
        alm.alm2map(sampling=sampling, nside=nside, frequencies=f)


def test_rot_alm_z(lmax):
    alm = hp.Alm.zeros(lmax=lmax)

    # rotate a single angle
    phi = jnp.array([jnp.pi / 2])
    phase = alm.rot_alm_z(phi=phi)
    ms = jnp.arange(-lmax, lmax + 1)
    assert phase.shape == (1, ms.size)
    assert jnp.allclose(phase, jnp.exp(-1j * ms * phi))

    # rotate a set of angles
    phi = jnp.linspace(0, 2 * jnp.pi, num=361)  # 1 deg spacing
    phase = alm.rot_alm_z(phi=phi)
    assert phase.shape == (phi.size, ms.size)
    assert jnp.allclose(phase[0], jnp.exp(-1j * ms * phi[0]))
    assert jnp.allclose(phase, jnp.exp(-1j * ms[None] * phi[:, None]))

    # check that phi = 0 and phi = 2pi give the same answer
    assert jnp.allclose(phase[0], phase[-1])

    # rotate in time
    alm = hp.Alm.zeros(lmax=lmax)
    div = jnp.array([1, 2, 4, 8])
    for d in div:
        dphi = jnp.array([2 * jnp.pi / d])
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
