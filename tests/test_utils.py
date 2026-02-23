import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

import croissant as cro
from croissant.constants import Y00

rng = np.random.default_rng(seed=0)


@pytest.mark.parametrize(
    "utils_func, rotations_func, func_args, func_kwargs",
    [
        (
            cro.utils.get_rot_mat,
            cro.rotations.get_rot_mat,
            ("fk5", "galactic"),
            {},
        ),
        (
            cro.utils.rotmat_to_euler,
            cro.rotations.rotmat_to_euler,
            (jnp.eye(3),),
            {"eulertype": "ZYX"},
        ),
        (
            cro.utils.rotmat_to_eulerZYX,
            cro.rotations.rotmat_to_eulerZYX,
            (jnp.eye(3),),
            {},
        ),
        (
            cro.utils.rotmat_to_eulerZYZ,
            cro.rotations.rotmat_to_eulerZYZ,
            (jnp.eye(3),),
            {},
        ),
    ],
)
def test_moved_function(utils_func, rotations_func, func_args, func_kwargs):
    """
    Test that the functions moved from cro.rotations to cro.utils are
    unchagned and raise a FutureWarning.
    """
    with pytest.warns(FutureWarning):
        result_utils = utils_func(*func_args, **func_kwargs)
    result_rotations = rotations_func(*func_args, **func_kwargs)
    assert jnp.allclose(
        jnp.asarray(result_utils), jnp.asarray(result_rotations)
    )


# --- healpix utils ---
def test_valid_nside():
    """
    Check that valid_nside returns True for valid nside values (powers
    of 2) and False for invalid nside values.
    """
    nside = 2 ** np.arange(0, 11)
    assert all(cro.utils.valid_nside(n) for n in nside)
    invalid_nside = [-2, 0, 3, 5, 6, 7, 9, 10, 12, 15, 20]
    assert all(not cro.utils.valid_nside(n) for n in invalid_nside)


def test_hp_npix2nside():
    """
    Check that hp_npix2nside correctly computes nside from npix for valid
    npix values and raises a ValueError for invalid npix values.
    """
    nside = 2 ** np.arange(0, 11)
    for n in nside:
        npix = 12 * n**2
        assert cro.utils.hp_npix2nside(npix) == n
    invalid_npix = [-100, 0, 1, 10, 100, 500, 1000, 5000]
    for npix in invalid_npix:
        with pytest.raises(ValueError):
            cro.utils.hp_npix2nside(npix)


def test_valid_npix():
    """
    Check that valid_npix returns True for valid npix values and False
    for invalid npix values.
    """
    nside = 2 ** np.arange(0, 11)
    valid_npix = [12 * n**2 for n in nside]
    assert all(cro.utils.hp_valid_npix(npix) for npix in valid_npix)
    invalid_npix = [-100, 0, 1, 10, 100, 500, 1000, 5000]
    assert all(not cro.utils.hp_valid_npix(npix) for npix in invalid_npix)


@pytest.mark.parametrize("lmax", [8, 16, 64, 128])
class TestAlmUtils:
    """
    Utils for alm coefficients.
    """

    def test_total_power(self, lmax):
        # make a map that is 1 everywhere so total power is 4pi:
        shape = cro.utils.shape_from_lmax(lmax)
        alm = jnp.zeros(shape, dtype=jnp.complex128)
        a00_idx = cro.utils.getidx(lmax, 0, 0)
        alm = alm.at[a00_idx].set(1 / Y00)
        power = cro.utils.total_power(alm, lmax)
        assert jnp.isclose(power, 4 * jnp.pi)

        # m(theta) = cos(theta)**2
        alm = jnp.zeros(shape, dtype=jnp.complex128)
        alm = alm.at[a00_idx].set(1 / (3 * Y00))
        a20_idx = cro.utils.getidx(lmax, 2, 0)
        alm = alm.at[a20_idx].set(4 * jnp.sqrt(jnp.pi / 5) * 1 / 3)
        power = cro.utils.total_power(alm, lmax)
        expected_power = 4 * jnp.pi / 3
        assert jnp.isclose(power, expected_power)

    def test_getidx(self, lmax):
        # using ints
        ell = 3
        emm = 2
        ix = cro.utils.getidx(lmax, ell, emm)
        ell_, emm_ = cro.utils.getlm(lmax, ix)
        assert ell == ell_
        assert emm == emm_

        # using arrays
        ls = lmax // jnp.arange(1, 10)
        ms = jnp.arange(-lmax, lmax + 1)
        ixs = cro.utils.getidx(lmax, ls, ms)
        ls_, ms_ = cro.utils.getlm(lmax, ixs)
        assert jnp.allclose(ls, ls_)
        assert jnp.allclose(ms, ms_)

    def test_getlm(self, lmax):
        alm = jnp.zeros(cro.utils.shape_from_lmax(lmax), dtype=jnp.complex128)
        nrows, ncols = alm.shape
        # l correspond to rows, m correspond to columns
        ls = jnp.arange(nrows)
        ms = jnp.arange(ncols) - lmax
        for i in range(nrows):
            for j in range(ncols):
                ix = (i, j)
                ell, emm = cro.utils.getlm(lmax, ix)
                assert ell == ls[i]
                assert emm == ms[j]

    def test_lmax_from_shape(self, lmax):
        shape = s2fft.sampling.s2_samples.flm_shape(lmax + 1)
        _lmax = cro.utils.lmax_from_shape(shape)
        assert _lmax == lmax

    def test_is_real(self, lmax):
        alm = jnp.zeros(cro.utils.shape_from_lmax(lmax), dtype=jnp.complex128)
        assert cro.utils.is_real(alm)
        val = 1.0 + 2.0j
        ix_21 = cro.utils.getidx(lmax, 2, 1)  # get index for l=2, m=1
        alm = alm.at[ix_21].set(val)  # set l=2, m=1 mode but not m=-1 mode
        assert not cro.utils.is_real(alm)
        ix_2m1 = cro.utils.getidx(lmax, 2, -1)  # get index for l=2, m=-1
        # set m=-1 mode to complex conjugate
        alm = alm.at[ix_2m1].set(-1 * val.conjugate())
        assert cro.utils.is_real(alm)

        # generate a real signal and check that alm.is_real is True
        alm = s2fft.utils.signal_generator.generate_flm(
            rng, lmax + 1, reality=True
        )
        assert cro.utils.is_real(alm)
        # complex
        alm = s2fft.utils.signal_generator.generate_flm(
            rng, lmax + 1, reality=False
        )
        assert not cro.utils.is_real(alm)

    def test_reduce_lmax(self, lmax):
        signal1 = s2fft.utils.signal_generator.generate_flm(rng, lmax + 1)
        # reduce lmax to new_lmax
        new_lmax = 5
        signal2 = cro.utils.reduce_lmax(signal1, new_lmax)
        assert cro.utils.lmax_from_shape(signal2.shape) == new_lmax
        # confirm that signal 2 has the expected shape
        expected_shape = cro.utils.shape_from_lmax(new_lmax)
        assert signal2.shape == expected_shape
        # check that the signals are the same for all ell, emm
        for ell in range(new_lmax + 1):
            for emm in range(-ell, ell + 1):
                # indexing differes since lmax differs
                ix1 = cro.utils.getidx(lmax, ell, emm)
                ix2 = cro.utils.getidx(new_lmax, ell, emm)
                assert signal1[ix1] == signal2[ix2]

        # check that smaller lmax throws error
        with pytest.raises(ValueError):
            cro.utils.reduce_lmax(signal1, lmax + 1)

    def test_shape_from_lmax(self, lmax):
        shape = cro.utils.shape_from_lmax(lmax)
        expected_shape = s2fft.sampling.s2_samples.flm_shape(lmax + 1)
        assert shape == expected_shape

    @pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "gl"])
    def test_lmax_from_ntheta_equiang(self, sampling, lmax):
        L = lmax + 1
        ntheta = s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling)
        expected = cro.utils.lmax_from_ntheta(ntheta, sampling)
        assert expected == lmax


@pytest.mark.parametrize("nside", [8, 16, 32, 1024])
def test_lmax_from_ntheta_hp(nside):
    npix = 12 * nside**2
    lmax = cro.utils.lmax_from_ntheta(npix, "healpix")
    assert lmax == 2 * nside


def test_lmax_from_ntheta_invalid_sampling():
    with pytest.raises(ValueError):
        cro.utils.lmax_from_ntheta(10, "invalid_sampling")
