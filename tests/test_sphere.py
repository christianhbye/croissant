"""Tests for the SphBase base class and sphere.compute_alm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from croissant import utils
from croissant.constants import Y00
from croissant.sphere import SphBase, compute_alm

LMAX_PARAMS = [8, 16, 25]
rng = np.random.default_rng(seed=0)

SAMPLING_PARAMS = [
    pytest.param("mwss"),
    pytest.param("mw"),
    pytest.param("dh"),
    pytest.param("gl"),
    pytest.param("healpix"),
]

# s2fft mw/mwss forward transforms call spin.size which fails on Python ints
# when jit is disabled; restrict disable_jit alm tests to healpix and dh
SAMPLING_PARAMS_JIT_SAFE = [
    pytest.param("dh"),
    pytest.param("gl"),
    pytest.param("healpix"),
]


def _make_data(lmax, sampling, N_freqs=50):
    """Create uniform data for a given lmax and sampling scheme."""
    if sampling == "healpix":
        # find largest nside that is <= lmax // 2
        nside = 1 << (lmax // 2).bit_length() - 1
        npix = 12 * nside**2
        return np.ones((N_freqs, npix))
    L = lmax + 1
    ntheta = s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling)
    nphi = s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling)
    return np.ones((N_freqs, ntheta, nphi))


@pytest.mark.parametrize("lmax", LMAX_PARAMS)
@pytest.mark.parametrize("sampling", SAMPLING_PARAMS)
def test_sphbase_init(sampling, lmax):
    """SphBase should initialize with correct attributes."""
    N_freqs = 50
    data = _make_data(lmax, sampling, N_freqs)
    freqs = np.linspace(50, 100, N_freqs)
    obj = SphBase(data, freqs, sampling, niter=0)

    if sampling == "healpix":
        npix = data.shape[1]
        expected_nside = utils.hp_npix2nside(npix)
        assert obj.nside == expected_nside
        expected_lmax = 2 * expected_nside
        assert obj.lmax == expected_lmax
        assert obj._L == expected_lmax + 1
    else:
        assert obj.nside is None
        assert obj.lmax == lmax
        assert obj._L == lmax + 1

    assert obj.sampling == sampling
    assert jnp.allclose(obj.freqs, freqs)
    assert obj.data.shape == data.shape


@pytest.mark.parametrize("lmax", LMAX_PARAMS)
@pytest.mark.parametrize("sampling", SAMPLING_PARAMS)
def test_sphbase_theta_phi_shape(sampling, lmax):
    """Theta and phi arrays should have consistent shapes."""
    data = _make_data(lmax, sampling)
    obj = SphBase(data, np.array([50.0]), sampling, niter=0)
    # For healpix, theta and phi have length npix
    if sampling == "healpix":
        npix = data.shape[1]
        assert obj.theta.shape == (npix,)
        assert obj.phi.shape == (npix,)
    else:
        L = lmax + 1
        ntheta = s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling)
        nphi = s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling)
        assert obj.theta.shape == (ntheta,)
        assert obj.phi.shape == (nphi,)


@pytest.mark.parametrize("lmax", LMAX_PARAMS)
@pytest.mark.parametrize("sampling", SAMPLING_PARAMS)
def test_sphbase_theta_range(sampling, lmax):
    """Theta values should be in [0, pi]."""
    data = _make_data(lmax, sampling)
    obj = SphBase(data, np.array([50.0]), sampling, niter=0)
    assert jnp.all(obj.theta >= 0)
    assert jnp.all(obj.theta < jnp.pi + 1e-10)


@pytest.mark.parametrize("lmax", LMAX_PARAMS)
@pytest.mark.parametrize("sampling", SAMPLING_PARAMS)
def test_sphbase_phi_range(sampling, lmax):
    """Phi values should be in [0, 2*pi)."""
    data = _make_data(lmax, sampling)
    obj = SphBase(data, np.array([50.0]), sampling, niter=0)
    assert jnp.all(obj.phi >= 0)
    assert jnp.all(obj.phi < 2 * jnp.pi + 1e-10)


@pytest.mark.parametrize(
    "disable_jit, lmax",
    [(True, 8), (False, 8), (False, 16), (False, 25)],
)
@pytest.mark.parametrize("sampling", SAMPLING_PARAMS_JIT_SAFE)
def test_compute_alm_shape(sampling, lmax, disable_jit):
    """compute_alm should return array of shape (N_freqs, lmax+1, 2*lmax+1)."""
    N_freqs = 3
    data = jnp.array(_make_data(lmax, sampling, N_freqs))
    if sampling == "healpix":
        npix = data.shape[1]
        nside = utils.hp_npix2nside(npix)
    else:
        nside = None
    with jax.disable_jit(disable_jit):
        alm = compute_alm(data, lmax, sampling, nside=nside)
    assert alm.shape == (N_freqs, lmax + 1, 2 * lmax + 1)


@pytest.mark.parametrize("lmax", LMAX_PARAMS)
@pytest.mark.parametrize("sampling", SAMPLING_PARAMS_JIT_SAFE)
def test_compute_alm_niter(sampling, lmax):
    """compute_alm with a non-default niter should return correct shape."""
    N_freqs = 3
    data = jnp.array(_make_data(lmax, sampling, N_freqs))
    if sampling == "healpix":
        npix = data.shape[1]
        nside = utils.hp_npix2nside(npix)
    else:
        nside = None
    alm = compute_alm(data, lmax, sampling, nside=nside, niter=1)
    assert alm.shape == (N_freqs, lmax + 1, 2 * lmax + 1)


@pytest.mark.parametrize("lmax", LMAX_PARAMS)
def test_compute_alm_healpix_niter_reduces_error(lmax):
    """
    niter=3 for healpix should reduce forward/inverse reconstruction
    error vs niter=0.
    """
    nside = 1 << (lmax // 2).bit_length() - 1
    npix = 12 * nside**2
    data = jnp.array(rng.standard_normal((1, npix)).astype(np.float32))

    alm0 = compute_alm(data, lmax, "healpix", nside=nside, niter=0)
    alm3 = compute_alm(data, lmax, "healpix", nside=nside, niter=3)

    rec0 = s2fft.inverse(
        np.array(alm0[0]),
        L=lmax + 1,
        spin=0,
        nside=nside,
        sampling="healpix",
        method="jax",
        reality=True,
    )
    rec3 = s2fft.inverse(
        np.array(alm3[0]),
        L=lmax + 1,
        spin=0,
        nside=nside,
        sampling="healpix",
        method="jax",
        reality=True,
    )

    err0 = float(jnp.mean(jnp.abs(jnp.array(rec0) - data[0])))
    err3 = float(jnp.mean(jnp.abs(jnp.array(rec3) - data[0])))
    assert err3 < err0


@pytest.mark.parametrize(
    "disable_jit, lmax",
    [(True, 8), (False, 8), (False, 16), (False, 25)],
)
@pytest.mark.parametrize("sampling", SAMPLING_PARAMS_JIT_SAFE)
def test_compute_alm_monopole(sampling, lmax, disable_jit):
    """Uniform map should produce a dominant monopole component."""

    T = 500.0
    N_freqs = 1
    data = T * jnp.array(_make_data(lmax, sampling, N_freqs))
    if sampling == "healpix":
        npix = data.shape[1]
        nside = utils.hp_npix2nside(npix)
    else:
        nside = None
    with jax.disable_jit(disable_jit):
        alm = compute_alm(data, lmax, sampling, nside=nside)
    l_ix, m_ix = utils.getidx(lmax, 0, 0)
    # monopole alm = T / Y00 for a uniform map
    assert jnp.isclose(alm[0, l_ix, m_ix].real, T / Y00, rtol=1e-3)
