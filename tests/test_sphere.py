"""Tests for the SphBase base class and sphere.compute_alm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from croissant import utils
from croissant.sphere import SphBase, compute_alm

rng = np.random.default_rng(seed=0)

# (sampling, lmax) parameter pairs for tests
SAMPLING_PARAMS = [
    pytest.param("mwss", 8, id="mwss-lmax8"),
    pytest.param("mw", 8, id="mw-lmax8"),
    pytest.param("dh", 8, id="dh-lmax8"),
    pytest.param("healpix", 8, id="healpix-lmax8"),
]


def _make_data(lmax, sampling, N_freqs=1):
    """Create uniform data for a given lmax and sampling scheme."""
    if sampling == "healpix":
        nside = lmax // 2
        npix = 12 * nside**2
        return np.ones((N_freqs, npix))
    L = lmax + 1
    ntheta = s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling)
    nphi = s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling)
    return np.ones((N_freqs, ntheta, nphi))


@pytest.mark.parametrize("sampling,lmax", SAMPLING_PARAMS)
def test_sphbase_init(sampling, lmax):
    """SphBase should initialize with correct attributes."""
    N_freqs = 2
    data = _make_data(lmax, sampling, N_freqs)
    freqs = np.linspace(50, 100, N_freqs)
    obj = SphBase(data, freqs, sampling)
    assert obj.sampling == sampling
    assert obj.lmax == lmax
    assert obj._L == lmax + 1
    assert jnp.allclose(obj.freqs, freqs)
    assert obj.data.shape == data.shape


@pytest.mark.parametrize("sampling,lmax", SAMPLING_PARAMS)
def test_sphbase_nside(sampling, lmax):
    """SphBase should infer nside for healpix and None otherwise."""
    data = _make_data(lmax, sampling)
    obj = SphBase(data, np.array([50.0]), sampling)
    if sampling == "healpix":
        assert obj.nside == lmax // 2
    else:
        assert obj.nside is None


@pytest.mark.parametrize("sampling,lmax", SAMPLING_PARAMS)
def test_sphbase_theta_phi_shape(sampling, lmax):
    """Theta and phi arrays should have consistent shapes."""
    data = _make_data(lmax, sampling)
    obj = SphBase(data, np.array([50.0]), sampling)
    # For healpix, theta and phi have length npix
    if sampling == "healpix":
        nside = lmax // 2
        npix = 12 * nside**2
        assert obj.theta.shape == (npix,)
        assert obj.phi.shape == (npix,)
    else:
        L = lmax + 1
        ntheta = s2fft.sampling.s2_samples.ntheta(L=L, sampling=sampling)
        nphi = s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling)
        assert obj.theta.shape == (ntheta,)
        assert obj.phi.shape == (nphi,)


@pytest.mark.parametrize("sampling,lmax", SAMPLING_PARAMS)
def test_sphbase_theta_range(sampling, lmax):
    """Theta values should be in [0, pi]."""
    data = _make_data(lmax, sampling)
    obj = SphBase(data, np.array([50.0]), sampling)
    assert jnp.all(obj.theta >= 0)
    assert jnp.all(obj.theta <= jnp.pi)


@pytest.mark.parametrize("sampling,lmax", SAMPLING_PARAMS)
def test_sphbase_phi_range(sampling, lmax):
    """Phi values should be in [0, 2*pi)."""
    data = _make_data(lmax, sampling)
    obj = SphBase(data, np.array([50.0]), sampling)
    assert jnp.all(obj.phi >= 0)
    assert jnp.all(obj.phi < 2 * jnp.pi + 1e-10)


@pytest.mark.parametrize("sampling,lmax", SAMPLING_PARAMS)
@pytest.mark.parametrize("disable_jit", [True, False])
def test_compute_alm_shape(sampling, lmax, disable_jit):
    """compute_alm should return array of shape (N_freqs, lmax+1, 2*lmax+1)."""
    N_freqs = 3
    data = jnp.array(_make_data(lmax, sampling, N_freqs))
    nside = (lmax // 2) if sampling == "healpix" else None
    with jax.disable_jit(disable_jit):
        alm = compute_alm(data, lmax, sampling, nside=nside)
    assert alm.shape == (N_freqs, lmax + 1, 2 * lmax + 1)


@pytest.mark.parametrize("sampling,lmax", SAMPLING_PARAMS)
@pytest.mark.parametrize("disable_jit", [True, False])
def test_compute_alm_monopole(sampling, lmax, disable_jit):
    """Uniform map should produce a dominant monopole component."""
    from croissant.constants import Y00

    T = 500.0
    N_freqs = 1
    data = T * jnp.array(_make_data(lmax, sampling, N_freqs))
    nside = (lmax // 2) if sampling == "healpix" else None
    with jax.disable_jit(disable_jit):
        alm = compute_alm(data, lmax, sampling, nside=nside)
    l_ix, m_ix = utils.getidx(lmax, 0, 0)
    # monopole alm = T / Y00 for a uniform map
    assert jnp.isclose(alm[0, l_ix, m_ix].real, T / Y00, rtol=1e-3)
