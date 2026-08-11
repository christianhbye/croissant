"""Tests for cached dense scalar and spin transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import s2fft

from croissant import utils
from croissant.dense import (
    DenseSphericalTransform,
    _build_analysis_matrix,
    dense_compute_alm,
)


def _random_valid_alm(lmax, spin, seed):
    rng = np.random.default_rng(seed)
    alm = np.zeros(
        (lmax + 1, 2 * lmax + 1),
        dtype=np.complex128,
    )
    for ell in range(abs(spin), lmax + 1):
        values = rng.normal(size=2 * ell + 1) + 1j * rng.normal(
            size=2 * ell + 1
        )
        alm[ell, lmax - ell : lmax + ell + 1] = values
    return alm


def test_dense_spin_transform_recovers_bandlimited_mwss_coefficients():
    lmax = 5
    spin = 2
    alm = _random_valid_alm(lmax, spin, seed=7)
    data = s2fft.inverse(
        jnp.asarray(alm),
        L=lmax + 1,
        spin=spin,
        sampling="mwss",
        method="jax",
        reality=False,
    )
    recovered = dense_compute_alm(
        data[None],
        lmax,
        "mwss",
        spin=spin,
    )[0]
    assert jnp.allclose(recovered, alm, rtol=2e-6, atol=2e-6)


def test_dense_transform_supports_low_lmax_healpix_and_gradients():
    nside = 4
    lmax = 4
    spin = -2
    transform = DenseSphericalTransform(
        lmax,
        "healpix",
        nside=nside,
        spin=spin,
        dtype=jnp.complex128,
    )
    data = jnp.linspace(0.0, 1.0, 12 * nside**2)
    result = transform(data)
    transform_lmax = max(lmax, 2 * nside - 1)
    reference_full = s2fft.forward(
        data,
        L=transform_lmax + 1,
        spin=spin,
        nside=nside,
        sampling="healpix",
        method="jax",
        reality=False,
        iter=0,
    )
    reference = utils.reduce_lmax(reference_full, lmax)

    def loss(values):
        return jnp.sum(jnp.abs(transform(values)) ** 2)

    def reference_loss(values):
        full = s2fft.forward(
            values,
            L=transform_lmax + 1,
            spin=spin,
            nside=nside,
            sampling="healpix",
            method="jax",
            reality=False,
            iter=0,
        )
        reduced = utils.reduce_lmax(full, lmax)
        return jnp.sum(jnp.abs(reduced) ** 2)

    gradient = jax.grad(loss)(data)
    reference_gradient = jax.grad(reference_loss)(data)
    assert result.shape == (lmax + 1, 2 * lmax + 1)
    assert jnp.allclose(result, reference, rtol=2e-12, atol=2e-12)
    assert gradient.shape == data.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(
        gradient,
        reference_gradient,
        rtol=2e-12,
        atol=2e-12,
    )


def test_dense_transform_matches_s2fft_for_complex_input():
    """Direct comparison on complex input at tight tolerance.

    Complex input distinguishes the analysis matrix from its conjugate,
    certifying the VJP row-extraction convention in
    ``_build_analysis_matrix``; the 1j-scaled batch entry would flip
    sign under a conjugating apply path.
    """
    nside = 8
    lmax = 6
    spin = 2
    rng = np.random.default_rng(41)
    npix = 12 * nside**2
    data = jnp.asarray(rng.normal(size=npix) + 1j * rng.normal(size=npix))
    transform = DenseSphericalTransform(
        lmax,
        "healpix",
        nside=nside,
        spin=spin,
        dtype=jnp.complex128,
    )
    dense = transform(data)
    transform_L = max(lmax, 2 * nside - 1) + 1
    reference = utils.reduce_lmax(
        s2fft.forward(
            data,
            L=transform_L,
            spin=spin,
            nside=nside,
            sampling="healpix",
            method="jax",
            reality=False,
            iter=0,
        ),
        lmax,
    )
    assert jnp.allclose(dense, reference, rtol=1e-12, atol=1e-12)

    batch = jnp.stack([data, 1j * data]).reshape(2, 1, npix)
    batched = transform(batch)
    assert batched.shape == (2, 1, lmax + 1, 2 * lmax + 1)
    assert jnp.allclose(batched[0, 0], dense, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(batched[1, 0], 1j * dense, rtol=1e-12, atol=1e-12)


def test_dense_cache_reuses_matrix_for_identical_geometry():
    hits_before = _build_analysis_matrix.cache_info().hits
    first = DenseSphericalTransform(3, "mwss", spin=0)
    second = DenseSphericalTransform(3, "mwss", spin=0)
    assert jnp.array_equal(first.matrix, second.matrix)
    assert _build_analysis_matrix.cache_info().hits > hits_before
