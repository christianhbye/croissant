"""Tests for the dense SHT engine: builders, cache and apply."""

import jax
import jax.numpy as jnp
import numpy as np
import s2fft

from croissant import dense, utils
from croissant.dense import (
    DenseSphericalTransform,
    _positive_lm_indices,
    dense_compute_alm,
    precompute_dense_matrix,
)
from croissant.footprints import spatial_shape


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
    """A repeated configuration is served from the cache, not rebuilt.

    The unified dict has no hit counter, so reuse is shown by the
    second transform holding the very array the first one stored: a
    rebuild would produce an equal but distinct array.
    """
    dense.clear_dense_matrix_cache()
    first = DenseSphericalTransform(3, "mwss", spin=0)
    second = DenseSphericalTransform(3, "mwss", spin=0)
    assert jnp.array_equal(first.matrix, second.matrix)
    assert first.matrix is second.matrix
    assert len(dense._DENSE_MATRIX_CACHE) == 1


def test_equiangular_builder_produces_the_packed_operator():
    """The relocated builder still reproduces s2fft's coefficients.

    The builder exists only to materialize s2fft's own transform, so
    s2fft.forward is the ground truth its packed output is pinned
    against. Comparing values rather than only the matrix shape is what
    makes this a regression test for the relocation.
    """
    lmax, sampling = 4, "dh"
    shape = spatial_shape(lmax, sampling, None)
    matrix = precompute_dense_matrix(shape, lmax, sampling)

    ncoeff = (lmax + 1) * (lmax + 2) // 2
    assert matrix.shape == (ncoeff, int(np.prod(shape)))

    rng = np.random.default_rng(seed=0)
    maps = jnp.asarray(rng.standard_normal(shape))
    expected = s2fft.forward(
        maps,
        L=lmax + 1,
        sampling=sampling,
        method="jax",
        reality=True,
    )
    ell, emm = _positive_lm_indices(lmax)
    packed = matrix @ maps.reshape(-1)
    np.testing.assert_allclose(
        packed, expected[ell, lmax + emm], rtol=1e-12, atol=1e-12
    )


def test_clear_releases_both_operator_flavours():
    """One clear function must empty the whole engine's cache.

    Before unification clear_dense_matrix_cache reached only the packed
    half; the VJP half was reachable only through an lru_cache's own
    cache_clear, which no public name exposed.
    """
    lmax, nside, npix = 4, 2, 48
    dense.clear_dense_matrix_cache()
    dense.precompute_dense_matrix((npix,), lmax, "healpix", nside=nside)
    dense.dense_compute_alm(
        jnp.zeros((1, npix)), lmax, "healpix", nside=nside, spin=2
    )
    assert len(dense._DENSE_MATRIX_CACHE) == 2

    dense.clear_dense_matrix_cache()
    assert len(dense._DENSE_MATRIX_CACHE) == 0


def test_packed_and_full_operators_do_not_collide():
    """Identical geometry, two flavours, two entries.

    Both are spin 0 at the same lmax, sampling, nside and niter. Only
    the packed flag separates them, so a key that omitted it would
    return the m >= 0 operator to a caller expecting the full one.
    """
    lmax, nside, npix = 4, 2, 48
    dense.clear_dense_matrix_cache()
    packed = dense.precompute_dense_matrix(
        (npix,), lmax, "healpix", nside=nside
    )
    dense.dense_compute_alm(
        jnp.zeros((1, npix)), lmax, "healpix", nside=nside, spin=0
    )

    assert len(dense._DENSE_MATRIX_CACHE) == 2
    shapes = {m.shape for m in dense._DENSE_MATRIX_CACHE.values()}
    ncoeff_packed = (lmax + 1) * (lmax + 2) // 2
    assert packed.shape == (ncoeff_packed, npix)
    assert shapes == {(ncoeff_packed, npix), ((lmax + 1) ** 2, npix)}
