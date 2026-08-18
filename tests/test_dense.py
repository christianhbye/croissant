"""Tests for the dense SHT engine: builders, cache and apply."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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

    What this cannot pin is the builder's own ``reality=True``: for the
    real one-hot basis maps it feeds s2fft, the general transform
    returns identical m >= 0 coefficients, so dropping the flag still
    passes here -- verified, not assumed. Losing it costs roughly a
    factor two in build work and memory, not correctness, so it is
    pinned by watching the call instead, in
    ``test_equiangular_builder_asks_s2fft_for_the_real_transform``.
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


def test_equiangular_builder_asks_s2fft_for_the_real_transform(monkeypatch):
    """Pin the builder's ``reality=True``, which values cannot pin.

    ``test_equiangular_builder_produces_the_packed_operator`` compares
    coefficients, and for the real one-hot basis maps this builder feeds
    s2fft the general transform returns identical ``m >= 0`` values --
    so dropping the flag costs roughly a factor two in build work and
    memory while passing every value assertion. The only way to hold on
    to it is to watch the call.
    """
    seen = []
    real_forward = s2fft.forward

    def recording_forward(*args, **kwargs):
        seen.append(kwargs.get("reality"))
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(s2fft, "forward", recording_forward)
    # The builder is filter_jit-wrapped and keys on geometry alone, so a
    # trace cached by an earlier test would never reach the patch.
    jax.clear_caches()
    dense.clear_dense_matrix_cache()

    lmax, sampling = 5, "dh"
    precompute_dense_matrix(
        spatial_shape(lmax, sampling, None), lmax, sampling
    )

    assert seen, "the builder never reached s2fft.forward"
    assert all(flag is True for flag in seen)


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


def test_no_tracer_ever_enters_the_dense_cache():
    """Building inside jax.jit must still cache concrete arrays.

    Retention is unbounded by design, so a tracer stored under a
    geometry's key poisons that key for the life of the process: every
    later use of it, traced or not, raises with an error naming an
    unrelated function. Both writers must therefore establish
    ``jax.ensure_compile_time_eval`` themselves rather than trusting
    their callers to have done it.
    """
    lmax, nside, npix = 4, 2, 48
    dense.clear_dense_matrix_cache()

    @jax.jit
    def build(scale):
        packed = dense.precompute_dense_matrix(
            (npix,), lmax, "healpix", nside=nside
        )
        # The full operator is reached through its writer rather than
        # through DenseSphericalTransform, whose __init__ establishes
        # the context itself: going via the class would pass on a
        # writer that had lost its own guard.
        full = dense._full_matrix_for(
            lmax, "healpix", nside, 2, 0, np.dtype(np.complex128)
        )
        return scale * (packed.sum() + full.sum())

    build(jnp.asarray(1.0))

    assert len(dense._DENSE_MATRIX_CACHE) == 2
    for matrix in dense._DENSE_MATRIX_CACHE.values():
        assert not isinstance(matrix, jax.core.Tracer)
        # The symptom of a cached tracer: the entry cannot be used
        # outside the trace that created it.
        np.asarray(matrix)


def test_packed_and_full_operators_do_not_collide():
    """Identical geometry, three operators, three entries.

    This covers both of the key's own discriminators. The packed and
    the spin-0 full operator agree on lmax, sampling, nside and niter,
    so only the packed flag separates them: a key that omitted it would
    return the m >= 0 operator to a caller expecting the full one. The
    spin-2 operator then agrees with the spin-0 full one on everything
    including the packed flag, so only spin separates those two.
    """
    lmax, nside, npix = 4, 2, 48
    dense.clear_dense_matrix_cache()
    packed = dense.precompute_dense_matrix(
        (npix,), lmax, "healpix", nside=nside
    )
    for spin in (0, 2):
        dense.dense_compute_alm(
            jnp.zeros((1, npix)), lmax, "healpix", nside=nside, spin=spin
        )

    assert len(dense._DENSE_MATRIX_CACHE) == 3
    shapes = {m.shape for m in dense._DENSE_MATRIX_CACHE.values()}
    ncoeff_packed = (lmax + 1) * (lmax + 2) // 2
    ncoeff_spin2 = (lmax + 1) ** 2 - 2**2
    assert packed.shape == (ncoeff_packed, npix)
    assert shapes == {
        (ncoeff_packed, npix),
        ((lmax + 1) ** 2, npix),
        (ncoeff_spin2, npix),
    }


def test_dense_cache_nbytes_tracks_both_flavours():
    """Retention is unbounded by design, so it must be inspectable."""
    lmax, nside, npix = 4, 2, 48
    dense.clear_dense_matrix_cache()
    assert dense.dense_cache_nbytes() == 0

    packed = dense.precompute_dense_matrix(
        (npix,), lmax, "healpix", nside=nside
    )
    assert dense.dense_cache_nbytes() == packed.nbytes

    dense.dense_compute_alm(
        jnp.zeros((1, npix)), lmax, "healpix", nside=nside, spin=2
    )
    (full,) = (
        matrix
        for matrix in dense._DENSE_MATRIX_CACHE.values()
        if matrix is not packed
    )
    assert dense.dense_cache_nbytes() == packed.nbytes + full.nbytes

    dense.clear_dense_matrix_cache()
    assert dense.dense_cache_nbytes() == 0


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_builders_reject_a_nonpositive_chunk_size(chunk_size):
    """Every builder guards its chunk size the same way.

    Unguarded, zero reaches ``range()`` as a step ("must not be zero")
    and a negative value produces no blocks at all, failing later in
    ``concatenate`` with nothing pointing at the caller's mistake.
    """
    lmax, nside, npix = 3, 2, 48
    dense.clear_dense_matrix_cache()
    with pytest.raises(ValueError, match="chunk_size"):
        dense.precompute_dense_matrix(
            (npix,), lmax, "healpix", nside=nside, chunk_size=chunk_size
        )
    shape = spatial_shape(lmax, "dh", None)
    with pytest.raises(ValueError, match="chunk_size"):
        dense.precompute_dense_matrix(shape, lmax, "dh", chunk_size=chunk_size)
    with pytest.raises(ValueError, match="chunk_size"):
        dense._build_analysis_matrix(
            lmax,
            "healpix",
            nside,
            2,
            0,
            "complex128",
            chunk_size=chunk_size,
        )


def test_full_operator_assembly_is_chunk_size_independent():
    """Row batching must not change the assembled operator.

    The builder pulls back coefficient basis vectors in chunks. If
    assembly and chunking are correctly separated, a one-row-at-a-time
    build and a batched one are bitwise identical.
    """
    lmax, spin, nside = 3, 2, 2
    args = (lmax, "healpix", nside, spin, 0, "complex128")
    batched = dense._build_analysis_matrix(*args, chunk_size=32)
    one_at_a_time = dense._build_analysis_matrix(*args, chunk_size=1)
    np.testing.assert_array_equal(batched, one_at_a_time)
