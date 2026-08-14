"""Tests for the SphBase base class and sphere.compute_alm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from croissant import utils
from croissant.constants import Y00
from croissant.sphere import (
    _DENSE_MATRIX_CACHE,
    SphBase,
    clear_dense_matrix_cache,
    compute_alm,
    precompute_dense_matrix,
)

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


# ---------------------------------------------------------------------------
# Reality contract
# ---------------------------------------------------------------------------


def _complex_healpix_data(nside, N_freqs=2):
    """A field whose imaginary part is independent of its real part."""
    npix = 12 * nside**2
    real = rng.standard_normal((N_freqs, npix))
    imag = rng.standard_normal((N_freqs, npix))
    return jnp.asarray(real), jnp.asarray(imag), jnp.asarray(real + 1j * imag)


@pytest.mark.parametrize("engine", ["s2fft", "dense"])
def test_compute_alm_rejects_complex_when_reality_is_true(engine):
    """reality=True is an assertion about the data, so it must be checked.

    Without this the packed real transform silently discards the
    imaginary part and returns coefficients that are wrong at order
    unity, with no warning.
    """
    nside = 2
    _, _, data = _complex_healpix_data(nside)

    with pytest.raises(ValueError, match="Complex input requires"):
        compute_alm(
            data,
            2 * nside,
            "healpix",
            nside=nside,
            reality=True,
            engine=engine,
        )


@pytest.mark.parametrize("engine", ["s2fft", "dense"])
def test_compute_alm_default_transforms_complex_data(engine):
    """The default must handle complex input, as s2fft's default does.

    The transform is linear, so the exact answer for a complex field is
    the transform of its real part plus i times that of its imaginary
    part. Each of those is computed on genuinely real data, which makes
    this oracle independent of the code path under test.
    """
    nside = 2
    lmax = 2 * nside
    real, imag, data = _complex_healpix_data(nside)

    kwargs = dict(nside=nside, engine=engine)
    expected = compute_alm(
        real, lmax, "healpix", reality=True, **kwargs
    ) + 1j * compute_alm(imag, lmax, "healpix", reality=True, **kwargs)
    actual = compute_alm(data, lmax, "healpix", **kwargs)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("engine", ["s2fft", "dense"])
def test_compute_alm_reality_is_only_an_optimization(engine):
    """On real data the packed transform must agree with the general one.

    reality=True exploits the exact Hermitian symmetry of a real
    field's coefficients, so it costs no accuracy. Changing the default
    is therefore safe for every real-valued caller.
    """
    nside = 2
    lmax = 2 * nside
    real, _, _ = _complex_healpix_data(nside)

    packed = compute_alm(
        real, lmax, "healpix", nside=nside, reality=True, engine=engine
    )
    general = compute_alm(real, lmax, "healpix", nside=nside, engine=engine)

    np.testing.assert_allclose(general, packed, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Dense transform engine
# ---------------------------------------------------------------------------
#
# The dense engine has two operators: the packed real matrix, reached
# with reality=True, and the general full-layout one in croissant.dense.
# The packed matrix is what Sky and Beam use, so the tests below state
# reality=True the way those classes do. Tests whose subject is the
# general operator say so in their names.


@pytest.mark.parametrize("niter", [0, 1])
def test_dense_engine_matches_s2fft_healpix(niter):
    """Dense transforms should reproduce s2fft, including refinement."""
    nside = 2
    lmax = 2 * nside
    data = jnp.asarray(rng.standard_normal((3, 12 * nside**2)))

    expected = compute_alm(
        data,
        lmax,
        "healpix",
        nside=nside,
        niter=niter,
        reality=True,
        engine="s2fft",
    )
    actual = compute_alm(
        data,
        lmax,
        "healpix",
        nside=nside,
        niter=niter,
        reality=True,
        engine="dense",
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_dense_engine_matches_s2fft_equiangular():
    """Dense transforms should retain support for non-HEALPix samplings."""
    lmax = 4
    data = jnp.asarray(rng.standard_normal(_make_data(lmax, "dh", 2).shape))

    expected = compute_alm(data, lmax, "dh", reality=True, engine="s2fft")
    actual = compute_alm(data, lmax, "dh", reality=True, engine="dense")

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_dense_matrix_is_packed_and_cached():
    """Only independent coefficients are stored and identical keys reuse it."""
    clear_dense_matrix_cache()
    matrix1 = precompute_dense_matrix((48,), 4, "healpix", nside=2)
    matrix2 = precompute_dense_matrix((48,), 4, "healpix", nside=2)

    assert matrix1 is matrix2
    assert matrix1.shape == (15, 48)


@pytest.mark.parametrize("sampling", ["healpix", "dh"])
def test_dense_engine_matches_s2fft_dtype(sampling):
    """The matrix precision must track s2fft's output, not the map dtype."""
    lmax = 4
    nside = 2 if sampling == "healpix" else None
    if sampling == "healpix":
        data = jnp.asarray(rng.standard_normal((2, 12 * nside**2)))
    else:
        shape = _make_data(lmax, "dh", 2).shape
        data = jnp.asarray(rng.standard_normal(shape))

    # float32 maps round differently in the two engines, so only the
    # output dtype is compared exactly.
    for dtype, tol in ((jnp.float32, 1e-6), (jnp.float64, 1e-11)):
        maps = data.astype(dtype)
        expected = compute_alm(maps, lmax, sampling, nside=nside, reality=True)
        actual = compute_alm(
            maps,
            lmax,
            sampling,
            nside=nside,
            reality=True,
            engine="dense",
        )
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "dtype, alm_dtype, tol",
    [
        (jnp.float32, jnp.complex64, 1e-6),
        (jnp.float64, jnp.complex128, 1e-11),
    ],
)
def test_dense_general_operator_accepts_single_precision(
    dtype, alm_dtype, tol
):
    """The general dense operator must build at the caller's precision.

    Its cotangent basis has to match the dtype s2fft actually returns,
    not the requested matrix dtype, or the VJP that materializes the
    matrix rejects single-precision maps outright.

    Unlike the packed matrix, whose precision tracks s2fft's output,
    this operator follows the map dtype, so the two agree on values but
    not on storage.
    """
    nside = 2
    lmax = 2 * nside
    data = jnp.asarray(rng.standard_normal((2, 12 * nside**2))).astype(dtype)

    expected = compute_alm(data, lmax, "healpix", nside=nside, reality=True)
    actual = compute_alm(data, lmax, "healpix", nside=nside, engine="dense")

    assert actual.dtype == np.dtype(alm_dtype)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


def test_dense_matrix_cache_is_dtype_independent():
    """float32 and float64 maps must share one cached matrix."""
    clear_dense_matrix_cache()
    npix = 48
    data = jnp.asarray(rng.standard_normal((1, npix)))
    for dtype in (jnp.float64, jnp.float32):
        compute_alm(
            data.astype(dtype),
            4,
            "healpix",
            nside=2,
            reality=True,
            engine="dense",
        )

    assert len(_DENSE_MATRIX_CACHE) == 1


def test_dense_engine_is_jittable_after_precompute():
    """A precomputed matrix can be reused from an enclosing jax.jit."""
    nside = 2
    lmax = 4
    npix = 12 * nside**2
    data = jnp.asarray(rng.standard_normal((2, npix)))
    precompute_dense_matrix((npix,), lmax, "healpix", nside=nside)

    transform = jax.jit(
        lambda maps: compute_alm(
            maps,
            lmax,
            "healpix",
            nside=nside,
            reality=True,
            engine="dense",
        )
    )

    expected = compute_alm(
        data, lmax, "healpix", nside=nside, reality=True, engine="s2fft"
    )
    np.testing.assert_allclose(
        transform(data), expected, rtol=1e-12, atol=1e-12
    )


def test_dense_engine_gradient_matches_s2fft():
    """Dense matrix multiplication should preserve end-to-end gradients."""
    nside = 2
    lmax = 4
    data = jnp.asarray(rng.standard_normal((2, 12 * nside**2)))

    # Warm the cache before tracing, as production SphBase objects do.
    compute_alm(
        data, lmax, "healpix", nside=nside, reality=True, engine="dense"
    ).block_until_ready()

    def loss(maps, engine):
        alm = compute_alm(
            maps, lmax, "healpix", nside=nside, reality=True, engine=engine
        )
        return jnp.sum(jnp.abs(alm) ** 2)

    expected = jax.grad(loss)(data, "s2fft")
    actual = jax.grad(loss)(data, "dense")
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_sphbase_dense_engine_precomputes_matrix():
    """SphBase should make dense transforms safe inside jitted methods."""
    data = jnp.asarray(rng.standard_normal((1, 48)))
    obj = SphBase(data, jnp.array([50.0]), "healpix", engine="dense")

    assert obj.engine == "dense"
    assert obj._dense_matrix.shape == (15, 48)


def test_invalid_sht_engine():
    """Unknown engine names should fail before any matrix construction."""
    with pytest.raises(ValueError, match="Unsupported SHT engine"):
        SphBase(jnp.ones((1, 48)), [50.0], "healpix", engine="unknown")


def test_explicit_lmax_is_healpix_only():
    """HEALPix can be truncated independently of its pixel resolution."""
    obj = SphBase(
        jnp.ones((1, 48)),
        jnp.array([50.0]),
        "healpix",
        lmax=3,
    )
    assert obj.lmax == 3
    assert obj._L == 4

    data = _make_data(4, "dh", 1)
    with pytest.raises(ValueError, match="only supported for HEALPix"):
        SphBase(data, jnp.array([50.0]), "dh", lmax=3)

    with pytest.raises(ValueError, match="non-negative"):
        SphBase(
            jnp.ones((1, 48)),
            jnp.array([50.0]),
            "healpix",
            lmax=-1,
        )


def test_dense_engine_supports_lmax_below_two_nside():
    """Dense HEALPix transforms should not inherit s2fft's low-L limit."""
    nside = 4
    lmax = 3
    temperature = 500.0
    data = temperature * jnp.ones((1, 12 * nside**2))

    alm = compute_alm(
        data,
        lmax,
        "healpix",
        nside=nside,
        reality=True,
        engine="dense",
    )

    assert alm.shape == (1, lmax + 1, 2 * lmax + 1)
    assert jnp.isclose(alm[0, 0, lmax].real, temperature / Y00, rtol=1e-12)


def test_dense_truncated_lmax_matches_truncated_s2fft():
    """A truncated dense transform is transform + low-pass in one step.

    With niter=0 every coefficient is an independent quadrature sum, so
    the dense engine at lmax < 2 * nside must equal the full-band s2fft
    transform truncated to the same lmax.
    """
    nside = 4
    lmax_full = 2 * nside
    lmax = 3
    data = jnp.asarray(rng.standard_normal((2, 12 * nside**2)))

    full = compute_alm(data, lmax_full, "healpix", nside=nside, reality=True)
    truncated = full[:, : lmax + 1, lmax_full - lmax : lmax_full + lmax + 1]

    actual = compute_alm(
        data, lmax, "healpix", nside=nside, reality=True, engine="dense"
    )

    np.testing.assert_allclose(actual, truncated, rtol=1e-12, atol=1e-12)
