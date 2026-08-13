"""
Tests for the precomputed-kernel spherical harmonic engine.

The engine caches s2fft's Wigner-d kernel and contracts it, instead of
recomputing the recursion per call (``engine="s2fft"``) or materialising
the whole operator (``engine="dense"``).

One test here is a regression guard rather than a behaviour test:
s2fft's *jax* kernel builder is silently wrong for spin != 0 on HEALPix
(it zeroes entries where the Price-McEwen recursion hits an exact
Wigner-d node), so croissant must build kernels with the *numpy*
builder. ``test_kernel_matches_on_the_fly_transform`` fails if a future
s2fft version breaks the builder we do rely on.
"""

import numpy as np
import pytest
import s2fft

from croissant import kernel

NSIDE = 8
LMAX = 2 * NSIDE - 1


@pytest.mark.parametrize("reality", [False, True])
def test_kernel_shape_and_size_prediction(reality):
    """kernel_nbytes predicts the footprint without building it.

    The last axis depends on ``reality``: a real-field kernel stores only
    m >= 0, so it is L wide rather than 2L-1.
    """
    predicted = kernel.kernel_nbytes(
        LMAX, "healpix", nside=NSIDE, reality=reality
    )
    k = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, spin=0, reality=reality
    )
    ntheta = 4 * NSIDE - 1
    nm = (LMAX + 1) if reality else (2 * LMAX + 1)
    assert k.shape == (ntheta, LMAX + 1, nm)
    assert predicted == k.nbytes


@pytest.mark.parametrize("reality", [False, True])
def test_kernel_is_built_at_the_healpix_floor(reality):
    """A band-limit below 2*nside-1 must still build at the floor.

    s2fft's HEALPix FFT requires L >= 2 * nside whatever band-limit the
    caller wants back, so the kernel is larger than the requested lmax
    implies. A regression dropping the floor would build a kernel s2fft
    cannot use, and kernel_nbytes would under-predict its size.
    """
    low_lmax = NSIDE - 1
    floor = kernel.transform_lmax(low_lmax, "healpix", nside=NSIDE)
    assert low_lmax < floor == 2 * NSIDE - 1
    predicted = kernel.kernel_nbytes(
        low_lmax, "healpix", nside=NSIDE, reality=reality
    )
    k = kernel.precompute_kernel(
        low_lmax, "healpix", nside=NSIDE, spin=0, reality=reality
    )
    nm = (floor + 1) if reality else (2 * floor + 1)
    assert k.shape == (4 * NSIDE - 1, floor + 1, nm)
    assert predicted == k.nbytes


def test_kernel_cache_returns_identical_object():
    """Repeated requests hit the cache rather than rebuilding."""
    kernel.clear_kernel_cache()
    first = kernel.precompute_kernel(LMAX, "healpix", nside=NSIDE, spin=2)
    second = kernel.precompute_kernel(LMAX, "healpix", nside=NSIDE, spin=2)
    assert first is second
    kernel.clear_kernel_cache()
    third = kernel.precompute_kernel(LMAX, "healpix", nside=NSIDE, spin=2)
    assert third is not first


@pytest.mark.parametrize("spin", [0, 2, -2])
def test_kernel_matches_on_the_fly_transform(spin):
    """Regression guard on the s2fft builder croissant depends on.

    s2fft's ``spin_spherical_kernel_jax`` drops modes for spin != 0 on
    HEALPix; the numpy ``spin_spherical_kernel`` does not. Croissant must
    use the latter. If this fails after an s2fft upgrade, the builder
    changed and the kernel engine cannot be trusted.
    """
    L = LMAX + 1
    rng = np.random.default_rng(3)
    flm = np.asarray(
        s2fft.utils.signal_generator.generate_flm(
            rng, L, spin=spin, reality=False
        )
    )
    field = np.asarray(
        s2fft.inverse(
            flm,
            L=L,
            spin=spin,
            nside=NSIDE,
            sampling="healpix",
            method="jax",
            reality=False,
        )
    )
    expected = np.asarray(
        s2fft.forward(
            field,
            L=L,
            spin=spin,
            nside=NSIDE,
            sampling="healpix",
            method="jax",
            reality=False,
            iter=0,
        )
    )
    k = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, spin=spin, forward=True
    )
    got = np.asarray(
        s2fft.precompute_transforms.spherical.forward(
            field,
            L=L,
            spin=spin,
            kernel=k,
            sampling="healpix",
            reality=False,
            method="jax",
            nside=NSIDE,
            iter=0,
        )
    )
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )


def test_kernel_compute_alm_matches_s2fft_engine_scalar():
    """The kernel engine reproduces the s2fft engine for real scalars,
    including the batch axis and the returned layout."""
    from croissant import sphere

    rng = np.random.default_rng(4)
    data = rng.normal(size=(3, 12 * NSIDE**2))
    kwargs = dict(lmax=LMAX, sampling="healpix", nside=NSIDE, niter=0)
    expected = np.asarray(sphere.compute_alm(data, engine="s2fft", **kwargs))
    got = np.asarray(kernel.kernel_compute_alm(data, **kwargs))
    assert got.shape == expected.shape == (3, LMAX + 1, 2 * LMAX + 1)
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )


def test_kernel_engine_follows_the_dtype_contract():
    """The engines share a dtype policy, so the kernel engine must too.

    Per ``sphere._dense_dtypes``, croissant's engines reproduce
    ``s2fft.forward``: complex128 out on an x64 runtime even for float32
    maps. A kernel engine that instead inherited the input dtype would
    silently change precision downstream, which the alm-value
    equivalence tests would not catch because they compare at x64.
    """
    from croissant import sphere

    rng = np.random.default_rng(13)
    for input_dtype in (np.float32, np.float64):
        data = rng.normal(size=(2, 12 * NSIDE**2)).astype(input_dtype)
        kwargs = dict(lmax=LMAX, sampling="healpix", nside=NSIDE, niter=0)
        expected = sphere.compute_alm(data, engine="s2fft", **kwargs)
        got = kernel.kernel_compute_alm(data, **kwargs)
        assert got.dtype == expected.dtype, (
            f"kernel engine returned {got.dtype} for {input_dtype} input, "
            f"s2fft engine returned {expected.dtype}"
        )


@pytest.mark.parametrize("spin", [2, -2])
def test_kernel_compute_alm_matches_s2fft_engine_spin(spin):
    """The kernel engine reproduces the s2fft engine for spin fields."""
    from croissant import sphere

    rng = np.random.default_rng(5)
    npix = 12 * NSIDE**2
    data = rng.normal(size=(2, npix)) + 1j * rng.normal(size=(2, npix))
    kwargs = dict(
        lmax=LMAX,
        sampling="healpix",
        nside=NSIDE,
        niter=0,
        spin=spin,
        reality=False,
    )
    expected = np.asarray(sphere.compute_alm(data, engine="s2fft", **kwargs))
    got = np.asarray(kernel.kernel_compute_alm(data, **kwargs))
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )


def test_refinement_converges_towards_a_band_limited_signal():
    """Refinement must actually converge, monotonically.

    HEALPix has no exact quadrature, so analysing a synthesised
    band-limited field does not recover its coefficients exactly. Each
    refinement step should shrink that error. s2fft's own precompute
    refinement fails this for spin != 0 -- it diverges -- which is why
    croissant runs the iteration itself.
    """
    L = LMAX + 1
    spin = 2
    rng = np.random.default_rng(6)
    flm = np.asarray(
        s2fft.utils.signal_generator.generate_flm(
            rng, L, spin=spin, reality=False
        )
    )
    field = np.asarray(
        s2fft.inverse(
            flm,
            L=L,
            spin=spin,
            nside=NSIDE,
            sampling="healpix",
            method="jax",
            reality=False,
        )
    )
    errors = []
    for niter in (0, 1, 2, 3):
        got = np.asarray(
            kernel.kernel_compute_alm(
                field[None],
                lmax=LMAX,
                sampling="healpix",
                nside=NSIDE,
                niter=niter,
                spin=spin,
                reality=False,
            )
        )[0]
        errors.append(np.abs(got - flm).max())
    for previous, current in zip(errors, errors[1:]):
        assert current < previous, f"refinement not converging: {errors}"
    assert errors[-1] < errors[0] / 10


@pytest.mark.parametrize("niter", [1, 3])
@pytest.mark.parametrize("spin", [0, 2])
def test_refined_kernel_engine_matches_s2fft_engine(niter, spin):
    """With refinement on, the kernel engine still matches s2fft's."""
    from croissant import sphere

    rng = np.random.default_rng(7)
    npix = 12 * NSIDE**2
    if spin == 0:
        data = rng.normal(size=(2, npix))
        reality = True
    else:
        data = rng.normal(size=(2, npix)) + 1j * rng.normal(size=(2, npix))
        reality = False
    kwargs = dict(
        lmax=LMAX,
        sampling="healpix",
        nside=NSIDE,
        niter=niter,
        spin=spin,
        reality=reality,
    )
    expected = np.asarray(sphere.compute_alm(data, engine="s2fft", **kwargs))
    got = np.asarray(kernel.kernel_compute_alm(data, **kwargs))
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )


def test_inverse_kernel_is_not_built_when_niter_is_zero():
    """niter=0 must not pay for the synthesis kernel."""
    kernel.clear_kernel_cache()
    rng = np.random.default_rng(8)
    data = rng.normal(size=(1, 12 * NSIDE**2))
    kernel.kernel_compute_alm(
        data, lmax=LMAX, sampling="healpix", nside=NSIDE, niter=0
    )
    forwards = [key[-1] for key in kernel._KERNEL_CACHE]
    assert forwards == [True]


def test_sphbase_accepts_the_kernel_engine():
    """Beam construction works with engine="kernel" and reports it."""
    from croissant import Beam

    rng = np.random.default_rng(9)
    data = rng.normal(size=(2, 12 * NSIDE**2)) ** 2
    beam = Beam(
        data,
        freqs=np.array([50.0, 60.0]),
        sampling="healpix",
        engine="kernel",
        niter=0,
    )
    assert beam.engine == "kernel"
    alm = beam.compute_alm()
    assert alm.shape[-2:] == (beam.lmax + 1, 2 * beam.lmax + 1)


def test_full_pipeline_visibilities_agree_across_engines():
    """One end-to-end check that the engines are interchangeable.

    The transform-level equivalence tests already pin the alm values, and
    everything downstream of the transform -- the convolve einsum, the
    phase rotation, the beam-integral normalisation -- is
    engine-independent, so this cannot fail on alm values alone. It is
    here to catch what those tests structurally cannot: dtype
    promotion through the pipeline, tracer/jit interaction (the dense
    engine raises inside jit unless precomputed, the kernel engine
    builds lazily), and cache side effects across repeated calls.

    Deliberately ONE test rather than parametrising
    ``tests/test_physics.py`` over engines: the physics file is
    ground-truth and is not to be modified, and tripling it would cost
    ~90 s to re-test a corollary of the equivalence theorem.
    """
    import jax.numpy as jnp
    from astropy.time import Time as AstroTime

    from croissant import Beam, Simulator, Sky

    npix = 12 * 8**2
    freqs = jnp.linspace(50.0, 250.0, 3)
    t0 = AstroTime("2022-01-01 00:00:00")
    times_jd = jnp.linspace(t0.jd, t0.jd + 0.5, 4, endpoint=False)
    beam_data = jnp.ones((len(freqs), npix))
    tsky = 1e4 * (freqs / 150.0) ** (-2.5)
    sky_data = tsky[:, None] * jnp.ones((npix,))

    visibilities = {}
    for engine in ("s2fft", "kernel", "dense"):
        beam = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            niter=0,
            engine=engine,
        )
        sky = Sky(
            sky_data,
            freqs,
            sampling="healpix",
            coord="galactic",
            niter=0,
            engine=engine,
        )
        sim = Simulator(beam, sky, times_jd, freqs, 0.0, 0.0, world="earth")
        visibilities[engine] = np.asarray(sim.sim())

    reference = visibilities["s2fft"]
    for engine, got in visibilities.items():
        assert got.dtype == reference.dtype, (
            f"engine {engine!r} changed the visibility dtype: "
            f"{got.dtype} vs {reference.dtype}"
        )
        np.testing.assert_allclose(
            got,
            reference,
            rtol=0,
            atol=1e-9 * np.abs(reference).max(),
            err_msg=f"engine {engine!r} changed the visibilities",
        )


def test_unknown_engine_is_rejected_with_the_full_list():
    from croissant import sphere

    rng = np.random.default_rng(10)
    data = rng.normal(size=(1, 12 * NSIDE**2))
    with pytest.raises(ValueError, match="kernel"):
        sphere.compute_alm(
            data,
            lmax=LMAX,
            sampling="healpix",
            nside=NSIDE,
            engine="nonsense",
        )
