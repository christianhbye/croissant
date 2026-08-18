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
@pytest.mark.parametrize(
    "sampling,expected_ntheta,expected_dtype",
    [
        ("healpix", 4 * NSIDE - 1, np.complex128),
        ("mwss", 2 * (LMAX + 1) + 1, np.float64),
        ("mw", 2 * (LMAX + 1) + 1, np.float64),
        ("dh", 2 * (LMAX + 1), np.float64),
        ("gl", LMAX + 1, np.float64),
    ],
)
def test_kernel_shape_and_size_prediction(
    sampling, expected_ntheta, expected_dtype, reality
):
    """kernel_nbytes predicts the footprint without building it, for
    every sampling scheme s2fft supports.

    Both the leading (theta) axis and the dtype are sampling-dependent
    (see ``footprints._kernel_ntheta`` and ``_kernel_itemsize``): only
    the HEALPix kernel is complex128, and ``mw``/``mwss`` build a
    larger leading axis (``2L + 1``) than their ordinary ``ntheta``
    would suggest. The last axis depends on ``reality``: a real-field
    kernel stores only m >= 0, so it is L wide rather than 2L-1. This
    is the test that compares against an actually-built kernel rather
    than re-deriving the prediction, so a wrong per-sampling formula
    cannot pass unnoticed.
    """
    nside = NSIDE if sampling == "healpix" else None
    predicted = kernel.kernel_nbytes(
        LMAX, sampling, nside=nside, reality=reality
    )
    k = kernel.precompute_kernel(
        LMAX, sampling, nside=nside, spin=0, reality=reality
    )
    L = LMAX + 1
    nm = L if reality else (2 * L - 1)
    assert k.shape == (expected_ntheta, L, nm)
    assert k.dtype == expected_dtype
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


def test_kernel_cache_key_tracks_the_x64_flag():
    """A kernel cached under one x64 setting must not be reused after
    the flag changes.

    jnp.asarray(built) bakes in the current x64 flag: a kernel built
    while it is off is complex64/float32, and reusing that array after
    jax.config.update("jax_enable_x64", True) would silently hand back
    an array with less precision than an x64 runtime callers expect.
    The cache key must include the resulting dtype (mirroring
    dense._dense_matrix_key, which includes dtype for exactly this
    reason) so toggling the flag misses the cache instead of reusing
    the old array.
    """
    import jax

    kernel.clear_kernel_cache()
    # Captured, not assumed: restoring a hardcoded True would silently
    # turn x64 ON for every later test whenever this file runs without
    # conftest.py's global enable, changing the dtype of every matrix
    # and kernel built for the rest of the session.
    was_enabled = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        low = kernel.precompute_kernel(LMAX, "healpix", nside=NSIDE, spin=0)
        assert low.dtype == np.complex64

        jax.config.update("jax_enable_x64", True)
        high = kernel.precompute_kernel(LMAX, "healpix", nside=NSIDE, spin=0)
        assert high.dtype == np.complex128
        assert high is not low
    finally:
        jax.config.update("jax_enable_x64", was_enabled)
        kernel.clear_kernel_cache()


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
    # Applied below with reality=False, so built with it too: the two
    # must agree or s2fft's einsum sees the wrong m axis.
    k = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, spin=spin, reality=False, forward=True
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

    Per ``utils.engine_dtypes``, croissant's engines reproduce
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
    # Index 5, not -1: the key's trailing entries are dtype and backend
    # (added so a kernel built under one x64/device setting is never
    # silently reused under another), so `forward` is no longer last.
    forwards = [key[5] for key in kernel._KERNEL_CACHE]
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


def test_beam_construction_populates_the_kernel_cache():
    """Regression test: constructing a kernel-engine Beam must actually
    cache a kernel, not merely return correct results.

    Beam.compute_alm is @jax.jit, so building the kernel lazily from
    inside it (the first implementation of this task) built the kernel
    as a jax tracer bound to that one trace and had to drop it rather
    than cache it, leaving _KERNEL_CACHE empty forever: correct output
    via XLA constant-folding, but none of precompute_kernel's cross-
    object sharing, and a multi-hundred-MiB kernel baked into every
    compiled executable at high nside. SphBase.__init__ now builds the
    kernel eagerly, before compute_alm ever runs, so the cache must be
    warm right after construction.
    """
    from croissant import Beam

    kernel.clear_kernel_cache()
    rng = np.random.default_rng(11)
    data = rng.normal(size=(2, 12 * NSIDE**2)) ** 2
    Beam(
        data,
        freqs=np.array([50.0, 60.0]),
        sampling="healpix",
        engine="kernel",
        niter=0,
    )
    assert len(kernel._KERNEL_CACHE) > 0


def test_two_beams_share_the_same_cached_kernel_object():
    """Cross-object sharing is the point of caching: two Beams with the
    same transform configuration must reuse one cached kernel array
    rather than each building (and each jit trace re-embedding) its
    own."""
    from croissant import Beam

    kernel.clear_kernel_cache()
    rng = np.random.default_rng(12)
    data = rng.normal(size=(2, 12 * NSIDE**2)) ** 2
    freqs = np.array([50.0, 60.0])
    beam1 = Beam(
        data, freqs=freqs, sampling="healpix", engine="kernel", niter=0
    )
    beam2 = Beam(
        data, freqs=freqs, sampling="healpix", engine="kernel", niter=0
    )
    shared = kernel.precompute_kernel(
        beam1.lmax, "healpix", nside=beam1.nside, spin=0, reality=True
    )
    assert beam1._kernel is shared
    assert beam2._kernel is shared


def test_compute_alm_inside_jit_without_precompute_raises():
    """engine="kernel" must refuse to build inside a caller's own
    jax.jit rather than silently caching a leaked tracer (the bug this
    fix round addresses). The caller must precompute_kernel(...)
    outside jax.jit and pass it in, exactly as engine="dense" requires
    for precompute_dense_matrix.
    """
    import jax

    from croissant import sphere

    kernel.clear_kernel_cache()
    rng = np.random.default_rng(14)
    data = rng.normal(size=(1, 12 * NSIDE**2))

    @jax.jit
    def call(x):
        return sphere.compute_alm(
            x,
            lmax=LMAX,
            sampling="healpix",
            nside=NSIDE,
            engine="kernel",
        )

    with pytest.raises(RuntimeError, match="precompute_kernel"):
        call(data)


def test_precompute_kernel_default_matches_the_apply_default():
    """The documented jit warm-up recipe must actually apply.

    ``kernel_compute_alm``, ``sphere.compute_alm`` and
    ``footprints.kernel_nbytes`` all default to ``reality=True``. A
    builder defaulting to False returns a kernel whose last axis is
    ``2L - 1`` where the apply path slices ``ftm`` to ``m >= 0`` and
    expects ``L``, so the README's own recipe raised a shape error --
    and the kernel engine has no pre-warmed-cache escape hatch, so this
    is the only supported path.
    """
    import jax

    from croissant import sphere

    kernel.clear_kernel_cache()
    rng = np.random.default_rng(21)
    data = rng.normal(size=(4, 12 * NSIDE**2))
    built = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, forward=True
    )

    @jax.jit
    def analyse(m):
        return sphere.compute_alm(
            m,
            LMAX,
            "healpix",
            nside=NSIDE,
            engine="kernel",
            kernel=built,
        )

    got = np.asarray(analyse(data))
    expected = np.asarray(
        sphere.compute_alm(data, LMAX, "healpix", nside=NSIDE, engine="s2fft")
    )
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )


def test_precompute_kernel_forces_reality_false_for_spin():
    """The builder applies the same rule the predictor and apply do.

    ``kernel_nbytes`` and ``kernel_compute_alm`` both force
    ``reality = reality and spin == 0`` because s2fft's real precompute
    path is only valid at spin 0. Leaving the builder out of that
    agreement is what lets a caller key, build and then fail to apply.
    """
    kernel.clear_kernel_cache()
    forced = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, spin=2, reality=True
    )
    explicit = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, spin=2, reality=False
    )
    assert forced is explicit
    assert forced.shape[-1] == 2 * (LMAX + 1) - 1


def test_sub_floor_band_limits_share_one_cached_kernel():
    """Two sub-floor band-limits build one kernel, so they must key alike.

    Both are built at ``transform_lmax(...) + 1``, so keying on the
    requested lmax stores byte-identical duplicates in a cache whose
    whole purpose is to hold a working set.
    """
    kernel.clear_kernel_cache()
    low = kernel.precompute_kernel(10, "healpix", nside=NSIDE)
    lower = kernel.precompute_kernel(14, "healpix", nside=NSIDE)
    assert low is lower
    assert len(kernel._KERNEL_CACHE) == 1


def test_rebuilding_a_polarized_pair_reuses_every_cached_kernel(monkeypatch):
    """One polarized simulation's working set must survive in the cache.

    A ``PairStokesBeam`` and a ``PolarizedSky`` at ``niter > 0`` need
    more kernels between them than the cache used to hold, so each
    construction evicted the other's. Results stayed correct -- live
    objects hold their own references -- but a parameter sweep or an MCMC
    step that rebuilds fields paid the full build cost every iteration,
    silently losing the reuse the cache exists to provide.
    """
    import s2fft.precompute_transforms.construct as construct

    from croissant.polarization import PairStokesBeam, PolarizedSky

    nside = 4
    npix = 12 * nside**2
    rng = np.random.default_rng(22)
    freqs = [10.0, 20.0]
    sky_data = rng.normal(size=(2, 4, npix))
    beam_data = rng.normal(size=(1, 2, 4, npix))

    def build_pair():
        PairStokesBeam(beam_data, freqs, [(0, 0)], sampling="healpix", niter=3)
        PolarizedSky(sky_data, freqs, sampling="healpix", niter=3)

    kernel.clear_kernel_cache()
    build_pair()

    calls = []
    real_builder = construct.spin_spherical_kernel

    def counting_builder(**kwargs):
        calls.append(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(construct, "spin_spherical_kernel", counting_builder)
    build_pair()
    assert calls == []
