"""
Tests for automatic SHT engine selection.

The engines agree numerically, so choosing between them is a resource
decision and croissant can own it. These tests pin the policy as a
table: what matters is that the choice is predictable, never exceeds the
memory cap, and degrades to the matrix-free engine when precomputing
cannot pay for itself.
"""

import pytest

from croissant import engine_select, footprints


def test_single_transform_uses_the_matrix_free_engine():
    """One transform has nothing to amortise a precompute against."""
    engine, reason = engine_select.resolve_engine(
        lmax=63, sampling="healpix", nside=32, batch_size=1
    )
    assert engine == "s2fft"
    assert reason


def test_repeated_transforms_prefer_the_kernel():
    """A batch amortises the kernel build."""
    engine, _ = engine_select.resolve_engine(
        lmax=63, sampling="healpix", nside=32, batch_size=64
    )
    assert engine == "kernel"


@pytest.mark.parametrize("niter", [0, 1, 3])
def test_refinement_does_not_by_itself_select_dense(niter):
    """niter>0 is not a reason for auto to pay for the dense operator.

    Dense does win on per-call cost at niter>0 (2.4x-8.3x in Task 6),
    but its build costs roughly 1.0x-25x more (nside=16/scalar builds
    are near-identical; nside=16/spin=2 costs 24.8x more), and the
    break-even call counts from the same measurements span 7 to 92338
    calls. Croissant transforms once at construction, below even the
    smallest of those, so that per-call saving is never repaid.
    Choosing dense for throughput needs knowledge only the caller has,
    and stays an explicit override.
    """
    engine, _ = engine_select.resolve_engine(
        lmax=31,
        sampling="healpix",
        nside=16,
        niter=niter,
        batch_size=64,
    )
    assert engine == "kernel"


def test_amortisation_threshold_scales_with_kernel_size():
    """The batch a kernel needs grows with the kernel, not a constant.

    Measured crossovers: batch 1 at nside=8 and nside=16 (kernels of 0.12
    and 0.98 MiB), batch 8 at nside=32 (7.94 MiB). A fixed threshold
    cannot serve both, so a batch of 4 must be enough at nside=16 and not
    enough at nside=32.
    """
    small, _ = engine_select.resolve_engine(
        lmax=31, sampling="healpix", nside=16, batch_size=4
    )
    large, reason = engine_select.resolve_engine(
        lmax=63, sampling="healpix", nside=32, batch_size=4
    )
    assert small == "kernel"
    assert large == "s2fft"
    assert "amortise" in reason


def test_low_bandlimit_on_a_high_resolution_map_selects_dense():
    """Dense's clearest remaining advantage: it can build at the HEALPix
    L >= 2*nside floor and keep only the requested low-ell rows, which
    the kernel engine cannot do."""
    engine, reason = engine_select.resolve_engine(
        lmax=15, sampling="healpix", nside=64, batch_size=64
    )
    assert engine == "dense"
    assert "floor" in reason


def test_low_bandlimit_falls_back_when_dense_will_not_fit():
    engine, _ = engine_select.resolve_engine(
        lmax=15,
        sampling="healpix",
        nside=512,
        batch_size=64,
    )
    assert engine == "s2fft"


def test_nothing_exceeds_the_memory_cap():
    """No configuration may auto-select a precomputing engine whose
    footprint exceeds the cap.

    Covers both spin=0 and spin=2 at every nside: a spin field's kernel
    cannot use the real-valued (m >= 0) storage (see
    ``footprints.kernel_nbytes``), so the spin=2 cases exercise the path
    where an under-prediction would most easily slip past the cap.
    """
    from croissant import kernel

    for nside in (8, 16, 32, 64, 128):
        lmax = 2 * nside - 1
        for spin in (0, 2):
            engine, _ = engine_select.resolve_engine(
                lmax=lmax,
                sampling="healpix",
                nside=nside,
                spin=spin,
                niter=3,
                batch_size=1024,
            )
            if engine == "dense":
                assert (
                    footprints.dense_nbytes(
                        lmax, "healpix", nside=nside, spin=spin
                    )
                    <= engine_select.DEFAULT_MEMORY_CAP_BYTES
                )
            elif engine == "kernel":
                # reality=True to match resolve_engine's own default,
                # which is what it actually used to decide "kernel";
                # kernel_nbytes itself defaults to reality=False and
                # would over-predict for spin=0. niter=3 above means
                # kernel_compute_alm builds a forward AND an inverse
                # kernel of matching size for the refinement iteration
                # (see engine_select.resolve_engine), so the resident
                # total is double a single kernel_nbytes call.
                assert (
                    2
                    * kernel.kernel_nbytes(
                        lmax,
                        "healpix",
                        nside=nside,
                        spin=spin,
                        reality=True,
                    )
                    <= engine_select.DEFAULT_MEMORY_CAP_BYTES
                )


@pytest.mark.parametrize("spin", [0, 2])
def test_kernel_nbytes_matches_a_built_kernel(spin):
    """Ground truth check: the prediction must match a real build.

    test_nothing_exceeds_the_memory_cap only re-derives the expected
    byte count with the same footprints function resolve_engine calls
    internally, so a wrong formula there could pass unnoticed by that
    test alone. This test instead builds the kernel with
    precompute_kernel and compares the prediction against its actual
    ``.nbytes``, at nside=8 so the build is fast.

    This is the test that would have caught kernel_nbytes lacking a
    ``spin`` parameter: ``kernel_compute_alm`` forces
    ``reality = reality and spin == 0`` because s2fft's real-valued
    (m >= 0) storage has no meaning for a spin field, so the kernel
    actually built for a requested ``reality=True`` at spin=2 is twice
    the size a prediction that ignored spin would give.
    """
    from croissant import kernel

    nside = 8
    lmax = 2 * nside - 1
    # What kernel_compute_alm actually builds for a caller requesting
    # reality=True: downgraded to False whenever spin != 0.
    built_reality = spin == 0
    built = kernel.precompute_kernel(
        lmax, "healpix", nside=nside, spin=spin, reality=built_reality
    )
    predicted = kernel.kernel_nbytes(
        lmax, "healpix", nside=nside, spin=spin, reality=True
    )
    assert predicted == built.nbytes


def test_a_tiny_cap_forces_the_matrix_free_engine():
    engine, _ = engine_select.resolve_engine(
        lmax=63,
        sampling="healpix",
        nside=32,
        niter=3,
        batch_size=1024,
        memory_cap=1024,
    )
    assert engine == "s2fft"


def test_dense_footprint_beats_kernel_only_at_small_nside():
    """The O(nside**4) vs O(nside**3) crossover the policy relies on."""
    from croissant import kernel

    small = footprints.dense_nbytes(
        15, "healpix", nside=8
    ) / kernel.kernel_nbytes(15, "healpix", nside=8)
    large = footprints.dense_nbytes(
        127, "healpix", nside=64
    ) / kernel.kernel_nbytes(127, "healpix", nside=64)
    assert large > small


@pytest.mark.parametrize("engine", ["s2fft", "kernel", "dense"])
def test_explicit_choices_are_returned_unchanged(engine):
    got, reason = engine_select.resolve_engine(
        lmax=63,
        sampling="healpix",
        nside=32,
        requested=engine,
    )
    assert got == engine
    assert "explicit" in reason


def test_auto_resolves_to_a_concrete_engine_on_the_object():
    """A Beam built with engine="auto" reports the mechanism it chose,
    not the word "auto" -- otherwise performance questions and bug
    reports cannot say which path ran."""
    import numpy as np

    from croissant import Beam

    nside = 8
    rng = np.random.default_rng(11)
    data = rng.normal(size=(2, 12 * nside**2)) ** 2
    beam = Beam(
        data,
        freqs=np.array([50.0, 60.0]),
        sampling="healpix",
        engine="auto",
    )
    assert beam.engine in {"s2fft", "kernel", "dense"}
    assert isinstance(beam.engine_reason, str) and beam.engine_reason


def test_auto_agrees_with_every_explicit_engine():
    """auto cannot change results, only cost."""
    import numpy as np

    from croissant import sphere

    nside, lmax = 8, 15
    rng = np.random.default_rng(12)
    data = rng.normal(size=(4, 12 * nside**2))
    kwargs = dict(lmax=lmax, sampling="healpix", nside=nside, niter=0)
    auto = np.asarray(sphere.compute_alm(data, engine="auto", **kwargs))
    for engine in ("s2fft", "kernel", "dense"):
        got = np.asarray(sphere.compute_alm(data, engine=engine, **kwargs))
        np.testing.assert_allclose(
            auto, got, rtol=0, atol=1e-10 * np.abs(got).max()
        )


def test_auto_is_the_default_engine():
    """Constructing without an explicit engine resolves automatically.

    The default was `"s2fft"` while `"auto"` was being validated. Flipping
    it changes performance for every existing caller and nothing else --
    the engines agree numerically -- so this test pins that the default
    now delegates the choice rather than hardcoding the matrix-free path.
    """
    import jax.numpy as jnp
    import numpy as np

    from croissant import Beam, Sky, sphere

    nside = 8
    npix = 12 * nside**2
    freqs = jnp.linspace(50.0, 150.0, 16)
    data = jnp.ones((len(freqs), npix))

    beam = Beam(data, freqs, sampling="healpix")
    sky = Sky(data, freqs, sampling="healpix", coord="galactic")
    for obj in (beam, sky):
        assert obj.engine in {"s2fft", "kernel", "dense"}
        assert "explicit request" not in obj.engine_reason

    # A batch this size at nside=8 amortises the kernel, so auto must
    # pick it -- proving the default really delegates.
    assert beam.engine == "kernel"

    # compute_alm's own default must delegate identically.
    resolved, _ = engine_select.resolve_engine(
        lmax=15, sampling="healpix", nside=nside, batch_size=len(freqs)
    )
    auto = sphere.compute_alm(data, lmax=15, sampling="healpix", nside=nside)
    explicit = sphere.compute_alm(
        data, lmax=15, sampling="healpix", nside=nside, engine=resolved
    )
    np.testing.assert_allclose(
        np.asarray(auto), np.asarray(explicit), rtol=0, atol=0
    )


def test_reason_strings_are_readable_for_tiny_and_singular_cases():
    """A reason must never read as "0.0 MiB" or "1 transforms".

    Sub-0.05 MiB kernels round to zero under plain MiB formatting, which
    reads as though the precompute were free, and a batch of one produced
    a plural. Both are user-facing strings shown by `engine_reason`.
    """
    _, reason = engine_select.resolve_engine(
        lmax=3, sampling="healpix", nside=2, batch_size=1
    )
    assert "0.0 MiB" not in reason, reason
    assert "1 transforms" not in reason, reason
    assert "1 transform" in reason, reason


def test_kernel_and_dense_size_predictors_share_a_reality_default():
    """The two footprint predictors must agree on their default.

    `reality=True` is the common case (real scalar fields), and
    `kernel_nbytes` applies the engine's own `reality and spin == 0` rule
    internally, so a True default is correct for scalar AND for spin,
    where it is forced back to False. A False default silently
    over-predicts the scalar kernel by 2x, which is how three separate
    call sites came to compare a packed dense operator against a
    full-size kernel.
    """
    import inspect

    from croissant import footprints

    k = inspect.signature(footprints.kernel_nbytes).parameters["reality"]
    d = inspect.signature(footprints.dense_nbytes).parameters["reality"]
    assert k.default == d.default is True

    # And the defaulted call must match what the scalar engine builds.
    from croissant import kernel

    built = kernel.precompute_kernel(
        15, "healpix", nside=8, spin=0, reality=True
    )
    assert footprints.kernel_nbytes(15, "healpix", nside=8) == built.nbytes


def test_polarized_analysis_pins_the_matrix_free_engine():
    """The polarized path must not inherit the ``"auto"`` default.

    `polarization._analysis_alm` runs inside jitted code and never routes
    through `SphBase.__init__`, which is what eagerly builds and threads
    the kernel engine's kernels -- and the kernel engine raises inside a
    trace when they are absent. Flipping the global default to `"auto"`
    broke three polarized tests exactly this way, so pin the dependency
    rather than rediscovering it next time a default moves.
    """
    import inspect

    from croissant import polarization

    src = inspect.getsource(polarization._analysis_alm)
    assert 'engine="s2fft"' in src, (
        "polarization._analysis_alm must pass an explicit engine; "
        "inheriting the auto default routes it into the kernel engine, "
        "which cannot build inside a jax trace"
    )


def test_auto_degrades_gracefully_inside_jit():
    """Auto must never pick an engine that then refuses to run.

    The kernel engine raises inside a trace when its kernels were not
    precomputed, which `SphBase` does eagerly but a bare `compute_alm`
    call cannot. Before `"auto"` became the default, jitting a function
    around `compute_alm` worked because the default was the matrix-free
    engine; auto must preserve that rather than turning ordinary user code
    into a RuntimeError. Falling back is safe because the engines agree
    numerically -- only cost differs.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from croissant import kernel, sphere

    nside, lmax = 32, 63
    npix = 12 * nside**2
    data = jnp.asarray(np.random.default_rng(0).normal(size=(64, npix)))
    # This configuration is one auto resolves to "kernel".
    assert (
        engine_select.resolve_engine(
            lmax=lmax, sampling="healpix", nside=nside, batch_size=64
        )[0]
        == "kernel"
    )

    @jax.jit
    def analyse(m):
        return sphere.compute_alm(m, lmax, "healpix", nside=nside)

    kernel.clear_kernel_cache()
    got = np.asarray(analyse(data))
    expected = np.asarray(
        sphere.compute_alm(data, lmax, "healpix", nside=nside, engine="s2fft")
    )
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )


def test_explicit_kernel_inside_jit_still_raises():
    """An explicit engine choice is honoured strictly, not softened.

    Only the automatic path degrades. A caller who asked for "kernel" by
    name gets the RuntimeError telling them to precompute, because
    silently giving them a different engine would hide the cost decision
    they made deliberately.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest as _pytest

    from croissant import kernel, sphere

    nside, lmax = 8, 15
    data = jnp.asarray(
        np.random.default_rng(0).normal(size=(4, 12 * nside**2))
    )

    @jax.jit
    def analyse(m):
        return sphere.compute_alm(
            m, lmax, "healpix", nside=nside, engine="kernel"
        )

    kernel.clear_kernel_cache()
    with _pytest.raises(RuntimeError, match="precompute"):
        analyse(data)
