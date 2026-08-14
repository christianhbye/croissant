"""
Tests for automatic SHT engine selection.

The engines agree numerically, so choosing between them is a resource
decision and croissant can own it. These tests pin the policy as a
table: what matters is that the choice is predictable, never exceeds the
memory cap, and degrades to the matrix-free engine when precomputing
cannot pay for itself.
"""

import math

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


def test_low_bandlimit_never_selects_an_engine_that_cannot_run():
    """Below the floor, dense is the only engine that works at all.

    An oversized dense operator is a reason to say so in the reason
    string, not a reason to hand back the matrix-free engine: s2fft
    cannot perform a HEALPix transform below ``L >= 2 * nside`` under any
    memory budget, so falling back to it turns a large allocation into a
    hard failure at the first ``compute_alm``. The memory cap is a
    policy; running at all is a correctness requirement, and the policy
    does not get to override it.
    """
    engine, reason = engine_select.resolve_engine(
        lmax=15,
        sampling="healpix",
        nside=512,
        batch_size=64,
    )
    assert engine == "dense"
    assert "floor" in reason


def test_the_matrix_free_engine_cannot_serve_a_sub_floor_band_limit():
    """The premise the rule above rests on, pinned directly."""
    import jax.numpy as jnp
    import numpy as np

    from croissant import sphere

    nside = 8
    data = jnp.asarray(
        np.random.default_rng(1).normal(size=(2, 12 * nside**2))
    )
    with pytest.raises(ValueError):
        sphere.compute_alm(data, 10, "healpix", nside=nside, engine="s2fft")


def test_amortisation_threshold_counts_the_inverse_kernel():
    """At niter > 0 both kernels are resident, so both must be paid for.

    ``kernel_fits`` already tests the doubled footprint. The threshold
    read the undoubled one a line later, so a batch large enough to
    amortise a single kernel was accepted for a configuration that
    builds two.
    """
    nside, lmax = 32, 63
    single = footprints.kernel_nbytes(lmax, "healpix", nside=nside)
    # Big enough to pay for one kernel, too small to pay for both.
    batch = math.ceil(single / 1024**2) + 1
    engine, reason = engine_select.resolve_engine(
        lmax=lmax,
        sampling="healpix",
        nside=nside,
        niter=3,
        batch_size=batch,
    )
    assert engine == "s2fft", reason


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
                # Both sides take the shared reality default, which is
                # what resolve_engine actually used to decide "kernel".
                # niter=3 above means kernel_compute_alm builds a
                # forward AND an inverse kernel of matching size for the
                # refinement iteration (see
                # engine_select.resolve_engine), so the resident total
                # is double a single kernel_nbytes call.
                assert (
                    2
                    * kernel.kernel_nbytes(
                        lmax,
                        "healpix",
                        nside=nside,
                        spin=spin,
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


def test_size_predictors_share_the_transform_reality_default():
    """A predictor's default must be the default of what it predicts.

    The two predictors have to agree with each other, or a call site
    that defaults both compares a packed operator against a full-size
    one -- which is how three separate call sites once came to do
    exactly that. They also have to agree with `sphere.compute_alm`,
    which assumes nothing about the caller's data: a `True` default here
    would under-predict the defaulted transform by 2x, and the objects
    that do know their field is real (`SphBase`, and the polarized
    containers) pass `reality=True` explicitly at every site.
    """
    import inspect

    from croissant import footprints, sphere

    k = inspect.signature(footprints.kernel_nbytes).parameters["reality"]
    d = inspect.signature(footprints.dense_nbytes).parameters["reality"]
    t = inspect.signature(sphere.compute_alm).parameters["reality"]
    assert k.default == d.default == t.default is False

    # And the defaulted call must match what the defaulted engine builds.
    from croissant import kernel

    built = kernel.precompute_kernel(15, "healpix", nside=8, spin=0)
    assert footprints.kernel_nbytes(15, "healpix", nside=8) == built.nbytes

    # The real-field claim must stay consistent end to end too.
    packed = kernel.precompute_kernel(
        15, "healpix", nside=8, spin=0, reality=True
    )
    assert (
        footprints.kernel_nbytes(15, "healpix", nside=8, reality=True)
        == packed.nbytes
    )


def test_polarized_fields_reach_the_kernel_engine():
    """The kernel engine must be reachable from a polarized field.

    Polarized HEALPix is part of what the kernel engine was built for,
    and it was unreachable for as long as `polarization._analysis_alm`
    pinned the matrix-free engine: the polarized classes now resolve and
    thread kernels themselves, so `auto` can pick it. This asserts both
    halves -- the resolved name and a kernel actually built -- because
    resolving to "kernel" while threading None would silently degrade
    inside `sphere.compute_alm`.
    """
    import numpy as np

    from croissant.polarization import PolarizedSky

    nside = 8
    data = np.random.default_rng(0).normal(size=(4, 4, 12 * nside**2))
    sky = PolarizedSky(data, [10.0, 20.0, 30.0, 40.0], sampling="healpix")

    assert sky.engine["IV"] == "kernel"
    assert sky.engine["P_MINUS"] == "kernel"
    assert sky._kernels[0] is not None
    assert sky._kernels[1] is not None


def test_polarized_blocks_resolve_independently():
    """One object must be able to report two different engines.

    A real sky's spin-0 kernel packs to ``m >= 0`` and so is about half
    the size of a spin-weighted one, and the I/V block is batched over
    twice as many maps as each Q/U block. At nside=16 with a single
    frequency those two facts land on opposite sides of the
    amortisation threshold, so a field that resolved once for the whole
    object could not produce this split whichever way it decided.
    """
    import numpy as np

    from croissant import footprints
    from croissant.polarization import PolarizedSky

    nside = 16
    lmax = 2 * nside
    assert footprints.kernel_nbytes(
        lmax, "healpix", nside=nside, spin=0, reality=True
    ) < footprints.kernel_nbytes(
        lmax, "healpix", nside=nside, spin=-2, reality=False
    ), "test needs the spin-0 block to be the cheaper one"

    data = np.random.default_rng(1).normal(size=(1, 4, 12 * nside**2))
    sky = PolarizedSky(data, [10.0], sampling="healpix")

    assert sky.engine["IV"] == "kernel"
    assert sky.engine["P_MINUS"] == "s2fft"
    assert sky._kernels[0] is not None
    assert sky._kernels[1] is None


def test_polarized_auto_degrades_inside_jit():
    """Auto must not break a polarized field constructed inside a trace.

    Kernels cannot be built while a trace is active, and constructing
    these objects inside `jax.jit` worked before they became
    engine-selectable, so the automatic choice degrades to the
    matrix-free engine instead of raising.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from croissant import kernel
    from croissant.polarization import PolarizedSky

    nside = 8
    freqs = [10.0, 20.0, 30.0, 40.0]
    data = jnp.asarray(
        np.random.default_rng(2).normal(size=(4, 4, 12 * nside**2))
    )

    @jax.jit
    def analyse(maps):
        return PolarizedSky(maps, freqs, sampling="healpix").compute_alm()

    kernel.clear_kernel_cache()
    got = np.asarray(analyse(data))
    expected = np.asarray(
        PolarizedSky(
            data, freqs, sampling="healpix", engine="s2fft"
        ).compute_alm()
    )
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )


def test_polarized_explicit_kernel_inside_jit_raises():
    """An explicit polarized engine choice is honoured strictly.

    The mirror of `test_explicit_kernel_inside_jit_still_raises` for the
    polarized classes: only the automatic path degrades.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from croissant import kernel
    from croissant.polarization import PolarizedSky

    nside = 8
    data = jnp.asarray(
        np.random.default_rng(3).normal(size=(4, 4, 12 * nside**2))
    )

    @jax.jit
    def analyse(maps):
        return PolarizedSky(
            maps,
            [10.0, 20.0, 30.0, 40.0],
            sampling="healpix",
            engine="kernel",
        ).compute_alm()

    kernel.clear_kernel_cache()
    with pytest.raises(RuntimeError, match="precompute_kernel"):
        analyse(data)


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


def test_scalar_field_auto_degrades_inside_jit():
    """`SphBase` must degrade like everything else `auto` touches.

    `sphere.compute_alm` already degrades an automatic kernel choice
    inside a trace, but `SphBase.__init__` used to raise instead. That
    contradicted the same invariant one screen up in its own module, and
    made a `Sky` built inside a `jax.grad` fail with a message about a
    precompute the caller never asked for -- the scalar twin of the
    polarized gradient path exercised in test_polarization.py.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from croissant import kernel
    from croissant.sky import Sky

    nside = 8
    freqs = jnp.asarray([10.0, 20.0])
    data = jnp.asarray(
        np.random.default_rng(4).normal(size=(2, 12 * nside**2))
    )

    def loss(maps):
        alm = Sky(maps, freqs, sampling="healpix").compute_alm()
        return jnp.sum(jnp.abs(alm) ** 2)

    kernel.clear_kernel_cache()
    gradient = jax.grad(loss)(data)
    assert gradient.shape == data.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_auto_degrades_when_only_the_forward_kernel_is_available():
    """``niter > 0`` needs two kernels, so one is not enough to proceed.

    The degrade check tested only the forward kernel, so an automatic
    kernel choice with a forward kernel threaded in still raised inside
    the caller's jit when the synthesis kernel was missing -- the same
    RuntimeError the degrade exists to prevent, one argument over.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from croissant import kernel, sphere

    nside, lmax = 8, 15
    data = jnp.asarray(
        np.random.default_rng(6).normal(size=(4, 12 * nside**2))
    )
    kernel.clear_kernel_cache()
    forward_only = kernel.precompute_kernel(
        lmax, "healpix", nside=nside, spin=0, reality=True, forward=True
    )

    @jax.jit
    def analyse(m):
        return sphere.compute_alm(
            m, lmax, "healpix", nside=nside, niter=3, kernel=forward_only
        )

    got = np.asarray(analyse(data))
    assert got.shape == (4, lmax + 1, 2 * lmax + 1)
    assert bool(np.all(np.isfinite(got)))


def test_auto_spin_below_the_floor_works_inside_jit():
    """The one configuration auto reaches for dense must survive a trace.

    ``croissant.dense`` builds under ``jax.ensure_compile_time_eval`` and
    so is perfectly buildable inside a trace; degrading it to s2fft --
    which cannot serve a sub-floor band-limit at all -- converted a
    working call into a shape error from inside s2fft.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from croissant import sphere

    nside, lmax = 8, 10
    rng = np.random.default_rng(7)
    npix = 12 * nside**2
    data = jnp.asarray(
        rng.normal(size=(4, npix)) + 1j * rng.normal(size=(4, npix))
    )

    @jax.jit
    def analyse(m):
        return sphere.compute_alm(
            m, lmax, "healpix", nside=nside, spin=2, reality=False
        )

    got = np.asarray(analyse(data))
    assert got.shape == (4, lmax + 1, 2 * lmax + 1)
    assert bool(np.all(np.isfinite(got)))


def test_scalar_field_auto_dense_degrades_inside_jit():
    """The dense twin of ``test_scalar_field_auto_degrades_inside_jit``.

    A sub-floor band-limit is the one case auto reaches for dense, so a
    ``Sky`` built inside ``jax.grad`` with a low explicit lmax resolved
    to dense and then raised about a precompute the caller never asked
    for. The kernel branch had already been fixed; the dense branch one
    screen down had not.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from croissant import kernel
    from croissant.sky import Sky

    nside = 8
    freqs = jnp.asarray([10.0, 20.0])
    data = jnp.asarray(
        np.random.default_rng(8).normal(size=(2, 12 * nside**2))
    )

    def loss(maps):
        alm = Sky(maps, freqs, sampling="healpix", lmax=10).compute_alm()
        return jnp.sum(jnp.abs(alm) ** 2)

    kernel.clear_kernel_cache()
    gradient = jax.grad(loss)(data)
    assert gradient.shape == data.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_polarized_fields_reject_engine_none_like_scalar_fields():
    """Both entry points must agree on what a valid engine argument is.

    ``resolve_engine`` treats ``None`` as "auto" for its own internal
    callers, but a field constructor is a public API: ``Sky`` rejects
    ``engine=None`` and the polarized classes silently auto-selected, so
    threading an optional engine through a config object behaved
    differently depending on which class received it.
    """
    import numpy as np

    from croissant.polarization import PolarizedSky

    nside = 8
    data = np.random.default_rng(9).normal(size=(1, 4, 12 * nside**2))
    with pytest.raises(ValueError, match="engine"):
        PolarizedSky(data, [10.0], sampling="healpix", engine=None)


def test_scalar_field_explicit_kernel_inside_jit_raises():
    """An explicit scalar engine choice is still never softened."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from croissant import kernel
    from croissant.sky import Sky

    nside = 8
    freqs = jnp.asarray([10.0, 20.0])
    data = jnp.asarray(
        np.random.default_rng(5).normal(size=(2, 12 * nside**2))
    )

    @jax.jit
    def analyse(maps):
        return Sky(
            maps, freqs, sampling="healpix", engine="kernel"
        ).compute_alm()

    kernel.clear_kernel_cache()
    with pytest.raises(RuntimeError, match="precompute_kernel"):
        analyse(data)
