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
    but its build costs 2x-25x more, and the break-even call counts from
    the same measurements are 168 to 11929 calls. Croissant transforms
    once at construction, so that per-call saving is never repaid.
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
    footprint exceeds the cap."""
    for nside in (8, 16, 32, 64, 128):
        lmax = 2 * nside - 1
        engine, _ = engine_select.resolve_engine(
            lmax=lmax,
            sampling="healpix",
            nside=nside,
            niter=3,
            batch_size=1024,
        )
        if engine == "dense":
            assert (
                footprints.dense_nbytes(lmax, "healpix", nside=nside)
                <= engine_select.DEFAULT_MEMORY_CAP_BYTES
            )
        elif engine == "kernel":
            from croissant import kernel

            # reality=True to match resolve_engine's own default, which
            # is what it actually used to decide "kernel"; kernel_nbytes
            # itself defaults to reality=False and would over-predict.
            assert (
                kernel.kernel_nbytes(
                    lmax, "healpix", nside=nside, reality=True
                )
                <= engine_select.DEFAULT_MEMORY_CAP_BYTES
            )


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
