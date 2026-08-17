"""
Automatic spherical harmonic engine selection.

Croissant's engines compute the same linear map to within ~1e-15, so
which one to use is a question about memory and reuse rather than about
results. That makes it croissant's decision to own: this module encodes
the rule that used to live in prose in the README.

The policy is deliberately conservative. It never selects a precomputing
engine whose footprint exceeds ``memory_cap``, and it falls back to the
matrix-free engine whenever precomputing cannot pay for itself. It is a
policy, not a promise: pin ``engine=`` explicitly to freeze behaviour.

Inputs are only quantities croissant can know at construction time. The
one thing it cannot know is whether an object will be reused across many
later calls, so ``batch_size`` stands in for the amortisation factor --
for ``Beam`` and ``Sky`` that is the number of frequencies.
"""

from .footprints import dense_nbytes, kernel_nbytes, transform_lmax

__all__ = [
    "DEFAULT_MEMORY_CAP_BYTES",
    "ENGINES",
    "degrade_for_trace",
    "resolve_engine",
    "validate_engine",
]

DEFAULT_MEMORY_CAP_BYTES = 512 * 1024**2

#: Kernel megabytes that one batched transform can amortise. MEASURED by
#: direct batch ladders, not reasoned: the crossover batch -- the
#: smallest batch at which the kernel engine's cold setup-plus-first call
#: beats the matrix-free engine's -- tracks the kernel's own size closely
#: at niter=0. Measured on CPU with x64, scalar HEALPix: crossover 1 at
#: lmax=31 (0.98 MiB), 10 at lmax=63 (7.94 MiB), 78 at lmax=127
#: (63.75 MiB), i.e. 1.02, 1.26 and 1.22 batch elements per MiB. Both
#: quantities grow as roughly nside**3, which is why a single fixed batch
#: count cannot serve every resolution. This constant is the reciprocal
#: of that measured ratio.
#:
#: Re-measure with
#: ``benchmarks/benchmark_engines.py --sections ladder`` before changing
#: it, and prefer the ladder to the cheaper ``--sections fit``: the fit
#: assumes a cold cost linear in batch, and at lmax=63 the cold curves
#: are nearly flat beyond a batch of ~12 because compilation dominates,
#: which made the fit overestimate that crossover by about 2x.
_MIB_PER_BATCHED_TRANSFORM = 0.81

ENGINES = ("s2fft", "kernel", "dense")


def _mib(nbytes):
    """Format a byte count for a human-readable reason string.

    Plain ``.1f`` MiB rounds anything under 0.05 MiB to "0.0 MiB", which
    reads as though the precompute were free. Fall back to KiB below that.
    """
    mib = nbytes / 1024**2
    if mib < 0.05:
        return f"{nbytes / 1024:.0f} KiB"
    return f"{mib:.1f} MiB"


def _transforms(count):
    """Pluralise a transform count for a reason string."""
    return f"{count} transform" + ("" if count == 1 else "s")


def _amortisation_threshold(build_bytes, niter):
    """
    Smallest batch size that can pay for BUILDING a kernel.

    What has to be repaid is build time, not resident bytes, and the two
    are deliberately different quantities here.

    ``build_bytes`` is a geometry proxy -- the size the kernel would have
    at spin 0 with ``reality=True`` -- rather than the footprint actually
    allocated. Measured build times are very nearly spin-independent
    (10.7 s scalar against 10.2 s at spin 2, lmax=127) even though the
    spin kernel is twice the bytes, because both builds run the same
    Wigner-d recursion over the same geometry. Sizing the threshold by
    the real footprint would therefore ask a spin field for twice the
    batch a scalar field needs to repay the same work. The true resident
    size still governs the memory cap in :func:`resolve_engine`; that is
    a separate question from amortisation and keeps its own accounting.

    Refinement divides the threshold by ``2 * niter + 1`` because it
    makes the kernel engine relatively cheaper, not dearer: s2fft repeats
    its full recursion on every refinement pass, while the kernel engine
    contracts the same cached table again. Both effects are real and they
    do not cancel -- ``niter > 0`` does build a second (synthesis) kernel,
    roughly doubling build cost, but s2fft's per-map cost rises by the
    larger factor. Measured at lmax=127: the crossover falls from 78 at
    niter=0 to 14 at niter=3.

    Rounds to nearest rather than up. The threshold estimates a measured
    crossover, so it is not a bound in either direction, and rounding up
    would bias every configuration toward the matrix-free engine.

    Parameters
    ----------
    build_bytes : int
        Geometry-proxy kernel size, in bytes.
    niter : int
        Number of iterative refinement steps requested.

    Returns
    -------
    int
        Minimum batch size, never below 1.

    """
    mib = build_bytes / 1024**2
    per_transform = _MIB_PER_BATCHED_TRANSFORM * (2 * int(niter) + 1)
    return max(1, round(mib / per_transform))


def degrade_for_trace(
    engine,
    *,
    has_kernel=False,
    has_inverse_kernel=False,
    niter=0,
    sub_floor=False,
):
    """
    Adjust an automatically chosen engine for what a live trace allows.

    Auto must never hand back an engine that then refuses to run. Only
    the kernel engine is actually blocked by an active trace: converting
    its numpy-built kernel to a ``jax.Array`` mid-trace would yield a
    tracer the module-level cache must never retain (see
    :func:`croissant.kernel.precompute_kernel`), so a kernel that was not
    precomputed and threaded in cannot be obtained. The dense operators
    are built from static geometry under
    ``jax.ensure_compile_time_eval`` and are unaffected.

    This is the ONE place that rule lives. It used to be written out at
    three call sites, each of which had forgotten a different half of it:
    one ignored ``inverse_kernel``, one degraded dense to an engine that
    cannot serve a sub-floor band-limit, and two let the dense path raise
    for a choice the caller never made.

    Call only for an automatic choice, and only while tracing. An
    explicit request is never softened: a caller who named an engine is
    told to precompute instead, so the cost decision they made
    deliberately is never silently swapped out from under them.

    Parameters
    ----------
    engine : str
        The engine ``resolve_engine`` chose.
    has_kernel : bool
        Whether a forward kernel was threaded in.
    has_inverse_kernel : bool
        Whether a synthesis kernel was threaded in. Only consulted when
        ``niter > 0``, which is when the refinement iteration needs one.
    niter : int
        Number of iterative refinement steps.
    sub_floor : bool
        Whether the band-limit is below the HEALPix ``L >= 2 * nside``
        floor. There the matrix-free engine is not a legal fallback --
        it cannot perform the transform at all -- so dense, the only
        engine that can low-pass in one step, is the degrade target.

    Returns
    -------
    str
        The engine to actually run.

    """
    if engine != "kernel":
        return engine
    if has_kernel and (niter == 0 or has_inverse_kernel):
        return engine
    return "dense" if sub_floor else "s2fft"


def validate_engine(engine):
    """
    Reject an engine name that is not a supported public choice.

    Shared by :class:`croissant.sphere.SphBase` and the polarized fields
    so the two entry points cannot drift on what counts as valid.
    :func:`resolve_engine` accepts ``None`` as a synonym for ``"auto"``
    for the convenience of its internal callers; that is not a licence
    for a public constructor to accept it, and a caller threading an
    optional engine through a config object should get the same answer
    whichever class receives it.

    Parameters
    ----------
    engine : str
        Engine name from the caller, or ``"auto"``.

    Returns
    -------
    str
        The engine name, unchanged.

    Raises
    ------
    ValueError
        If ``engine`` is not ``"auto"`` or one of :data:`ENGINES`.

    """
    if engine != "auto" and engine not in ENGINES:
        raise ValueError(
            f"Unsupported SHT engine {engine!r}. Supported engines are "
            f"{set(ENGINES) | {'auto'}}."
        )
    return engine


def resolve_engine(
    lmax,
    sampling,
    nside=None,
    spin=0,
    niter=0,
    reality=False,
    batch_size=1,
    memory_cap=None,
    requested=None,
):
    """
    Choose a spherical harmonic engine for one configuration.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    spin : int
        Spin weight of the field.
    niter : int
        Number of iterative refinement steps requested.
    reality : bool
        Whether the field is real. Defaults to False, matching
        :func:`croissant.sphere.compute_alm`, so that the footprints
        this weighs are the ones the defaulted transform would build.
        Callers that know their own field is real must say so, or the
        engine is chosen by sizing an operator twice as large as the one
        they will actually build.
    batch_size : int
        Number of transforms the choice will be amortised over, i.e. the
        product of the leading batch axes.
    memory_cap : int or None
        Largest precomputed footprint to consider, in bytes. Defaults to
        :data:`DEFAULT_MEMORY_CAP_BYTES`.
    requested : str or None
        An explicit engine name, returned unchanged. ``None`` or
        ``"auto"`` selects automatically.

    Returns
    -------
    tuple of (str, str)
        The engine name and a short human-readable reason, suitable for
        logging or for reporting which mechanism an object chose.

    """
    if requested is not None and requested != "auto":
        if requested not in ENGINES:
            raise ValueError(
                f"Unsupported SHT engine {requested!r}. Supported "
                f"engines are {set(ENGINES)}."
            )
        return requested, f"explicit request for {requested!r}"

    cap = DEFAULT_MEMORY_CAP_BYTES if memory_cap is None else int(memory_cap)
    kernel_bytes = kernel_nbytes(
        lmax, sampling, nside=nside, spin=spin, reality=reality
    )
    # niter > 0 makes kernel_compute_alm build a synthesis (inverse)
    # kernel alongside the forward one, to run the refinement iteration
    # itself (see kernel.kernel_compute_alm's module docstring). Both
    # are resident in memory at once, so the actual footprint can be up
    # to double what kernel_nbytes reports for the forward kernel alone.
    # Measured equal for healpix/dh/gl and smaller for mw/mwss, so the
    # forward kernel is always the larger of the two and doubling it is
    # a safe upper bound, never an under-prediction.
    needed_kernel_bytes = kernel_bytes * (2 if niter > 0 else 1)
    dense_bytes = dense_nbytes(
        lmax, sampling, nside=nside, spin=spin, reality=reality
    )
    kernel_fits = needed_kernel_bytes <= cap
    dense_fits = dense_bytes <= cap

    # The kernel engine cannot serve a band-limit below the HEALPix
    # L >= 2*nside floor; the dense engine can, by building at the floor
    # and keeping only the requested low-ell rows. That row selection is
    # dense's clearest remaining advantage, so it is checked first.
    needs_row_selection = transform_lmax(lmax, sampling, nside=nside) != int(
        lmax
    )
    if needs_row_selection:
        reason = (
            f"lmax={lmax} is below the HEALPix floor for "
            f"nside={nside}; only the dense engine can low-pass in "
            "one step"
        )
        if not dense_fits:
            # Correctness outranks the cap here, and this is the one
            # place the two can conflict. Neither s2fft nor the kernel
            # engine can perform a HEALPix transform below the floor
            # under any memory budget, so there is no cheaper engine to
            # fall back TO: naming one would trade a large allocation
            # for a hard failure at the caller's first transform. Report
            # the footprint instead and let the caller choose a higher
            # lmax or a coarser map.
            reason += f"; its {_mib(dense_bytes)} exceeds the {_mib(cap)} cap"
        return "dense", reason

    # Sized from BUILD cost, which is a different quantity from the
    # resident footprint `kernel_fits` tests above, and deliberately so:
    # a spin kernel is twice the bytes of a scalar one at the same
    # geometry but takes the same time to build, so the batch needed to
    # repay it is the same. See _amortisation_threshold.
    build_bytes = kernel_nbytes(
        lmax, sampling, nside=nside, spin=0, reality=True
    )
    threshold = _amortisation_threshold(build_bytes, niter)
    if batch_size < threshold:
        return (
            "s2fft",
            f"batch of {batch_size} cannot amortise a "
            f"{_mib(needed_kernel_bytes)} kernel "
            f"(needs {threshold})",
        )

    # NOTE, and this is the one place a per-call benchmark misleads.
    # Dense does win on cached-apply at niter > 0 -- 2.4x to 8.3x in
    # Task 6 -- because its refinement folds into the cached matrix while
    # the kernel engine pays 2*niter+1 passes. But what decides whether
    # that win is worth having is the BUILD cost, and build ratios
    # (dense/kernel) across the niter=3 configs in the same table span
    # only roughly 1.0x to 25x, not a fixed multiple: nside=16/scalar
    # builds are near-identical (1.02x) while nside=16/spin=2 costs
    # 24.8x more. The resulting break-even call counts span 7 to 92338
    # calls -- 7 for the near-identical nside=16/scalar build (the
    # smallest gap the table has, and the case the build-cost range
    # above must not omit), up to 92338 for spin=2/nside=16/niter=0.
    # Croissant transforms once at construction, batched over
    # frequencies -- one call, below even that smallest break-even. So
    # selecting dense for per-call speed would routinely trade a build
    # whose extra cost is never repaid in a single call.
    # Dense-for-throughput is therefore an explicit user override, not
    # an automatic choice: only the caller knows it will re-apply the
    # same transform thousands of times. The structural case above (a
    # band-limit below the HEALPix floor) is the only reason auto
    # reaches for dense.
    if kernel_fits:
        return (
            "kernel",
            f"{_mib(needed_kernel_bytes)} kernel amortises "
            f"over {_transforms(batch_size)}"
            + (
                f"; dense would need {_mib(dense_bytes)}"
                if not dense_fits
                else ""
            ),
        )

    return (
        "s2fft",
        f"no precomputed engine fits under {_mib(cap)} "
        f"(kernel {_mib(needed_kernel_bytes)}, dense "
        f"{_mib(dense_bytes)})",
    )
