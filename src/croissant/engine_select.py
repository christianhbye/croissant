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

import math

from .footprints import dense_nbytes, kernel_nbytes, transform_lmax

__all__ = [
    "DEFAULT_MEMORY_CAP_BYTES",
    "ENGINES",
    "degrade_for_trace",
    "resolve_engine",
    "validate_engine",
]

DEFAULT_MEMORY_CAP_BYTES = 512 * 1024**2

#: Kernel megabytes that one batched transform can amortise. MEASURED in
#: Task 6, not reasoned: the batch size at which the kernel engine's
#: setup-plus-first call overtakes the matrix-free engine tracks the
#: kernel's own size almost 1:1 -- crossover 1 at nside=8 and nside=16,
#: where the kernel is 0.12 and 0.98 MiB, and crossover 8 at nside=32,
#: where it is 7.94 MiB. Both quantities grow as roughly nside**3.
#: A single fixed batch threshold cannot express this: it would have to be
#: 1 to serve nside=16 and 8 to serve nside=32. Only three resolutions
#: were fitted, so treat the 1:1 coefficient as order-of-magnitude, and
#: re-measure with benchmarks/benchmark_engines.py before changing it.
#:
#: KNOWN MISCALIBRATED AT niter > 0, and this coefficient is why. All
#: three crossovers above were measured at niter=0. The threshold below
#: correctly counts both resident kernels at niter > 0, so the batch it
#: asks for doubles -- but the kernel engine gets CHEAPER per transform
#: there, not dearer: s2fft pays 2*niter+1 full Wigner-d recursions where
#: the kernel pays cheap contractions. The true crossover falls while the
#: predicted one rises. Measured counterexample at nside=32/lmax=63/
#: niter=3: the kernel wins at a batch of 4 (7.25s vs 9.74s
#: setup-plus-first) while this policy asks for 16, and batches of 8-15
#: select s2fft where the undoubled threshold selected kernel. Correct
#: but sometimes slower, never wrong -- the engines agree to ~1e-13. The
#: same under-fitting shows at nside >= 64, where a 64.7 MiB kernel asks
#: for 65 transforms and is effectively unreachable. Fixing it needs a
#: batch sweep at niter > 0 and at nside >= 64; do not hand-tune the
#: constant without one.
_MIB_PER_BATCHED_TRANSFORM = 1.0

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


def _amortisation_threshold(resident_bytes):
    """
    Smallest batch size that can pay for a kernel footprint this size.

    Parameters
    ----------
    resident_bytes : int
        Total predicted size of the kernels that would be held at once,
        which is both of them when ``niter > 0``.

    Returns
    -------
    int
        Minimum batch size, never below 1.

    """
    mib = resident_bytes / 1024**2
    return max(1, math.ceil(mib / _MIB_PER_BATCHED_TRANSFORM))


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
    reality=True,
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
        Whether the field is real.
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

    # Sized from the resident footprint, not the forward kernel alone:
    # at niter > 0 both kernels are held at once, which is the same
    # quantity `kernel_fits` tests two lines up. Reading it two different
    # ways would size the batch for half the memory actually spent.
    threshold = _amortisation_threshold(needed_kernel_bytes)
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
