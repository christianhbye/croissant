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

__all__ = ["DEFAULT_MEMORY_CAP_BYTES", "ENGINES", "resolve_engine"]

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
_MIB_PER_BATCHED_TRANSFORM = 1.0

ENGINES = ("s2fft", "kernel", "dense")


def _amortisation_threshold(kernel_bytes):
    """
    Smallest batch size that can pay for a kernel of this size.

    Parameters
    ----------
    kernel_bytes : int
        Predicted size of the kernel that would be built.

    Returns
    -------
    int
        Minimum batch size, never below 1.

    """
    mib = kernel_bytes / 1024**2
    return max(1, math.ceil(mib / _MIB_PER_BATCHED_TRANSFORM))


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
    kernel_bytes = kernel_nbytes(lmax, sampling, nside=nside, reality=reality)
    dense_bytes = dense_nbytes(
        lmax, sampling, nside=nside, spin=spin, reality=reality
    )
    kernel_fits = kernel_bytes <= cap
    dense_fits = dense_bytes <= cap

    # The kernel engine cannot serve a band-limit below the HEALPix
    # L >= 2*nside floor; the dense engine can, by building at the floor
    # and keeping only the requested low-ell rows. That row selection is
    # dense's clearest remaining advantage, so it is checked first.
    needs_row_selection = transform_lmax(lmax, sampling, nside=nside) != int(
        lmax
    )
    if needs_row_selection:
        if dense_fits:
            return (
                "dense",
                f"lmax={lmax} is below the HEALPix floor for "
                f"nside={nside}; only the dense engine can low-pass in "
                "one step",
            )
        return (
            "s2fft",
            f"lmax={lmax} is below the HEALPix floor for nside={nside} "
            f"and the dense operator needs {dense_bytes >> 20} MiB",
        )

    threshold = _amortisation_threshold(kernel_bytes)
    if batch_size < threshold:
        return (
            "s2fft",
            f"batch of {batch_size} cannot amortise a "
            f"{kernel_bytes / 1024**2:.1f} MiB kernel "
            f"(needs {threshold})",
        )

    # NOTE, and this is the one place a per-call benchmark misleads.
    # Dense does win on cached-apply at niter > 0 -- 2.4x to 8.3x in
    # Task 6 -- because its refinement folds into the cached matrix while
    # the kernel engine pays 2*niter+1 passes. But its BUILD costs 2x to
    # 25x more, and the break-even call counts computed from the same
    # table are 168, 803, 7926 and 11929 calls (and 92338 for
    # spin 2/nside 16/niter 0). Croissant transforms once at
    # construction, batched over frequencies -- one call. So selecting
    # dense for per-call speed would trade a 25 s build for a 0.03 s
    # saving that is never repaid. Dense-for-throughput is therefore an
    # explicit user override, not an automatic choice: only the caller
    # knows it will re-apply the same transform thousands of times. The
    # structural case above (a band-limit below the HEALPix floor) is the
    # only reason auto reaches for dense.
    if kernel_fits:
        return (
            "kernel",
            f"{kernel_bytes >> 20} MiB kernel amortises over "
            f"{batch_size} transforms"
            + (
                f"; dense would need {dense_bytes >> 20} MiB"
                if not dense_fits
                else ""
            ),
        )

    return (
        "s2fft",
        f"no precomputed engine fits under {cap >> 20} MiB "
        f"(kernel {kernel_bytes >> 20} MiB, dense "
        f"{dense_bytes >> 20} MiB)",
    )
