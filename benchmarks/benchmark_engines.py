"""
Benchmark croissant's three spherical harmonic engines.

Reports setup cost, resident size of the precomputed object, and per-call
cost for engine="s2fft", "kernel" and "dense" across HEALPix
resolutions, and checks that all three agree numerically at each
configuration so the timings cannot be read as a speed/accuracy
trade-off.

Also answers the niter question: dense folds refinement into its cached
matrix, so its per-call cost is independent of niter, while the kernel
engine pays 2*niter+1 applications. Per-call cost is roughly one pass
over the precomputed object, and dense's is ~1.5*nside times larger, so
the FLOP count says the kernel still wins for nside > 4 -- this measures
whether wall-clock agrees.

Dense configurations whose predicted footprint exceeds MEMORY_CAP_MIB are
skipped rather than built, and reported as skipped.

The `ladder` section finds the amortisation crossover -- the batch at
which building a kernel starts paying for itself against the matrix-free
engine -- by timing both engines from cold at a ladder of batch sizes
and reporting where the winner flips. It is the ground truth
`croissant.engine_select` is calibrated against, and it supersedes the
two sections below, which are kept for provenance and must be asked for
by name.

The `sweep` section is the original version of that measurement: the
same idea at three fixed resolutions (scalar, nside=8/16/32, niter=0),
but timing each point once rather than medianing repeats.

The `fit` section tried to derive the crossover instead of measuring it,
on the grounds that the sweep looked noise-limited: at nside=32 the two
setup-plus-first curves sit within 0.12% of each other across the whole
batch range. That framing was half right. The curves really are that
close, but because BOTH are dominated by compilation, not because the
measurement was worthless -- and the linear model below then
overestimated the nside=32 crossover by about 2x, where the sweep had
been roughly correct. Use `ladder`. Model the cold cost of one batched
call as

    T(B) = a + m * B

where `a` collects compilation plus, for the kernel engine, the Wigner-d
build. Fit one such line per engine and the crossover follows in closed
form,

    B* = (a_kernel - a_s2fft) / (m_s2fft - m_kernel)

Both `a` and `m` are fitted from COLD calls -- medianed at batch 1 and
at a high batch -- rather than mixing a cold intercept with a warm
slope. That mixing was the first version of this section and it was
wrong: a cold call's marginal cost per map is higher than a warm one's,
so a warm slope understates `m_s2fft` and biases B* high. The
confirmation pass is what caught it, and it stays in place for exactly
that reason: it times the cold call either side of each derived B* and
reports MISMATCH when the observed winner falls on the wrong side. A
MISMATCH means the linear model does not describe that configuration
and the fitted number should not be used.
"""

import argparse
import math
import os
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from croissant import dense, footprints, kernel, sphere  # noqa: E402

MEMORY_CAP_MIB = 1024
SCALAR_NSIDES = (8, 16, 32)
SPIN_NSIDES = (8, 16)
NITERS = (0, 3)
REPEATS = 3
BATCH_SWEEP_NSIDES = (8, 16, 32)
BATCH_SWEEP_SIZES = (1, 2, 4, 8, 16, 32)

SECTIONS = ("grid", "sweep", "fit", "ladder")

#: Sections run when none are named. `ladder` produces the numbers the
#: amortisation threshold is calibrated against and `grid` the engine
#: comparison the README quotes. `sweep` and `fit` are diagnostics kept
#: for provenance and must be asked for by name.
DEFAULT_SECTIONS = ("grid", "ladder")

#: Ladder configurations as (nside, niter, spin, batches). The batch
#: lists bracket rather than sweep: each was aimed with a cheap `fit`
#: run and then narrowed until the winner flipped between two adjacent
#: points, which is why they differ per configuration. nside=64/niter=3
#: is the case the shipped policy used to get worst.
LADDER_CONFIGS = (
    (16, 0, 0, (1, 2, 4)),
    (16, 3, 0, (1, 2, 4)),
    (32, 0, 0, (4, 8, 12, 16, 24, 32)),
    (32, 0, 2, (4, 8, 12, 16, 24, 32)),
    (32, 3, 0, (1, 2, 4)),
    (64, 0, 0, (16, 24, 32, 48, 64, 96)),
    (64, 3, 0, (4, 8, 12, 16, 24)),
)

#: Resolutions for the component fit. nside=64 is the point of the
#: exercise: the MiB-scaled threshold it replaces asks for 65 batch
#: elements there, which no realistic frequency axis supplies, so the
#: kernel engine is effectively unreachable at nside >= 64 today.
FIT_NSIDES = (16, 32, 64)
FIT_SPINS = (0, 2)
FIT_NITERS = (0, 3)

#: High batch point for the slope estimate, per resolution. Kept small at
#: nside=64, where one s2fft transform at niter=3 costs of order a
#: second per map and a batch of 32 would dominate the whole run.
FIT_BATCH_HI = {8: 32, 16: 32, 32: 32, 64: 8}

#: Cold measurements are medianed rather than best-of: a from-cold call
#: is dominated by compilation, which scatters by a few hundred ms, and
#: the crossover formula takes a DIFFERENCE of two such numbers. Five
#: repeats at batch 1 is affordable precisely because batch 1 is the
#: cheapest point on the curve.
COLD_REPEATS = 5

#: Confirmation batches are capped so a large derived crossover cannot
#: turn the check into the expensive sweep this section exists to avoid.
CONFIRM_BATCH_CAP = 64


def _median(values):
    """Median of a short sequence of timings."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _clear_caches():
    """Drop every precomputed object, including JAX's compilation cache."""
    kernel.clear_kernel_cache()
    dense.clear_dense_matrix_cache()
    jax.clear_caches()


def _time_call(fn, *args, **kwargs):
    """Return (first_call_seconds, best_of_REPEATS_cached_seconds)."""
    started = perf_counter()
    result = fn(*args, **kwargs)
    result.block_until_ready()
    first = perf_counter() - started
    best = float("inf")
    for _ in range(REPEATS):
        started = perf_counter()
        fn(*args, **kwargs).block_until_ready()
        best = min(best, perf_counter() - started)
    return first, best


def _cold_seconds(run, repeats=COLD_REPEATS):
    """Median wall-clock of a from-cold setup-plus-first call."""
    times = []
    for _ in range(repeats):
        _clear_caches()
        started = perf_counter()
        run().block_until_ready()
        times.append(perf_counter() - started)
    return _median(times)


def _warm_seconds(run, repeats=REPEATS):
    """Best-of wall-clock of a call whose precompute is already cached."""
    run().block_until_ready()
    best = float("inf")
    for _ in range(repeats):
        started = perf_counter()
        run().block_until_ready()
        best = min(best, perf_counter() - started)
    return best


def _healpix_data(batch, npix, reality):
    """Random map stack of shape (batch, npix), complex unless real."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=(batch, npix))
    if not reality:
        values = values + 1j * rng.normal(size=(batch, npix))
    return jnp.asarray(values)


def _make_run(data, lmax, nside, niter, spin, reality, engine):
    """Zero-argument closure running one configured analysis."""

    def run():
        return sphere.compute_alm(
            data,
            lmax,
            "healpix",
            nside=nside,
            niter=niter,
            spin=spin,
            reality=reality,
            engine=engine,
        )

    return run


def _build_seconds(lmax, nside, spin, reality, niter):
    """
    Median seconds to build the kernels a configuration keeps resident.

    Both kernels are built at niter > 0: croissant drives the refinement
    iteration itself, so it needs the synthesis kernel alongside the
    analysis one (see croissant.kernel's module docstring).
    """
    total = 0.0
    directions = (True,) if niter == 0 else (True, False)
    for forward in directions:
        times = []
        for _ in range(REPEATS):
            kernel.clear_kernel_cache()
            started = perf_counter()
            kernel.precompute_kernel(
                lmax,
                "healpix",
                nside=nside,
                spin=spin,
                reality=reality,
                forward=forward,
            ).block_until_ready()
            times.append(perf_counter() - started)
        total += _median(times)
    return total


def grid_benchmark():
    """Time all three engines across the spin/nside/niter grid."""
    rows = []
    for spin, nsides in ((0, SCALAR_NSIDES), (2, SPIN_NSIDES)):
        reality = spin == 0
        for nside in nsides:
            lmax = 2 * nside - 1
            npix = 12 * nside**2
            data = _healpix_data(4, npix, reality)

            kernel_mib = (
                footprints.kernel_nbytes(
                    lmax,
                    "healpix",
                    nside=nside,
                    spin=spin,
                    reality=reality,
                )
                / 2**20
            )
            dense_mib = (
                footprints.dense_nbytes(
                    lmax, "healpix", nside=nside, spin=spin, reality=reality
                )
                / 2**20
            )

            for niter in NITERS:
                reference = None
                for engine in ("s2fft", "kernel", "dense"):
                    if engine == "dense" and dense_mib > MEMORY_CAP_MIB:
                        print(
                            f"spin={spin:+d} nside={nside} niter={niter} "
                            f"engine=dense SKIPPED "
                            f"predicted_mib={dense_mib:.1f}"
                        )
                        continue
                    _clear_caches()
                    run = _make_run(
                        data, lmax, nside, niter, spin, reality, engine
                    )
                    setup_and_first, cached = _time_call(run)
                    got = np.asarray(run())
                    if reference is None:
                        reference = got
                        agreement = 0.0
                    else:
                        agreement = np.abs(got - reference).max() / max(
                            np.abs(reference).max(), 1e-300
                        )
                    mib = {
                        "s2fft": 0.0,
                        "kernel": kernel_mib,
                        "dense": dense_mib,
                    }[engine]
                    print(
                        f"spin={spin:+d} nside={nside} niter={niter} "
                        f"engine={engine} "
                        f"setup_plus_first_seconds={setup_and_first:.4f} "
                        f"cached_apply_seconds={cached:.6f} "
                        f"precompute_mib={mib:.2f} "
                        f"rel_vs_s2fft={agreement:.2e}"
                    )
                    rows.append(
                        (
                            spin,
                            nside,
                            niter,
                            engine,
                            setup_and_first,
                            cached,
                            mib,
                            agreement,
                        )
                    )

    print()
    print(
        "| spin | nside | niter | engine | setup+first (s) | "
        "cached apply (s) | precompute (MiB) | rel vs s2fft |"
    )
    print("|---:|---:|---:|:--|---:|---:|---:|---:|")
    for spin, nside, niter, engine, setup, cached, mib, agree in rows:
        print(
            f"| {spin:+d} | {nside} | {niter} | {engine} | {setup:.3f} | "
            f"{cached:.6f} | {mib:.2f} | {agree:.1e} |"
        )


def batch_sweep():
    """Sweep batch size looking for the kernel/s2fft crossover."""
    for sweep_nside in BATCH_SWEEP_NSIDES:
        sweep_lmax = 2 * sweep_nside - 1
        sweep_npix = 12 * sweep_nside**2
        sweep_kernel_mib = (
            footprints.kernel_nbytes(
                sweep_lmax, "healpix", nside=sweep_nside, reality=True
            )
            / 2**20
        )
        print(
            f"# batch sweep: scalar, nside={sweep_nside}, "
            f"lmax={sweep_lmax}, niter=0, "
            f"kernel_mib={sweep_kernel_mib:.2f}"
        )
        crossover = None
        for batch_size in BATCH_SWEEP_SIZES:
            data = _healpix_data(batch_size, sweep_npix, True)
            setup_times = {}
            for engine in ("s2fft", "kernel"):
                _clear_caches()
                run = _make_run(
                    data, sweep_lmax, sweep_nside, 0, 0, True, engine
                )
                setup_and_first, cached = _time_call(run)
                setup_times[engine] = setup_and_first
                print(
                    f"nside={sweep_nside} batch_size={batch_size} "
                    f"engine={engine} "
                    f"setup_plus_first_seconds={setup_and_first:.4f} "
                    f"cached_apply_seconds={cached:.6f}"
                )
            winner = (
                "kernel"
                if setup_times["kernel"] < setup_times["s2fft"]
                else "s2fft"
            )
            print(
                f"nside={sweep_nside} batch_size={batch_size} winner={winner}"
            )
            if crossover is None and winner == "kernel":
                crossover = batch_size
        if crossover is None:
            print(
                f"nside={sweep_nside} crossover_batch_size=NEVER "
                f"(kernel did not beat s2fft up to "
                f"batch_size={BATCH_SWEEP_SIZES[-1]})"
            )
        else:
            print(f"nside={sweep_nside} crossover_batch_size={crossover}")


def _confirm(
    crossover, lmax, nside, niter, spin, reality, npix, cap=CONFIRM_BATCH_CAP
):
    """
    Time the cold call either side of a derived crossover.

    Reports the observed winner below and above B*. A correct B* should
    show s2fft below and kernel above; anything else means the linear
    cost model does not describe this configuration.
    """
    below = max(1, int(math.floor(crossover / 2)))
    above = int(math.ceil(crossover)) * 2
    if above > cap:
        print(f"  confirm SKIPPED (would need batch {above} > {cap})")
        return
    for batch in (below, above):
        data = _healpix_data(batch, npix, reality)
        times = {}
        for engine in ("s2fft", "kernel"):
            run = _make_run(data, lmax, nside, niter, spin, reality, engine)
            times[engine] = _cold_seconds(run, repeats=REPEATS)
        winner = "kernel" if times["kernel"] < times["s2fft"] else "s2fft"
        expected = "s2fft" if batch < crossover else "kernel"
        print(
            f"  confirm batch={batch} s2fft={times['s2fft']:.4f} "
            f"kernel={times['kernel']:.4f} winner={winner} "
            f"expected={expected} "
            f"{'OK' if winner == expected else 'MISMATCH'}"
        )


def ladder_benchmark(configs):
    """
    Locate each crossover directly, by timing a ladder of batch sizes.

    This is the ground truth the amortisation threshold is calibrated
    against, and it supersedes both `sweep` and `fit`. Against `sweep`
    it medians several from-cold repeats per point instead of timing
    each once. Against `fit` it makes no assumption about the shape of
    the cost curve, which matters: at lmax=63 the cold curves are nearly
    flat beyond a batch of ~12 because compilation dominates, so the
    straight line `fit` assumes overestimated that crossover by ~2x.

    Reports the winner at each batch and the bracket the crossover falls
    in. A bracket, not a number, is the honest output -- the crossover
    lies between the last batch s2fft wins and the first the kernel
    wins, and narrowing it further costs machine time for precision the
    policy cannot use.
    """
    for nside, niter, spin, batches in configs:
        reality = spin == 0
        lmax = 2 * nside - 1
        npix = 12 * nside**2
        print(
            f"# ladder: spin={spin:+d} nside={nside} lmax={lmax} niter={niter}"
        )
        previous = None
        bracket = None
        for batch in batches:
            data = _healpix_data(batch, npix, reality)
            times = {}
            for engine in ("s2fft", "kernel"):
                run = _make_run(
                    data, lmax, nside, niter, spin, reality, engine
                )
                times[engine] = _cold_seconds(run, repeats=REPEATS)
            winner = "kernel" if times["kernel"] < times["s2fft"] else "s2fft"
            margin = (
                100
                * abs(times["kernel"] - times["s2fft"])
                / min(times.values())
            )
            print(
                f"spin={spin:+d} nside={nside} niter={niter} "
                f"batch={batch:>3} s2fft={times['s2fft']:.4f} "
                f"kernel={times['kernel']:.4f} winner={winner} "
                f"margin={margin:.1f}%"
            )
            if bracket is None and winner == "kernel":
                bracket = (previous, batch)
            previous = batch
        if bracket is None:
            print(
                f"spin={spin:+d} nside={nside} niter={niter} "
                f"crossover=ABOVE {batches[-1]}"
            )
        elif bracket[0] is None:
            print(
                f"spin={spin:+d} nside={nside} niter={niter} "
                f"crossover=AT OR BELOW {batches[0]}"
            )
        else:
            print(
                f"spin={spin:+d} nside={nside} niter={niter} "
                f"crossover=IN ({bracket[0]}, {bracket[1]}]"
            )


def component_fit(nsides, spins, niters, confirm, cap=CONFIRM_BATCH_CAP):
    """Derive the kernel/s2fft crossover from measured cost components."""
    rows = []
    for spin in spins:
        reality = spin == 0
        for nside in nsides:
            lmax = 2 * nside - 1
            npix = 12 * nside**2
            batch_hi = FIT_BATCH_HI[nside]
            kernel_mib = (
                footprints.kernel_nbytes(
                    lmax,
                    "healpix",
                    nside=nside,
                    spin=spin,
                    reality=reality,
                )
                / 2**20
            )
            for niter in niters:
                resident_mib = kernel_mib * (2 if niter > 0 else 1)
                build = _build_seconds(lmax, nside, spin, reality, niter)

                cold_lo = {}
                cold_hi = {}
                warm_slope = {}
                intercept = {}
                slope = {}
                for engine in ("s2fft", "kernel"):
                    data_lo = _healpix_data(1, npix, reality)
                    run_lo = _make_run(
                        data_lo, lmax, nside, niter, spin, reality, engine
                    )
                    data_hi = _healpix_data(batch_hi, npix, reality)
                    run_hi = _make_run(
                        data_hi, lmax, nside, niter, spin, reality, engine
                    )
                    cold_lo[engine] = _cold_seconds(run_lo)
                    cold_hi[engine] = _cold_seconds(run_hi)
                    slope[engine] = (cold_hi[engine] - cold_lo[engine]) / (
                        batch_hi - 1
                    )
                    intercept[engine] = cold_lo[engine] - slope[engine]
                    # Reported for contrast only. An earlier version of
                    # this fit took the slope from warm cached applies,
                    # which is a different regime: a cold call's marginal
                    # cost per map is higher, so the warm slope
                    # understates slope_s2fft and biases the crossover
                    # HIGH. The confirmation pass caught it (predicted
                    # 30.1 at spin=2/nside=32/niter=0, kernel actually
                    # already winning at batch 15). Both are printed so
                    # the size of that gap stays visible.
                    warm_slope[engine] = (
                        _warm_seconds(run_hi) - _warm_seconds(run_lo)
                    ) / (batch_hi - 1)

                denom = slope["s2fft"] - slope["kernel"]
                if denom <= 0:
                    crossover = None
                else:
                    crossover = max(
                        1.0,
                        (intercept["kernel"] - intercept["s2fft"]) / denom,
                    )

                crossover_text = (
                    "NEVER" if crossover is None else f"{crossover:.2f}"
                )
                print(
                    f"spin={spin:+d} nside={nside} niter={niter} "
                    f"resident_mib={resident_mib:.2f} "
                    f"build_seconds={build:.4f} "
                    f"cold1_s2fft={cold_lo['s2fft']:.4f} "
                    f"cold1_kernel={cold_lo['kernel']:.4f} "
                    f"coldhi_batch={batch_hi} "
                    f"coldhi_s2fft={cold_hi['s2fft']:.4f} "
                    f"coldhi_kernel={cold_hi['kernel']:.4f} "
                    f"slope_s2fft={slope['s2fft']:.6f} "
                    f"slope_kernel={slope['kernel']:.6f} "
                    f"warmslope_s2fft={warm_slope['s2fft']:.6f} "
                    f"warmslope_kernel={warm_slope['kernel']:.6f} "
                    f"crossover={crossover_text}"
                )
                rows.append(
                    (
                        spin,
                        nside,
                        niter,
                        resident_mib,
                        build,
                        cold_lo["s2fft"],
                        cold_lo["kernel"],
                        slope["s2fft"],
                        slope["kernel"],
                        crossover,
                    )
                )
                if confirm and crossover is not None:
                    _confirm(
                        crossover,
                        lmax,
                        nside,
                        niter,
                        spin,
                        reality,
                        npix,
                        cap=cap,
                    )

    print()
    print(
        "| spin | nside | niter | resident (MiB) | build (s) | "
        "cold s2fft (s) | cold kernel (s) | slope s2fft (s/map) | "
        "slope kernel (s/map) | crossover B* |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        spin, nside, niter, mib, build = row[:5]
        cold_s, cold_k, slope_s, slope_k, crossover = row[5:]
        crossover_text = "never" if crossover is None else f"{crossover:.1f}"
        print(
            f"| {spin:+d} | {nside} | {niter} | {mib:.2f} | {build:.3f} | "
            f"{cold_s:.3f} | {cold_k:.3f} | {slope_s:.6f} | "
            f"{slope_k:.6f} | {crossover_text} |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=SECTIONS,
        default=list(DEFAULT_SECTIONS),
        help=(
            "Which benchmark sections to run. Defaults to "
            f"{' '.join(DEFAULT_SECTIONS)}; 'sweep' and 'fit' are "
            "superseded diagnostics, kept for provenance."
        ),
    )
    parser.add_argument(
        "--fit-nsides",
        nargs="+",
        type=int,
        default=list(FIT_NSIDES),
        help="Resolutions for the component fit.",
    )
    parser.add_argument(
        "--fit-spins",
        nargs="+",
        type=int,
        default=list(FIT_SPINS),
        help="Spin weights for the component fit.",
    )
    parser.add_argument(
        "--fit-niters",
        nargs="+",
        type=int,
        default=list(FIT_NITERS),
        help="Refinement counts for the component fit.",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the confirmation pass around each derived crossover.",
    )
    parser.add_argument(
        "--confirm-cap",
        type=int,
        default=CONFIRM_BATCH_CAP,
        help=(
            "Largest batch the confirmation pass may time. Raise it to "
            "confirm a large crossover, at proportionate cost."
        ),
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    if "grid" in args.sections:
        grid_benchmark()
        print()
    if "sweep" in args.sections:
        batch_sweep()
        print()
    if "ladder" in args.sections:
        ladder_benchmark(LADDER_CONFIGS)
        print()
    if "fit" in args.sections:
        component_fit(
            args.fit_nsides,
            args.fit_spins,
            args.fit_niters,
            not args.no_confirm,
            cap=args.confirm_cap,
        )


if __name__ == "__main__":
    main()
