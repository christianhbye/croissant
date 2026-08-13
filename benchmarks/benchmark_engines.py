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

Finally, sweeps batch size at three fixed resolutions (scalar,
nside=8/16/32, niter=0) to find the crossover where the kernel
engine's setup-plus-first-call starts beating s2fft's at each
resolution -- the kernel build cost grows with nside, so the
crossover is expected to move with it, and the sweep checks whether a
single `_AMORTISATION_THRESHOLD` (Task 7) can represent that or
whether the threshold needs to scale with nside (or, as a proxy for
build cost, with `kernel_nbytes`).
"""

import os
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "1")

MEMORY_CAP_MIB = 1024
SCALAR_NSIDES = (8, 16, 32)
SPIN_NSIDES = (8, 16)
NITERS = (0, 3)
REPEATS = 3
BATCH_SWEEP_NSIDES = (8, 16, 32)
BATCH_SWEEP_SIZES = (1, 2, 4, 8, 16, 32)


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


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from croissant import footprints, kernel, sphere

    jax.config.update("jax_enable_x64", True)
    rows = []

    for spin, nsides in ((0, SCALAR_NSIDES), (2, SPIN_NSIDES)):
        reality = spin == 0
        for nside in nsides:
            lmax = 2 * nside - 1
            npix = 12 * nside**2
            rng = np.random.default_rng(0)
            data = rng.normal(size=(4, npix))
            if not reality:
                data = data + 1j * rng.normal(size=(4, npix))
            data = jnp.asarray(data)

            kernel_mib = (
                kernel.kernel_nbytes(
                    lmax, "healpix", nside=nside, reality=reality
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
                    kernel.clear_kernel_cache()
                    sphere.clear_dense_matrix_cache()
                    jax.clear_caches()

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
    for sweep_nside in BATCH_SWEEP_NSIDES:
        sweep_lmax = 2 * sweep_nside - 1
        sweep_npix = 12 * sweep_nside**2
        sweep_kernel_mib = (
            kernel.kernel_nbytes(
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
            rng = np.random.default_rng(0)
            data = jnp.asarray(rng.normal(size=(batch_size, sweep_npix)))
            setup_times = {}
            for engine in ("s2fft", "kernel"):
                kernel.clear_kernel_cache()
                sphere.clear_dense_matrix_cache()
                jax.clear_caches()

                def run():
                    return sphere.compute_alm(
                        data,
                        sweep_lmax,
                        "healpix",
                        nside=sweep_nside,
                        niter=0,
                        spin=0,
                        reality=True,
                        engine=engine,
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


if __name__ == "__main__":
    main()
