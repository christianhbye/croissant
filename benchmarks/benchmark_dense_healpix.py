"""Benchmark the low-bandlimit HEALPix dense-analysis cache."""

import os
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "1")


def main():
    import jax
    import jax.numpy as jnp

    from croissant.dense import DenseSphericalTransform

    jax.config.update("jax_enable_x64", True)
    nside = 32
    lmax = 30
    npix = 12 * nside**2
    data = jnp.linspace(-1.0, 1.0, npix, dtype=jnp.float64)
    transforms = []
    started = perf_counter()
    for spin in (0, -2, 2):
        transform_started = perf_counter()
        transform = DenseSphericalTransform(
            lmax,
            "healpix",
            nside=nside,
            spin=spin,
            dtype=jnp.complex128,
        )
        transform.matrix.block_until_ready()
        built = perf_counter() - transform_started
        transforms.append(transform)
        print(
            f"spin={spin:+d} build_seconds={built:.6f} "
            f"matrix_mib={transform.matrix.nbytes / 2**20:.3f}"
        )
    total_matrix_mib = (
        sum(transform.matrix.nbytes for transform in transforms) / 2**20
    )
    print(f"three_matrix_mib={total_matrix_mib:.3f}")
    print(f"total_build_seconds={perf_counter() - started:.6f}")
    from croissant.dense import dense_cache_nbytes

    print(f"cache_mib={dense_cache_nbytes() / 2**20:.3f}")

    for spin, transform in zip((0, -2, 2), transforms, strict=True):
        first_started = perf_counter()
        result = transform(data)
        result.block_until_ready()
        first = perf_counter() - first_started
        cached_started = perf_counter()
        transform(data).block_until_ready()
        cached = perf_counter() - cached_started
        print(
            f"spin={spin:+d} first_apply_seconds={first:.6f} "
            f"cached_apply_seconds={cached:.6f}"
        )


if __name__ == "__main__":
    main()
