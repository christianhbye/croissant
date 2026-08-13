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
def test_kernel_shape_and_size_prediction(reality):
    """kernel_nbytes predicts the footprint without building it.

    The last axis depends on ``reality``: a real-field kernel stores only
    m >= 0, so it is L wide rather than 2L-1.
    """
    predicted = kernel.kernel_nbytes(
        LMAX, "healpix", nside=NSIDE, reality=reality
    )
    k = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, spin=0, reality=reality
    )
    ntheta = 4 * NSIDE - 1
    nm = (LMAX + 1) if reality else (2 * LMAX + 1)
    assert k.shape == (ntheta, LMAX + 1, nm)
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
    k = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, spin=spin, forward=True
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
