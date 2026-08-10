"""
Regression tests certifying the pinned s2fft revision.

The polarized HEALPix path requires the fix in slosar/s2fft@cefdf46
(upstream PR astro-informatics/s2fft#387): the on-the-fly Wigner-d
recursion hits exact zero nodes at the rational cos(theta) values of
HEALPix rings, injecting NaNs that nansum silently drops and producing
percent-level spin-2 errors in both transform directions.

These tests fail on s2fft 1.4.0 and pass at the pinned revision. They
deliberately avoid forward/inverse round trips, whose related errors
partially cancel:

- the forward transform is checked against analytically sampled
  spin-weighted spherical harmonics (external ground truth), exercising
  the exact call croissant's ``compute_alm`` makes;
- the inverse transform is checked against s2fft's independent
  Turok-recursion base implementation, mirroring the upstream fix's own
  test.
"""

from math import factorial

import healpy
import numpy as np
import pytest
import s2fft
from s2fft.base_transforms import spherical as s2fft_base

NSIDE = 16
L = 2 * NSIDE  # s2fft's HEALPix transforms require L >= 2 * nside


def wigner_d(ell, m, n, beta):
    """Wigner small-d matrix element from the explicit sum formula."""
    pref = np.sqrt(
        factorial(ell + m)
        * factorial(ell - m)
        * factorial(ell + n)
        * factorial(ell - n)
    )
    d = np.zeros_like(beta)
    for k in range(max(0, m - n), min(ell + m, ell - n) + 1):
        denom = (
            factorial(ell + m - k)
            * factorial(k)
            * factorial(ell - n - k)
            * factorial(n - m + k)
        )
        d = d + (
            (-1) ** (n - m + k)
            / denom
            * np.cos(beta / 2) ** (2 * ell + m - n - 2 * k)
            * np.sin(beta / 2) ** (n - m + 2 * k)
        )
    return pref * d


def spin_spherical_harmonic(spin, ell, m, theta, phi):
    """Spin-weighted spherical harmonic sYlm (McEwen & Wiaux convention,
    as used by s2fft, including the Condon-Shortley phase)."""
    return (
        (-1) ** (spin + abs(m))
        * np.sqrt((2 * ell + 1) / (4 * np.pi))
        * wigner_d(ell, m, -spin, theta)
        * np.exp(1j * m * phi)
    )


@pytest.mark.parametrize("spin", [-2, 2])
@pytest.mark.parametrize("ell, emm", [(2, 0), (3, 1), (5, -3)])
def test_healpix_spin_forward_recovers_analytic_harmonic(spin, ell, emm):
    """Forward spin transform of an analytically sampled sYlm must
    return a delta in (ell, m). This is croissant's compute_alm call."""
    theta, phi = healpy.pix2ang(NSIDE, np.arange(12 * NSIDE**2))
    f = spin_spherical_harmonic(spin, ell, emm, theta, phi)
    flm = np.asarray(
        s2fft.forward(
            f,
            L,
            spin=spin,
            nside=NSIDE,
            sampling="healpix",
            method="jax",
            reality=False,
            iter=3,
        )
    )
    expected = np.zeros((L, 2 * L - 1), dtype=complex)
    expected[ell, L - 1 + emm] = 1.0
    np.testing.assert_allclose(flm, expected, atol=1e-5)


@pytest.mark.parametrize("spin", [-2, 2])
def test_healpix_spin_inverse_matches_base_transform(spin):
    """On-the-fly inverse spin transform must match s2fft's independent
    Turok-recursion base implementation on the same HEALPix grid."""
    rng = np.random.default_rng(8128)
    flm = np.zeros((L, 2 * L - 1), dtype=complex)
    for ell in range(abs(spin), L):
        for emm in range(-ell, ell + 1):
            flm[ell, L - 1 + emm] = rng.normal() + 1j * rng.normal()
    f = np.asarray(
        s2fft.inverse(
            flm,
            L,
            spin=spin,
            nside=NSIDE,
            sampling="healpix",
            method="jax",
            reality=False,
        )
    )
    f_base = s2fft_base.inverse(
        flm, L, spin=spin, nside=NSIDE, sampling="healpix"
    )
    np.testing.assert_allclose(f, f_base, atol=1e-12)
