"""
Tests for the shared analytic spin-harmonic reference helpers.

``harmonics_reference`` holds the naive, closed-form spin-weighted
spherical harmonics used as an independent oracle by the s2fft pin tests
and the spin-transport tests. It is deliberately NOT production code:
the explicit factorial sum overflows a float64 for ell >= 58, and its
value lies in being derived from a textbook formula rather than from the
recursions croissant and s2fft actually use.

These tests pin the oracle itself against an outside implementation, so
that a convention slip in the reference cannot silently propagate into
the tests that depend on it.
"""

import numpy as np
import pytest
from scipy import special

from .harmonics_reference import spin_spherical_harmonic, wigner_d


@pytest.mark.parametrize("ell, emm", [(0, 0), (2, 0), (3, 1), (5, -3), (4, 2)])
def test_spin_zero_matches_scipy_spherical_harmonic(ell, emm):
    """At spin 0 the reference reduces to the ordinary Ylm.

    scipy's ``sph_harm_y`` is an independent implementation with the same
    Condon-Shortley phase, so agreement here fixes the normalisation,
    the phase convention and the theta/phi argument order all at once.
    """
    theta = np.linspace(0.2, np.pi - 0.2, 5)[:, None]
    phi = np.linspace(0.0, 2 * np.pi, 4, endpoint=False)[None, :]

    got = spin_spherical_harmonic(0, ell, emm, theta, phi)
    expected = special.sph_harm_y(ell, emm, theta, phi)

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("spin", [-2, -1, 0, 1, 2])
def test_conjugation_relation(spin):
    """sYlm* = (-1)^(s+m) (-s)Yl(-m), the Goldberg conjugation relation.

    This is the identity that fixes the relative sign between opposite
    spins, and the one an odd-spin sign error in ``wigner_d`` breaks --
    the failure mode that produced the earlier (-1)^spin offset.
    """
    ell, emm = 4, 2
    theta = np.linspace(0.2, np.pi - 0.2, 5)[:, None]
    phi = np.linspace(0.0, 2 * np.pi, 4, endpoint=False)[None, :]

    lhs = np.conj(spin_spherical_harmonic(spin, ell, emm, theta, phi))
    rhs = (-1.0) ** (spin + emm) * spin_spherical_harmonic(
        -spin, ell, -emm, theta, phi
    )

    np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-13)


def test_wigner_d_is_orthogonal_at_fixed_ell():
    """The Wigner-d matrix is real orthogonal: sum_n d_{m,n} d_{m',n} =
    delta_{m,m'}. A transposed-index or prefactor slip breaks this."""
    ell, beta = 5, np.array(0.7)
    ms = range(-ell, ell + 1)
    d = np.array([[float(wigner_d(ell, m, n, beta)) for n in ms] for m in ms])

    np.testing.assert_allclose(d @ d.T, np.eye(2 * ell + 1), atol=1e-12)


def test_reference_overflows_above_stable_range():
    """Document the ceiling that keeps this out of production code.

    The explicit factorial sum forms a product of four factorials before
    casting to float, so it dies for ell >= 58 rather than degrading. Any
    production use of spin harmonics must therefore come from a stable
    recursion, not from this oracle.
    """
    theta, phi = np.array([0.7]), np.array([0.3])

    spin_spherical_harmonic(2, 55, 0, theta, phi)  # inside the range

    with pytest.raises(OverflowError):
        spin_spherical_harmonic(2, 58, 0, theta, phi)
