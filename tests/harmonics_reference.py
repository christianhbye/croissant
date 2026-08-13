"""
Analytic spin-weighted spherical harmonics, as a test oracle.

These are closed-form, deliberately naive implementations, kept
independent of the recursions that croissant and s2fft actually use so
that they can certify them. They live here rather than in a test module
because several test files depend on them: the s2fft pin tests
(``test_s2fft_pin.py``, deleted when the pin is dropped) and the
spin-transport tests, which must keep working across that deletion.

NOT production code. The explicit factorial sum builds a product of four
factorials before casting to float, so it raises ``OverflowError`` for
ell >= 58 rather than losing precision gracefully; ``test_harmonics_
reference.py`` pins that ceiling. Production spin harmonics must come
from a numerically stable recursion.
"""

from math import factorial

import numpy as np


def wigner_d(ell, m, n, beta):
    """Wigner small-d matrix element d^ell_{m,n}(beta) from the
    explicit sum formula (Wikipedia/Varshalovich convention). The
    factorial products are cast to float before sqrt/division since
    they exceed int64 for ell + |m| >= 21."""
    pref = np.sqrt(
        float(
            factorial(ell + m)
            * factorial(ell - m)
            * factorial(ell + n)
            * factorial(ell - n)
        )
    )
    d = np.zeros_like(beta)
    for k in range(max(0, n - m), min(ell + n, ell - m) + 1):
        denom = float(
            factorial(ell + n - k)
            * factorial(k)
            * factorial(ell - m - k)
            * factorial(m - n + k)
        )
        d = d + (
            (-1) ** (m - n + k)
            / denom
            * np.cos(beta / 2) ** (2 * ell + n - m - 2 * k)
            * np.sin(beta / 2) ** (m - n + 2 * k)
        )
    return pref * d


def spin_spherical_harmonic(spin, ell, m, theta, phi):
    """Spin-weighted spherical harmonic sYlm in the Goldberg /
    McEwen & Wiaux convention used by s2fft (Condon-Shortley phase
    included): (-1)^s sqrt((2l+1)/4pi) d^l_{m,-s}(theta) *
    e^{i m phi}.

    An earlier version computed the transposed d^l_{n,m} and
    compensated with a (-1)^(s+|m|) prefactor; the cancellation is
    exact only for even spin and flipped the sign at odd spin."""
    return (
        (-1) ** spin
        * np.sqrt((2 * ell + 1) / (4 * np.pi))
        * wigner_d(ell, m, -spin, theta)
        * np.exp(1j * m * phi)
    )
