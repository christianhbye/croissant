"""
Regression tests for polarized spin transport under frame rotation.

Croissant's public Stokes convention is IAU (U_IAU = -U_COSMO), and its
harmonic machinery is s2fft's, whose spin-weighted basis is the Goldberg
/ McEwen & Wiaux one. Together these fix, with no freedom left, which
complex Stokes combination is the spin +2 object:

    Q - i U_IAU = Q + i U_COSMO = sum_lm -(E + iB)_lm 2Y_lm,

the standard CMB relation, certified against healpy's
alm2map(pol=True) at machine precision by the pairing test in this
file. So Q - iU must be analyzed at spin +2 and Q + iU at spin -2. If
the labels are mismatched, statics remain exact (a fixed-spin harmonic
contraction is complete for any spin, and sky and beam use matching
labels) and z-rotations commute with the error (zero transport phase)
-- but frame rotations with beta != 0 apply the complex conjugate
exp(+2i psi) of the physical transport phase exp(-2i psi), an
order-unity Q/U error.

These tests pin the healpy pairing itself (alm2map(pol=True) against
analytically sampled spin harmonics), the matched pairing through
croissant's own analysis (band-limited duals of a single E-mode), and
the transport itself (gal->FK5 rotation of an E-only sky against a
reference that needs only scalar rotation of E, since E-modes rotate
as scalars).
"""

import healpy as hp
import numpy as np
import s2fft

from croissant.polarization import PolarizedSky

from .test_s2fft_pin import spin_spherical_harmonic

LMAX = 8
L = LMAX + 1


def _mwss_grid():
    theta = np.asarray(s2fft.sampling.s2_samples.thetas(L=L, sampling="mwss"))
    phi = np.asarray(
        s2fft.sampling.s2_samples.phis_equiang(L=L, sampling="mwss")
    )
    return theta[:, None], phi[None, :]


def _to_2d(packed):
    """healpy-packed alm of a real field -> s2fft 2D (ell, m) layout."""
    out = np.zeros((L, 2 * LMAX + 1), dtype=complex)
    for ell in range(L):
        for emm in range(ell + 1):
            val = packed[hp.Alm.getidx(LMAX, ell, emm)]
            out[ell, LMAX + emm] = val
            if emm > 0:
                out[ell, LMAX - emm] = (-1) ** emm * np.conj(val)
    return out


def _qu_iau_from_e(alm_e_2d, theta, phi):
    """(Q, U_IAU) maps of an E-only sky via the certified relation
    Q - iU_IAU = sum_lm -(E)_lm 2Y_lm."""
    fm = np.zeros(np.broadcast(theta, phi).shape, dtype=complex)
    for ell in range(2, L):
        for emm in range(-ell, ell + 1):
            coeff = alm_e_2d[ell, LMAX + emm]
            if coeff != 0:
                fm -= coeff * spin_spherical_harmonic(2, ell, emm, theta, phi)
    return fm.real, -fm.imag


def _random_real_alm(rng):
    """Random healpy-packed alm of a real field, ell >= 2 only."""
    alm = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    for ell in range(2, L):
        for emm in range(ell + 1):
            alm[hp.Alm.getidx(LMAX, ell, emm)] = rng.normal() + (
                1j * rng.normal() if emm > 0 else 0.0
            )
    return alm


def test_healpy_polarized_synthesis_matches_analytic_pairing():
    """hp.alm2map(pol=True) must implement the pairing

        (Q + i U_COSMO) = sum_lm -(E + iB)_lm 2Y_lm

    with 2Y_lm the Goldberg / s2fft harmonic of
    spin_spherical_harmonic (itself certified against s2fft.forward).
    This is the one external convention relation the transport tests
    in this file rest on. Measured agreement is ~6e-15 of field scale;
    the sign-flipped pairing +(E + iB) and the conjugate pairing
    -(E - iB) both miss at order unity."""
    nside = 16
    rng = np.random.default_rng(20260812)
    alm_t = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    alm_e = _random_real_alm(rng)
    alm_b = _random_real_alm(rng)
    _, q_map, u_map = hp.alm2map(
        [alm_t, alm_e, alm_b], nside, lmax=LMAX, pol=True
    )
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    e_2d = _to_2d(alm_e)
    b_2d = _to_2d(alm_b)
    p_cosmo = np.zeros(theta.shape, dtype=complex)
    for ell in range(2, L):
        for emm in range(-ell, ell + 1):
            coeff = e_2d[ell, LMAX + emm] + 1j * b_2d[ell, LMAX + emm]
            p_cosmo -= coeff * spin_spherical_harmonic(2, ell, emm, theta, phi)
    scale = np.abs(p_cosmo).max()
    np.testing.assert_allclose(q_map, p_cosmo.real, atol=1e-13 * scale)
    np.testing.assert_allclose(u_map, p_cosmo.imag, atol=1e-13 * scale)


def _e_mode_sky(packed_e):
    theta, phi = _mwss_grid()
    q_map, u_map = _qu_iau_from_e(_to_2d(packed_e), theta, phi)
    data = np.zeros((1, 4) + q_map.shape)
    data[0, 1] = q_map
    data[0, 2] = u_map
    return PolarizedSky(data, [10.0], sampling="mwss", coord="galactic")


def test_polarized_duals_of_single_e_mode_are_band_limited():
    """A single E-mode must produce polarized duals supported only at
    its own (ell, |m|). Mismatched spin labels expand each dual in the
    opposite-spin basis, smearing O(1) power over every ell."""
    ell0, emm0 = 3, 1
    packed = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    packed[hp.Alm.getidx(LMAX, ell0, emm0)] = 1.0
    sky = _e_mode_sky(packed)
    alm = np.asarray(sky.compute_alm())
    for comp in (2, 3):
        dual = alm[0, comp]
        support = np.zeros(dual.shape, dtype=bool)
        support[ell0, LMAX + emm0] = True
        support[ell0, LMAX - emm0] = True
        assert np.abs(dual[support]).max() > 0.5
        np.testing.assert_allclose(dual[~support], 0.0, atol=1e-8)


def test_gal_to_fk5_rotation_transports_polarization():
    """gal->FK5 rotation of an E-only sky must match the reference
    built from scalar rotation of E. The conjugate transport (or any
    missing transport phase) misses at order unity.

    Under the matched pairing, component 2 (spin -2) analyzes
    Q + iU_IAU, so synthesizing it at spin -2 returns Q + iU_IAU in
    the target frame."""
    packed = _random_real_alm(np.random.default_rng(2026))
    sky = _e_mode_sky(packed)
    alm_eq = np.asarray(sky.compute_alm_eq(world="earth"))

    rotated_e = hp.Rotator(coord=["G", "C"]).rotate_alm(packed, lmax=LMAX)
    theta, phi = _mwss_grid()
    q_true, u_true = _qu_iau_from_e(_to_2d(rotated_e), theta, phi)

    f_minus = np.asarray(
        s2fft.inverse(
            alm_eq[0, 2],
            L,
            spin=-2,
            sampling="mwss",
            method="jax",
            reality=False,
        )
    )
    scale = np.abs(q_true + 1j * u_true).max()
    np.testing.assert_allclose(f_minus.real, q_true, atol=1e-8 * scale)
    np.testing.assert_allclose(f_minus.imag, u_true, atol=1e-8 * scale)
