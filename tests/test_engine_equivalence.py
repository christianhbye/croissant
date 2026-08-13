"""
Cross-engine equivalence for the spherical harmonic transform.

Croissant's engines differ only in how much of the pixels-to-alm map is
precomputed, never in the map itself, so they must agree to near machine
precision. ``engine="auto"`` relies on that: if the engines could
disagree, switching engines for resource reasons would change results.

HEALPix with ``niter > 0`` is the case worth pinning. The engines reach
refinement by different routes -- the s2fft engine iterates inside s2fft,
the dense engine folds the same iteration into its cached matrix in gram
form -- so agreement here is a real constraint rather than a tautology.
"""

import numpy as np
import pytest

from croissant import sphere

NSIDE = 8
# s2fft's HEALPix transforms require L = lmax + 1 >= 2 * nside.
LMAX = 2 * NSIDE - 1

ENGINES = ["s2fft", "dense"]


def _healpix_data(rng, nfreq=2, complex_=False):
    """Random HEALPix data of shape (nfreq, npix)."""
    npix = 12 * NSIDE**2
    data = rng.normal(size=(nfreq, npix))
    if complex_:
        data = data + 1j * rng.normal(size=(nfreq, npix))
    return data


def _assert_engines_agree(a, b, atol_rel=1e-10):
    """Assert two alm arrays agree relative to the larger one's scale."""
    a = np.asarray(a)
    b = np.asarray(b)
    scale = max(np.abs(a).max(), np.abs(b).max())
    np.testing.assert_allclose(a, b, rtol=0, atol=atol_rel * scale)


@pytest.mark.parametrize("niter", [0, 1, 3])
def test_scalar_healpix_engines_agree(niter):
    """Real scalar HEALPix analysis is engine-independent."""
    data = _healpix_data(np.random.default_rng(0))
    kwargs = dict(lmax=LMAX, sampling="healpix", nside=NSIDE, niter=niter)
    reference = sphere.compute_alm(data, engine="s2fft", **kwargs)
    for engine in ENGINES[1:]:
        got = sphere.compute_alm(data, engine=engine, **kwargs)
        _assert_engines_agree(reference, got)


@pytest.mark.parametrize("niter", [0, 3])
@pytest.mark.parametrize("spin", [2, -2])
def test_spin_healpix_engines_agree(niter, spin):
    """Spin-weighted HEALPix analysis is engine-independent."""
    data = _healpix_data(np.random.default_rng(1), complex_=True)
    kwargs = dict(
        lmax=LMAX,
        sampling="healpix",
        nside=NSIDE,
        niter=niter,
        spin=spin,
        reality=False,
    )
    reference = sphere.compute_alm(data, engine="s2fft", **kwargs)
    for engine in ENGINES[1:]:
        got = sphere.compute_alm(data, engine=engine, **kwargs)
        _assert_engines_agree(reference, got)


def test_mwss_engines_agree():
    """Equiangular sampling is engine-independent (niter is irrelevant
    there: MWSS satisfies a sampling theorem, so analysis is exact)."""
    import s2fft

    L = LMAX + 1
    ntheta = s2fft.sampling.s2_samples.ntheta(L=L, sampling="mwss")
    nphi = s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling="mwss")
    rng = np.random.default_rng(2)
    data = rng.normal(size=(2, ntheta, nphi))
    kwargs = dict(lmax=LMAX, sampling="mwss")
    reference = sphere.compute_alm(data, engine="s2fft", **kwargs)
    for engine in ENGINES[1:]:
        got = sphere.compute_alm(data, engine=engine, **kwargs)
        _assert_engines_agree(reference, got)
