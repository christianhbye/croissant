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
from croissant.polarization import PairStokesBeam, PolarizedSky

NSIDE = 8
# s2fft's HEALPix transforms require L = lmax + 1 >= 2 * nside.
LMAX = 2 * NSIDE - 1

ENGINES = ["s2fft", "dense", "kernel"]

FREQS = [10.0, 20.0]


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


# Polarized fields select an engine per transform block rather than once
# per object, so equivalence has to be asserted on the assembled dual --
# a per-block slip (wrong kernel for a spin, or a kernel reused across
# the reality=True and reality=False spin-0 blocks) shows up there and
# nowhere else.


def _mwss_spatial_shape():
    """(ntheta, nphi) of the MWSS grid whose band-limit is ``LMAX``."""
    import s2fft

    L = LMAX + 1
    return (
        s2fft.sampling.s2_samples.ntheta(L=L, sampling="mwss"),
        s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling="mwss"),
    )


def _sky_data(rng, spatial_shape):
    """Real IQUV maps of shape (nfreq, 4, spatial...)."""
    return rng.normal(size=(len(FREQS), 4) + spatial_shape)


def _beam_data(rng, spatial_shape, npair=2):
    """Complex pair-response maps of shape (npair, nfreq, 4, spatial...)."""
    shape = (npair, len(FREQS), 4) + spatial_shape
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.mark.parametrize("niter", [0, 3])
def test_polarized_sky_healpix_engines_agree(niter):
    """The IQUV harmonic dual is engine-independent on HEALPix."""
    data = _sky_data(np.random.default_rng(3), (12 * NSIDE**2,))
    kwargs = dict(sampling="healpix", niter=niter)
    reference = PolarizedSky(
        data, FREQS, engine="s2fft", **kwargs
    ).compute_alm()
    for engine in ENGINES[1:]:
        got = PolarizedSky(data, FREQS, engine=engine, **kwargs).compute_alm()
        _assert_engines_agree(reference, got)


def test_polarized_sky_mwss_engines_agree():
    """MWSS exercises the second spin-2 transform.

    On HEALPix the P+ block is derived from P- by conjugation rather
    than transformed, so only MWSS proves the spin +2 block picks up its
    own engine and kernel.
    """
    data = _sky_data(np.random.default_rng(4), _mwss_spatial_shape())
    reference = PolarizedSky(
        data, FREQS, sampling="mwss", engine="s2fft"
    ).compute_alm()
    for engine in ENGINES[1:]:
        got = PolarizedSky(
            data, FREQS, sampling="mwss", engine=engine
        ).compute_alm()
        _assert_engines_agree(reference, got)


@pytest.mark.parametrize("niter", [0, 3])
def test_pair_response_healpix_engines_agree(niter):
    """The pair-response dual is engine-independent on HEALPix.

    The response's spin-0 block is complex, so it needs a different
    kernel from the sky's real spin-0 block at the same configuration.
    """
    data = _beam_data(np.random.default_rng(5), (12 * NSIDE**2,))
    pairs = [(0, 0), (0, 1)]
    kwargs = dict(sampling="healpix", niter=niter)
    reference = PairStokesBeam(
        data, FREQS, pairs, engine="s2fft", **kwargs
    ).compute_alm()
    for engine in ENGINES[1:]:
        got = PairStokesBeam(
            data, FREQS, pairs, engine=engine, **kwargs
        ).compute_alm()
        _assert_engines_agree(reference, got)


def test_pair_response_mwss_engines_agree():
    """The pair-response dual is engine-independent on MWSS."""
    data = _beam_data(np.random.default_rng(6), _mwss_spatial_shape())
    pairs = [(0, 0), (0, 1)]
    reference = PairStokesBeam(
        data, FREQS, pairs, sampling="mwss", engine="s2fft"
    ).compute_alm()
    for engine in ENGINES[1:]:
        got = PairStokesBeam(
            data, FREQS, pairs, sampling="mwss", engine=engine
        ).compute_alm()
        _assert_engines_agree(reference, got)
