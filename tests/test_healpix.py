from copy import deepcopy
import healpy
import numpy as np
import pytest
from croissant import healpix as hp


freqs = np.linspace(1, 50, 50)


def test_healpix2lonlat():
    nside = 32
    pix = 1  # should be the north pole
    lon, lat = hp.healpix2lonlat(nside, pix=pix)
    assert lat > 85  # should be reasonably close to 90

    lons, lats = hp.healpix2lonlat(nside, pix=None)
    assert lons.shape == lats.shape == (healpy.nside2npix(nside),)
    assert 0 <= lons.all() < 360
    assert -90 <= lats.all() < 90


def test_nested_check():
    nside = 10  # this is invalid for NESTED but OK for RING
    npix = healpy.nside2npix(nside)
    data = np.random.rand(freqs.size, npix)
    args = [nside]
    kwargs = {"data": data, "nested_input": True, "frequencies": freqs}
    # it should raise an error
    with pytest.raises(ValueError):
        hp.HealpixMap(*args, **kwargs)
    kwargs["nested_input"] = False
    hp.HealpixMap(*args, **kwargs)  # should work


NSIDE = 8  # make it valid for nested so that the ud_grade can run
args = [NSIDE]
NPIX = healpy.nside2npix(NSIDE)
kwargs = {"nested_input": False, "frequencies": freqs}


def test_npix():
    data = np.ones((freqs.size, NPIX))
    kwargs["data"] = data
    hpm = hp.HealpixMap(*args, **kwargs)
    assert hpm.npix == NPIX
    assert hpm.npix == hpm.data.shape[-1]


def test_ud_grade():
    data = np.ones((freqs.size, NPIX)) * np.arange(NPIX).reshape(1, -1)
    kwargs["data"] = data
    hpm = hp.HealpixMap(*args, **kwargs)
    nside_out = [1, 2, 8, 32]
    for ns in nside_out:
        hp_copy = deepcopy(hpm)
        hp_copy.ud_grade(ns)
        assert hp_copy.nside == ns
        assert np.allclose(hp_copy.data, healpy.ud_grade(hpm.data, ns))


def test_alm():
    kwargs["frequencies"] = None
    AVG_VALUE = 10.0
    kwargs["data"] = AVG_VALUE * np.ones(NPIX)  # constant map
    hpm = hp.HealpixMap(*args, **kwargs)
    # constant map
    alm = hp.Alm.from_healpix(hpm)
    assert alm.lmax == 3 * NSIDE - 1  # default lmax
    a00 = alm.get_coeff(0, 0)
    assert np.allclose(a00, AVG_VALUE * np.sqrt(4 * np.pi))
    alm.set_coeff(0, 0, 0)  # set a00 to 0, all coeffs should be 0 after this
    assert np.allclose(alm.alm.real, 0, atol=1e-2)
    assert np.allclose(alm.alm.imag, 0, atol=1e-2)
