from copy import deepcopy
import healpy
import numpy as np
import pytest
from croissant import healpix as hp

NSIDE = 10  # this is invalid for NESTED but OK for RING
NPIX = hp.nside2npix(NSIDE)
freqs = np.linspace(50, 100, 51)
data = np.random.rand(freqs.size, NPIX)
args = [NSIDE]
kwargs = {"data": data, "nested_input": False, "frequencies": freqs}

def test_nested_check():
    hpm = hp.HealpixMap(*args, **kwargs)  # should work
    kwargs["nested_input"] = True  # now it should raise an error
    with pytest.raises(ValueError) as e:
        hpm = hp.HealpixMap(*args, **kwargs)


hpm = hp.HealpixMap(*args, **kwargs)

def test_npix():
    assert hpm.npix == NPIX
    assert hpm.npix == hpm.data.shape[-1]

def test_ud_grade():
    nside_out = [1, 2, 10, 20, 100, 1000]
    for ns in nside_out:
        hp_copy = deepcopy(hpm)
        hp_copy.ud_grade(ns)
        assert hp_copy.nside == ns
        assert np.allclose(hp_copy.data, healpy.ud_grade(hpm.data, ns))

# test alm
kwargs["frequencies"] = None
AVG_VALUE = 10.
kwargs["data"] = AVG_VALUE * np.ones(NPIX)  # constant map
hpm = hp.HealpixMap(*args, **kwargs)

def test_alm():
    # constant map
    alm = hp.Alm.from_healpix(hpm)
    assert alm.lmax == 3 * NSIDE - 1  # default lmax
    a00 = alm.get_coeff(0, 0)
    assert np.isclose(AVG_VALUE, a00)
    alm.set_coeff(0, 0, 0)  # set a00 to 0, all coeffs should be 0 after this
    assert np.allclose(alm.alm, 0)
