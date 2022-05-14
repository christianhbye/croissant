import numpy as np
import pytest
from croissant import healpix as hp

NSIDE = 10  # this is invalid for NESTED but OK for RINGG
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


hpm = hp.HealpixMap(*args, **kwargs)  # should work

def test_npix():
    assert hpm.npix == NPIX
    assert hpm.npix == data.shape[-1]
