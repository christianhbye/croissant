import healpy as hp
import numpy as np
import pytest

from croissant import sky, healpix

def test_power_law_map():
    nside = 1
    npix = hp.nside2npix(nside)
    freq = 1
    data = np.arange(npix).reshape(1, -1)
    test_sky = sky.Sky(data, frequencies=freq)
    freq_out = np.linspace(1, 50)
    beta = 3
    test_sky.power_law_map(freq_out, spectral_index=beta)
    assert np.allclose(test_sky.frequencies, freq_out)
    for pix in range(npix):  # check that there's a power law at each pixel
        temp = test_sky.data[:, pix]
        pl = data[0, pix] * freq_out**beta
        assert np.allclose(temp, pl)
