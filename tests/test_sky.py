import healpy as hp
import numpy as np

from croissant.sky import Sky


def test_power_law_map():
    nside = 32
    npix = hp.nside2npix(nside)
    freq = 25.0
    data = np.arange(npix)
    sky = Sky(data, frequencies=freq)
    freq_out = np.linspace(1, 50)
    beta = -2.5
    sky.power_law_map(freq_out, spectral_index=beta)
    assert np.allclose(sky.frequencies, freq_out)
    for pix in range(npix):  # check that there's a power law at each pixel
        temp = sky.data[:, pix]
        f_ratio = freq_out / freq
        f_ratio.shape = (1, -1)
        pl = data[pix] * f_ratio**beta
        assert np.allclose(temp, pl)
