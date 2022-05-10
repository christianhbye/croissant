import healpy as hp
import numpy as np
import warnings


def nside2npix(nside):
    npix = 12 * nside**2
    return npix

def check_shapes(npix, data, alm, frequencies):
    data_check = True
    alm_check = True
    
    if frequencies is None:
        expected_data_shape = (npix,)
    else:
        nfreq = len(frequencies)
        expected_data_shape = (nfreq, npix)
        if alm is not None:
            alm_check = np.shape(alm)[0] == nfreq
        

    if data is not None:
        data_check = np.shape(data) == expected_data_shape

    if not (data_check and alm_check):
        raise ValueError("Shape mismatch in npix, data, alm, frequencies.")

# nside's for which pixel weights exist
PIX_WEIGHTS_NSIDE = [32, 64, 128, 256, 512, 1024, 2048, 4096]


class HealpixBase:
    def __init__(
            self,
            nside,
            data=None,
            nested_input=False,
            alm=None,
            frequencies=None
        ):
        hp.check_nside(nside, nest=nested_input)
        self.nside = nside
        self.npix = nside2npix(self.nside)
        check_shapes(self.npix, data, alm, frequencies)

        if data is not None:
            if nested_input:
                ix = hp.nest2ring(self.nside, np.arange(self.npix))
                if frequencies is None:
                    data = data[ix]
                else:
                    data = data[:, ix]
            if alm is None:
                alm = self.hp2alm()
            elif not np.allclose(alm, self.hp2alm()):
                raise ValueError("Provided alm and map are inconsistent.")
        else:
            if alm is not None:
                data = self.alm2hp()

        self.data = data
        self.alm = alm
        self.frequencies = frequencies

    def alm2hp(self, lmax=None):
        if self.alm is None:
            raise ValueError("alm is None, cannot convert to map.")
        hp_map = hp.alm2map(self.alm, self.nside, lmax=lmax, mmax=lmax)
        return hp_map

    def alm2hp(self, lmax=None):
        if self.data is None:
            raise ValueError("data is None, cannot compute alms.")
        if lmax is None:
            lmax = 3 * self.nside - 1
        use_pix_weights = self.nside in PIX_WEIGHTS_NSIDE
        use_ring_weights = not use_pix_weights
        alm = hp.map2alm(
            self.data,
            lmax=lmax,
            mmax=lmax,
            use_ring_weights=use_ring_weights,
            use_pixel_weights=use_pix_weights,
        )
        return alm

    def plot(self, **kwargs):
        m = kwargs.pop("m", self.data)
        _ = hp.projview(m=m, projection_type=projection)
