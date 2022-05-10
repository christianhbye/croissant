import healpy as hp

def nside2npix(nside):
    npix = 12 * nside **2
    return npix

# nside's for which pixel weights exist
PIX_WEIGHTS_NSIDE = [32, 64, 128, 256, 512, 1024, 2048, 4096]

class HealpixBase:
    def __init__(self, nside, data=None, alm=None, nested_input=False):
        hp.check_nside(nside, nest=nested_input)
        self.nside = nside
        self.npix = nside2npix(self.nside)
        if nested_input and data is not None:
            ix = hp.nest2ring(self.nside, np.arange(npix))
            data = data[ix]
        if data is None:
            if alm is not None:
                data = alm2hp(alm)
        else:
            if alm is None:
                alm = hp2alm(data)
            else:
                # consistency check
                # np.allclose(alm, hp2alm(data))

        self.data = data
        self.alm = alm

    def hp2alm(self, lmax=None):
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
        use_ring_weights = not use_pix_weigths
        alm =  hp.map2alm(
                self.data,
                lmax=lmax,
                mmax=lmax,
                use_ring_weights=use_ring_weights,
                use_pixel_weights=use_pix_weights,
            )
        return alm

    def plot(self, projection="mollweide", **kwargs):
        if self.data is None:
            raise ValueError("data is None, there is nothing to plot.")
        views = {
                "mollweide": hp.mollview,
                "gnomonic": hp.gnomview,
                "cartesian": hp.cartview,
                "ortographic": hp.orthview,
                }
        if not projection in views:
            raise ValueError(
                    f"Projection must be in {list[views.keys()]}, not"
                    f"{projection}."
            )
        func = views[projection]
        func(self.data, **kwargs)
