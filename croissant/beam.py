from .healpix import HealpixBase, Alm


class Beam(Alm):
    def __init__(self, power, frequencies=None, from_grid=False, **kwargs):
        if from_grid:
            req_kwargs = ["theta", "phi"]
        else:
            req_kwargs = ["nside"]

        if not all([k in kwargs for k in req_kwargs]):
            raise ValueError(f"Not all kwargs in {req_kwargs} are provided.")

        if from_grid:
            theta = kwargs.pop["theta"]
            phi = kwargs.pop["phi"]
            super().from_grid()
        else:
            nside = kwargs.pop["nside"]
            super().__init__(nside, data=power, frequencies=frequencies)

        self.power = self.data

    @classmethod
    def from_file(path):
        raise NotImplementedError

    def to_file(fname):
        raise NotImplementedError
