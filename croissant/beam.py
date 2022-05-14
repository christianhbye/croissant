from .healpix import Alm


class Beam(Alm):
    def __init__(self, data=None, frequencies=None, from_grid=False, **kwargs):
        if from_grid:
            req_kwargs = ["theta", "phi"]
        else:
            req_kwargs = []

        if not all([k in kwargs for k in req_kwargs]):
            raise ValueError(f"Not all kwargs in {req_kwargs} are provided.")

        if from_grid:
            theta = kwargs["theta"]
            phi = kwargs["phi"]
            if data is None:
                raise ValueError("No data is provided.")
            super().from_grid(data, theta, phi, frequencies=frequencies)
        else:
            lmax = kwargs.pop("lmax", None)
            super().__init__(alm=data, lmax=lmax, frequencies=frequencies)

    @classmethod
    def from_file(path):
        raise NotImplementedError

    def to_file(fname):
        raise NotImplementedError
