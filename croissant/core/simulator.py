import numpy as np
from ..simulatorbase import SimulatorBase


class Simulator(SimulatorBase):
    def run(self, dpss=True, **dpss_kwargs):
        """
        Compute the convolution for a range of times.
        """
        # the rotation phases
        phases = self.sky.rot_alm_z(times=self.dt, world=self.world)
        phases.shape = (self.N_times, 1, -1)  # add freq axis
        if self.frequencies is None:
            dpss = False  # no need to compute dpss if no frequencies
            sky_alm = self.sky.alm.reshape(1, 1, -1)  # add time and freq axes
        else:
            sky_alm = np.expand_dims(self.sky.alm, axis=0)  # add time axis
        rot_sky_coeffs = sky_alm * phases

        if dpss:
            self.compute_dpss(**dpss_kwargs)
            # m = 0 modes
            res = (
                rot_sky_coeffs[:, :, : self.lmax + 1].real
                @ self.beam.coeffs[:, : self.lmax + 1].T.real
            )
            # for m != 0 we must account for +m and -m
            res += 2 * np.real(
                rot_sky_coeffs[:, :, self.lmax + 1 :]
                @ self.beam.coeffs[:, self.lmax + 1 :].T.conj()
            )

            waterfall = np.einsum("ijk, jk -> ij", res, self.design_matrix)

        else:
            beam_coeffs = np.expand_dims(self.beam.alm, axis=0)
            if self.beam.frequencies is None:  # add freq axis
                beam_coeffs = beam_coeffs.reshape(1, 1, -1)
            # m = 0 modes (already real)
            waterfall = np.einsum(
                "ijk, ijk -> ij",
                rot_sky_coeffs[:, :, : self.lmax + 1].real,
                beam_coeffs[:, :, : self.lmax + 1].real,
            )
            # for m != 0 we must account for +m and -m
            waterfall += 2 * np.real(
                np.einsum(
                    "ijk, ijk -> ij",
                    rot_sky_coeffs[:, :, self.lmax + 1 :],
                    beam_coeffs.conj()[:, :, self.lmax + 1 :],
                )
            )

        self.waterfall = np.squeeze(waterfall) / self.beam.total_power
