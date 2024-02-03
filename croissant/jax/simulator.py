import jax
import jax.numpy as jnp
from ..simulatorbase import SimulatorBase

@jax.jit
def convolve(sky_alm, beam_alm, phases):
    """
    Compute the convolution for a range of times in jax. The convolution is
    a dot product in l,m space. Axes are in the order: time, freq, ell, emm.
    
    Parameters
    ----------
    sky_alm : jnp.ndarray
        The sky alms. Shape (N_freqs, lmax+1, 2*lmax+1).
    beam_alm : jnp.ndarray
        The beam alms. Shape (N_freqs, lmax+1, 2*lmax+1).
    phases : jnp.ndarray
        The phases that roate the sky, of the form exp(-i*m*phi(t)).
        Shape (N_times, 2*lmax+1).

    Returns
    -------
    res : jnp.ndarray
        The convolution. Shape (N_times, N_freqs).
    """
    s = sky_alm[None, :, :, :]  # add time axis
    p = phases[:, None, None, :]  # add freq and ell axes
    b = beam_alm.conjugate()[None, :, :, :]  # add time axis and conjugate
    res = jnp.sum(s * p * b, axes=(2, 3))  # dot product in l,m space
    return res

def convolve_dpss():
    raise NotImplementedError

class Simulator(SimulatorBase):
    def run(self, dpss=True, **dpss_kwargs):
        """
        Compute the convolution for a range of times in jax.
        
        Parameters
        ----------
        dpss : bool
            Whether to use a dpss basis or not.
        dpss_kwargs : dict
            Passed to SimulatorBase().compute_dpss.

        """
        if dpss:
            res = convolve_dpss()
        else:
            res = convolve(
                self.sky.alm,
                self.beam.alm,
                self.sky.rot_alm_z(self.dt, self.world)
            )
        self.waterfall = res / self.beam.total_power
