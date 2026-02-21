"""
Multi-pair visibility simulation via vmapped alm dot products.

This module extends CROISSANT to compute visibilities for multiple antenna pairs
simultaneously. It works with complex-valued pair beams in s2fft alm format,
handling both auto-correlations (real beams) and cross-correlations (complex beams)
uniformly.

The core operation is a vmap of the existing convolve function over the pair axis.
"""

import jax
import jax.numpy as jnp

from .simulator import convolve
from .alm import total_power, lmax_from_shape

# vmap convolve over the pair axis (axis 0 of beam_alm)
# sky_alm and phases are broadcast (not mapped)
_multi_convolve = jax.vmap(convolve, in_axes=(0, None, None))


@jax.jit
def multi_convolve(beam_alm, sky_alm, phases):
    """
    Compute the convolution for multiple antenna pairs via vmap.

    This is a thin wrapper around the existing convolve function that
    vmaps over the pair axis (axis 0) of the beam array.

    Parameters
    ----------
    beam_alm : jnp.ndarray
        The beam alms for all pairs. Shape (N_pairs, N_freqs, lmax+1, 2*lmax+1).
        Each slice along axis 0 holds the s2fft alm of one pair beam.
        dtype complex128.
    sky_alm : jnp.ndarray
        The sky alms. Shape (N_freqs, lmax+1, 2*lmax+1). dtype complex128.
    phases : jnp.ndarray
        The phases that rotate the sky, of the form exp(-i*m*phi(t)).
        Shape (N_times, 2*lmax+1). See simulator.rot_alm_z.

    Returns
    -------
    vis : jnp.ndarray
        The unnormalized visibilities. Shape (N_pairs, N_times, N_freqs).
        dtype complex128.

    """
    return _multi_convolve(beam_alm, sky_alm, phases)


@jax.jit
def compute_visibilities(beam_alm, sky_alm, phases, norm):
    """
    Compute normalized visibilities for multiple antenna pairs.

    Parameters
    ----------
    beam_alm : jnp.ndarray
        The beam alms for all pairs. Shape (N_pairs, N_freqs, lmax+1, 2*lmax+1).
        Each slice along axis 0 holds the s2fft alm of one pair beam.
        dtype complex128.
    sky_alm : jnp.ndarray
        The sky alms. Shape (N_freqs, lmax+1, 2*lmax+1). dtype complex128.
    phases : jnp.ndarray
        The phases that rotate the sky, of the form exp(-i*m*phi(t)).
        Shape (N_times, 2*lmax+1). See simulator.rot_alm_z.
    norm : jnp.ndarray
        Normalization factors for each pair. Shape (N_pairs,) for scalar
        normalization or (N_pairs, N_freqs) for frequency-dependent
        normalization. For pair (p, q), this should be
        sqrt(total_power_p * total_power_q) computed from the
        auto-correlation beams.

    Returns
    -------
    vis : jnp.ndarray
        The normalized visibilities. Shape (N_times, N_pairs, N_freqs).
        dtype complex128. For auto-correlations, the imaginary part will
        be at numerical noise level.

    """
    # Compute raw visibilities: shape (N_pairs, N_times, N_freqs)
    vis_raw = multi_convolve(beam_alm, sky_alm, phases)

    # Normalize: broadcast norm over time axis (and freq axis if scalar)
    # norm shape (N_pairs,) -> (N_pairs, 1, 1)
    # norm shape (N_pairs, N_freqs) -> (N_pairs, 1, N_freqs)
    norm_broadcast = norm[:, None, None] if norm.ndim == 1 else norm[:, None, :]
    vis_normalized = vis_raw / norm_broadcast

    # Transpose to (N_times, N_pairs, N_freqs)
    return jnp.transpose(vis_normalized, (1, 0, 2))


def compute_normalization(auto_beam_alm):
    """
    Compute normalization factors for antenna pairs from auto-correlation beams.

    For pair (p, q), the normalization is sqrt(total_power_p * total_power_q).
    This function takes the auto-correlation beam alms and computes the
    total power for each antenna.

    Parameters
    ----------
    auto_beam_alm : jnp.ndarray
        The auto-correlation beam alms. Shape (N_antennas, N_freqs, lmax+1, 2*lmax+1).
        Entry i holds the beam alm for antenna i (auto-correlation).

    Returns
    -------
    antenna_powers : jnp.ndarray
        The total power for each antenna at each frequency.
        Shape (N_antennas, N_freqs).

    """
    lmax = lmax_from_shape(auto_beam_alm.shape)
    # vmap total_power over antenna and frequency axes
    # total_power expects last two dims to be (lmax+1, 2*lmax+1)
    # auto_beam_alm has shape (N_antennas, N_freqs, lmax+1, 2*lmax+1)
    # We need to apply total_power to each (lmax+1, 2*lmax+1) slice
    return jax.vmap(jax.vmap(lambda x: total_power(x, lmax)))(auto_beam_alm)


def pair_normalization(antenna_powers, pairs):
    """
    Compute the normalization for each pair from antenna powers.

    Parameters
    ----------
    antenna_powers : jnp.ndarray
        The total power for each antenna. Shape (N_antennas,) or (N_antennas, N_freqs).
        If frequency-dependent, normalization will be frequency-dependent.
    pairs : Sequence[Tuple[int, int]]
        List of (p, q) tuples indicating antenna pairs.

    Returns
    -------
    norm : jnp.ndarray
        Normalization for each pair. Shape (N_pairs,) or (N_pairs, N_freqs).

    """
    # For each pair (p, q), compute sqrt(power_p * power_q)
    norms = []
    for p, q in pairs:
        norm_pq = jnp.sqrt(antenna_powers[p] * antenna_powers[q])
        norms.append(norm_pq)
    return jnp.stack(norms, axis=0)
