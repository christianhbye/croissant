"""Fractional visibility at horizon boundaries."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import s2fft


def horizon_weights(theta, phi=None, theta_h=jnp.pi / 2):
    """Approximate visible cell fractions on a separable theta/phi grid.

    Cell edges lie halfway between neighboring theta samples, with the
    first and last edges at 0 and pi. Visibility varies linearly in theta
    within a boundary cell: a horizon through its center has weight 1/2
    on an equiangular grid. This is a cell-edge approximation, not exact
    solid-angle integration or azimuthal averaging of terrain.

    Parameters
    ----------
    theta : array_like
        Strictly increasing one-dimensional colatitudes in radians,
        with at least two samples. Use a regular grid's theta axis,
        not HEALPix's repeated per-pixel colatitudes.
    phi : array_like or None
        One-dimensional longitude axis in radians. Required for a
        callable horizon; optional for a scalar or array horizon.
    theta_h : float, array_like, or callable
        Maximum visible colatitude in radians, as a scalar, one value
        per phi, or a callable evaluated as ``theta_h(phi)``. Values
        at or below 0 block all cells; at or above pi keep all cells.

    Returns
    -------
    weights : jax.Array
        Fractions in [0, 1], shape ``(ntheta, 1)`` for a scalar horizon
        or ``(ntheta, nphi)`` for an azimuth-dependent horizon. Pass
        directly as ``horizon=`` to Beam or PairStokesBeam. Explicit
        boolean masks still retain their original hard boundary.

    Notes
    -----
    The arithmetic supports JAX tracing and differentiation with respect
    to the horizon height inside a cell. Callers supply valid coordinates;
    HEALPix pixel coverage requires a different geometry.
    """
    theta = jnp.asarray(theta)
    if theta.ndim != 1 or theta.size < 2:
        raise ValueError("theta must be a 1-D grid with at least two samples.")
    if not isinstance(theta, jax.core.Tracer):
        # Validate concrete coordinates on the host: JAX reductions on
        # a closed-over array would still become tracers inside jit.
        concrete_theta = np.asarray(theta)
        if np.any(~np.isfinite(concrete_theta)) or np.any(
            (concrete_theta < 0) | (concrete_theta > np.pi)
        ):
            raise ValueError("theta must be finite colatitudes in [0, pi].")
        if np.any(np.diff(concrete_theta) <= 0):
            raise ValueError("theta must be strictly increasing.")
    if phi is not None:
        phi = jnp.asarray(phi)
        if phi.ndim != 1:
            raise ValueError("phi must be a 1-D longitude axis.")
    if callable(theta_h):
        if phi is None:
            raise ValueError("A callable theta_h requires phi.")
        theta_h = theta_h(phi)
    theta_h = jnp.asarray(theta_h)
    if theta_h.ndim > 1 or theta_h.size == 0:
        raise ValueError("theta_h must be scalar or one value per phi.")
    if theta_h.ndim == 1 and phi is not None:
        if theta_h.shape != phi.shape:
            raise ValueError("theta_h must have one value per phi.")
    mid = (theta[:-1] + theta[1:]) / 2
    lower = jnp.concatenate((jnp.zeros(1), mid))
    upper = jnp.concatenate((mid, jnp.full(1, jnp.pi)))
    weights = jnp.clip(
        (theta_h.reshape(1, -1) - lower[:, None]) / (upper - lower)[:, None],
        0.0,
        1.0,
    )
    return weights


def _default_horizon(theta, sampling):
    """Upper-hemisphere weights, including half of equatorial pixels."""
    theta = jnp.asarray(theta)
    if sampling != "healpix":
        return horizon_weights(theta)
    # Equatorial HEALPix pixels are symmetric about the equator. No
    # rectangular theta/phi cell model is assumed for other pixels.
    return jnp.where(theta == jnp.pi / 2, 0.5, (theta < jnp.pi / 2) * 1.0)


@lru_cache(maxsize=32)
def _healpix_ring_metadata(nside):
    """Host constants for RING ordering; never cache JAX tracers."""
    counts = np.array(
        [
            s2fft.sampling.s2_samples.nphi_ring(t, nside)
            for t in range(4 * nside - 1)
        ]
    )
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    return starts, counts


def _horizon_in_beam_frame(
    horizon, horizon_frame, beam_rot, sampling, spatial_shape, nside=None
):
    """Sample a ground-fixed mask at phi_ground = phi_beam - beam_rot.

    Periodic linear interpolation preserves bounded visibility weights.
    HEALPix interpolation stays within each RING latitude. This shifts
    existing weights; it does not estimate terrain's true pixel coverage.
    """
    if horizon_frame == "beam":
        return horizon
    # Scalars and theta-only masks are invariant under azimuth rotation.
    if horizon.ndim == 0 or horizon.shape[-1] == 1:
        return horizon
    horizon = jnp.broadcast_to(horizon, spatial_shape)
    rotation = jnp.mod(beam_rot, 360.0)
    if sampling == "healpix":
        starts, counts = _healpix_ring_metadata(nside)
        # Cache only O(nside) host metadata, not O(npix) index arrays.
        starts = jnp.asarray(starts, dtype=jnp.int32)
        counts = jnp.asarray(counts, dtype=jnp.int32)
        pixels = jnp.arange(spatial_shape[0])
        ring = jnp.searchsorted(starts, pixels, side="right") - 1
        starts, counts = starts[ring], counts[ring]
        position = pixels - starts
        position = position - rotation * counts / 360.0
        lower = jnp.floor(position).astype(jnp.int32)
        fraction = position - jnp.floor(position)
        left = starts + jnp.mod(lower, counts)
        right = starts + jnp.mod(lower + 1, counts)
        return (1 - fraction) * horizon[left] + fraction * horizon[right]
    nphi = spatial_shape[-1]
    position = jnp.arange(nphi) - rotation * nphi / 360.0
    lower = jnp.floor(position).astype(jnp.int32)
    fraction = position - jnp.floor(position)
    left = jnp.mod(lower, nphi)
    right = jnp.mod(lower + 1, nphi)
    return (1 - fraction) * horizon[:, left] + fraction * horizon[:, right]
