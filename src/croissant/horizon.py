"""Fractional visibility at horizon boundaries."""

import math
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
    lower, upper = _band_edges(theta)
    return _band_weights(
        lower[:, None], upper[:, None], theta_h.reshape(1, -1)
    )


def _band_edges(centers):
    """Band edges halfway between increasing centers, 0 and pi at ends."""
    mid = (centers[:-1] + centers[1:]) / 2
    lower = jnp.concatenate((jnp.zeros(1), mid))
    upper = jnp.concatenate((mid, jnp.full(1, jnp.pi)))
    return lower, upper


def _band_weights(lower, upper, theta_h):
    """Fraction of each [lower, upper] band at colatitudes below theta_h."""
    return jnp.clip((theta_h - lower) / (upper - lower), 0.0, 1.0)


def _check_colatitudes(theta_h):
    """Reject concrete horizon colatitudes outside [0, pi]."""
    if isinstance(theta_h, jax.core.Tracer):
        return
    values = np.asarray(theta_h)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > np.pi)):
        raise ValueError(
            "horizon_theta must be finite colatitudes in radians within "
            f"[0, pi]; got values from {values.min()} to {values.max()}. "
            "Convert degrees with np.deg2rad."
        )


def _horizon_from_theta(theta_h, theta, phi, sampling):
    """Weights on the ground grid for a scalar or compass-azimuth profile.

    Each sample's band in theta has edges halfway between neighbouring
    rows (regular grids) or RING colatitudes (HEALPix), as in
    ``horizon_weights``. A callable is evaluated once at the compass
    azimuth ``A = (pi/2 - phi) mod 2 pi`` of every column or pixel.
    """
    theta = np.asarray(theta)
    if sampling == "healpix":
        rings, ring = np.unique(theta, return_inverse=True)
        lower, upper = _band_edges(jnp.asarray(rings))
        lower, upper = lower[ring], upper[ring]
    else:
        lower, upper = _band_edges(jnp.asarray(theta))
        lower, upper = lower[:, None], upper[:, None]
    if callable(theta_h):
        azimuth = jnp.mod(jnp.pi / 2 - jnp.asarray(phi), 2 * jnp.pi)
        values = jnp.asarray(theta_h(azimuth))
        if values.ndim > 0 and values.shape != azimuth.shape:
            raise ValueError(
                "A callable horizon_theta must return a scalar or one "
                f"colatitude per azimuth, shape {azimuth.shape}; got "
                f"shape {values.shape}."
            )
        theta_h = jnp.broadcast_to(values, azimuth.shape)
    else:
        theta_h = jnp.asarray(theta_h)
    _check_colatitudes(theta_h)
    return _band_weights(lower, upper, theta_h)


def _resolve_horizon(
    horizon, horizon_theta, horizon_frame, theta, phi, sampling
):
    """Initial horizon weights for Beam and PairStokesBeam."""
    if horizon_theta is None:
        if horizon is None:
            return _default_horizon(theta, sampling)
        return jnp.asarray(horizon)
    if horizon is not None:
        raise ValueError("Pass either horizon or horizon_theta, not both.")
    if callable(horizon_theta):
        if horizon_frame != "topocentric":
            raise ValueError(
                "A callable horizon_theta is a function of compass "
                "azimuth, which is fixed to the ground; pass "
                "horizon_frame='topocentric'."
            )
    elif np.ndim(horizon_theta) != 0:
        raise ValueError(
            "horizon_theta must be a scalar colatitude or a callable of "
            "compass azimuth A, e.g. lambda A: np.interp(A, az, theta_h, "
            "period=2 * np.pi)."
        )
    return _horizon_from_theta(horizon_theta, theta, phi, sampling)


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


def rotate_horizon(horizon, delta_phi_deg, sampling, nside=None):
    """Shift horizon visibility weights periodically in longitude.

    The output at longitude phi is the input at ``phi - delta_phi_deg``,
    so every feature of the mask moves to larger phi by
    ``delta_phi_deg``. Phi is the grid longitude of ``utils.generate_phi``
    for the same sampling. On the ground grid of
    ``horizon_frame="topocentric"``, phi = 0 is East and phi = 90 deg is
    North, so compass azimuth is ``A = 90 deg - phi``. A mask written on
    a grid with ``A = A0 - phi`` (the same handedness, phi = 0 pointing
    to ``A0``) moves onto that grid with ``delta_phi_deg = 90 - A0``.
    For example, a North-zero grid with phi = 90 deg West (``A = -phi``)
    needs ``delta_phi_deg = 90``. A grid of opposite handedness must be
    mirrored in phi first.

    Parameters
    ----------
    horizon : array_like
        Visibility weights with longitude (or, for HEALPix, the RING
        pixel index) on the last axis. Leading axes are shifted
        independently. A scalar, or an array whose last axis has length
        one (a theta-only mask), is returned unchanged.
    delta_phi_deg : float or jax.Array
        Shift in degrees; positive moves features to larger phi. It may
        be traced, and the result is differentiable with respect to it
        between grid columns.
    sampling : {"mw", "mwss", "dh", "gl", "healpix"}
        Sampling scheme of the grid. Every scheme except HEALPix shifts
        along a uniform longitude axis that starts at phi = 0.
    nside : int or None
        HEALPix resolution. Inferred from the last axis's length when
        None; when given, it must match that length. Ignored for other
        samplings.

    Returns
    -------
    rotated : jax.Array
        Shifted weights, same shape as ``horizon``. Values stay within
        the input's range, so weights in [0, 1] stay in [0, 1].

    Raises
    ------
    ValueError
        For HEALPix, if the last axis is not a valid pixel count or does
        not match ``nside``.

    Notes
    -----
    Each sample takes the periodic linear interpolation of its two
    neighbours along its longitude row (along its latitude ring for
    HEALPix). The shift is exact when ``delta_phi_deg`` is a whole number
    of columns: 90 deg is exact on the 1-degree MWSS grid (360 columns)
    and on every HEALPix ring, but never on MW, DH or GL grids, whose
    column count is odd. Otherwise it softens sharp edges, and the
    effect accumulates when shifts are chained. This moves existing
    weights; it does not estimate terrain's true pixel coverage.
    """
    horizon = jnp.asarray(horizon)
    # Scalars and theta-only masks are invariant under azimuth rotation.
    if horizon.ndim == 0 or horizon.shape[-1] == 1:
        return horizon
    rotation = jnp.mod(jnp.asarray(delta_phi_deg), 360.0)
    npix = horizon.shape[-1]
    if sampling == "healpix":
        inferred = math.isqrt(npix // 12)
        if 12 * inferred**2 != npix:
            raise ValueError(
                f"A HEALPix horizon of {npix} pixels does not match "
                "any nside (npix = 12 * nside**2)."
            )
        if nside is not None and int(nside) != inferred:
            raise ValueError(
                f"A HEALPix horizon of {npix} pixels has nside "
                f"{inferred}, not {nside}."
            )
        starts, counts = _healpix_ring_metadata(inferred)
        # Cache only O(nside) host metadata, not O(npix) index arrays.
        starts = jnp.asarray(starts, dtype=jnp.int32)
        counts = jnp.asarray(counts, dtype=jnp.int32)
        pixels = jnp.arange(npix)
        ring = jnp.searchsorted(starts, pixels, side="right") - 1
        starts, counts = starts[ring], counts[ring]
        position = pixels - starts
        position = position - rotation * counts / 360.0
        lower = jnp.floor(position).astype(jnp.int32)
        fraction = position - jnp.floor(position)
        left = starts + jnp.mod(lower, counts)
        right = starts + jnp.mod(lower + 1, counts)
    else:
        position = jnp.arange(npix) - rotation * npix / 360.0
        lower = jnp.floor(position).astype(jnp.int32)
        fraction = position - jnp.floor(position)
        left = jnp.mod(lower, npix)
        right = jnp.mod(lower + 1, npix)
    rotated = (1 - fraction) * horizon[..., left]
    return rotated + fraction * horizon[..., right]


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
    return rotate_horizon(horizon, beam_rot, sampling, nside=nside)
