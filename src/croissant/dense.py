"""Cached dense spherical harmonic analysis for scalar and spin fields."""

from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import s2fft

from .footprints import spatial_shape as _spatial_shape
from .footprints import transform_lmax


def _valid_lm_indices(lmax, spin):
    ell_indices = []
    m_indices = []
    offset = lmax
    for ell in range(abs(spin), lmax + 1):
        for emm in range(-ell, ell + 1):
            ell_indices.append(ell)
            m_indices.append(emm + offset)
    return np.asarray(ell_indices), np.asarray(m_indices)


@lru_cache(maxsize=6)
def _build_analysis_matrix(
    lmax,
    sampling,
    nside,
    spin,
    niter,
    complex_dtype_name,
    device_key=None,
):
    """Materialize selected rows of corrected s2fft's linear operator."""
    del device_key
    complex_dtype = np.dtype(complex_dtype_name)
    spatial_shape = _spatial_shape(lmax, sampling, nside)
    ell_indices, m_indices = _valid_lm_indices(lmax, spin)
    ncoeff = ell_indices.size

    # s2fft's HEALPix FFT requires L >= 2*nside even when only lower modes
    # are retained. Build that supported operator and select the requested
    # low-l rows so low-lmax, high-nside dense analysis remains available.
    build_lmax = transform_lmax(lmax, sampling, nside=nside)
    transform_L = build_lmax + 1
    selected_m = np.asarray(
        [
            emm + build_lmax
            for ell in range(abs(spin), lmax + 1)
            for emm in range(-ell, ell + 1)
        ]
    )

    def selected_forward(data):
        full = s2fft.forward(
            data,
            L=transform_L,
            spin=spin,
            nside=nside,
            sampling=sampling,
            method="jax",
            reality=False,
            precomps=None,
            spmd=False,
            L_lower=0,
            iter=niter,
        )
        return full[ell_indices, selected_m]

    zero_map = jnp.zeros(spatial_shape, dtype=complex_dtype)
    coefficients, pullback = jax.vjp(selected_forward, zero_map)
    # s2fft may transform at a wider precision than the map it was given,
    # and a VJP only accepts cotangents in the dtype its primal output
    # actually has. Seed the basis from that dtype rather than from the
    # requested one, which is a statement about the stored matrix.
    cotangent_dtype = coefficients.dtype
    matrix = jnp.empty(
        (ncoeff, int(np.prod(spatial_shape))),
        dtype=complex_dtype,
    )
    chunk_size = 32
    for start in range(0, ncoeff, chunk_size):
        stop = min(start + chunk_size, ncoeff)
        coefficient_basis = jax.nn.one_hot(
            jnp.arange(start, stop),
            ncoeff,
            dtype=cotangent_dtype,
        )
        rows = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
            coefficient_basis
        ).reshape(stop - start, -1)
        matrix = matrix.at[start:stop].set(rows.astype(complex_dtype))
    # JAX's holomorphic VJP uses the complex transpose convention, so each
    # pulled-back coefficient basis vector is already one analysis row.
    return matrix, ell_indices, m_indices, spatial_shape


class DenseSphericalTransform(eqx.Module):
    """A cached dense analysis matrix differentiable with respect to maps."""

    matrix: jax.Array
    ell_indices: jax.Array
    m_indices: jax.Array
    lmax: int = eqx.field(static=True)
    sampling: str = eqx.field(static=True)
    nside: int | None = eqx.field(static=True)
    spin: int = eqx.field(static=True)
    niter: int = eqx.field(static=True)
    spatial_shape: tuple = eqx.field(static=True)

    def __init__(
        self,
        lmax,
        sampling,
        nside=None,
        spin=0,
        niter=0,
        dtype=jnp.complex128,
    ):
        dtype = np.dtype(dtype)
        if dtype.kind != "c":
            raise ValueError("Dense transform dtype must be complex.")
        with jax.ensure_compile_time_eval():
            device_key = str(jnp.empty((), dtype=jnp.uint8).device)
            matrix, ell_indices, m_indices, spatial_shape = (
                _build_analysis_matrix(
                    int(lmax),
                    str(sampling),
                    None if nside is None else int(nside),
                    int(spin),
                    int(niter),
                    dtype.name,
                    device_key,
                )
            )
        self.matrix = jnp.asarray(matrix)
        self.ell_indices = jnp.asarray(ell_indices, dtype=jnp.int32)
        self.m_indices = jnp.asarray(m_indices, dtype=jnp.int32)
        self.lmax = int(lmax)
        self.sampling = str(sampling)
        self.nside = None if nside is None else int(nside)
        self.spin = int(spin)
        self.niter = int(niter)
        self.spatial_shape = tuple(spatial_shape)

    @jax.jit
    def __call__(self, data):
        """Apply the cached analysis matrix to arbitrary leading batches."""
        data = jnp.asarray(data)
        if data.shape[-len(self.spatial_shape) :] != self.spatial_shape:
            raise ValueError(
                f"Expected trailing spatial shape {self.spatial_shape}; "
                f"got {data.shape}."
            )
        batch_shape = data.shape[: -len(self.spatial_shape)]
        flat = data.reshape((-1, int(np.prod(self.spatial_shape))))
        valid = jnp.einsum("kn,bn->bk", self.matrix, flat)
        shape = (flat.shape[0], self.lmax + 1, 2 * self.lmax + 1)
        full = jnp.zeros(shape, dtype=valid.dtype)
        full = full.at[:, self.ell_indices, self.m_indices].set(valid)
        return full.reshape(batch_shape + (self.lmax + 1, 2 * self.lmax + 1))


def dense_compute_alm(
    data,
    lmax,
    sampling,
    nside=None,
    spin=0,
    niter=0,
    dtype=None,
):
    """Convenience wrapper around :class:`DenseSphericalTransform`."""
    if dtype is None:
        input_dtype = np.dtype(jnp.asarray(data).dtype)
        use_128 = (input_dtype.kind == "f" and input_dtype.itemsize >= 8) or (
            input_dtype.kind == "c" and input_dtype.itemsize >= 16
        )
        dtype = jnp.complex128 if use_128 else jnp.complex64
    transform = DenseSphericalTransform(
        lmax,
        sampling,
        nside=nside,
        spin=spin,
        niter=niter,
        dtype=dtype,
    )
    return transform(data)
