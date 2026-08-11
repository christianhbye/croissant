"""Tests for full-Stokes transforms and component convolution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from croissant.beam import Beam
from croissant.polarization import (
    PairStokesBeam,
    PolarizedSky,
    convert_stokes_convention,
    polarized_convolve,
)
from croissant.simulator import convolve, rot_alm_z
from croissant.sky import Sky
from croissant.sphere import compute_alm


def _mwss_shape(lmax):
    L = lmax + 1
    return (
        s2fft.sampling.s2_samples.ntheta(L=L, sampling="mwss"),
        s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling="mwss"),
    )


def test_compute_alm_accepts_complex_spin_and_batch_axes():
    lmax = 5
    shape = _mwss_shape(lmax)
    data = jnp.ones((2, 3) + shape, dtype=jnp.complex128)
    alm = compute_alm(
        data,
        lmax,
        "mwss",
        spin=2,
        reality=False,
    )
    assert alm.shape == (2, 3, lmax + 1, 2 * lmax + 1)
    assert jnp.iscomplexobj(alm)


def test_nonzero_spin_rejects_reality_optimization():
    lmax = 3
    with pytest.raises(ValueError, match="reality=False"):
        compute_alm(
            jnp.ones((1,) + _mwss_shape(lmax)),
            lmax,
            "mwss",
            spin=2,
        )


def test_iau_cosmo_conversion_is_involutive_and_contragredient():
    sky = jnp.asarray([2.0, 0.4, -0.7, 0.2])
    response = jnp.asarray([1.0 + 2j, -3.0, 0.5j, 4.0])
    sky_cosmo = convert_stokes_convention(sky, "IAU", "COSMO")
    response_cosmo = convert_stokes_convention(response, "IAU", "COSMO")
    assert jnp.allclose(
        convert_stokes_convention(sky_cosmo, "COSMO", "IAU"),
        sky,
    )
    assert jnp.allclose(
        jnp.sum(response * sky),
        jnp.sum(response_cosmo * sky_cosmo),
    )


def test_positive_v_fixture_selects_exp_plus_i_jones_state():
    positive_v_field = jnp.asarray([1.0, 1.0j]) / jnp.sqrt(2.0)
    matched_receive = jnp.asarray([1.0, -1.0j]) / jnp.sqrt(2.0)
    rejected_receive = jnp.asarray([1.0, 1.0j]) / jnp.sqrt(2.0)

    def auto_response(receive):
        theta, phi = receive
        return jnp.asarray(
            [
                jnp.abs(theta) ** 2 + jnp.abs(phi) ** 2,
                jnp.abs(theta) ** 2 - jnp.abs(phi) ** 2,
                2 * jnp.real(theta * phi.conjugate()),
                1j * (phi * theta.conjugate() - theta * phi.conjugate()),
            ]
        )

    coherency = jnp.outer(
        positive_v_field,
        positive_v_field.conjugate(),
    )
    assert jnp.allclose(
        coherency,
        jnp.asarray([[1.0, -1.0j], [1.0j, 1.0]]) / 2.0,
    )
    matched = auto_response(matched_receive)
    rejected = auto_response(rejected_receive)
    assert jnp.allclose(
        matched,
        jnp.asarray([1.0, 0.0, 0.0, 1.0]),
    )
    assert jnp.allclose(
        rejected,
        jnp.asarray([1.0, 0.0, 0.0, -1.0]),
    )
    sky = jnp.asarray([1.0, 0.0, 0.0, 1.0])
    assert jnp.vdot(matched, sky) == pytest.approx(2.0)
    assert jnp.vdot(rejected, sky) == pytest.approx(0.0)


def test_polarized_objects_validate_shape_and_metadata():
    lmax = 4
    shape = _mwss_shape(lmax)
    sky = PolarizedSky(
        np.zeros((2, 4) + shape),
        [10.0, 20.0],
        sampling="mwss",
    )
    beam = PairStokesBeam(
        np.zeros((3, 2, 4) + shape, dtype=np.complex128),
        [10.0, 20.0],
        [(0, 0), (0, 1), (1, 1)],
        sampling="mwss",
        horizon=np.ones(shape, dtype=bool),
    )
    assert sky.lmax == beam.lmax == lmax
    assert sky.compute_alm().shape == (2, 4, lmax + 1, 2 * lmax + 1)
    assert beam.compute_alm().shape == (
        3,
        2,
        4,
        lmax + 1,
        2 * lmax + 1,
    )


def test_polarized_healpix_objects_accept_low_target_lmax():
    nside = 2
    npix = 12 * nside**2
    pixel = np.linspace(-1.0, 1.0, npix)
    sky_data = np.zeros((1, 4, npix))
    sky_data[0, 0] = 2.0 + pixel
    sky_data[0, 3] = 0.2 * pixel
    beam_data = np.zeros((1, 1, 4, npix), dtype=np.complex128)
    beam_data[0, 0, 0] = 1.0 + 0.1j * pixel
    sky = PolarizedSky(
        sky_data,
        [10.0],
        sampling="healpix",
    )
    beam = PairStokesBeam(
        beam_data,
        [10.0],
        [(0, 0)],
        sampling="healpix",
    )
    assert sky.lmax > 2
    assert sky._niter == beam._niter == 0
    sky_alm = sky.compute_alm(lmax=2)
    beam_alm = beam.compute_alm(lmax=2)
    assert sky_alm.shape == (1, 4, 3, 5)
    assert beam_alm.shape == (1, 1, 4, 3, 5)
    assert jnp.any(jnp.abs(sky_alm) > 0)
    assert jnp.any(jnp.abs(beam_alm) > 0)


def test_topocentric_sky_analysis_stays_local():
    lmax = 4
    shape = _mwss_shape(lmax)
    sky_data = np.zeros((1, 4) + shape)
    sky_data[0, 0] = 1.0
    sky = PolarizedSky(
        sky_data,
        [10.0],
        sampling="mwss",
        coord="topo",
    )

    assert sky.coord == sky.frame == "topo"
    assert sky.compute_alm().shape == (1, 4, lmax + 1, 2 * lmax + 1)
    with pytest.raises(
        ValueError,
        match="observer location and reference epoch",
    ):
        sky.compute_alm_eq(world="moon")


def test_polarized_mepa_rotation_matches_scalar_sky():
    lmax = 4
    L = lmax + 1
    theta = s2fft.sampling.s2_samples.thetas(L=L, sampling="mwss")
    phi = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling="mwss")
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    intensity = 2.0 + 0.2 * np.sin(tt) * np.cos(pp)
    polarized_data = np.zeros((1, 4) + intensity.shape)
    polarized_data[0, 0] = intensity

    polarized = PolarizedSky(
        polarized_data,
        [10.0],
        sampling="mwss",
        coord="galactic",
    )
    scalar = Sky(
        intensity[None],
        jnp.asarray([10.0]),
        sampling="mwss",
        coord="galactic",
    )
    polarized_mepa = polarized.compute_alm_eq(world="moon", et=0.0)
    scalar_mepa = scalar.compute_alm_eq(world="moon", et=0.0)
    assert jnp.allclose(
        polarized_mepa[0, 0],
        scalar_mepa[0],
        rtol=2e-6,
        atol=2e-6,
    )


def test_pair_beam_rotation_matches_scalar_beam_convention():
    lmax = 4
    L = lmax + 1
    ntheta, nphi = _mwss_shape(lmax)
    phi = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling="mwss")
    intensity = np.broadcast_to(1.0 + 0.3 * np.cos(phi), (ntheta, nphi))
    pair_data = np.zeros((1, 1, 4, ntheta, nphi), dtype=np.complex128)
    pair_data[0, 0, 0] = intensity
    horizon = np.ones((ntheta, nphi), dtype=bool)

    pair_beam = PairStokesBeam(
        pair_data,
        [10.0],
        [(0, 0)],
        sampling="mwss",
        horizon=horizon,
        beam_rot=37.0,
    )
    scalar_beam = Beam(
        intensity[None],
        jnp.asarray([10.0]),
        sampling="mwss",
        horizon=horizon,
        beam_rot=37.0,
    )
    assert jnp.allclose(
        pair_beam.compute_alm()[0, 0, 0],
        scalar_beam.compute_alm()[0],
        rtol=2e-6,
        atol=2e-6,
    )


def test_unpolarized_reduction_matches_scalar_pipeline():
    """An (I, 0, 0, 0) sky through an I-only pair response reproduces
    the scalar croissant visibility end to end, phases included."""
    lmax = 6
    L = lmax + 1
    ntheta, nphi = _mwss_shape(lmax)
    theta = s2fft.sampling.s2_samples.thetas(L=L, sampling="mwss")
    phi = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling="mwss")
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    freqs = [10.0, 20.0]

    intensity = np.stack(
        [
            2.0 + 0.3 * np.sin(tt) * np.cos(pp),
            1.5 + 0.2 * np.cos(tt) * np.sin(2 * pp),
        ]
    )
    response = np.stack(
        [
            1.0 + 0.4 * np.sin(tt) ** 2 * np.cos(pp),
            0.8 + 0.3 * np.cos(tt) ** 2,
        ]
    )
    sky_maps = np.zeros((2, 4, ntheta, nphi))
    sky_maps[:, 0] = intensity
    beam_maps = np.zeros((1, 2, 4, ntheta, nphi), dtype=np.complex128)
    beam_maps[0, :, 0] = response
    horizon = np.ones((ntheta, nphi), dtype=bool)

    polarized_sky = PolarizedSky(
        sky_maps, freqs, sampling="mwss", coord="mepa"
    )
    pair_beam = PairStokesBeam(
        beam_maps, freqs, [(0, 0)], sampling="mwss", horizon=horizon
    )
    scalar_sky = Sky(
        intensity, jnp.asarray(freqs), sampling="mwss", coord="mepa"
    )
    scalar_beam = Beam(
        response, jnp.asarray(freqs), sampling="mwss", horizon=horizon
    )

    phases = rot_alm_z(lmax, times=jnp.asarray([0.0, 3600.0, 7200.0]))
    polarized_vis = polarized_convolve(
        pair_beam.compute_alm(), polarized_sky.compute_alm(), phases
    )
    scalar_vis = convolve(
        scalar_beam.compute_alm(), scalar_sky.compute_alm(), phases
    )
    assert polarized_vis.shape == (3, 1, 2)
    np.testing.assert_allclose(
        polarized_vis[:, 0, :], scalar_vis, rtol=1e-10, atol=1e-10
    )


def test_harmonic_dual_matches_direct_full_stokes_quadrature():
    lmax = 8
    L = lmax + 1
    ntheta, nphi = _mwss_shape(lmax)
    theta = s2fft.sampling.s2_samples.thetas(L=L, sampling="mwss")
    phi = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling="mwss")
    tt, pp = np.meshgrid(theta, phi, indexing="ij")

    stokes_i = 2.0 + 0.1 * np.cos(tt)
    stokes_q = 0.2 * np.sin(tt) ** 2 * np.cos(2 * pp)
    stokes_u = -0.15 * np.sin(tt) ** 2 * np.sin(2 * pp)
    stokes_v = 0.05 * np.cos(tt)
    sky_maps = np.stack((stokes_i, stokes_q, stokes_u, stokes_v))[None]

    BI = 0.9 + 0.2j + 0.05 * np.cos(tt)
    BQ = (0.3 - 0.1j) * np.sin(tt) ** 2 * np.cos(2 * pp)
    BU = (-0.2 + 0.4j) * np.sin(tt) ** 2 * np.sin(2 * pp)
    BV = (0.1 + 0.3j) * np.cos(tt)
    beam_maps = np.stack((BI, BQ, BU, BV))[None, None]

    sky = PolarizedSky(
        sky_maps,
        [15.0],
        sampling="mwss",
        coord="mepa",
    )
    beam = PairStokesBeam(
        beam_maps,
        [15.0],
        [(0, 1)],
        sampling="mwss",
        horizon=np.ones((ntheta, nphi), dtype=bool),
    )
    phases = rot_alm_z(lmax, times=jnp.asarray([0.0]))
    harmonic = polarized_convolve(
        beam.compute_alm(), sky.compute_alm(), phases
    )[0, 0, 0]

    weights = s2fft.utils.quadrature_jax.quad_weights(
        L=L,
        sampling="mwss",
    )
    direct = jnp.einsum(
        "stp,stp,t->",
        jnp.asarray(beam_maps[0, 0]),
        jnp.asarray(sky_maps[0]),
        weights,
    )
    assert jnp.allclose(harmonic, direct, rtol=2e-5, atol=2e-5)


def test_polarized_convolution_gradients_cover_sky_and_complex_beam():
    lmax = 3
    shape = _mwss_shape(lmax)
    sky_data = jnp.ones((1, 4) + shape)
    beam_real = jnp.ones((1, 1, 4) + shape)
    beam_imag = 0.2 * jnp.ones_like(beam_real)
    phases = rot_alm_z(lmax, times=jnp.asarray([0.0, 10.0]))

    def loss(sky_values, response_real, response_imag):
        sky = PolarizedSky(
            sky_values,
            [10.0],
            sampling="mwss",
            coord="mepa",
        )
        beam = PairStokesBeam(
            response_real + 1j * response_imag,
            [10.0],
            [(0, 0)],
            sampling="mwss",
            horizon=jnp.ones(shape, dtype=bool),
        )
        vis = polarized_convolve(beam.compute_alm(), sky.compute_alm(), phases)
        return jnp.sum(jnp.abs(vis) ** 2)

    gradients = jax.grad(loss, argnums=(0, 1, 2))(
        sky_data, beam_real, beam_imag
    )
    for gradient in gradients:
        assert gradient.shape in {sky_data.shape, beam_real.shape}
        assert jnp.all(jnp.isfinite(gradient))
