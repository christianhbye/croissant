"""Tests for full-Stokes transforms and component convolution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

from croissant.beam import Beam
from croissant.dense import dense_compute_alm
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
    """A spin field has no real transform, so the pair must be rejected.

    reality=True is passed explicitly here: it is not the default, so
    spin=2 on its own is an ordinary spin-weighted transform.
    """
    lmax = 3
    with pytest.raises(ValueError, match="reality=False"):
        compute_alm(
            jnp.ones((1,) + _mwss_shape(lmax)),
            lmax,
            "mwss",
            spin=2,
            reality=True,
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


def test_sky_p_plus_dual_matches_direct_transform_on_healpix():
    """On quadrature samplings the P_PLUS dual is derived from the
    P_MINUS analysis by the conjugate-flip identity
    P+_lm = (-1)^m conj(P-_l,-m) instead of a second transform; it
    must equal the explicit spin +2 analysis of Q - iU to machine
    precision on both the native-lmax and low-lmax dense paths."""
    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(133)
    data = rng.standard_normal((1, 4, npix))
    sky = PolarizedSky(data, [10.0], sampling="healpix")
    q_minus_iu = jnp.asarray(data[:, 1] - 1j * data[:, 2])

    alm = np.asarray(sky.compute_alm())
    direct = np.asarray(
        compute_alm(
            q_minus_iu,
            sky.lmax,
            "healpix",
            nside=nside,
            spin=2,
            reality=False,
        )
    )
    scale = np.abs(direct).max()
    np.testing.assert_allclose(alm[:, 3], direct, atol=1e-12 * scale)

    low_lmax = 6
    alm_low = np.asarray(sky.compute_alm(lmax=low_lmax))
    direct_low = np.asarray(
        dense_compute_alm(
            q_minus_iu, low_lmax, "healpix", nside=nside, spin=2, niter=0
        )
    )
    scale_low = np.abs(direct_low).max()
    np.testing.assert_allclose(
        alm_low[:, 3], direct_low, atol=1e-12 * scale_low
    )


def test_sky_p_plus_dual_keeps_explicit_transform_on_mwss():
    """The mw/mwss sampling-theorem transforms alias out-of-band power
    asymmetrically between spins +2 and -2, so the conjugate-flip
    identity fails for generic pixel data and the P_PLUS dual must
    come from the explicit spin +2 transform. The final assertion is
    the non-vacuity guard: on this input the flip misses at order
    percent or worse, so this test discriminates a de-gated
    implementation."""
    lmax = 8
    shape = _mwss_shape(lmax)
    rng = np.random.default_rng(2026)
    data = rng.standard_normal((1, 4) + shape)
    sky = PolarizedSky(data, [10.0], sampling="mwss")
    alm = np.asarray(sky.compute_alm())
    direct = np.asarray(
        compute_alm(
            jnp.asarray(data[:, 1] - 1j * data[:, 2]),
            lmax,
            "mwss",
            spin=2,
            reality=False,
        )
    )
    scale = np.abs(direct).max()
    np.testing.assert_allclose(alm[:, 3], direct, atol=1e-12 * scale)

    emms = np.arange(-lmax, lmax + 1)
    flip = (-1.0) ** emms * np.conj(alm[:, 2][..., ::-1])
    assert np.abs(flip - direct).max() > 1e-2 * scale


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


def test_low_band_limit_healpix_bypasses_the_block_engine():
    """A band-limit under the HEALPix floor stays on croissant.dense.

    No kernel exists below ``L >= 2 * nside - 1``, so this branch cannot
    honour a block's resolved engine and must not try. Requesting
    "kernel" explicitly is the sharp case: the block resolves to kernel,
    and the low-lmax call has to route around it rather than raise.
    """
    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(77)
    data = rng.standard_normal((2, 4, npix))

    kernel_sky = PolarizedSky(
        data, [10.0, 20.0], sampling="healpix", engine="kernel"
    )
    assert kernel_sky.engine["P_MINUS"] == "kernel"

    reference = PolarizedSky(
        data, [10.0, 20.0], sampling="healpix", engine="s2fft"
    ).compute_alm(lmax=2)
    got = kernel_sky.compute_alm(lmax=2)

    assert got.shape == (2, 4, 3, 5)
    scale = np.abs(np.asarray(reference)).max()
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(reference), rtol=0, atol=1e-10 * scale
    )


def test_conjugation_samplings_build_no_plus_kernel():
    """The block conjugation supplies must not precompute anything.

    On HEALPix the P+ dual comes from P- by conjugation, so building a
    spin +2 kernel for it would be pure waste -- the largest single
    kernel a polarized sky would hold.
    """
    nside = 8
    rng = np.random.default_rng(78)
    data = rng.standard_normal((4, 4, 12 * nside**2))
    healpix_sky = PolarizedSky(
        data, [10.0, 20.0, 30.0, 40.0], sampling="healpix"
    )
    assert healpix_sky.engine["P_PLUS"] is None
    assert healpix_sky._kernels[2] is None

    lmax = 8
    shape = _mwss_shape(lmax)
    mwss_sky = PolarizedSky(
        rng.standard_normal((4, 4) + shape),
        [10.0, 20.0, 30.0, 40.0],
        sampling="mwss",
    )
    assert mwss_sky.engine["P_PLUS"] == "kernel"
    assert mwss_sky._kernels[2] is not None


def test_pair_response_spin0_block_is_complex():
    """The response's spin-0 block must not reuse the sky's kernel.

    Both are spin 0 at the same band-limit, but a sky's I/V block is
    real and packs to ``m >= 0`` while a response's is complex. Applying
    a packed kernel to complex data raises inside s2fft's einsum, so the
    two must resolve and build separately.
    """
    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(79)
    freqs = [10.0, 20.0, 30.0, 40.0]
    beam = PairStokesBeam(
        rng.standard_normal((1, 4, 4, npix))
        + 1j * rng.standard_normal((1, 4, 4, npix)),
        freqs,
        [(0, 0)],
        sampling="healpix",
    )
    sky = PolarizedSky(
        rng.standard_normal((4, 4, npix)), freqs, sampling="healpix"
    )
    assert beam.engine["IV"] == sky.engine["IV"] == "kernel"
    assert beam._kernels[0].shape != sky._kernels[0].shape


def test_dense_polarized_inside_jit_needs_a_warmed_cache():
    """The packed-real block cannot build its operator inside a trace.

    Only the sky's real spin-0 block takes sphere.py's packed-real dense
    route and so needs its matrix threaded in; the spin-weighted blocks
    reach croissant.dense, which builds under
    ``jax.ensure_compile_time_eval`` and is unaffected. So an explicit
    "dense" field constructed inside jax.jit must raise unless that one
    matrix was precomputed first, exactly as SphBase requires.
    """
    from croissant import sphere

    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(80)
    freqs = [10.0, 20.0]
    data = jnp.asarray(rng.standard_normal((2, 4, npix)))

    @jax.jit
    def analyse(maps):
        return PolarizedSky(
            maps, freqs, sampling="healpix", engine="dense"
        ).compute_alm()

    sphere.clear_dense_matrix_cache()
    with pytest.raises(RuntimeError, match="precompute_dense_matrix"):
        analyse(data)

    lmax = PolarizedSky(data, freqs, sampling="healpix").lmax
    sphere.precompute_dense_matrix(
        (npix,), lmax, "healpix", nside=nside, niter=0
    )
    got = analyse(data)
    expected = PolarizedSky(
        data, freqs, sampling="healpix", engine="dense"
    ).compute_alm()
    scale = np.abs(np.asarray(expected)).max()
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=0, atol=1e-10 * scale
    )


def test_a_band_limit_at_the_floor_does_not_take_the_dense_bypass(
    monkeypatch,
):
    """At exactly the HEALPix floor the resolved engine still serves.

    The bypass compared the target against the native band-limit
    (``2 * nside``) rather than the floor (``2 * nside - 1``), so a
    request sitting exactly on the floor -- which the kernel engine
    serves fine -- silently built an ``O(nside**4)`` dense operator
    instead. At nside=64 that is ~12.9 GiB per spin block from a call
    that had resolved to a 127 MiB kernel.
    """
    from croissant import dense as _dense

    nside = 16
    npix = 12 * nside**2
    rng = np.random.default_rng(313)
    data = rng.standard_normal((1, 4, npix))
    sky = PolarizedSky(data, [10.0], sampling="healpix")

    def refuse(*args, **kwargs):
        raise AssertionError(
            "the dense bypass was taken for a band-limit at the floor"
        )

    monkeypatch.setattr(_dense, "dense_compute_alm", refuse)
    floor = 2 * nside - 1
    alm = sky.compute_alm(lmax=floor)
    assert alm.shape == (1, 4, floor + 1, 2 * floor + 1)
