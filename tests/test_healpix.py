from copy import deepcopy
import healpy
import numpy as np
import pytest
from croissant import healpix as hp


def test_healpix2lonlat():
    nside = 32
    pix = 1  # the north pole
    lon, lat = hp.healpix2lonlat(nside, pix=pix)
    assert lat > 85  # should be close to 90

    lons, lats = hp.healpix2lonlat(nside, pix=None)
    assert lons.shape == lats.shape == (healpy.nside2npix(nside),)
    assert 0 <= lons.all() < 360
    assert -90 <= lats.all() < 90


def test_grid_interp():
    # create mock data at 1 deg spacing in theta/phi
    theta = np.linspace(0, np.pi, 181)
    phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    data = np.cos(theta) ** 2
    data = np.repeat(data.reshape(-1, 1), phi.size, axis=1)

    # test that we get the same thing back if interpolate to same values
    # interpolation must be called on the flattened arrays
    to_phi, to_theta = [d.ravel() for d in np.meshgrid(phi, theta)]
    interp_data = hp.grid_interp(data, theta, phi, to_theta, to_phi)
    # interp data has a frequency axis as the 0th axis
    assert interp_data.shape == (1, to_theta.size)
    # assert np.allclose(data.ravel(), interp_data[0])


def test_grid2healpix():
    # constant map
    theta = np.linspace(0, np.pi, 181)
    phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    data = np.ones((theta.size, phi.size))
    nside = 32
    npix = healpy.nside2npix(nside)
    hp_map = hp.grid2healpix(data, nside, theta=theta, phi=phi)
    expected_map = np.ones((1, npix))
    assert np.allclose(hp_map, expected_map)

    # map that depends on theta
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    data = np.cos(theta_grid) ** 2
    hp_map = hp.grid2healpix(data, nside, theta=theta, phi=phi)
    # angles of healpix map:
    lat = hp.healpix2lonlat(nside)[1]
    hp_theta = np.pi / 2 - np.deg2rad(lat)
    expected_map = np.cos(hp_theta) ** 2
    expected_map.shape = (1, -1)  # add frequency axis
    assert np.allclose(hp_map, expected_map)

    # set pixel centers to be the first npix values of theta/phi
    pc_theta = theta_grid.ravel()[:npix]
    pc_phi = phi_grid.ravel()[:npix]
    pix_centers = np.transpose([pc_theta, pc_phi])
    hp_map = hp.grid2healpix(
        data, nside, theta=theta, phi=phi, pixel_centers=pix_centers
    )
    expected_map = np.cos(pc_theta) ** 2
    expected_map.shape = (1, -1)  # add frequency axis
    assert np.allclose(hp_map, expected_map)


def test_map2alm():
    nside = 32
    npix = healpy.nside2npix(nside)
    data = np.ones(npix)
    lmax = 5
    alm = hp.map2alm(data, lmax)
    assert np.isclose(alm[0], np.sqrt(4 * np.pi))  # a00 = Y00 * 4pi
    assert np.allclose(alm[1:], 0)  # rest of alms should be 0

    # multiple maps at once
    c = 10  # constant extra value
    d0 = np.arange(npix)
    d1 = d0 + c
    alm0 = hp.map2alm(d0, lmax)
    alm1 = hp.map2alm(d1, lmax)
    data = np.array([d0, d1])
    alm = hp.map2alm(data, lmax)
    # check that computing jointly is the same as individually
    assert np.allclose(alm[0], alm0)
    assert np.allclose(alm[1], alm1)
    # d1 is d0 + constant value so they should only differ at a00
    assert np.allclose(alm0[1:], alm1[1:])
    assert np.isclose(alm0[0] + c * np.sqrt(4 * np.pi), alm1[0])


def test_nested_check():
    nside = 10  # this is invalid for NESTED but OK for RING
    freqs = np.linspace(1, 50, 50)
    npix = healpy.nside2npix(nside)
    data = np.ones((freqs.size, npix)) * 10
    kwargs = {"data": data, "nested_input": True, "frequencies": freqs}
    # it should raise an error
    with pytest.raises(ValueError):
        hp.HealpixMap(nside, **kwargs)
    kwargs["nested_input"] = False
    hp.HealpixMap(nside, **kwargs)  # should work


def test_npix():
    nside = 8  # make it valid for nested so that the ud_grade can run
    npix = healpy.nside2npix(nside)
    freqs = np.linspace(1, 50, 50)
    data = np.ones((freqs.size, npix))
    kwargs = {"data": data, "nested_input": False, "frequencies": freqs}
    hpm = hp.HealpixMap(nside, **kwargs)
    assert hpm.npix == npix
    assert hpm.npix == hpm.data.shape[-1]


def test_ud_grade():
    freqs = np.linspace(1, 50, 50)
    nside = 8
    npix = healpy.nside2npix(nside)
    data = np.ones((freqs.size, npix)) * np.arange(npix).reshape(1, -1)
    kwargs = {"data": data, "nested_input": False, "frequencies": freqs}
    hpm = hp.HealpixMap(nside, **kwargs)
    nside_out = [1, 2, 8, 32]
    for ns in nside_out:
        hp_copy = deepcopy(hpm)
        hp_copy.ud_grade(ns)
        assert hp_copy.nside == ns
        assert np.allclose(hp_copy.data, healpy.ud_grade(hpm.data, ns))


def test_alm():
    nside = 8
    npix = healpy.nside2npix(nside)
    avg = 10.0
    data = avg * np.ones(npix)  # constant map
    kwargs = {"data": data, "nested_input": False, "frequencies": None}
    hpm = hp.HealpixMap(nside, **kwargs)
    # constant map
    alm = hp.Alm.from_healpix(hpm)
    assert alm.lmax == 3 * nside - 1  # default lmax
    a00 = alm.get_coeff(0, 0)
    assert np.allclose(a00, avg * np.sqrt(4 * np.pi))
    alm.set_coeff(0, 0, 0)  # set a00 to 0, all coeffs should be 0 after this
    assert np.allclose(alm.alm, 0, atol=1e-2)
