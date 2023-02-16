from copy import deepcopy
import healpy
import numpy as np
import pytest
from croissant import healpix as hp, sphtransform as spht
from croissant.constants import sidereal_day_earth, sidereal_day_moon, Y00


def test_coord_rep():
    coords = ["galactic", "equatorial", "ecliptic", "mcmf", "topocentric"]
    short = ["G", "C", "E", "M", "T"]
    for i in range(len(coords)):
        assert hp.coord_rep(coords[i]) == short[i]


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
    assert np.allclose(data.ravel(), interp_data[0])

    # test with data that depends on phi
    data = np.repeat(phi.reshape(1, -1), theta.size, axis=0)
    interp_data = hp.grid_interp(data, theta, phi, to_theta, to_phi)
    assert interp_data.shape == (1, to_theta.size)
    # we need to disregard the poles here
    no_poles = (0 < to_theta) & (to_theta < np.pi)
    assert np.allclose(data.ravel()[no_poles], interp_data[0, no_poles])


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
    data = np.sin(theta_grid)
    hp_map = hp.grid2healpix(data, nside, theta=theta, phi=phi)
    # angles of healpix map:
    lat = hp.healpix2lonlat(nside)[1]
    hp_theta = np.pi / 2 - np.deg2rad(lat)
    expected_map = np.sin(hp_theta)
    assert np.allclose(hp_map, expected_map)

    # map that depends on phi
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    data = np.sin(phi_grid)
    hp_map = hp.grid2healpix(data, nside, theta=theta, phi=phi)
    # angles of healpix map:
    lon = hp.healpix2lonlat(nside)[0]
    hp_phi = np.deg2rad(lon)
    expected_map = np.sin(hp_phi)
    # for map that depends on phi we expect errors at the pole
    pole_th = 0.15  # radians, where we allow errors (theta)
    pix = healpy.ang2pix(nside, pole_th, 0)
    assert np.allclose(hp_map[0, pix:-pix], expected_map[pix:-pix])

    # set pixel centers to be the first npix values of theta/phi
    data = np.sin(theta_grid)
    pc_theta = theta_grid.ravel()[:npix]
    pc_phi = phi_grid.ravel()[:npix]
    pix_centers = np.transpose([pc_theta, pc_phi])
    hp_map = hp.grid2healpix(
        data, nside, theta=theta, phi=phi, pixel_centers=pix_centers
    )
    expected_map = np.sin(pc_theta)
    assert np.allclose(hp_map, expected_map)


def test_nested_input():
    freqs = np.linspace(1, 50, 50)
    nside = 10  # this is invalid for NESTED but OK for RING
    npix = healpy.nside2npix(nside)
    data = 10 * np.ones((freqs.size, npix))
    kwargs = {
        "data": data,
        "nest": False,
        "frequencies": freqs,
    }
    hp.HealpixMap(**kwargs)  # should work
    # should raise an error if nested is True
    kwargs["nest"] = True
    with pytest.raises(ValueError):
        hp.HealpixMap(**kwargs)

    # valid nested input
    nside = 8
    npix = healpy.nside2npix(nside)
    data = np.arange(npix) ** 2
    kwargs = {
        "data": data,
        "nest": True,
        "frequencies": None,
    }
    hp_map = hp.HealpixMap(**kwargs)
    # hp_map data is in the RING scheme:
    ring_ix = healpy.nest2ring(nside, np.arange(npix))
    data_ring = data[ring_ix]
    assert np.allclose(hp_map.data, data_ring)


def test_from_alm():
    nside = 10
    lmax = 3 * nside - 1
    alm_arr = np.zeros((healpy.Alm.getsize(lmax)), dtype=np.complex128)
    alm = hp.Alm(alm_arr, lmax=lmax)
    a00 = 10
    alm[0, 0] = a00
    hp_map = hp.HealpixMap.from_alm(alm, nside=None)
    # healpix map should be able to infer the nside from lmax
    assert hp_map.nside == nside
    npix = healpy.nside2npix(nside)
    # the map should just be = a00 * Y00 everywhere
    expected_map = np.full((1, npix), a00 * Y00)
    assert np.allclose(hp_map.data, expected_map)


def test_from_grid():
    nside = 32
    theta = np.linspace(0, np.pi, 181)
    phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    data = np.cos(phi)[None] * np.sin(theta)[:, None]  # mock data
    hp_map = hp.HealpixMap.from_grid(data, nside, theta, phi, coord="T")
    assert hp_map.nside == nside
    assert hp_map.frequencies is None
    assert hp_map.coord == "T"
    assert np.allclose(hp_map.data, hp.grid2healpix(data, nside, theta, phi))


def test_ud_grade():
    freqs = np.linspace(1, 50, 50)
    nside = 8
    npix = healpy.nside2npix(nside)
    data = np.repeat(np.arange(npix).reshape(1, -1), freqs.size, axis=0)
    kwargs = {
        "data": data,
        "nest": False,
        "frequencies": freqs,
    }
    hpm = hp.HealpixMap(**kwargs)
    nside_out = [1, 2, 8, 32]
    for ns in nside_out:
        hp_copy = deepcopy(hpm)
        hp_copy.ud_grade(ns)
        assert hp_copy.nside == ns
        assert np.allclose(hp_copy.data, healpy.ud_grade(hpm.data, ns))


def test_switch_coords():
    nside = 64
    npix = healpy.nside2npix(nside)
    data = np.arange(npix)
    # switch from galactic to equatorial
    coord = "G"
    new_coord = "C"
    hp_map = hp.HealpixMap(data, coord=coord)
    assert hp_map.coord == coord
    hp_map.switch_coords(new_coord)
    assert hp_map.coord == new_coord
    rot = healpy.Rotator(coord="gc")
    expected_data = rot.rotate_map_alms(data)
    assert np.allclose(hp_map.data, expected_data)

    # several maps at once
    freqs = np.arange(1, 11).reshape(-1, 1)
    data = np.arange(npix, dtype=np.float64).reshape(1, -1) * freqs
    hp_map = hp.HealpixMap(data, frequencies=freqs, coord=coord)
    assert hp_map.coord == coord
    hp_map.switch_coords(new_coord)
    assert hp_map.coord == new_coord
    expected_data = np.empty_like(data)
    for i in range(freqs.size):
        expected_data[i] = rot.rotate_map_alms(data[i])
    assert np.allclose(hp_map.data, expected_data)


def test_alm():
    nside = 32
    npix = healpy.nside2npix(nside)
    data = np.arange(npix)
    hp_map = hp.HealpixMap(data)
    # test default lmax
    alm = hp_map.alm(lmax=None)
    expected_lmax = 3 * nside - 1  # should be default
    expected_size = healpy.Alm.getsize(expected_lmax)
    assert alm.shape == (expected_size,)
    # specify lmax
    lmax = 10
    alm = hp_map.alm(lmax=lmax)
    expected_size = healpy.Alm.getsize(lmax)
    assert alm.shape == (expected_size,)

    # several maps at once
    freqs = np.arange(10).reshape(-1, 1)
    data = np.arange(npix).reshape(1, -1) * freqs
    hp_map = hp.HealpixMap(data, frequencies=freqs)
    alm = hp_map.alm(lmax=lmax)
    assert alm.shape == (freqs.size, expected_size)


def test_alm_indexing():
    lmax = 10
    freqs = np.linspace(1, 50, 50)
    nfreqs = freqs.size
    # initialize all alms to 0
    alm_arr = np.zeros(
        (nfreqs, healpy.Alm.getsize(lmax)),
        dtype=np.complex128,
    )
    alm = hp.Alm(alm=alm_arr, lmax=lmax, frequencies=freqs)
    # set a00 = 1 for first half of frequencies
    alm[: nfreqs // 2, 0, 0] = 1.0
    # check __setitem__ acted correctly on alm.alm
    assert np.allclose(alm.alm[: nfreqs // 2, 0], 1)
    assert np.allclose(alm.alm[nfreqs // 2 :, 0], 0)
    assert np.allclose(alm.alm[:, 1:], 0)
    # check that __getitem__ agrees:
    assert np.allclose(alm[: nfreqs // 2, 0, 0], 1)
    assert np.allclose(alm[nfreqs // 2 :, 0, 0], 0)
    # __getitem__ can't get multiple l-modes or m-modes at once...
    for ell in range(1, lmax + 1):
        for emm in range(ell + 1):
            assert np.allclose(alm[:, ell, emm], 0)

    # set everything back to 0
    alm.alm = np.zeros_like(alm.alm)
    # negative indexing
    val = 3.0 + 2.3j
    alm[-1, 10, 7] = val
    assert alm[-1, 10, 7] == val
    ix = healpy.Alm.getidx(lmax, 10, 7)
    assert alm[-1, 10, 7] == alm.alm[-1, ix]

    # negative emm
    alm.alm = np.zeros_like(alm.alm)
    alm[0, 3, 2] = val
    assert np.isclose(alm[0, 3, -2], (-1) ** 2 * np.conj(val))

    # frequency index not specified
    with pytest.raises(IndexError):
        alm[3, 2] = 5
        alm[7, -1]

    # no frequencies
    alm = hp.Alm(alm=alm_arr[0], lmax=lmax, frequencies=None)
    alm[5, 2] = 3.0
    assert alm[5, 2] == 3.0


def test_reduce_lmax():
    old_lmax = 10
    new_lmax = 5
    old_size = hp.Alm.getsize(old_lmax)
    alm = hp.Alm(np.arange(old_size))
    ell, emm = healpy.Alm.getlm(new_lmax)
    ix = alm.getidx(ell, emm)
    old_alms = alm.alm[ix]
    alm.reduce_lmax(new_lmax)
    assert alm.lmax == new_lmax
    with pytest.raises(IndexError):
        alm[7, 0]  # asking for ell > new_lmax should raise error
    assert np.allclose(alm.alm, old_alms)


def test_from_healpix():
    nside = 8
    npix = healpy.nside2npix(nside)
    freqs = np.linspace(1, 50, 50)
    data = np.arange(npix).reshape(1, -1) * freqs.reshape(-1, 1) ** 2
    coord = "equatorial"
    hp_map = hp.HealpixMap(data, frequencies=freqs, coord=coord)
    lmax = 10
    alm = hp.Alm.from_healpix(hp_map, lmax=lmax)
    assert alm.lmax == lmax
    assert np.allclose(alm.frequencies, freqs)
    assert alm.coord == hp.coord_rep(coord)
    assert np.allclose(alm.alm, spht.map2alm(data, lmax=lmax))


def test_alm_from_grid():
    lmax = 45
    nside = 32
    theta = np.linspace(0, np.pi, 181)
    phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    data = np.cos(phi)[None] * np.sin(theta)[:, None]  # mock data
    alm = hp.Alm.from_grid(data, theta, phi, lmax, nside=nside)
    assert alm.lmax == lmax
    hp_map = hp.HealpixMap.from_grid(data, nside, theta, phi)
    alm2 = hp.Alm.from_healpix(hp_map, lmax=lmax)
    assert np.allclose(alm.alm, alm2.alm)


def test_alm_switch_coords():
    lmax = 10
    size = healpy.Alm.getsize(lmax)
    data = np.arange(size, dtype=np.complex128)
    # switch from galactic to equatorial
    coord = "G"
    new_coord = "C"
    alm = hp.Alm(alm=data, lmax=lmax, coord=coord)
    assert alm.coord == coord
    alm.switch_coords(new_coord)
    assert alm.coord == new_coord
    rot = healpy.Rotator(coord="gc")
    expected_data = rot.rotate_alm(data)
    assert np.allclose(alm.alm, expected_data)

    # several alms at once
    freqs = np.arange(10).reshape(-1, 1)
    data = np.arange(size, dtype=np.complex128).reshape(1, -1) * freqs
    alm = hp.Alm(alm=data, lmax=lmax, frequencies=freqs, coord=coord)
    assert alm.coord == coord
    alm.switch_coords(new_coord)
    assert alm.coord == new_coord
    expected_data = np.empty_like(data)
    for i in range(freqs.size):
        expected_data[i] = rot.rotate_alm(data[i])
    assert np.allclose(alm.alm, expected_data)


def test_getidx():
    lmax = 5
    alm_arr = np.zeros(healpy.Alm.getsize(lmax), dtype=np.complex128)
    alm = hp.Alm(alm_arr, lmax=lmax)
    ell = 3
    emm = 2
    bad_ell = 2 * lmax  # bigger than lmax
    bad_emm = 4  # bigger than ell
    with pytest.raises(IndexError):
        alm.getidx(bad_ell, emm)
        alm.getidx(ell, bad_emm)
        alm.getidx(-ell, emm)  # should fail since l < 0
        alm.getidx(ell, -emm)  # shoul fail since m < 0

    # try convert back and forth ell, emm <-> index
    ix = alm.getidx(ell, emm)
    ell_, emm_ = alm.getlm(i=ix)
    assert ell == ell_
    assert emm == emm_


def test_hp_map():
    # make constant map
    lmax = 10
    nside = 100
    alm_arr = np.zeros(healpy.Alm.getsize(lmax), dtype=np.complex128)
    alm = hp.Alm(alm_arr, lmax=lmax)
    a00 = 5
    alm[0, 0] = a00
    hp_map = alm.hp_map(nside=nside)
    npix = hp_map.shape[-1]
    assert nside == healpy.npix2nside(npix)
    expected_map = np.full(npix, a00 * Y00)
    assert np.allclose(hp_map, expected_map)

    # make many maps
    frequencies = np.linspace(1, 50, 50)
    alm_arr = np.repeat(alm_arr.reshape(1, -1), frequencies.size, axis=0)
    alm = hp.Alm(alm_arr, lmax=lmax, frequencies=frequencies)
    alm[:, 0, 0] = a00 * frequencies
    hp_map = alm.hp_map(nside=nside)
    expected_maps = np.full((frequencies.size, npix), a00 * Y00)
    expected_maps *= frequencies.reshape(-1, 1)
    assert np.allclose(hp_map, expected_maps)


def test_rot_alm_z():
    lmax = 10
    alm_arr = np.zeros(healpy.Alm.getsize(lmax), dtype=np.complex128)
    alm = hp.Alm(alm_arr, lmax=lmax)

    # rotate a single angle
    phi = np.pi / 2
    phase = alm.rot_alm_z(phi=phi)
    for ell in range(lmax + 1):
        for emm in range(ell + 1):
            ix = alm.getidx(ell, emm)
            assert np.isclose(phase[ix], np.exp(-1j * emm * phi))

    # rotate a set of angles
    phi = np.linspace(0, 2 * np.pi, num=361)  # 1 deg spacing
    phase = alm.rot_alm_z(phi=phi)
    for ell in range(lmax + 1):
        for emm in range(ell + 1):
            ix = alm.getidx(ell, emm)
            assert np.allclose(phase[:, ix], np.exp(-1j * emm * phi))

    # check that phi = 0 and phi = 2pi give the same answer
    assert np.allclose(phase[0], phase[-1])

    # rotate in time
    alm = hp.Alm(alm_arr, lmax=20)
    div = [1, 2, 4, 8]
    for d in div:
        dphi = 2 * np.pi / d
        # earth
        dt = sidereal_day_earth / d
        assert np.allclose(
            alm.rot_alm_z(times=dt, world="earth"), alm.rot_alm_z(phi=dphi)
        )
        # moon
        dt = sidereal_day_moon / d
        assert np.allclose(
            alm.rot_alm_z(times=dt, world="moon"), alm.rot_alm_z(phi=dphi)
        )
