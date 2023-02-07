from copy import deepcopy
import healpy
import numpy as np
import pytest
from croissant import rotations, healpix as hp
from croissant.constants import sidereal_day_earth, sidereal_day_moon, Y00


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
    assert np.allclose(data.ravel(), interp_data[0])


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

    # map that depends on phi
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    data = phi_grid.copy()
    hp_map = hp.grid2healpix(data, nside, theta=theta, phi=phi)
    # angles of healpix map:
    lon = hp.healpix2lonlat(nside)[0]
    hp_phi = np.deg2rad(lon)
    expected_map = hp_phi.copy()
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
    assert np.isclose(alm[0], Y00 * 4 * np.pi)  # a00 = Y00 * 4pi
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
    assert np.isclose(alm0[0] + c * Y00 * 4 * np.pi, alm1[0])


def test_alm2map():
    # make constant map with mmax = lmax
    lmax = 10
    mmax = lmax
    a00 = 3
    size = healpy.Alm.getsize(lmax, mmax=mmax)
    alm = np.zeros(size, dtype=np.complex128)
    alm[0] = a00
    nside = 32
    hp_map = hp.alm2map(alm, nside=nside, mmax=mmax)
    npix = hp_map.shape[-1]
    assert nside == healpy.npix2nside(npix)
    expected_map = np.full(npix, a00 * Y00)
    assert np.allclose(hp_map, expected_map)

    # constant map with mmax < lmax
    mmax = 5
    size = healpy.Alm.getsize(lmax, mmax=mmax)
    alm = np.zeros(size, dtype=np.complex128)
    alm[0] = a00
    hp_map = hp.alm2map(alm, nside=nside, mmax=mmax)
    npix = hp_map.shape[-1]
    assert nside == healpy.npix2nside(npix)
    expected_map = np.full(npix, a00 * Y00)
    assert np.allclose(hp_map, expected_map)

    # make many maps
    frequencies = np.linspace(1, 50, 50)
    mmax = lmax
    size = healpy.Alm.getsize(lmax, mmax=mmax)
    alm = np.zeros((frequencies.size, size), dtype=np.complex128)
    alm[:, 0] = a00 * frequencies**2.5
    hp_map = hp.alm2map(alm, nside=nside, mmax=mmax)
    expected_maps = np.full((frequencies.size, npix), a00 * Y00)
    expected_maps *= frequencies.reshape(-1, 1) ** 2.5
    assert np.allclose(hp_map, expected_maps)

    # inverting map2alm
    lmax = 10
    mmax = lmax
    size = healpy.Alm.getsize(lmax, mmax=mmax)
    alm = np.zeros(size, dtype=np.complex128)
    lm_dict = {
        (0, 0): 5.1,
        (2, 0): 7.4,
        (3, 2): 3 + 5j,
        (4, 1): -4.3 + 1.3j,
        (7, 7): 11,
    }
    for ell, emm in lm_dict:
        ix = healpy.Alm.getidx(lmax, ell, emm)
        alm[ix] = lm_dict[(ell, emm)]

    nside = 64
    m = hp.alm2map(alm, nside=nside)
    alm_ = hp.map2alm(m, lmax)
    assert np.allclose(alm, alm_)
    m_ = hp.alm2map(alm_, nside=nside)
    assert np.allclose(m, m_)


def test_nested_input():
    freqs = np.linspace(1, 50, 50)
    nside = 10  # this is invalid for NESTED but OK for RING
    npix = healpy.nside2npix(nside)
    data = 10 * np.ones((freqs.size, npix))
    kwargs = {
        "data": data,
        "nside": nside,
        "nested_input": False,
        "frequencies": freqs,
    }
    hp.HealpixMap(**kwargs)  # should work
    # should raise an error if nested is True
    kwargs["nested_input"] = True
    with pytest.raises(ValueError):
        hp.HealpixMap(**kwargs)

    # valid nested input
    nside = 8
    npix = healpy.nside2npix(nside)
    data = np.arange(npix) ** 2
    kwargs = {
        "data": data,
        "nside": nside,
        "nested_input": True,
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
    alm = hp.Alm(lmax=lmax)
    a00 = 10
    alm[0, 0] = a00
    hp_map = hp.HealpixMap.from_alm(alm, nside=None)
    # healpix map should be able to infer the nside from lmax
    assert hp_map.nside == nside
    npix = healpy.nside2npix(nside)
    # the map should just be = a00 * Y00 everywhere
    expected_map = np.full((1, npix), a00 * Y00)
    assert np.allclose(hp_map.data, expected_map)


def test_ud_grade():
    freqs = np.linspace(1, 50, 50)
    nside = 8
    npix = healpy.nside2npix(nside)
    data = np.repeat(np.arange(npix).reshape(1, -1), freqs.size, axis=0)
    kwargs = {
        "data": data,
        "nside": nside,
        "nested_input": False,
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
    nside = 8
    npix = healpy.nside2npix(nside)
    data = np.arange(npix)
    # switch from galactic to equatorial
    coords = "galactic"
    new_coords = "equatorial"
    hp_map = hp.HealpixMap(data=data, nside=nside, coords=coords)
    assert hp_map.coords == coords
    hp_map.switch_coords(new_coords)
    assert hp_map.coords == new_coords
    expected_data = rotations.rotate_map(
        data, from_coords=coords, to_coords=new_coords
    )
    assert np.allclose(hp_map.data, expected_data)

    # several maps at once
    freqs = np.arange(10).reshape(-1, 1)
    data = np.arange(npix).reshape(1, -1) * freqs
    hp_map = hp.HealpixMap(
        data=data, nside=nside, frequencies=freqs, coords=coords
    )
    assert hp_map.coords == coords
    hp_map.switch_coords(new_coords)
    assert hp_map.coords == new_coords
    expected_data = rotations.rotate_map(
        data, from_coords=coords, to_coords=new_coords
    )
    assert np.allclose(hp_map.data, expected_data)


def test_alm():
    nside = 32
    npix = healpy.nside2npix(nside)
    data = np.arange(npix)
    hp_map = hp.HealpixMap(data=data, nside=nside)
    # test default lmax
    alm = hp_map.alm(lmax=None)
    expected_lmax = 3 * nside - 1  # should be default
    expected_size = healpy.Alm.getsize(expected_lmax, mmax=expected_lmax)
    assert alm.shape == (1, expected_size)
    # specify lmax
    lmax = 10
    alm = hp_map.alm(lmax=lmax)
    expected_size = healpy.Alm.getsize(lmax, mmax=lmax)
    assert alm.shape == (1, expected_size)

    # several maps at once
    freqs = np.arange(10).reshape(-1, 1)
    data = np.arange(npix).reshape(1, -1) * freqs
    hp_map = hp.HealpixMap(data=data, nside=nside, frequencies=freqs)
    alm = hp_map.alm(lmax=lmax)
    assert alm.shape == (freqs.size, expected_size)


def test_Alm_init():
    lmax = 5
    size = healpy.Alm.getsize(lmax, mmax=lmax)
    freqs = np.linspace(1, 50, 50)
    # alm = None
    alm = hp.Alm(alm=None, lmax=lmax, frequencies=freqs)
    expected_alm = np.zeros((freqs.size, size), dtype=np.complex128)
    assert np.allclose(alm.alm, expected_alm)
    # lmax = None
    alms = np.arange(size).reshape(1, -1) * freqs.reshape(-1, 1)
    alm = hp.Alm(alm=alms, lmax=None, frequencies=freqs)
    assert alm.lmax == lmax  # init should infer lmax from alm size
    # both alm and lmax are specfied but inconsistent
    wrong_lmax = 10
    with pytest.raises(ValueError):
        hp.Alm(alm=alms, lmax=wrong_lmax, frequencies=freqs)


def test_alm_indexing():
    lmax = 10
    freqs = np.linspace(1, 50, 50)
    nfreqs = freqs.size
    # initialize all alms to 0
    alm = hp.Alm(alm=None, lmax=lmax, frequencies=freqs)
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
    alm.all_zero()
    # negative indexing
    val = 3.0 + 2.3j
    alm[-1, 10, 7] = val
    assert alm[-1, 10, 7] == val
    ix = healpy.Alm.getidx(lmax, 10, 7)
    assert alm[-1, 10, 7] == alm.alm[-1, ix]

    # negative emm
    alm.all_zero()
    alm[0, 3, 2] = val
    assert np.isclose(alm[0, 3, -2], (-1) ** 2 * np.conj(val))

    # frequency index not specified
    with pytest.raises(IndexError):
        alm[3, 2] = 5
        alm[7, -1]

    # no frequencies
    alm = hp.Alm(alm=None, lmax=lmax, frequencies=None)
    alm[5, 2] = 3.0
    assert alm[5, 2] == 3.0
    assert alm[0, 5, 2] == 3.0  # can optionally have freq idx = 0
    assert alm[-1, 5, 2] == 3.0  # ... or -1


def test_from_healpix():
    nside = 8
    npix = healpy.nside2npix(nside)
    freqs = np.linspace(1, 50, 50)
    data = np.arange(npix).reshape(1, -1) * freqs.reshape(-1, 1) ** 2
    coords = "equatorial"
    hp_map = hp.HealpixMap(
        data=data, nside=nside, frequencies=freqs, coords=coords
    )
    lmax = 10
    alm = hp.Alm.from_healpix(hp_map, lmax=lmax)
    assert alm.lmax == lmax
    assert np.allclose(alm.frequencies, freqs)
    assert alm.coords == coords
    assert np.allclose(alm.alm, hp.map2alm(data, lmax))


def test_alm_switch_coords():
    lmax = 10
    size = healpy.Alm.getsize(lmax, mmax=lmax)
    data = np.arange(size)
    # switch from galactic to equatorial
    coords = "galactic"
    new_coords = "equatorial"
    alm = hp.Alm(alm=data, lmax=lmax, coords=coords)
    assert alm.coords == coords
    alm.switch_coords(new_coords)
    assert alm.coords == new_coords
    expected_data = rotations.rotate_alm(
        data, from_coords=coords, to_coords=new_coords
    )
    assert np.allclose(alm.alm, expected_data)

    # several alms at once
    freqs = np.arange(10).reshape(-1, 1)
    data = np.arange(size).reshape(1, -1) * freqs
    alm = hp.Alm(alm=data, lmax=lmax, frequencies=freqs, coords=coords)
    assert alm.coords == coords
    alm.switch_coords(new_coords)
    assert alm.coords == new_coords
    expected_data = rotations.rotate_alm(
        data, from_coords=coords, to_coords=new_coords
    )
    assert np.allclose(alm.alm, expected_data)


def test_getidx():
    lmax = 5
    alm = hp.Alm(lmax=lmax)
    ell = 3
    emm = 2
    bad_ell = 2 * lmax  # bigger than lmax
    bad_emm = 4  # bigger than ell
    with pytest.raises(ValueError):
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
    alm = hp.Alm(lmax=lmax)
    a00 = 5
    alm[0, 0] = a00
    hp_map = alm.hp_map(nside=nside)
    npix = hp_map.shape[-1]
    assert nside == healpy.npix2nside(npix)
    expected_map = np.full((1, npix), a00 * Y00)
    assert np.allclose(hp_map, expected_map)

    # make many maps
    frequencies = np.linspace(1, 50, 50)
    alm = hp.Alm(lmax=lmax, frequencies=frequencies)
    alm[:, 0, 0] = a00 * frequencies
    hp_map = alm.hp_map(nside=nside)
    expected_maps = np.full((frequencies.size, npix), a00 * Y00)
    expected_maps *= frequencies.reshape(-1, 1)
    assert np.allclose(hp_map, expected_maps)


def test_rotate_alm_angle():
    lmax = 10
    alm = hp.Alm(lmax=lmax)
    phi = np.pi / 2
    phase = alm.rotate_alm_angle(phi)
    for ell in range(lmax + 1):
        for emm in range(ell + 1):
            ix = alm.getidx(ell, emm)
            assert np.isclose(phase[0, 0, ix], np.exp(1j * emm * phi))

    # rotate a set of angles
    phi = np.linspace(0, 2 * np.pi, num=361)  # 1 deg spacing
    phase = alm.rotate_alm_angle(phi)
    for ell in range(lmax + 1):
        for emm in range(ell + 1):
            ix = alm.getidx(ell, emm)
            assert np.allclose(phase[:, 0, ix], np.exp(1j * emm * phi))

    # check that phi = 0 and phi = 2pi give the same answer
    assert np.allclose(phase[0], phase[-1])


def test_rotate_alm_time():
    alm = hp.Alm(lmax=20, frequencies=np.linspace(1, 50, 50))
    div = [1, 2, 4, 8]
    for d in div:
        dphi = 2 * np.pi / d
        # earth
        dt = sidereal_day_earth / d
        assert np.allclose(
            alm.rotate_alm_time(dt, world="earth"), alm.rotate_alm_angle(dphi)
        )
        # moon
        dt = sidereal_day_moon / d
        assert np.allclose(
            alm.rotate_alm_time(dt, world="moon"), alm.rotate_alm_angle(dphi)
        )
