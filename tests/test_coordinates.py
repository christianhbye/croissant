from astropy.coordinates import AltAz, EarthLocation, ICRS
from astropy.time import Time
from astropy import units
import healpy as hp
import numpy as np

from croissant import coordinates as coord


def test_top2radec():
    phi = np.linspace(0, 2 * np.pi, num=360, endpoint=False)
    theta = np.linspace(0, np.pi, num=181)
    loc = EarthLocation(
        lat=40.5 * units.deg, lon=130.2 * units.deg, height=10.3 * units.m
    )
    time = "2022-06-09 16:12:00"
    ra, dec = coord.topo2radec(theta, phi, time, loc, grid=True)
    assert -90 <= dec.all() < 90
    assert 0 <= ra.all() < 360
    az = phi.reshape(1, -1) * units.rad
    alt = (np.pi / 2 - theta.reshape(-1, 1)) * units.rad
    aa = AltAz(alt=alt, az=az, location=loc, obstime=time)
    icrs = aa.transform_to(ICRS())
    assert np.allclose(ra, icrs.ra.rad.ravel())
    assert np.allclose(dec, icrs.dec.rad.ravel())

    # change phi/theta from coord axes to datapoints
    phi, theta = [ang.ravel() for ang in np.meshgrid(phi, theta)]
    ra, dec = coord.topo2radec(theta, phi, time, loc, grid=False)
    assert -90 <= dec.all() < 90
    assert 0 <= ra.all() < 360
    assert np.allclose(ra, icrs.ra.rad.ravel())
    assert np.allclose(dec, icrs.dec.rad.ravel())

    # pass astropy time to top2radec:
    time = Time(time)
    ra, dec = coord.topo2radec(theta, phi, time, loc, grid=False)
    assert -90 <= dec.all() < 90
    assert 0 <= ra.all() < 360
    assert np.allclose(ra, icrs.ra.rad.ravel())
    assert np.allclose(dec, icrs.dec.rad.ravel())

    # invert radec2topo
    phi = np.linspace(0, 2 * np.pi, num=360, endpoint=False)
    theta = np.linspace(0, np.pi, num=181)[1:-1]  # remove poles
    phi, theta = [ang.ravel() for ang in np.meshgrid(phi, theta)]
    ra, dec = np.rad2deg(coord.topo2radec(theta, phi, time, loc, grid=False))
    theta_, phi_ = coord.radec2topo(ra, dec, time, loc)
    # handle precision issue
    phi_ = np.where(phi_ >= 2 * np.pi - 1e-10, phi_ - 2 * np.pi, phi_)
    assert np.allclose(theta, theta_)
    assert np.allclose(phi, phi_)


def test_radec2topo():
    ra = np.linspace(0, 360, num=360, endpoint=False)
    dec = np.linspace(90, -90, num=181)
    ra, dec = [ang.ravel() for ang in np.meshgrid(ra, dec)]
    loc = EarthLocation(
        lat=40.5 * units.deg, lon=130.2 * units.deg, height=10.3 * units.m
    )
    time = "2022-06-09 16:12:00"
    theta, phi = coord.radec2topo(ra, dec, time, loc)
    assert 0 <= theta.all() < np.pi / 2
    assert 0 <= phi.all() < 2 * np.pi
    icrs = ICRS(ra=ra * units.deg, dec=dec * units.deg)
    aa = icrs.transform_to(AltAz(location=loc, obstime=time))
    assert np.allclose(theta, np.pi / 2 - aa.alt.rad)
    assert np.allclose(phi, aa.az.rad)

    # use astropy time
    time = Time(time)
    theta, phi = coord.radec2topo(ra, dec, time, loc)
    assert 0 <= theta.all() < np.pi / 2
    assert 0 <= phi.all() < 2 * np.pi
    assert np.allclose(theta, np.pi / 2 - aa.alt.rad)
    assert np.allclose(phi, aa.az.rad)

    # invert topo2radec
    ra = np.linspace(0, 360, num=360, endpoint=False)
    dec = np.linspace(90, -90, num=181)
    dec = dec[1:-1]  # remove poles since it will mess with ra
    ra, dec = [ang.ravel() for ang in np.meshgrid(ra, dec)]
    theta, phi = coord.radec2topo(ra, dec, time, loc)
    ra_, dec_ = np.rad2deg(coord.topo2radec(theta, phi, time, loc, grid=False))
    # handle precision issue
    ra_ = np.where(ra_ >= 360 - 1e-10, ra_ - 360, ra_)
    assert np.allclose(dec, dec_)
    assert np.allclose(ra, ra_)


def test_hp_rotate():
    # verify that this method yields the same as the healpy example in the docs
    rot = coord.hp_rotate("galactic", "ecliptic")
    theta_gal, phi_gal = np.pi / 2, 0
    theta_ecl, phi_ecl = rot(theta_gal, phi_gal)
    assert np.isclose(theta_ecl, 1.66742347999)  # from healpy docs
    assert np.isclose(phi_ecl, -1.6259571125)
    vec_gal = np.array([1, 0, 0])
    vec_ecl = rot(vec_gal)
    assert np.allclose(vec_ecl, [-0.05487563, -0.99382135, -0.09647686])


def test_rotate_map():
    # pixel weights are not computed for nside = 8
    nside = 8
    npix = hp.nside2npix(nside)
    m = np.arange(npix)
    rm = coord.rotate_map(m, from_coords="galactic", to_coords="equatorial")
    hp_rot = hp.Rotator(coord=["G", "C"])
    hprm = hp_rot.rotate_map_alms(m, use_pixel_weights=False)
    assert np.allclose(rm, hprm)

    # compare for nside 64 which uses pixel weights
    nside = 64
    npix = hp.nside2npix(nside)
    m = np.arange(npix)
    rm = coord.rotate_map(m, from_coords="equatorial", to_coords="ecliptic")
    hp_rot = hp.Rotator(coord=["C", "E"])
    hprm = hp_rot.rotate_map_alms(m, use_pixel_weights=True)
    assert np.allclose(rm, hprm)
    # without pixel weights it should be different
    hprm = hp_rot.rotate_map_alms(m, use_pixel_weights=False)
    assert not np.allclose(rm, hprm)

    # several maps at once
    f = np.linspace(1, 50, 50).reshape(-1, 1)
    maps = m.reshape(1, -1) * f**3.2
    rm = coord.rotate_map(maps, from_coords="galactic", to_coords="equatorial")
    hp_rot = hp.Rotator(coord=["G", "C"])
    hprm = np.empty((f.size, npix))
    for i, m in enumerate(maps):
        hprm[i] = hp_rot.rotate_map_alms(m, use_pixel_weights=True)
    assert np.allclose(rm, hprm)


def test_rotate_alm():
    lmax = 10
    size = hp.Alm.getsize(lmax)
    alm = np.arange(size) + 5j * np.arange(size) ** 2
    rot_alm = coord.rotate_alm(
        alm, from_coords="galactic", to_coords="equatorial"
    )
    hp_rot = hp.Rotator(coord=["G", "C"])
    hp_rot_alm = hp_rot.rotate_alm(alm)
    assert np.allclose(rot_alm, hp_rot_alm)

    # multiple sets of alms at once
    alm.shape = (1, size)
    f = np.linspace(1, 50, 50).reshape(-1, 1)
    alms = np.repeat(alm, f.size, axis=0) * f**2
    assert alms.shape == (50, size)
    rot_alms = coord.rotate_alm(
        alms, from_coords="galactic", to_coords="equatorial"
    )
    hp_rot_alms = np.empty_like(alms)
    for i, a in enumerate(alms):
        hp_rot_alms[i] = hp_rot.rotate_alm(a)
    assert np.allclose(rot_alms, hp_rot_alms)
