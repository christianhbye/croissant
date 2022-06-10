from astropy.coordinates import AltAz, EarthLocation, ICRS
from astropy.time import Time
from astropy import units
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
