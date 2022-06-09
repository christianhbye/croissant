from astropy.coordianates import AltAz, EarthLocation, ICRS
from astropy.time import Time
from astropy import units
import numpy as np

from croissant import coordinates as coord


def test_top2radec():
    phi = np.linspace(0, 2 * np.pi, num=360, endpoint=False)
    theta = np.linspace(0, np.pi, num=181)
    loc = (40.5, 130.2, 10.3)
    time = "2022-06-09 16:12:00"
    ra, dec = coord.topo2radec(theta, phi, time, loc, grid=True)
    assert -90 <= dec.all() < 90
    assert 0 <= ra.all() < 360
    az = phi.reshape(1, -1) * units.rad
    alt = (np.pi / 2 - theta.reshape(-1, 1)) * units.rad
    aa = AltAz(alt=alt, az=az, location=loc, obstime=time)
    icrs = aa.transform_to(ICRS())
    assert np.allclose(ra, icrs.ra.rad)
    assert np.allclose(dec, icrs.dec.rad)

    # change phi/theta from coord axes to datapoints
    phi, theta = [ang.ravel() for ang in np.meshgrid(phi, theta)]
    ra, dec = coord.topo2radec(theta, phi, time, loc, grid=False)
    assert -90 <= dec.all() < 90
    assert 0 <= ra.all() < 360
    assert np.allclose(ra, icrs.ra.rad)
    assert np.allclose(dec, icrs.dec.rad)

    # pass astropy objects to top2radec:
    loc = EarthLocation(loc)
    time = Time(time)
    ra, dec = coord.topo2radec(theta, phi, time, loc, grid=False)
    assert -90 <= dec.all() < 90
    assert 0 <= ra.all() < 360
    assert np.allclose(ra, icrs.ra.rad)
    assert np.allclose(dec, icrs.dec.rad)

    # topo2radec doesn't care about added multiples of 2pi
    ra, dec = coord.topo2radec(
        theta + 16 * np.pi, phi - 8 * np.pi, time, loc, grid=False
    )
    assert -90 <= dec.all() < 90
    assert 0 <= ra.all() < 360
    assert np.allclose(ra, icrs.ra.rad)
    assert np.allclose(dec, icrs.dec.rad)

    # invert radec2topo
    ra, dec = coord.topo2radec(theta, phi, time, loc, grid=False)
    theta_, phi_ = coord.radec2topo(ra, dec, time, loc)
    assert np.allclose(theta, theta_)
    assert np.allclose(phi, phi_)


def test_radec2topo():
    ra = np.linspace(0, 360, num=360, endpoint=False)
    dec = np.linspace(90, -90, num=181)
    loc = (40.5, 130.2, 10.3)
    time = "2022-06-09 16:12:00"
    theta, phi = coord.radec2topo(ra, dec, time, loc)
    assert 0 <= theta.all() < np.pi / 2
    assert 0 <= phi.all() < 2 * np.pi
    icrs = ICRS(ra=ra * units.deg, dec=dec * units.deg)
    aa = icrs.transform_to(AltAz(location=loc, obstime=time))
    assert np.allclose(theta, np.pi / 2 - aa.alt.rad)
    assert np.allclose(phi, aa.az.rad)

    # use astropy objects
    loc = EarthLocation(loc)
    time = Time(time)
    theta, phi = coord.radec2topo(ra, dec, time, loc)
    assert 0 <= theta.all() < np.pi / 2
    assert 0 <= phi.all() < 2 * np.pi
    assert np.allclose(theta, np.pi / 2 - aa.alt.rad)
    assert np.allclose(phi, aa.az.rad)

    # radec2topo doesn't care about added mutliples of 360
    theta, phi = coord.topo2radec(ra - 4 * 360, dec + 8 * 360, time, loc)
    assert 0 <= theta.all() < np.pi / 2
    assert 0 <= phi.all() < 2 * np.pi
    assert np.allclose(theta, np.pi / 2 - aa.alt.rad)
    assert np.allclose(phi, aa.az.rad)

    # invert topo2radec
    theta, phi = coord.radec2topo(ra, dec, time, loc)
    ra_, dec_ = coord.topo2radec(theta, phi, time, loc, grid=False)
    assert np.allclose(ra, ra_)
    assert np.allclose(dec, dec_)
