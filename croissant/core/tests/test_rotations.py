from astropy.coordinates import EarthLocation
import healpy as hp
from lunarsky import MoonLocation, Time
import numpy as np
import pytest

from croissant import rotations

rng = np.random.default_rng(1234)


def test_rotator_init():
    # invalid eulertype
    with pytest.raises(ValueError):
        rotations.Rotator(eulertype="foo")

    # too many coords
    with pytest.raises(ValueError):
        rotations.Rotator(coord=["G", "C", "M"])

    # invalid coord
    with pytest.raises(KeyError):
        rotations.Rotator(coord=["foo", "C"])

    # topocentric without location
    time = Time("2022-06-16 17:00:00")
    with pytest.raises(ValueError):
        rotations.Rotator(coord=["T", "M"], time=time)
    loc = EarthLocation(lon=0, lat=40)
    with pytest.raises(ValueError):  # location is EarthLocation
        rotations.Rotator(coord=["T", "M"], loc=loc, time=time)
    # should work
    loc = MoonLocation(lon=0, lat=40)
    _ = rotations.Rotator(coord=["T", "M"], loc=loc, time=time)
    # no time:
    with pytest.raises(ValueError):
        rotations.Rotator(coord=["T", "M"], loc=loc)


@pytest.mark.parametrize("lmax", [8, 16, 32, 64])
def test_rotate_alm(lmax):
    size = hp.Alm.getsize(lmax)
    alm = rng.standard_normal(size) + 1j * rng.standard_normal(size)

    # galactic -> equatorial
    rot = rotations.Rotator(coord=["G", "C"])
    rot_alm = rot.rotate_alm(alm, lmax=lmax)
    hp_rot = hp.Rotator(coord=["G", "C"])
    hp_rot_alm = hp_rot.rotate_alm(alm)
    assert np.allclose(rot_alm, hp_rot_alm)

    # galactic -> equatorial with subsequent rotation in equatorial coords
    eul = (np.pi / 2, 0, np.pi / 2)
    rot = rotations.Rotator(rot=eul, coord=["G", "C"])
    rot_alm = rot.rotate_alm(alm, lmax=lmax)
    hp_rot = hp.Rotator(rot=eul, coord=["G", "C"])
    hp_rot_alm = hp_rot.rotate_alm(alm)
    assert np.allclose(rot_alm, hp_rot_alm)

    # test polarization maps
    t = alm.copy()
    e = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    b = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    alm_pol = np.array([t, e, b])
    rot = rotations.Rotator(coord=["G", "C"])
    rot_alm = rot.rotate_alm(alm_pol, lmax=lmax, polarized=True)
    hp_rot = hp.Rotator(coord=["G", "C"])
    hp_rot_alm = hp_rot.rotate_alm(alm_pol, lmax=lmax)
    assert np.allclose(rot_alm, hp_rot_alm)

    # multiple sets of alms at once
    alm.shape = (1, size)
    f = np.linspace(1, 50, 50).reshape(-1, 1)
    alms = np.repeat(alm, f.size, axis=0) * f**2
    assert alms.shape == (50, size)
    rot = rotations.Rotator(coord=["G", "C"])
    rot_alms = rot.rotate_alm(alms, lmax=lmax)
    hp_rot_alms = np.empty_like(alms)
    hp_rot = hp.Rotator(coord=["G", "C"])
    for i, a in enumerate(alms):
        hp_rot_alms[i] = hp_rot.rotate_alm(a, lmax=lmax)
    assert np.allclose(rot_alms, hp_rot_alms)


def test_rotate_map():
    # compare for nside 64 which uses pixel weights
    nside = 64
    npix = hp.nside2npix(nside)
    m = np.arange(npix)
    rot = rotations.Rotator(coord=["G", "C"])
    rm = rot.rotate_map_alms(m)
    hp_rot = hp.Rotator(coord=["G", "C"])
    hprm = hp_rot.rotate_map_alms(m, use_pixel_weights=True)
    assert np.allclose(rm, hprm)
    # without pixel weights it should be different
    hprm = hp_rot.rotate_map_alms(m, use_pixel_weights=False)
    assert not np.allclose(rm, hprm)

    # rotate in pixel space
    rm = rot.rotate_map_pixel(m)
    hprm = hp_rot.rotate_map_pixel(m)
    assert np.allclose(rm, hprm)

    # test polarization maps
    i_map = m.copy()
    q_map = rng.standard_normal(npix)
    u_map = rng.standard_normal(npix)
    map_pol = np.array([i_map, q_map, u_map])
    rm = rot.rotate_map_alms(map_pol, polarized=True)
    hprm = hp_rot.rotate_map_alms(map_pol, use_pixel_weights=True)
    assert np.allclose(rm, hprm)
    rm = rot.rotate_map_pixel(map_pol, polarized=True)
    hprm = hp_rot.rotate_map_pixel(map_pol)
    assert np.allclose(rm, hprm)

    # several maps at once
    f = np.linspace(1, 50, 50).reshape(-1, 1)
    maps = m.reshape(1, -1) * f**3.2
    rm = rot.rotate_map_alms(maps)
    hp_rot = hp.Rotator(coord=["G", "C"])
    hprm = np.empty((f.size, npix))
    for i, m in enumerate(maps):
        hprm[i] = hp_rot.rotate_map_alms(m, use_pixel_weights=True)
    assert np.allclose(rm, hprm)
