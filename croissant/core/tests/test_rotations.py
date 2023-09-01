from astropy.coordinates import AltAz, EarthLocation
import healpy as hp
from lunarsky import LunarTopo, MCMF, MoonLocation, SkyCoord, Time
import numpy as np
import pytest

from croissant import rotations


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


def test_get_rot_mat():
    # check that we agree with healpy for galactic -> equatorial
    rot_mat = rotations.get_rot_mat("galactic", "fk5")
    rot = hp.Rotator(coord=["G", "C"])
    assert np.allclose(rot_mat, rot.mat)

    # equatorial -> galactic
    rot_mat = rotations.get_rot_mat("fk5", "galactic")
    rot = hp.Rotator(coord=["C", "G"])
    assert np.allclose(rot_mat, rot.mat)

    # check that we agree with astropy for equatorial -> AltAz
    time = Time("2022-06-16 17:00:00")
    loc = EarthLocation(lon=0, lat=40)
    to_frame = AltAz(obstime=time, location=loc)
    rot_mat = rotations.get_rot_mat("fk5", to_frame)
    x, y, z = np.eye(3)
    xp, yp, zp = (
        SkyCoord(x=x, y=y, z=z, frame="fk5", representation_type="cartesian")
        .transform_to(to_frame)
        .cartesian.xyz.value
    )
    assert np.allclose(rot_mat, np.array([xp, yp, zp]))

    # MCMF -> AltAz
    loc = MoonLocation(lon=0, lat=40)
    to_frame = LunarTopo(obstime=time, location=loc)
    rot_mat = rotations.get_rot_mat("mcmf", to_frame)
    xp, yp, zp = (
        SkyCoord(x=x, y=y, z=z, frame="mcmf", representation_type="cartesian")
        .transform_to(to_frame)
        .cartesian.xyz.value
    )
    assert np.allclose(rot_mat, np.array([xp, yp, zp]))

    # galactic -> MCMF
    # in this case we have to invert the matrix that does MCMF -> galactic
    # since we cannot instantiate a galactic frame from cartesian coords
    rot_mat = rotations.get_rot_mat("galactic", MCMF())
    xp, yp, zp = (
        SkyCoord(x=x, y=y, z=z, frame="mcmf", representation_type="cartesian")
        .transform_to("galactic")
        .cartesian.xyz.value
    )
    assert np.allclose(rot_mat, np.array([xp, yp, zp]).T)


def test_rotmat_to_euler():
    # check that rotmat_to_euler is the inverse of euler_matrix_new
    rot_mat = rotations.get_rot_mat("galactic", "fk5")
    eul = rotations.rotmat_to_euler(rot_mat)
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)

    rot_mat = rotations.get_rot_mat("galactic", "mcmf")
    eul = rotations.rotmat_to_euler(rot_mat)
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)

    rot_mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    eul = rotations.rotmat_to_euler(rot_mat)
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)


def test_rotate_alm():
    lmax = 10
    size = hp.Alm.getsize(lmax)
    alm = np.arange(size) + 5j * np.arange(size) ** 2

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

    # several maps at once
    f = np.linspace(1, 50, 50).reshape(-1, 1)
    maps = m.reshape(1, -1) * f**3.2
    rm = rot.rotate_map_alms(maps)
    hp_rot = hp.Rotator(coord=["G", "C"])
    hprm = np.empty((f.size, npix))
    for i, m in enumerate(maps):
        hprm[i] = hp_rot.rotate_map_alms(m, use_pixel_weights=True)
    assert np.allclose(rm, hprm)
