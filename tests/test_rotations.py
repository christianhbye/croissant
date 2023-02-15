from astropy.coordinates import AltAz, EarthLocation
import healpy as hp
from lunarsky import LunarTopo, MCMF, MoonLocation, SkyCoord, Time
import numpy as np
import pytest

from croissant import rotations


def test_get_rotmat():
    # check that we agree with healpy for galactic -> equatorial
    rot_mat = rotations.get_rotmat("galactic", "fk5")
    rot = hp.Rotator(coord=["G", "C"])
    assert np.allclose(rot_mat, rot.mat)

    # equatorial -> galactic
    rot_mat = rotations.get_rotmat("equatorial", "galactic")
    rot = hp.Rotator(coord=["C", "G"])
    assert np.allclose(rot_mat, rot.mat)

    # check that we agree with astropy for equatorial -> AltAz
    time = Time("2022-06-16 17:00:00")
    loc = EarthLocation(lon=0, lat=40)
    to_frame = AltAz(obstime=time, location=loc)
    rot_mat = rotations.get_rotmat("equatorial", to_frame)
    x, y, z = np.eye(3)
    xp, yp, zp = (
        SkyCoord(x=x, y=y, z=z, frame="fk5")
        .transform_to(to_frame)
        .cartesian.xyz.value
    )
    assert np.allclose(rot_mat, np.array([xp, yp, zp]).T)

    # MCMF -> AltAz
    loc = MoonLocation(lon=0, lat=40)
    to_frame = LunarTopo(obstime=time, location=loc)
    rot_mat = rotations.get_rotmat("mcmf", to_frame)
    xp, yp, zp = (
        SkyCoord(x=x, y=y, z=z, frame=MCMF())
        .transform_to(to_frame)
        .cartesian.xyz.value
    )
    assert np.allclose(rot_mat, np.array([xp, yp, zp]).T)

    # galactic -> MCMF
    rot_mat = rotations.get_rotmat("galactic", MCMF())
    xp, yp, zp = (
        SkyCoord(x=x, y=y, z=z, frame="galactic")
        .transform_to(MCMF())
        .cartesian.xyz.value
    )
    assert np.allclose(rot_mat, np.array([xp, yp, zp]).T)


eul = np.repeat(np.linspace(0, 2 * np.pi, 10), 3).reshape(3, -1)


@pytest.mark.parametrize("alpha, beta, gamma", eul)
def test_rotmat_to_euler(alpha, beta, gamma):
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)

    Z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    Y = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    X = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    rot_mat = X @ Y @ Z
    euler = rotations.rotmat_to_euler(rot_mat)
    assert np.allclose(euler, [alpha, -beta, gamma])  # weird sign convention


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
    # pixel weights are not computed for nside = 8
    nside = 8
    npix = hp.nside2npix(nside)
    m = np.arange(npix)
    rot = rotations.Rotator(coord=["G", "C"])
    rm = rot.rotate_map_alms(m)
    hp_rot = hp.Rotator(coord=["G", "C"])
    hprm = hp_rot.rotate_map_alms(m, use_pixel_weights=False)
    assert np.allclose(rm, hprm)

    # compare for nside 64 which uses pixel weights
    nside = 64
    npix = hp.nside2npix(nside)
    m = np.arange(npix)
    rot = rotations.Rotator(coord=["G", "C"])
    rm = rotations.rotate_map_alms(m)
    hp_rot = hp.Rotator(coord=["C", "G"])
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
