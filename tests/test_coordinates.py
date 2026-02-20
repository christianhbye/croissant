import healpy as hp
import numpy as np
from astropy.coordinates import AltAz, EarthLocation
from lunarsky import MCMF, LunarTopo, MoonLocation, SkyCoord, Time

from croissant import utils


def test_coord_rep():
    coords = ["galactic", "equatorial", "ecliptic", "mcmf", "topocentric"]
    short = ["G", "C", "E", "M", "T"]
    for i in range(len(coords)):
        assert utils.coord_rep(coords[i]) == short[i]


def test_get_rot_mat():
    # check that we agree with healpy for galactic -> equatorial
    rot_mat = utils.get_rot_mat("galactic", "fk5")
    rot = hp.Rotator(coord=["G", "C"])
    assert np.allclose(rot_mat, rot.mat)

    # equatorial -> galactic
    rot_mat = utils.get_rot_mat("fk5", "galactic")
    rot = hp.Rotator(coord=["C", "G"])
    assert np.allclose(rot_mat, rot.mat)

    # check that we agree with astropy for equatorial -> AltAz
    time = Time("2022-06-16 17:00:00")
    loc = EarthLocation(lon=0, lat=40)
    to_frame = AltAz(obstime=time, location=loc)
    rot_mat = utils.get_rot_mat("fk5", to_frame)
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
    rot_mat = utils.get_rot_mat("mcmf", to_frame)
    xp, yp, zp = (
        SkyCoord(x=x, y=y, z=z, frame="mcmf", representation_type="cartesian")
        .transform_to(to_frame)
        .cartesian.xyz.value
    )
    assert np.allclose(rot_mat, np.array([xp, yp, zp]))

    # galactic -> MCMF
    # in this case we have to invert the matrix that does MCMF -> galactic
    # since we cannot instantiate a galactic frame from cartesian coords
    rot_mat = utils.get_rot_mat("galactic", MCMF())
    xp, yp, zp = (
        SkyCoord(x=x, y=y, z=z, frame="mcmf", representation_type="cartesian")
        .transform_to("galactic")
        .cartesian.xyz.value
    )
    assert np.allclose(rot_mat, np.array([xp, yp, zp]).T)


def test_rotmat_to_euler():
    # check that rotmat_to_euler is the inverse of euler_matrix_new
    rot_mat = utils.get_rot_mat("galactic", "fk5")
    eul = utils.rotmat_to_euler(rot_mat)
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)

    rot_mat = utils.get_rot_mat("galactic", "mcmf")
    eul = utils.rotmat_to_euler(rot_mat)
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)

    rot_mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    eul = utils.rotmat_to_euler(rot_mat)
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)
