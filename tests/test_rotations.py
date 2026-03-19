import healpy as hp
import numpy as np
from astropy.coordinates import AltAz, EarthLocation
from lunarsky import MCMF, LunarTopo, MoonLocation, SkyCoord, Time

from croissant import rotations


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
    eul = rotations.rotmat_to_euler(rot_mat, eulertype="ZYX")
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)

    rot_mat = rotations.get_rot_mat("galactic", MCMF())
    eul = rotations.rotmat_to_euler(rot_mat, eulertype="ZYX")
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)

    rot_mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    eul = rotations.rotmat_to_euler(rot_mat, eulertype="ZYX")
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)


def test_mepa_rotation_matrix():
    """MEPA rotation matrix should be a proper rotation (det=+1)."""
    R = rotations.get_mepa_rotation_matrix()
    assert R.shape == (3, 3)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_topo_to_mepa_time_dependent():
    """topo→MEPA Euler angles must change with observation time."""
    loc = MoonLocation(lon=0, lat=40)
    t1 = Time("2022-01-01 00:00:00")
    t2 = Time("2022-01-15 00:00:00")
    lmax = 4

    topo1 = LunarTopo(location=loc, obstime=t1)
    eul1, _ = rotations.topo_to_mepa_euler_dl(lmax, topo1)

    topo2 = LunarTopo(location=loc, obstime=t2)
    eul2, _ = rotations.topo_to_mepa_euler_dl(lmax, topo2)

    # Euler angles should differ (Moon has rotated ~180° in 14 days)
    assert not np.allclose(eul1, eul2)


def test_topo_to_mepa_beta_constant():
    """
    The beta Euler angle (colatitude of beam center in MEPA) should be
    the same for different observation times at the same location. This
    verifies that the MEPA epoch is set to the frame's obstime by
    default, so the MCMF→J2000→MEPA chain cancels to topo→MCMF
    (time-independent).
    """
    loc = MoonLocation(lon=0, lat=40)
    t1 = Time("2022-01-01 00:00:00")
    t2 = Time("2022-01-15 00:00:00")
    lmax = 4

    topo1 = LunarTopo(location=loc, obstime=t1)
    eul1, _ = rotations.generate_euler_dl(lmax, topo1, "mepa")

    topo2 = LunarTopo(location=loc, obstime=t2)
    eul2, _ = rotations.generate_euler_dl(lmax, topo2, "mepa")

    # all three Euler angles should be the same because the MEPA epoch
    # defaults to obstime, collapsing topo->MCMF->J2000->MEPA to topo->MCMF
    # compare modulo 2pi to handle branch-cut wrapping (e.g. pi vs -pi)
    twopi = 2 * np.pi
    assert np.isclose(eul1[0] % twopi, eul2[0] % twopi)
    assert np.isclose(eul1[1] % twopi, eul2[1] % twopi)
    assert np.isclose(eul1[2] % twopi, eul2[2] % twopi)
