import healpy as hp
import numpy as np
from astropy.coordinates import AltAz, EarthLocation
from lunarsky import LunarTopo, MoonLocation, SkyCoord, Time

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

    # equatorial -> AltAz: det must be +1 (ENU swap fixes NEU
    # handedness).  Astropy's FK5->AltAz includes aberration so the
    # matrix is not perfectly orthogonal (~1e-4); we only check det
    # and the row-swap relationship to the raw SkyCoord output.
    time = Time("2022-06-16 17:00:00")
    loc = EarthLocation(lon=0, lat=40)
    to_frame = AltAz(obstime=time, location=loc)
    rot_mat = rotations.get_rot_mat("fk5", to_frame)
    assert np.isclose(np.linalg.det(rot_mat), 1.0, atol=1e-6)
    # verify relationship to raw SkyCoord (NEU): our matrix has
    # swapped rows compared to the raw NEU-based SkyCoord result
    x, y, z = np.eye(3)
    raw = (
        SkyCoord(
            x=x,
            y=y,
            z=z,
            frame="fk5",
            representation_type="cartesian",
        )
        .transform_to(to_frame)
        .cartesian.xyz.value
    )
    assert np.allclose(rot_mat, raw[[1, 0, 2], :])

    # LunarTopo -> MEPA
    loc = MoonLocation(lon=0, lat=40)
    topo_frame = LunarTopo(obstime=time, location=loc)
    rot_mat = rotations.get_rot_mat(topo_frame, "mepa")
    # must be a proper rotation (det = +1) and orthogonal
    assert np.isclose(np.linalg.det(rot_mat), 1.0)
    assert np.allclose(rot_mat @ rot_mat.T, np.eye(3), atol=1e-10)
    # round-trip: to_mepa then back should give identity
    rot_mat_inv = rotations.get_rot_mat("mepa", topo_frame)
    assert np.allclose(rot_mat_inv @ rot_mat, np.eye(3), atol=1e-10)

    # galactic -> MEPA
    rot_mat = rotations.get_rot_mat("galactic", "mepa")
    assert np.isclose(np.linalg.det(rot_mat), 1.0)
    assert np.allclose(rot_mat @ rot_mat.T, np.eye(3), atol=1e-10)
    # round-trip
    rot_mat_inv = rotations.get_rot_mat("mepa", "galactic")
    assert np.allclose(rot_mat_inv @ rot_mat, np.eye(3), atol=1e-10)


def test_rotmat_to_euler():
    # check that rotmat_to_euler is the inverse of euler_matrix_new
    rot_mat = rotations.get_rot_mat("galactic", "fk5")
    eul = rotations.rotmat_to_euler(rot_mat, eulertype="ZYX")
    rmat = hp.rotator.get_rotation_matrix(eul)[0]
    assert np.allclose(rot_mat, rmat)

    rot_mat = rotations.get_rot_mat("galactic", "mepa")
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
    """topo→MEPA Euler angles must change with observation time
    when the MEPA epoch is fixed."""
    loc = MoonLocation(lon=0, lat=40)
    t1 = Time("2022-01-01 00:00:00")
    t2 = Time("2022-01-15 00:00:00")
    lmax = 4

    # Use a fixed MEPA epoch (J2000) so the time-dependent parts
    # of the chain do NOT cancel
    et_fixed = 0.0

    topo1 = LunarTopo(location=loc, obstime=t1)
    eul1, _ = rotations.generate_euler_dl(lmax, topo1, "mepa", et=et_fixed)

    topo2 = LunarTopo(location=loc, obstime=t2)
    eul2, _ = rotations.generate_euler_dl(lmax, topo2, "mepa", et=et_fixed)

    # Euler angles should differ (Moon has rotated ~180° in 14 days)
    assert not np.allclose(eul1, eul2)


def test_topo_to_mepa_beta_constant():
    """
    The beta Euler angle (colatitude of beam center in MEPA) should be
    the same for different observation times at the same location. This
    verifies that the MEPA epoch is set to the frame's obstime by
    default, so the time-dependent parts of the chain cancel out
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
    # defaults to obstime, so the time-dependent parts cancel
    # compare modulo 2pi to handle branch-cut wrapping (e.g. pi vs -pi)
    twopi = 2 * np.pi
    assert np.isclose(eul1[0] % twopi, eul2[0] % twopi)
    assert np.isclose(eul1[1] % twopi, eul2[1] % twopi)
    assert np.isclose(eul1[2] % twopi, eul2[2] % twopi)
