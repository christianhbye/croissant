# Bug: topo→MEPA rotation matrix has det=-1 (improper rotation)

## Summary

The composed topo→MEPA rotation matrix has determinant -1, meaning it's an improper rotation (includes a reflection). ZYZ Euler angles can only represent proper rotations (SO(3), det=+1), so `rotmat_to_eulerZYZ` returns angles for a *different* matrix than the actual transform. This means `rotate_flms` applies an incorrect rotation to beam alm in the topo→MEPA path.

## Root cause

In `_rot_mat_to_mepa` (rotations.py:266):

```python
R_from_mcmf = get_rot_mat(from_frame, "mcmf")  # det = -1!
```

This goes through the SkyCoord path, which produces a matrix with det=-1 for LunarTopo→MCMF. The SPICE-based matrices (MCMF→J2000 and J2000→MEPA) both have det=+1, so the final product inherits det=-1.

## Diagnosis needed

- Determine whether the det=-1 is a lunarsky convention mismatch (handedness difference between LunarTopo and MCMF) or a genuine bug in `get_rot_mat`.
- Check if other frame pairs (e.g., FK5→AltAz on Earth) also produce det=-1 through the SkyCoord path.
- Assess the numerical impact on simulated visibilities.

## Possible fixes

- If it's a handedness issue: adjust the SkyCoord-based matrix (e.g., negate one axis) before composing with the SPICE matrices.
- Alternatively, compute the topo→MCMF step via SPICE instead of SkyCoord, bypassing the issue entirely.
- Add a det=+1 assertion or correction in `_rot_mat_to_mepa` as a safeguard.
