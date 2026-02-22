# Suggested Improvements to CROISSANT

This document summarises areas where the simulator could be made more physically
accurate or easier to use correctly.  Each section is written for a human reader
first, followed by a concrete LLM implementation prompt.

---

## 1. Sky rotation model assumes a fixed rotation axis

### Human summary

The sky is rotated in spherical-harmonic space by applying a phase
`exp(-i·m·φ(t))` to every *m*-mode (see `rot_alm_z`).  This is exact when the
rotation axis is the same as the *z*-axis of the simulation frame (i.e. the
Earth's/Moon's spin axis is aligned with the equatorial frame pole).

On the Moon this is fine: MCMF already has its *z*-axis along the lunar spin
pole.  On Earth the simulator uses FK5 (mean equatorial of J2000), which also
has *z* along the celestial pole, so the approximation holds for Earth too.

The subtle problem is that the Euler angles used to rotate the *beam* from
topocentric to equatorial frame (`eul_topo`) are computed at **a single
reference epoch** (`times_jd[0]`) and then held fixed for the entire
observation.  Over a short observation this is fine, but:

- On Earth, precession and nutation move the celestial pole at roughly
  50"/year (precession) and up to 17" (nutation), so over months the fixed
  rotation is noticeably wrong.
- On the Moon the effect is smaller but the MCMF frame itself drifts relative
  to inertial space on timescales of a lunar month.
- The `phases` array (sky rotation) is computed using a fixed sidereal-day
  constant, but the actual rotation rate changes slightly between the synodic
  and sidereal periods and is not constant over a year.

**Impact:** For observations spanning more than a few days the fixed beam
orientation introduces a systematic error that grows with time.

---

## 2. Ground model is oversimplified

### Human summary

The ground contribution is currently:

```python
vis_gnd = beam.compute_fgnd() * Tgnd          # in Simulator.sim()
vis_sky /= beam.compute_norm()                 # normalise by full-sphere integral
vis = vis_sky + vis_gnd                        # add ground
```

This is a simple model with three main limitations:

**1. Constant, isotropic, unpolarised ground temperature.**  Real sites have
terrain, and the ground temperature/emissivity depends on both azimuth and
frequency.  The interface only accepts a scalar `Tgnd` with no spatial
variation.

**2. No frequency dependence in `Tgnd`.**  At MHz frequencies relevant to
cosmological 21-cm studies, the ground contribution can have a nontrivial
spectral dependence (e.g. via soil emissivity ~1 - reflectance).

**3. No reflection or scattering.**  The model assumes the ground is a perfect
absorber, but in reality the ground can reflect and scatter radiation, which
can produce additional contributions to the visibility. The reflected light should show up at a delay set by the path length difference between direct and reflected rays, which can be important for foreground contamination in 21-cm experiments.

---

## 4. Beam tilt is not implemented

### Human summary

`Beam.__init__` accepts a `beam_tilt` argument (the angle between the beam
boresight and the local zenith) but raises `NotImplementedError` when it is
non-zero.  A tilted antenna is common in real deployments — e.g. phased arrays
on uneven terrain.  The implementation would require an additional Wigner
rotation in the beam-to-equatorial transform.

---

## 5. `sim()` recomputes beam/sky alms on every call

### Human summary

`Simulator.sim()` calls `compute_beam_eq()` and `sky.compute_alm_eq()` every
time it is invoked.  These are expensive SHTs.  If a user iterates over a grid
of sky or beam parameters (as is common in Bayesian inference or design
optimisation) the alms are recomputed redundantly.

The pattern used in the code is `@jax.jit`, which memoises the *compiled*
function but still executes it on new values.  What would help for repeated
calls with *the same* beam/sky is an explicit `precompute()` method that stores
the equatorial alms and a lazy-evaluation flag.

---

## 7. `correct_ground_loss` is a public API that inverts the wrong normalisation

### Human summary

`simulator.correct_ground_loss(vis, fgnd, Tgnd)` is exposed as a public
function but assumes the visibility was produced with `fgnd` exactly matching
the beam's own `compute_fgnd()`.  If a user passes a different `fgnd`
(e.g. derived from an external model) the correction will be inconsistent with
the sky convolution that used `compute_norm()`.  The docstring should spell out
the exact formula and when it is valid.

---

## 8. The topocentric-to-equatorial rotation does not account for the observer's altitude

### Human summary

`EarthLocation(lon, lat, height=alt)` is constructed with the altitude, and
`MoonLocation(lon, lat, height=alt)` likewise.  However, the Euler angles
computed from `get_rot_mat(topo_frame, sim_frame)` describe a pure rotation
(no translation), so altitude affects the orientation only through relativistic
aberration and parallax — both negligible at radio frequencies for ground-based
observations.  This is physically fine.  However, `alt` is stored as an
`eqx.field` with no documentation that it is effectively unused, which could
confuse contributors.

---

## LLM Implementation Prompt

```
You are working on the `croissant` radio-astronomy simulator (version 5.0.0).
The codebase is in `src/croissant/`. The main modules are:

  simulator.py  – Simulator class and helper functions (rot_alm_z, convolve,
                  correct_ground_loss)
  beam.py       – Beam class (inherits SphBase)
  sky.py        – Sky class (inherits SphBase)
  sphere.py     – SphBase base class and compute_alm function
  rotations.py  – Coordinate-rotation helpers
  constants.py  – Physical constants (sidereal days, Y00)
  utils.py      – SHT helpers (getidx, lmax_from_*, generate_theta/phi, etc.)

The simulator computes the antenna temperature seen by a radio telescope as a
function of time and frequency, using a spherical-harmonic convolution:

  T_ant(t, ν) = Σ_{l,m} sky_alm*(ν) · beam_eq_alm(ν) · exp(-i m φ(t))
                / ∫B dΩ  +  f_gnd · T_gnd

where φ(t) = 2π t / T_sid is the sidereal-rotation angle.

Please implement the following improvements.  Make minimal, surgical changes to
the existing code; do not refactor unrelated parts.  After each change, verify
that the existing tests in tests/ still pass.

### Change 4 – Clarify `correct_ground_loss` docstring

In `simulator.py`, update the `correct_ground_loss` docstring to state:

* The function assumes `vis` was produced by `Simulator.sim()` with the same
  `fgnd` and `Tgnd`.
* It is the inverse of adding `fgnd * Tgnd` and dividing by `fsky = 1 - fgnd`.
* Warn that passing `fgnd` inconsistent with the beam model will produce
  incorrect results.

### Change 5 – Implement beam tilt (small-angle rotation)

In `Beam.__init__` (beam.py), remove the `NotImplementedError` for `beam_tilt`
and implement it as a second Wigner rotation applied after `beam_az_rot`.

The tilt is a rotation about the local *y*-axis (East–West axis in antenna
frame) by angle `beam_tilt` (in degrees).  In alm space this corresponds to a
Wigner D rotation with Euler angles `(0, beam_tilt_rad, 0)` in the ZYZ
convention.

The exact s2fft function signatures (confirmed against the installed library) are:

    s2fft.generate_rotate_dls(L: int, beta: float) -> jax.Array
    s2fft.utils.rotation.rotate_flms(
        flm: jax.Array,
        L: int,
        rotation: tuple[float, float, float],
        dl_array: jax.Array = None,
    ) -> jax.Array

Steps:
1. After the azimuthal phase is applied to `alm` in `compute_alm()`, compute
   the Wigner d-array for the tilt angle using
   `dl_tilt = s2fft.generate_rotate_dls(self._L, tilt_rad)`.
2. Apply the rotation with
   `s2fft.utils.rotation.rotate_flms(alm_freq, L=self._L,
       rotation=(0.0, tilt_rad, 0.0), dl_array=dl_tilt)`.
3. Use `jax.vmap` to apply over the frequency axis (same pattern as
   `compute_beam_eq` in `simulator.py`).

Restrict `beam_tilt` to `[-90, 90]` degrees and raise `ValueError` for values
outside this range.

Add tests in tests/test_beam.py:
- `beam_tilt=0` gives the same result as before.
- A non-zero `beam_tilt` changes the alms but preserves total power
  (`compute_norm()` should be unchanged).
- `beam_tilt=±90` does not raise an error (edge case of pointing at horizon).
- `|beam_tilt| > 90` raises `ValueError`.

### Change 6 – Add `n_jobs` / chunked time evaluation for long observations

In `Simulator.sim()`, add an optional `chunk_size` integer parameter (default
`None`).  When provided, split `self.phases` into chunks of `chunk_size` time
steps and evaluate `convolve()` iteratively, accumulating results with
`jnp.concatenate`.  This allows simulating long time series without running
out of GPU memory (the current `einsum` over N_times × N_freqs × lmax² can
require many GB for large lmax and many times).

The interface should be:

    vis = sim.sim(chunk_size=100)  # process 100 time steps at once

When `chunk_size` is None, the existing behaviour (all times at once) is
preserved.  Do not change the output format.

Add a test in tests/test_sim_class.py that verifies `sim(chunk_size=N)` gives
the same result as `sim()` for N=1, N=N_times//2, and N=N_times.

### Implementation notes

* All new code must pass `ruff check` and `ruff format --check`.
* Use `jax_enable_x64 = True` (already configured in `tests/conftest.py`).
* Do not change `core_tests/`; those are legacy and will be removed.
* After all changes, run `python -m pytest tests/ -q` and confirm all
  pre-existing tests still pass (2 expected timeouts in
  `test_rot_alm_z[24-*-32]` are acceptable).
```
