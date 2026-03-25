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

- On Earth, precession moves the celestial pole at roughly 50"/year (cumulative,
  grows linearly with time). Nutation adds an oscillatory term up to ~17"
  (bounded, not cumulative). For observations spanning days to weeks, precession
  is the dominant error source.
- On the Moon the effect is smaller but the MCMF frame itself drifts relative
  to inertial space on timescales of a lunar month.
- The `phases` array (sky rotation) is computed using a fixed sidereal-day
  constant, but the actual rotation rate changes slightly between the synodic
  and sidereal periods and is not constant over a year.

**Impact:** For typical 21-cm observations (days to weeks), the beam orientation
error is well below the beam uncertainty itself. For observations spanning
months or longer, the fixed beam orientation introduces a systematic error that
grows with time — but such long baselines are uncommon in practice. This is a
low-priority improvement.

---

## 2. Ground model — known limitations

The ground contribution is currently:

```python
vis_gnd = beam.compute_fgnd() * Tgnd          # in Simulator.sim()
vis_sky /= beam.compute_norm()                 # normalise by full-sphere integral
vis = vis_sky + vis_gnd                        # add ground
```

This is a simple model with known limitations:

1. **Constant, isotropic, unpolarised ground temperature.** Real sites have
   terrain, and the ground temperature/emissivity depends on azimuth and
   frequency.
2. **No frequency dependence in `Tgnd`.**
3. **No reflection or scattering.**

A spatially-varying, frequency-dependent, scattering ground model would require
its own SHT pipeline and represents a significant architectural change. This is
outside the scope of incremental improvements and would be better addressed in a
dedicated design document if needed.

---

## 3. Beam tilt is not implemented

### Human summary

`Beam.__init__` accepts a `beam_tilt` argument (the angle between the beam
boresight and the local zenith) but raises `NotImplementedError` when it is
non-zero.  A tilted antenna is common in real deployments — e.g. phased arrays
on uneven terrain.  The implementation requires an additional Wigner rotation in
the beam-to-equatorial transform.

**Performance note:** The proposed approach (forward SHT → rotate alm → inverse
SHT → apply horizon → forward SHT) requires three SHTs per `compute_alm()`
call for a tilted beam, which triples compile time and runtime. For small tilts,
a pixel-space interpolation might be cheaper, though less exact at high lmax.

---

## 4. SHT recomputation in `sim()` — by design for differentiability

### Human summary

`Simulator.sim()` calls `compute_beam_eq()` and `sky.compute_alm_eq()` every
time it is invoked.  These are expensive SHTs.

**This is intentional.** `beam.data` and `sky.data` are traced arrays in
`eqx.Module`. The SHT inside `sim()` is part of the JAX computation graph, so
`jax.grad` can differentiate through beam/sky data. If alms were precomputed and
stored in `__init__`, they would become static pytree leaves and gradients would
no longer flow through the beam and sky parameters.

Additionally, `@jax.jit` on `sim()` means that for identical inputs, JAX
dispatches the cached compiled function — the recomputation exists in the XLA
graph but XLA can fuse and optimise it.

**For users who don't need gradients** through beam/sky data (e.g. fixed
instrument, varying sky model parameters only in alm space), the inner
`convolve()` function is public and can be called directly:

```python
beam_alm = sim.compute_beam_eq()
sky_alm = sim.sky.compute_alm_eq(world=sim.world, et=sim._et_ref)
beam_alm = utils.reduce_lmax(beam_alm, sim.lmax)
sky_alm = utils.reduce_lmax(sky_alm, sim.lmax)
vis = convolve(beam_alm, sky_alm, sim.phases)
```

This avoids redundant SHTs when iterating over parameters that don't affect the
beam or sky pixel data.

---

## 5. HEALPix SHT compile time

### Human summary

The HEALPix spherical harmonic transform via `s2fft` with `method="jax"` has
long JIT compile times, especially with iterative refinement (`niter > 0`).
Each iteration requires an additional forward/inverse SHT pass, all of which
must be compiled. This is the biggest practical pain point for day-to-day
usability.

The available options and their tradeoffs:

| Approach | Compile time | Accuracy | GPU | Gradients |
|---|---|---|---|---|
| `method="jax"`, `niter=3` | Very slow | Good | Yes | Yes |
| `method="jax"`, `niter=0` (current default) | Fast | Approximate | Yes | Yes |
| `method="jax_healpy"` | Fast | Good | **No** | Partial |

As of v5.1.2, the default is `niter=0` for all sampling schemes. Users who need
higher accuracy for HEALPix can set `niter=3` explicitly, accepting the compile
cost. The `method="jax_healpy"` path uses healpy's C backend via JAX callbacks
and works only on CPU; it also breaks some gradient paths.

A `method` parameter could be added to `SphBase.__init__` to let users opt into
`"jax_healpy"` for CPU workflows without auto-detecting the device internally.

---

## 6. Observer altitude is unused

`EarthLocation(lon, lat, height=alt)` and `MoonLocation(lon, lat, height=alt)`
are constructed with the altitude, but the Euler angles from
`get_rot_mat(topo_frame, sim_frame)` describe a pure rotation (no translation).
Altitude affects the orientation only through relativistic aberration and
parallax — both negligible at radio frequencies for ground-based observations.
This is physically correct but the `alt` parameter could use a docstring note
explaining this.

---

## LLM Implementation Prompt

```
You are working on the `croissant` radio-astronomy simulator (version 5.1.2).
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

### Change 1 – Implement beam tilt (small-angle rotation)

In `Beam.__init__` (beam.py), remove the `NotImplementedError` for `beam_tilt`
and implement it as a second Wigner rotation applied after `beam_rot`.

The tilt is a rotation about the local *y*-axis (East–West axis in antenna
frame) by angle `beam_tilt` (in degrees).  In alm space this corresponds to a
Wigner D rotation with Euler angles `(0, beam_tilt_rad, 0)` in the ZYZ
convention.

Note that the tilt affects the interaction between the beam and the horizon. In the current implementation of `compute_alm` and `compute_fgnd`, the beam is multiplied with the horizon mask in pixel space before the SHT as `self.data * self.horizon`. The tilt has to act on `self.data` before the horizon mask is applied. This can either be achieved directly in pixel space or by doing the SHT, rotating the alms, and then doing the inverse SHT. The latter is more consistent with the existing code structure and allows us to reuse the Wigner rotation functions from s2fft.

The exact s2fft function signatures (confirmed against the installed library) are:

    s2fft.generate_rotate_dls(L: int, beta: float) -> jax.Array
    s2fft.utils.rotation.rotate_flms(
        flm: jax.Array,
        L: int,
        rotation: tuple[float, float, float],
        dl_array: jax.Array = None,
    ) -> jax.Array

Steps (assuming the tilt is applied in alm space):
In `Beam.compute_alm()`, if `not isclose(self.beam_tilt, 0.0)`, compute the tilt rotation:
   1. Convert `beam_tilt` to radians: `tilt_rad = jnp.radians(self.beam_tilt)`.
   2. Generate the Wigner d-array for the tilt: `dl_tilt = s2fft.generate_rotate_dls(self._L, tilt_rad)`.
   3. Compute the alm using `sphere.compute_alm` on `self.data`. No horizon mask is applied yet.
   4. Apply the rotation with
   `s2fft.utils.rotation.rotate_flms(alm_freq, L=self._L,
       rotation=(0.0, tilt_rad, 0.0), dl_array=dl_tilt)`.
   5. Use `jax.vmap` to apply over the frequency axis (same pattern as
   `compute_beam_eq` in `simulator.py`).
   6. Compute tilted_data by doing an inverse SHT.
   7. Pass tilted data into the existing code replacing `self.data` (which is the untilted beam pattern) with `tilted_data` (the tilted beam pattern).

Note: the inverse SHT must also use `method="jax"` and consistent `niter`
settings to avoid asymmetry with the forward transform.

A similar fix is needed in `compute_fgnd()` to ensure the horizon mask is applied after the tilt. You are allowed to refactor the code to avoid duplication, e.g. by creating a helper method that applies the tilt to the beam pattern and is called from both `compute_alm` and `compute_fgnd`.

Restrict `beam_tilt` to `[-90, 90]` degrees and raise `ValueError` for values
outside this range.

Add tests in tests/test_beam.py:
- `beam_tilt=0` gives the same result as before.
- A non-zero `beam_tilt` changes the alms but preserves total power
  (`compute_norm()` should be unchanged).
- `beam_tilt=±90` does not raise an error (edge case of pointing at horizon).
- `|beam_tilt| > 90` raises `ValueError`.

### Change 2 – Add chunked time evaluation for long observations

In `Simulator.sim()`, add an optional `chunk_size` integer parameter (default
`None`).  When provided, split `self.phases` into chunks of `chunk_size` time
steps and evaluate `convolve()` iteratively, accumulating results.  This allows
simulating long time series without running out of GPU memory.

Prefer `jax.lax.map` or `jax.lax.scan` over a Python loop with
`jnp.concatenate` to avoid recompilation for different chunk counts.

The interface should be:

    vis = sim.sim(chunk_size=100)  # process 100 time steps at once

When `chunk_size` is None, the existing behaviour (all times at once) is
preserved.  Do not change the output format.

Add a test in tests/test_sim_class.py that verifies `sim(chunk_size=N)` gives
the same result as `sim()` for N=1, N=N_times//2, and N=N_times.

### Implementation notes

* All new code must pass `ruff check` and `ruff format --check`.
* Use `jax_enable_x64 = True` (already configured in `tests/conftest.py`).
* After all changes, run `python -m pytest tests/ -q` and confirm all
  pre-existing tests still pass.
```
