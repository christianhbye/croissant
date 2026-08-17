# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CROISSANT (spheriCal haRmOnics vISibility SimulAtor iN pyThon) is a fast, differentiable visibility simulator for radio astronomy. It decomposes sky and antenna beam patterns into spherical harmonic coefficients, computes visibilities as dot products in harmonic space, and handles time evolution via phase rotation. Fully JAX-based since v5.0.0.

## Common Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_simulator.py
uv run pytest tests/test_simulator.py::test_function_name

# Lint and format
uv run ruff check
uv run ruff check --fix
uv run ruff format

# Install pre-commit hooks
uv run pre-commit install
```

Python 3.11–3.13 (`requires-python = ">=3.11, <3.14"`; CI tests all three). Tests enable 64-bit JAX precision globally via `conftest.py`.

Engine benchmarks live in `benchmarks/` (nothing imports them); run via `uv run python benchmarks/benchmark_engines.py --help`, results land in `benchmarks/results/`.

## Architecture

All core classes inherit from `eqx.Module` (Equinox/JAX) and are JIT-compilable.

- **`Simulator`** (`simulator.py`) — Main orchestrator. Holds beam, sky, observer location, times, frequencies. Computes visibilities via `sim()` which calls `convolve()` (an einsum over alm coefficients). Key helper: `rot_alm_z()` handles sky rotation over time as phase factors `exp(-i*m*φ(t))`.
- **`Beam`** (`beam.py`) — Antenna beam patterns. Inherits `SphBase`. Supports multiple sampling schemes (mwss, mw, dh, gl, healpix). Includes horizon masking and azimuthal rotation.
- **`Sky`** (`sky.py`) — Sky models in galactic, equatorial, or MEPA coordinates. Inherits `SphBase`. Transforms between coordinate systems via Euler rotations of alm.
- **`SphBase`** (`sphere.py`) — Base class for fields on the sphere. Manages data in various samplings, computes lmax from shape, runs spherical harmonic transforms via `compute_alm`.
- **SHT engines** — `sphere.compute_alm` takes `engine=`: `"s2fft"` (matrix-free `s2fft.forward`), `"dense"` (`dense.py`; materializes the transform matrix, cached), `"kernel"` (`kernel.py`; precomputed Wigner-d kernels, module-level LRU cache), or `"auto"` (the default; `engine_select.py` picks from predicted memory footprint and amortisation). Inside a jax trace, `auto` degrades to an engine that can run there (`engine_select.degrade_for_trace`, the single definition of that rule); an **explicit** engine request is never softened. `footprints.py` predicts kernel/dense sizes and shapes without building anything.
- **`polarization.py`** — Full-Stokes skies (`PolarizedSky`) and beams (`PairStokesBeam`) via spin-weighted harmonics; IAU↔COSMO Stokes conventions. Engines resolve **per spin block** (spin-0 I/V vs spin-∓2 Q/U), so `engine`/`engine_reason` are dicts keyed by block.
- **`multipair.py`** — Multi-antenna pair visibilities using `jax.vmap`.
- **`rotations.py`** — Euler angle computation and coordinate transforms (galactic↔equatorial↔MEPA).
- **`spice_utils.py`** — Explicit SPICE kernel management for lunar frames; croissant furnishes its own kernels rather than relying on lunarsky import side effects.
- **`utils.py`** — Spherical harmonic indexing (`getidx`, `getlm`), lmax calculations, coordinate helpers.
- `alm.py` and `croissant.jax` are deprecated re-export shims — do not add new code there.

### Core Data Flow

1. **Input** — Beam/Sky data on a grid `(N_freqs, theta, phi)` or HEALPix pixels
2. **Transform** — Compute alm via `sphere.compute_alm` (vmapped over frequencies), which dispatches to one of the SHT engines above (`engine="auto"` by default). It defaults to the general complex transform, as s2fft does; `Beam` and `Sky` pass `reality=True` because their own fields are real, which exploits Hermitian symmetry. `reality=True` is an assertion about the data and is rejected for complex or nonzero-spin input.
3. **Rotation** — Apply phase factors `exp(-i*m*φ(t))` for sky rotation with sidereal time
4. **Convolution** — Einsum `"flm,tm,flm->tf"` over beam and sky alm → visibility `(time, frequency)`
5. **Normalization** — Divide by beam integral (monopole mode) to recover sky temperature

### Spherical Harmonic Indexing

alm arrays have shape `(N_freqs, lmax+1, 2*lmax+1)` indexed as `(freq, ell, m)` where m ranges from -lmax to +lmax. Use `utils.getidx(lmax, ell, emm)` to convert (ell, m) to array index.

### Coordinate Conventions

- Theta: colatitude [0, π], Phi: longitude [0, 2π). Euler angles use ZYZ convention.
- Supports Earth (FK5/AltAz) and Moon (MEPA/LunarTopo) observations.

## Physics Tests

`tests/test_physics.py` contains physical invariant tests that validate fundamental properties of the simulator (linearity, sidereal periodicity, spectral scaling, beam symmetries, ground loss, multipair consistency, full-Stokes polarization invariants). These tests should **always pass** and should **not be modified** unless there is a deliberate physics-level breaking change to the simulator. If a code change causes a physics test to fail, the code change is wrong — fix the code, not the test.

## Code Style

- Line length: 79 characters (ruff enforced)
- Ruff lint rules: E, F, W, I (pycodestyle errors/warnings, pyflakes, isort)
- NumPy-style docstrings
- Use `eqx.field(static=True)` for non-traced fields in Module classes
- Use `jnp` (JAX NumPy) for array operations, not `numpy`
- Floating point comparisons in tests: `np.testing.assert_allclose`
- Test timeout: 120s per test
