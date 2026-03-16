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

## Architecture

All core classes inherit from `eqx.Module` (Equinox/JAX) and are JIT-compilable.

- **`Simulator`** (`simulator.py`) — Main orchestrator. Holds beam, sky, observer location, times, frequencies. Computes visibilities via `sim()` which calls `convolve()` (an einsum over alm coefficients). Key helper: `rot_alm_z()` handles sky rotation over time as phase factors `exp(-i*m*φ(t))`.
- **`Beam`** (`beam.py`) — Antenna beam patterns. Inherits `SphBase`. Supports multiple sampling schemes (mwss, mw, dh, gl, healpix). Includes horizon masking and azimuthal rotation.
- **`Sky`** (`sky.py`) — Sky models in galactic, equatorial, or MCMF coordinates. Inherits `SphBase`. Transforms between coordinate systems via Euler rotations of alm.
- **`SphBase`** (`sphere.py`) — Base class for fields on the sphere. Manages data in various samplings, computes lmax from shape, runs spherical harmonic transforms via `s2fft`.
- **`multipair.py`** — Multi-antenna pair visibilities using `jax.vmap`.
- **`rotations.py`** — Euler angle computation and coordinate transforms (galactic↔equatorial↔MCMF).
- **`utils.py`** — Spherical harmonic indexing (`getidx`, `getlm`), lmax calculations, coordinate helpers.

### Spherical Harmonic Indexing

alm arrays have shape `(N_freqs, lmax+1, 2*lmax+1)` indexed as `(freq, ell, m)` where m ranges from -lmax to +lmax. Use `utils.getidx(lmax, ell, emm)` to convert (ell, m) to array index.

### Coordinate Conventions

- Theta: colatitude [0, π], Phi: longitude [0, 2π). Euler angles use ZYZ convention.
- Supports Earth (FK5/AltAz) and Moon (MCMF/LunarTopo) observations.

## Code Style

- Line length: 79 characters (ruff enforced)
- NumPy-style docstrings
- Use `eqx.field(static=True)` for non-traced fields in Module classes
- Floating point comparisons in tests: `np.testing.assert_allclose`
- Test timeout: 120s per test
