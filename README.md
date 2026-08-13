# CROISSANT: spheriCal haRmOnics vISibility SimulAtor iN pyThon

[![codecov](https://codecov.io/gh/christianhbye/croissant/branch/main/graph/badge.svg?token=pj1hkgcazd)](https://codecov.io/gh/christianhbye/croissant)

CROISSANT is a rapid visiblity simulator in python based on spherical harmonics. Given an antenna design and a sky model, CROISSANT simulates the visbilities - that is, the perceived sky temperature.

CROISSANT uses spherical harmonics to decompose the sky and antenna beam to a set of coefficients. Since the spherical harmonics represents a complete, orthormal basis on the sphere, the visibility computation reduces nicely from a convolution to a dot product.

Moreover, the time evolution of the simulation is very natural in this representation. In the antenna reference frame, the sky rotates overhead with time. To account for this rotation, it is enough to rotate the spherical harmonics coefficients. In the right choice of coordinates (that is, one where the z-axis is aligned with the rotation axis of the earth or the moon), this rotation is simply achieved by multiplying the spherical coefficient by a phase.


> **New:** CROISSANT supports differentiable full-Stokes
> skies and arbitrary complex pair-response beams while retaining the
> diagonal-in-m time kernel. The exact IAU/COSMO, spin, and Stokes-V
> conventions are documented in
> [`docs/polarization.md`](docs/polarization.md).
>
> Version 5.0.0 moved CROISSANT fully to JAX and dropped the legacy
> NumPy/healpy implementation. Spherical harmonic transforms (built on
> [s2fft](https://github.com/astro-informatics/s2fft/)), coordinate
> transformations, rotations, and the simulator can all be differentiated
> using JAX autograd.

Overall, this makes CROISSANT a very fast visibility simulator. CROISSANT can therefore be used to simulate a large combination of antenna models and sky models - allowing for the exploration of a range of proposed designs before choosing an antenna for an experiment.

### Dense low-band-limit transforms

For repeated transforms at low spherical-harmonic band-limits, `Beam` and
`Sky` accept `engine="dense"`:

```python
sky = croissant.Sky(
    maps,
    frequencies,
    sampling="healpix",
    engine="dense",
    lmax=30,
)
alm = sky.compute_alm()
```

On first construction, Croissant evaluates the spherical-harmonic basis in
bounded chunks and caches the resulting analysis matrix on the current JAX
device. Later transforms are native JAX matrix multiplications, so they
support JIT compilation, batching, and automatic differentiation without
external callbacks. The cached matrix includes the selected `niter`
refinement count and stores only the independent `m >= 0` coefficients.
HEALPix inputs may set `lmax` below the usual `2 * nside` default, which is
particularly useful for high-resolution maps with low-band-limit science.

### Transform engines

A spherical harmonic transform has two stages: an FFT around each ring of
constant latitude (phi to m), and a Wigner-d recursion with quadrature weights
(theta to ell). The second is the expensive one. Croissant's three engines
compute the same linear map — they agree to ~1e-15, verified in
`tests/test_engine_equivalence.py` — and differ only in how much of it they
precompute.

| engine | precomputes | per call | precompute size |
|:--|:--|:--|:--|
| `"s2fft"` | nothing | FFT + full recursion | — |
| `"kernel"` | the theta-to-ell table | FFT + one contraction | `O(nside**3)` |
| `"dense"` | the whole pixels-to-alm operator | one matrix multiply | `O(nside**4)` |

`"dense"` sits at the far end: its matrix absorbs the FFT, the recursion, the
quadrature weights *and* any `niter` refinement, which is why it is the most
expensive to build and reduces each transform to a single `einsum`.

Measured on CPU with x64 enabled (reproduce with
`uv run python benchmarks/benchmark_engines.py`; recorded output in
`benchmarks/results/`):

| | nside=8 | nside=16 | nside=32 |
|:--|---:|---:|---:|
| memory, dense/kernel (scalar) | 13.3x | 25.3x | 49.1x |
| memory, dense/kernel (spin) | 12.8x | 24.6x | not tested |
| setup, dense/kernel (scalar) | 1.15x | 0.82x | 5.9x |
| setup, dense/kernel (spin) | 22x–25x | 22x–25x | not tested |

Memory ratios grow with `nside` as the `O(nside**4)` vs `O(nside**3)`
footprints predict. Setup ratios behave differently and are reported
separately for scalar and spin fields because they differ sharply: for
scalar fields the dense build is roughly break-even with the kernel build
at nside=8–16 and only pulls ahead at nside=32, while for spin fields it is
already 22x–25x more expensive at every tested resolution — the dense
engine's NumPy spin Wigner-d builder is disproportionately expensive
relative to the kernel builder.

The footprints differ by a full power of `nside` — `~32*nside**3` for the kernel
against `~48*nside**4` for the dense operator, a ratio of `~1.5*nside` — so at
nside=32 the dense operator needs ~768 MiB where the kernel needs ~16 MiB, and
by nside=64 dense needs ~12 GiB against the kernel's ~130 MiB.

### Choosing an engine

Because the engines agree numerically, the choice is about memory and reuse, not
results — so `engine="auto"` makes it for you:

1. **A band-limit below the HEALPix floor** (`lmax < 2*nside - 1`) selects
   `"dense"`. s2fft's HEALPix FFT requires `L >= 2*nside` whatever you want
   back, and only the dense engine can build at that floor and keep just the
   requested low-`ell` rows, low-passing in a single step. The kernel engine
   contracts the whole `ell` range at once and so raises for these
   configurations.
2. **A batch too small to amortise a precompute** selects `"s2fft"`. There is
   nothing to pay the build cost back.
3. **Otherwise** `"kernel"`, provided it fits the 512 MiB precompute budget,
   falling back to `"s2fft"` if nothing fits.

Note that `niter > 0` is deliberately *not* a reason to choose `"dense"`. Its
folded refinement makes per-call cost independent of `niter`, while the kernel
engine pays `2*niter+1` applications — but per-call cost is essentially one pass
over the precomputed object, and dense's is `~1.5*nside` times larger, so the
kernel still wins for any `nside > 4`.

The resolved choice and the reason for it are reported on the object:

```python
beam = Beam(data, freqs, sampling="healpix", engine="auto")
print(beam.engine)  # e.g. "kernel"
print(beam.engine_reason)  # e.g. "16 MiB kernel amortises over 64 transforms"
```

`"auto"` is a policy, not a promise: it may change between versions. Pin
`engine=` explicitly to freeze behaviour, and prefer an explicit engine when you
want the dense operator itself (for a Fisher or gram matrix, or an explicit
Jacobian), when you have a memory budget croissant cannot see, or when you know
about reuse it cannot see — for example the same `Beam` driving many thousands of
likelihood evaluations, where the batch size understates the amortisation.

`engine="s2fft"` remains the default. Applications that call
`croissant.sphere.compute_alm` from inside an enclosing `jax.jit` should
build the matrix once with `croissant.precompute_dense_matrix` and pass it
to the jitted function as an argument via `dense_matrix=...`, so it enters
the trace as a runtime input. (A pre-warmed cache alone also works — the
matrix is then captured as a compile-time constant, which can increase
compilation time and keeps the matrix alive as long as the compiled
function.) `Beam` and `Sky` handle this automatically: they precompute the
matrix during initialization and thread it through their jitted methods as
a dynamic argument. Use `croissant.clear_dense_matrix_cache()` to release
Croissant's in-process matrix references.

## Installation
To install the package for standard use, you can use your preferred Python package manager:

**Using `uv` (Recommended)**
```bash
uv pip install croissant-sim

```

**Using `pip`**

```bash
pip install croissant-sim

```

CROISSANT supports Python 3.11-3.13. The corrected pinned s2fft
revision required for reliable spin-2 HEALPix transforms requires Python
3.11 or newer; croissant-sim 5.2.1 remains the last Python 3.10-compatible
release.
Python 3.14 and newer versions are experimental.

## Development

We recommend using [`uv`](https://github.com/astral-sh/uv) to manage the development environment. It is exceptionally fast and handles virtual environments, dependencies, and lockfiles automatically. However, standard `pip` workflows are also fully supported.

### 1. Set Up the Environment

**Option A: Using `uv` (Recommended)**

`uv` will automatically read the `pyproject.toml`, create a virtual environment (`.venv`), and install all core and development dependencies.

```bash
# Clone the repository
git clone git@github.com:christianhbye/croissant.git
cd croissant

# Sync the project and install all dependencies
uv sync

```

**Option B: Using `pip`**

If you prefer standard Python tools, you will need to manually create the environment and install the package in editable mode.

```bash
# Clone the repository
git clone git@github.com:christianhbye/croissant.git
cd croissant

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with development dependencies
pip install -e .
pip install -r requirements-dev.txt

```

### 2. Install Pre-commit Hooks

We use `pre-commit` to automatically format and lint code before every commit. This ensures all code follows our style guidelines (enforced by `ruff`).

**With `uv`:**

```bash
uv run pre-commit install

```

**With `pip`:**

```bash
pre-commit install

```

### 3. Running Tests and Linters

We use `pytest` for testing and `ruff` for linting and formatting. Prepend these commands with `uv run` if using `uv`.

```bash
pytest                         # Run the test suite
ruff format                    # Auto-format code
ruff check --fix               # Run linter and fix auto-fixable errors

```

## Demo
Jupyter Notebook: https://nbviewer.org/github/christianhbye/croissant/blob/main/notebooks/example_sim.ipynb

## Contributing
Contributions are welcome - please see the [contribution guidelines](https://github.com/christianhbye/croissant/blob/add_contributing/CONTRIBUTING.md).
