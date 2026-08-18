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
compute the same linear map — they agree to better than 1e-12 in every
benchmarked configuration (worst case 1.4e-13) and are pinned to 1e-10 by
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
| setup, dense/kernel (spin) | 14.7x–24.9x | 22.5x–24.8x | not tested |

Memory ratios grow with `nside` as the `O(nside**4)` vs `O(nside**3)`
footprints predict. Setup ratios behave differently and are reported
separately for scalar and spin fields because they differ sharply: for
scalar fields the dense build is roughly break-even with the kernel build
at nside=8–16 and only pulls ahead at nside=32, while for spin fields it is
already 14.7x–24.9x more expensive at every tested resolution (as low as
14.7x at nside=8/niter=3; see
[`benchmarks/results/engines-2026-08-13.md`](benchmarks/results/engines-2026-08-13.md)
for the per-configuration figures) — the dense engine's NumPy spin
Wigner-d builder is disproportionately expensive relative to the kernel
builder.

The footprints differ by a full power of `nside` — `~32*nside**3` for the kernel
against `~48*nside**4` for the dense operator, a ratio of `~1.5*nside`. These
are computed footprints (from `croissant.footprints`, whose formula is
verified in `tests/test_engine_select.py` against actually-built kernels),
not timings; the nside=32 row matches the memory table above, and nside=64
and nside=128 are not benchmarked, only computed the same way:

| nside | kernel (scalar / spin) | dense (scalar / spin) |
|---:|---:|---:|
| 32 | 7.9 MiB / 15.8 MiB | 390.0 MiB / 767.2 MiB |
| 64 | 63.8 MiB / 127.0 MiB | 6.0 GiB / 12.0 GiB |
| 128 | 511.0 MiB / 1020.0 MiB | 96.4 GiB / 192.0 GiB |

Dense already exceeds the default 512 MiB `engine="auto"` budget by
nside=32 for spin fields (767.2 MiB) and by nside=64 for scalar fields
(6.0 GiB), while the kernel is still under it at nside=128 for scalar
fields (511.0 MiB) and only exceeds it there for spin fields (1020 MiB).
That gap is why the kernel engine exists at all.

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

The batch in rule 2 is compared against a measured crossover — the batch at
which building a kernel starts costing less than recomputing the recursion
per call. Two things set it. It grows with the kernel, because the build does;
and it *falls* with `niter`, because refinement makes s2fft repeat its whole
Wigner-d recursion `2*niter+1` times while the kernel engine re-contracts a
table it already has. Measured on CPU with x64 (ladders in
[`benchmarks/results/engines-2026-08-14-amortisation.md`](benchmarks/results/engines-2026-08-14-amortisation.md),
reproduce with `--sections ladder`):

| | nside=16 | nside=32 | nside=64 |
|:--|---:|---:|---:|
| crossover batch, `niter=0` | ≤1 | 8–12 | 64–96 |
| crossover batch, `niter=3` | ≤1 | ≤1 | 12–16 |

so at `nside=64` with refinement the kernel pays for itself from about 14
frequencies, while without refinement it needs closer to 80. The threshold is
sized from the *scalar* kernel footprint even for spin fields, because a
spin-weighted kernel is twice the bytes but takes the same time to build
(10.7 s against 10.2 s at `nside=64`) — it is build cost that has to be repaid,
not bytes. Bytes still govern the 512 MiB budget in rule 3, and there both
kernels count when `niter > 0`.

Note that `niter > 0` is deliberately *not* a reason for `auto` to choose
`"dense"`. Dense does win on per-call cost there — its refinement folds into
the cached matrix, while the kernel engine pays `2*niter+1` applications — but
its build costs roughly 1.0x to 25x more, and the measured break-even ranges
from 7 to 92338 calls depending on configuration. Croissant transforms once at
construction, batched over frequencies, so that per-call saving is never
repaid. If you re-apply the same transform thousands of times, pin
`engine="dense"` explicitly — only you know that.

The resolved choice and the reason for it are reported on the object:

```python
beam = Beam(data, freqs, sampling="healpix", engine="auto")
print(beam.engine)  # e.g. "kernel"
print(
    beam.engine_reason
)  # e.g. "16.0 MiB kernel amortises over 64 transforms"
```

`"auto"` is a policy, not a promise: it may change between versions. Pin
`engine=` explicitly to freeze behaviour, and prefer an explicit engine when you
want the dense operator itself (for a Fisher or gram matrix, or an explicit
Jacobian), when you have a memory budget croissant cannot see, or when you know
about reuse it cannot see — for example the same `Beam` driving many thousands of
likelihood evaluations, where the batch size understates the amortisation.

Applications that call
`croissant.sphere.compute_alm` from inside an enclosing `jax.jit` should
build the matrix once with `croissant.precompute_dense_matrix` and pass it
to the jitted function as an argument via `dense_matrix=...`, so it enters
the trace as a runtime input. (A pre-warmed cache alone also works — the
matrix is then captured as a compile-time constant, which can increase
compilation time and keeps the matrix alive as long as the compiled
function.) `Beam` and `Sky` handle this automatically: they precompute the
matrix during initialization and thread it through their jitted methods as
a dynamic argument. Dense operators are cached for the life of the process
and are never evicted, so that a matrix warmed with
`croissant.precompute_dense_matrix` is always still there when a later
`jax.jit`-ed call needs it. The cost is that retention grows with the
number of distinct configurations you touch: each `lmax`, `niter`, `nside`,
sampling, `spin`, packed-vs-full flavour, dtype and backend combination is
a separate entry. Measured on CPU with x64 enabled, a band-limit sweep at
`nside=32` over seven values of `lmax` retains about 904 MiB (about 452
MiB under JAX's own x64-disabled default, since the operator dtype
follows `jax.config.x64_enabled` between complex128 and complex64). Call
`croissant.dense_cache_nbytes()` to see the current total and
`croissant.clear_dense_matrix_cache()` to release all of it.

`engine="kernel"` needs the same care, with its own functions:
`croissant.precompute_kernel(..., forward=True)` for the analysis kernel,
plus `forward=False` for the synthesis kernel if `niter > 0`, passed to the
jitted call as `kernel=...` and `inverse_kernel=...`. Its `reality`
defaults to `True` to match the apply path, and is forced to `False` for
spin-weighted fields, so the same call is correct for scalar and spin
blocks alike. Called from inside
`jax.jit` with no kernel available at all, `compute_alm(..., engine="kernel")`
raises `RuntimeError` rather than silently building. A warm cache counts as
available: a trace forbids *building* a kernel, not using one this process
already holds, so a single `precompute_kernel(...)` outside `jax.jit` serves
later traced calls whether or not you also thread the result through as
`kernel=`. Passing it explicitly is still the clearer form, and the only one
that survives a `clear_kernel_cache()`.
`Beam` and `Sky` handle this automatically too, precomputing the forward
kernel (and the inverse kernel, if `niter > 0`) during initialization.

Unlike the dense cache, the kernel cache is bounded — but by a kernel
*count*, currently 32, not by bytes. The floor on that number is one
polarized simulation's working set: a `PairStokesBeam` and a `PolarizedSky`
at `niter > 0` need a forward and an inverse kernel per transformed block,
and a smaller cache makes the two evict each other on every construction,
silently losing the reuse the cache exists to provide. The ceiling is the
other side of the same coin: 32 kernels at `nside=128` is about 16 GiB,
against the 512 MiB budget rule 3 applies to a single choice. The two govern
different things, but they add up. Call `croissant.kernel_cache_nbytes()` to
see what is actually resident and `croissant.clear_kernel_cache()` to release
it — that release valve is offered deliberately in place of a byte-based
eviction policy, which would drop whichever kernel is largest rather than
whichever is least likely to be reused.

`engine="auto"` never turns a working call into a `RuntimeError` this way.
Call `croissant.sphere.compute_alm` from inside your own `jax.jit` with
nothing precomputed and nothing cached, and an automatic choice of
`"kernel"` degrades to an engine that can actually run: `"s2fft"` normally, or `"dense"` for a
band-limit below the HEALPix floor, where the matrix-free engine cannot
serve the transform at all. Only an *explicit* `engine="kernel"` raises,
because a named engine is a cost decision croissant will not silently swap
out. `croissant.kernel.kernel_compute_alm`, which has no `"auto"` to fall
back on, raises exactly as before.

Constructing a field *inside* a trace — differentiating through the
construction itself, as `jax.grad(lambda m: Sky(m, freqs, ...).compute_alm())`
does — follows the same rule. No kernel can be built there, so an automatic
choice of `"kernel"` degrades and the `engine_reason` says so, while an
explicit `engine="kernel"` still raises. An automatic choice of `"dense"`
does build rather than refuse: its operator depends only on static
geometry, so it is materialised as a compile-time constant. This holds for
`Beam`, `Sky`, `PolarizedSky` and `PairStokesBeam` alike.

### Engines for polarized fields

`PolarizedSky` and `PairStokesBeam` take the same `engine=` argument, but a
polarized field is not one transform — it is three, and they do not share a
footprint. So an engine is resolved per *block*: the spin-0 block carrying
I and V, and the two spin-∓2 blocks carrying the Q/U duals. A real sky's
spin-0 kernel packs to `m >= 0` and is roughly half the size of a
spin-weighted one, and the blocks are batched over different numbers of
maps, so one verdict for the whole object would be wrong for some of it.
`engine` and `engine_reason` are therefore mappings rather than strings:

```python
sky = PolarizedSky(data, freqs, sampling="healpix")
print(sky.engine)  # {"IV": "kernel", "P_MINUS": "kernel", "P_PLUS": None}
```

`P_PLUS` reports `None` on HEALPix, DH and GL, where a real sky's P+ dual
follows from P- by conjugation and is never transformed at all. An explicit
`engine=` pins every block.

One behaviour differs from `Beam` and `Sky`: `compute_alm(lmax=...)` below
the HEALPix floor stays on the dense engine whatever the block resolved to,
because no kernel exists at that band-limit.

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
