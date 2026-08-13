# Kernel SHT Engine and Automatic Engine Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third spherical-harmonic engine that contracts a precomputed
Wigner-d kernel — 73x faster to set up and 49x smaller than the dense operator
at nside=32/L=64 — and make croissant choose the engine automatically instead of
documenting the choice in prose.

**Architecture:** Croissant's three engines differ only in how much of the
pixels-to-alm map is precomputed. `s2fft` precomputes nothing and recomputes the
Wigner-d recursion per call. `dense` precomputes the entire operator as an
`ncoeff x npix` matrix. The new `kernel` engine sits between them: it caches
s2fft's `(ntheta, L, 2L-1)` Wigner-d table, leaving the per-ring FFT to run per
call. Since all three agree to ~1e-15, engine choice is a pure resource
decision, so `engine="auto"` resolves it from statically-known quantities
(sampling, lmax, nside, spin, niter, batch size) against a memory cap.

**Tech Stack:** JAX, Equinox, s2fft (pinned fork
`slosar/s2fft@cefdf46`), numpy, scipy, pytest.

**Spec:** No separate spec document — the design rationale and every
measurement quoted below were established in the investigation of 2026-08-13 and
are reproduced in the Background section, which serves as the spec.

## Global Constraints

- Line length 79 characters, enforced by ruff (`uv run ruff check`,
  `uv run ruff format`).
- Ruff lint rules: `E`, `F`, `W`, `I`.
- NumPy-style docstrings on all public functions.
- Use `jnp` for array operations in library code, not `numpy`. NumPy is
  permitted at precompute time (outside traces), following the existing
  precedent in `sphere.py:_build_dense_matrix_healpix`.
- Use `eqx.field(static=True)` for non-traced fields on `eqx.Module` classes.
- Floating point comparisons in tests use `np.testing.assert_allclose`.
- Test timeout is 120 s per test (`--timeout=120`). Every config in this plan is
  chosen to run in single-digit seconds.
- Python 3.11–3.13.
- **`tests/test_physics.py` must not be modified.** If a change here breaks a
  physics test, the change is wrong.
- **Never call `s2fft.precompute_transforms.construct.spin_spherical_kernel_jax`.**
  It is silently wrong for spin != 0 on HEALPix (see Background). Always use the
  numpy builder `spin_spherical_kernel`.
- **Never pass `iter > 0` to `s2fft.precompute_transforms.spherical.forward`.**
  Its refinement branch builds an inverse kernel with the broken jax builder.

## Background (the spec)

### Why a kernel engine

Measured on CPU with x64 enabled, HEALPix, cache cleared:

| config | dense operator build | kernel build | dense memory | kernel memory |
|---|---|---|---|---|
| nside=16, L=32 | 11–13 s | 0.26–0.29 s | 48 MiB | 1.94 MiB |
| nside=32, L=64 | 173–197 s | 2.4 s | 768 MiB | 15.75 MiB |

The footprints scale differently: kernel is `ntheta * L * (2L-1)` ~ `32*nside^3`,
dense is `ncoeff * npix` ~ `48*nside^4`. One full power of nside apart, ratio
~`1.5*nside`, which matches the measured 48.8x at nside=32. At nside=64 the
dense operator needs ~12 GiB and the kernel ~130 MiB, so dense hits a wall the
kernel does not.

Accuracy is not a differentiator: the kernel path agrees with the on-the-fly
transform to 1.2e-15, and the dense path to 3.1e-15.

### The s2fft bug this design routes around

`recursions/price_mcewen.py` renormalises with `bigi = 1/abs(dl_entry)` and
`lbig = log(abs(dl_entry))`, which go to inf/-inf at an exact Wigner-d zero
node. HEALPix rings sit at rational cos(theta), and for spin != 0 some land
exactly on nodes. `precompute_transforms/construct.py:238` calls that recursion
for `spin_spherical_kernel_jax` and then masks the NaNs at line 241 with
`dl = dl.at[jnp.where(dl != dl)].set(0)`, turning a numerical failure into a
silently wrong answer.

Verified at nside=16/L=32/spin=2: the rings where the recursion emits NaN are
exactly the rings the finished kernel zeroes — `[15, 23, 25, 34, 37, 39, 47]` —
all at exact rational cos(theta) (2/3, 1/3, 1/4, -1/8, -1/4, -1/3, -2/3).
The default JAX precompute path is 7.0e-2 relative wrong at spin 2.

The numpy builder `spin_spherical_kernel` is unaffected because it routes
through `recursions.turok.compute_slice` (`construct.py:93`), and it agrees with
the on-the-fly transform to 2.3e-15. Hence the constraint above: build with
numpy, apply with `method="jax"`.

### Why croissant owns refinement

`spherical.forward(..., iter>0)` builds its inverse kernel with the broken jax
builder, so its refinement diverges for spin != 0 — worse than not refining, and
never converging. Driving the same iteration ourselves with numpy-built kernels
reproduces the on-the-fly convergence digit-for-digit:

```
iter:      0          1          2          3          4          5
error:  5.691e-02  5.426e-03  6.469e-04  7.869e-05  9.499e-06  1.147e-06
```

The iteration is `flm = F(f)` then `flm += F(f - I(flm))`, where `F` is forward
and `I` is inverse. This is the same iteration `sphere.py:169-181` already
applies to the scalar dense matrix in gram form, which is why the engines agree
at `niter > 0`.

Consequence: `niter > 0` needs an inverse kernel as well as a forward one, so
build the inverse lazily — `niter=0` must pay nothing. And unlike `dense`, the
kernel engine cannot fold refinement into a cached operator, so its per-call
cost is ~`2*niter+1` kernel applications. `dense` therefore remains the fastest
per call at `niter > 0`, gated on whether its footprint is affordable.

---

## File Structure

- **Create `src/croissant/footprints.py`** — pure "predict a precomputed
  object's size without building it" helpers: `transform_lmax`,
  `kernel_nbytes`, `dense_nbytes`. Imported by `kernel.py`,
  `engine_select.py` and `benchmarks/benchmark_engines.py`, so all three share
  one formula. It imports nothing from croissant, so it cannot create a cycle.
- **Create `src/croissant/kernel.py`** — the kernel engine: build, bounded
  cache, apply, croissant-side refinement. A separate module because it is a
  distinct mechanism with its own cache, mirroring how `dense.py` is separate
  from `sphere.py`. `sphere.py` is already 622 lines.
- **Create `src/croissant/engine_select.py`** — the `auto` policy as a pure
  function of statically-known quantities, with no imports from `sphere.py` or
  `dense.py`, so it is trivially testable and cannot create an import cycle.
- **Modify `src/croissant/sphere.py`** — accept `"kernel"` and `"auto"` in
  `compute_alm` (line 353 signature, validation at 432) and `SphBase.__init__`
  (validation at ~547), resolve `auto` to a concrete engine name.
- **Modify `src/croissant/beam.py`** — extend the `engine` docstring and
  validation (line 22 default, 69 docstring).
- **Modify `src/croissant/__init__.py`** — export the new public names.
- **Modify `README.md`** — replace the prose engine-selection rule (lines 51–55)
  with a description of `auto` plus the override.
- **Create `tests/test_engine_equivalence.py`** — the cross-engine invariant,
  including `niter > 0`. This is the gate for the whole plan.
- **Create `tests/test_kernel_engine.py`** — kernel build, cache, guard against
  future s2fft regressions, refinement convergence.
- **Create `tests/test_engine_select.py`** — the `auto` policy table.
- **Create `benchmarks/benchmark_engines.py`** — cost and agreement across all
  three engines, following the conventions of the existing
  `benchmarks/benchmark_dense_healpix.py` (x64 via the environment before
  importing jax, imports inside `main`, `key=value` output lines,
  `block_until_ready` around every timing).
- **Create `benchmarks/results/engines-<YYYY-MM-DD>.md`** — the committed
  benchmark output. It is evidence, not a build artifact: Task 7's thresholds and
  Task 8's README table both cite it, so re-running the benchmark and not
  updating this file makes the README wrong.

---

## Task 1: Cross-engine equivalence at niter > 0

Establishes the invariant the rest of the plan depends on: engines are
interchangeable, so `auto` may switch between them freely. Written against the
two engines that exist today, then extended to `kernel` in Task 5.

**Files:**
- Test: `tests/test_engine_equivalence.py` (create)

**Interfaces:**
- Consumes: `croissant.sphere.compute_alm(data, lmax, sampling, nside=,
  niter=, spin=, reality=, engine=)`, already public.
- Produces: `NSIDE`, `LMAX`, `_healpix_data(rng, nfreq, complex_)` and
  `_assert_engines_agree(a, b, atol_rel)` helpers, reused by Task 5.

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_equivalence.py`:

```python
"""
Cross-engine equivalence for the spherical harmonic transform.

Croissant's engines differ only in how much of the pixels-to-alm map is
precomputed, never in the map itself, so they must agree to near machine
precision. ``engine="auto"`` relies on that: if the engines could
disagree, switching engines for resource reasons would change results.

HEALPix with ``niter > 0`` is the case worth pinning. The engines reach
refinement by different routes -- the s2fft engine iterates inside s2fft,
the dense engine folds the same iteration into its cached matrix in gram
form -- so agreement here is a real constraint rather than a tautology.
"""

import numpy as np
import pytest

from croissant import sphere

NSIDE = 8
# s2fft's HEALPix transforms require L = lmax + 1 >= 2 * nside.
LMAX = 2 * NSIDE - 1

ENGINES = ["s2fft", "dense"]


def _healpix_data(rng, nfreq=2, complex_=False):
    """Random HEALPix data of shape (nfreq, npix)."""
    npix = 12 * NSIDE**2
    data = rng.normal(size=(nfreq, npix))
    if complex_:
        data = data + 1j * rng.normal(size=(nfreq, npix))
    return data


def _assert_engines_agree(a, b, atol_rel=1e-10):
    """Assert two alm arrays agree relative to the larger one's scale."""
    a = np.asarray(a)
    b = np.asarray(b)
    scale = max(np.abs(a).max(), np.abs(b).max())
    np.testing.assert_allclose(a, b, rtol=0, atol=atol_rel * scale)


@pytest.mark.parametrize("niter", [0, 1, 3])
def test_scalar_healpix_engines_agree(niter):
    """Real scalar HEALPix analysis is engine-independent."""
    data = _healpix_data(np.random.default_rng(0))
    kwargs = dict(
        lmax=LMAX, sampling="healpix", nside=NSIDE, niter=niter
    )
    reference = sphere.compute_alm(data, engine="s2fft", **kwargs)
    for engine in ENGINES[1:]:
        got = sphere.compute_alm(data, engine=engine, **kwargs)
        _assert_engines_agree(reference, got)


@pytest.mark.parametrize("niter", [0, 3])
@pytest.mark.parametrize("spin", [2, -2])
def test_spin_healpix_engines_agree(niter, spin):
    """Spin-weighted HEALPix analysis is engine-independent."""
    data = _healpix_data(np.random.default_rng(1), complex_=True)
    kwargs = dict(
        lmax=LMAX,
        sampling="healpix",
        nside=NSIDE,
        niter=niter,
        spin=spin,
        reality=False,
    )
    reference = sphere.compute_alm(data, engine="s2fft", **kwargs)
    for engine in ENGINES[1:]:
        got = sphere.compute_alm(data, engine=engine, **kwargs)
        _assert_engines_agree(reference, got)


def test_mwss_engines_agree():
    """Equiangular sampling is engine-independent (niter is irrelevant
    there: MWSS satisfies a sampling theorem, so analysis is exact)."""
    import s2fft

    L = LMAX + 1
    ntheta = s2fft.sampling.s2_samples.ntheta(L=L, sampling="mwss")
    nphi = s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling="mwss")
    rng = np.random.default_rng(2)
    data = rng.normal(size=(2, ntheta, nphi))
    kwargs = dict(lmax=LMAX, sampling="mwss")
    reference = sphere.compute_alm(data, engine="s2fft", **kwargs)
    for engine in ENGINES[1:]:
        got = sphere.compute_alm(data, engine=engine, **kwargs)
        _assert_engines_agree(reference, got)
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_engine_equivalence.py -v --no-cov`

Expected: **PASS**. This test characterises existing behaviour, so unlike the
rest of the plan it is not expected to fail first — its job is to lock the
invariant in before `kernel` and `auto` can rely on it.

**If any case fails, stop and report.** A genuine `s2fft`-vs-`dense`
disagreement at `niter > 0` is a pre-existing croissant bug, and it must be
understood before building on the assumption that engines are interchangeable.
Do not loosen the tolerance to make it pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_equivalence.py
git commit -m "test: pin cross-engine SHT equivalence including niter>0"
```

---

## Task 2: Build and cache the Wigner-d kernel

**Files:**
- Create: `src/croissant/footprints.py`
- Create: `src/croissant/kernel.py`
- Test: `tests/test_kernel_engine.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, in `footprints.py`:
  - `transform_lmax(lmax, sampling, nside=None) -> int`
  - `kernel_nbytes(lmax, sampling, nside=None, reality=False) -> int`
  - `dense_nbytes(lmax, sampling, nside=None, spin=0, reality=True) -> int`
- Produces, in `kernel.py`:
  - `precompute_kernel(lmax, sampling, nside=None, spin=0, reality=False,
    forward=True) -> jax.Array`
  - `clear_kernel_cache() -> None`
  - `_KERNEL_CACHE_MAXSIZE` (int, 8)
  - re-exports `transform_lmax` and `kernel_nbytes` from `footprints`, so
    `croissant.kernel.kernel_nbytes` also resolves.

**Note on `reality` (controller ruling R2, verified empirically).** The kernel's
`m` extent depends on the `reality` flag: with `reality=True` s2fft's precompute
path slices `ftm` to `m >= 0` and expects a kernel of shape
`(ntheta, L, L)` rather than `(ntheta, L, 2L-1)`. Building with one flag and
applying with the other raises
`ValueError: Size of label 'm' for operand 1 (31) does not match previous terms
(16)`. So `reality` is a build parameter and part of the cache key, and
`kernel_compute_alm` (Task 3) must pass the same value to both. Matching flags
also halve the kernel for real scalar fields — 124 KiB against 240 KiB at
nside=8/L=16 — at identical accuracy (1.4e-15).

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernel_engine.py`:

```python
"""
Tests for the precomputed-kernel spherical harmonic engine.

The engine caches s2fft's Wigner-d kernel and contracts it, instead of
recomputing the recursion per call (``engine="s2fft"``) or materialising
the whole operator (``engine="dense"``).

One test here is a regression guard rather than a behaviour test:
s2fft's *jax* kernel builder is silently wrong for spin != 0 on HEALPix
(it zeroes entries where the Price-McEwen recursion hits an exact
Wigner-d node), so croissant must build kernels with the *numpy*
builder. ``test_kernel_matches_on_the_fly_transform`` fails if a future
s2fft version breaks the builder we do rely on.
"""

import numpy as np
import pytest
import s2fft

from croissant import kernel

NSIDE = 8
LMAX = 2 * NSIDE - 1


@pytest.mark.parametrize("reality", [False, True])
def test_kernel_shape_and_size_prediction(reality):
    """kernel_nbytes predicts the footprint without building it.

    The last axis depends on ``reality``: a real-field kernel stores only
    m >= 0, so it is L wide rather than 2L-1.
    """
    predicted = kernel.kernel_nbytes(
        LMAX, "healpix", nside=NSIDE, reality=reality
    )
    k = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, spin=0, reality=reality
    )
    ntheta = 4 * NSIDE - 1
    nm = (LMAX + 1) if reality else (2 * LMAX + 1)
    assert k.shape == (ntheta, LMAX + 1, nm)
    assert predicted == k.nbytes


def test_kernel_cache_returns_identical_object():
    """Repeated requests hit the cache rather than rebuilding."""
    kernel.clear_kernel_cache()
    first = kernel.precompute_kernel(LMAX, "healpix", nside=NSIDE, spin=2)
    second = kernel.precompute_kernel(LMAX, "healpix", nside=NSIDE, spin=2)
    assert first is second
    kernel.clear_kernel_cache()
    third = kernel.precompute_kernel(LMAX, "healpix", nside=NSIDE, spin=2)
    assert third is not first


@pytest.mark.parametrize("spin", [0, 2, -2])
def test_kernel_matches_on_the_fly_transform(spin):
    """Regression guard on the s2fft builder croissant depends on.

    s2fft's ``spin_spherical_kernel_jax`` drops modes for spin != 0 on
    HEALPix; the numpy ``spin_spherical_kernel`` does not. Croissant must
    use the latter. If this fails after an s2fft upgrade, the builder
    changed and the kernel engine cannot be trusted.
    """
    L = LMAX + 1
    rng = np.random.default_rng(3)
    flm = np.asarray(
        s2fft.utils.signal_generator.generate_flm(
            rng, L, spin=spin, reality=False
        )
    )
    field = np.asarray(
        s2fft.inverse(
            flm,
            L=L,
            spin=spin,
            nside=NSIDE,
            sampling="healpix",
            method="jax",
            reality=False,
        )
    )
    expected = np.asarray(
        s2fft.forward(
            field,
            L=L,
            spin=spin,
            nside=NSIDE,
            sampling="healpix",
            method="jax",
            reality=False,
            iter=0,
        )
    )
    k = kernel.precompute_kernel(
        LMAX, "healpix", nside=NSIDE, spin=spin, forward=True
    )
    got = np.asarray(
        s2fft.precompute_transforms.spherical.forward(
            field,
            L=L,
            spin=spin,
            kernel=k,
            sampling="healpix",
            reality=False,
            method="jax",
            nside=NSIDE,
            iter=0,
        )
    )
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_kernel_engine.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'croissant.kernel'`

- [ ] **Step 3: Write the minimal implementation**

First create `src/croissant/footprints.py`:

```python
"""
Predict the size of a precomputed transform without building it.

Croissant's engines precompute different amounts of the pixels-to-alm
map, and both the automatic engine policy and the benchmarks need to know
what a choice would cost before paying for it. These helpers are pure
arithmetic over the transform's geometry; they import nothing from
croissant, so any module may use them.
"""

import numpy as np
import s2fft

_COMPLEX_ITEMSIZE = np.dtype(np.complex128).itemsize


def transform_lmax(lmax, sampling, nside=None):
    """
    Band-limit a transform must actually be performed at.

    s2fft's HEALPix FFT requires ``L >= 2 * nside`` even when only lower
    modes are wanted, the same floor ``croissant.dense`` handles at
    ``dense.py:52``.

    Parameters
    ----------
    lmax : int
        Requested maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.

    Returns
    -------
    int
        The band-limit to transform at, always ``>= lmax``.

    """
    if sampling != "healpix":
        return int(lmax)
    if nside is None:
        raise ValueError("nside is required for HEALPix transforms.")
    return max(int(lmax), 2 * int(nside) - 1)


def _ntheta(lmax, sampling, nside=None):
    """Number of latitude rings for a sampling scheme."""
    if sampling == "healpix":
        if nside is None:
            raise ValueError("nside is required for HEALPix transforms.")
        return 4 * int(nside) - 1
    return s2fft.sampling.s2_samples.ntheta(L=lmax + 1, sampling=sampling)


def _npix(lmax, sampling, nside=None):
    """Number of spatial samples for a sampling scheme."""
    if sampling == "healpix":
        if nside is None:
            raise ValueError("nside is required for HEALPix transforms.")
        return 12 * int(nside) ** 2
    L = lmax + 1
    return s2fft.sampling.s2_samples.ntheta(
        L=L, sampling=sampling
    ) * s2fft.sampling.s2_samples.nphi_equiang(L=L, sampling=sampling)


def kernel_nbytes(lmax, sampling, nside=None, reality=False):
    """
    Predict a Wigner-d kernel's memory footprint.

    Reports the footprint at the band-limit the kernel would really be
    built at, i.e. after applying the HEALPix ``L >= 2 * nside`` floor.
    Reporting the requested ``lmax`` instead would under-predict by
    ``(2 * nside / (lmax + 1)) ** 2`` whenever a caller asks for a low
    band-limit on a high-resolution map.

    Parameters
    ----------
    lmax : int
        Requested maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    reality : bool
        Whether the kernel is built for a real field. Real kernels store
        only ``m >= 0``, halving the last axis.

    Returns
    -------
    int
        Size in bytes of the complex128 kernel.

    """
    L = transform_lmax(lmax, sampling, nside=nside) + 1
    nm = L if reality else 2 * L - 1
    return _ntheta(lmax, sampling, nside) * L * nm * _COMPLEX_ITEMSIZE


def dense_nbytes(lmax, sampling, nside=None, spin=0, reality=True):
    """
    Predict the dense analysis operator's memory footprint.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    spin : int
        Spin weight of the field.
    reality : bool
        Whether the field is real. Real scalar fields store only the
        independent ``m >= 0`` coefficients.

    Returns
    -------
    int
        Size in bytes of the complex128 operator.

    """
    L = lmax + 1
    if spin == 0 and reality:
        ncoeff = (lmax + 1) * (lmax + 2) // 2
    else:
        ncoeff = L * L - spin * spin
    return ncoeff * _npix(lmax, sampling, nside) * _COMPLEX_ITEMSIZE
```

Then create `src/croissant/kernel.py`:

```python
"""
Precomputed-kernel spherical harmonic analysis.

This engine caches the Wigner-d kernel that carries the theta-to-ell
stage of the transform and contracts it per call, leaving only the
per-ring FFT to be recomputed. It sits between ``engine="s2fft"``, which
precomputes nothing, and ``engine="dense"``, which materialises the whole
``ncoeff x npix`` operator: the kernel is ``O(nside**3)`` where the dense
operator is ``O(nside**4)``, which is what makes moderate and high nside
reachable.

Two s2fft constraints are load-bearing and must not be relaxed:

1. Kernels are built with the *numpy* ``spin_spherical_kernel``, never
   ``spin_spherical_kernel_jax``. The jax builder routes through the
   Price-McEwen recursion, which hits exact Wigner-d zero nodes at
   HEALPix's rational cos(theta) values for spin != 0; the NaNs are then
   masked to zero, silently dropping those modes (~7e-2 relative error
   at spin 2). The numpy builder uses Turok's recursion and is exact.
2. ``iter > 0`` is never passed to s2fft's precompute transform. Its
   refinement branch builds an inverse kernel with the broken jax
   builder and diverges for spin != 0. Croissant runs the refinement
   iteration itself in :func:`kernel_compute_alm`.
"""

from collections import OrderedDict
from threading import Lock

import jax.numpy as jnp
import s2fft

from .footprints import kernel_nbytes, transform_lmax

__all__ = [
    "clear_kernel_cache",
    "kernel_compute_alm",
    "kernel_nbytes",
    "precompute_kernel",
    "transform_lmax",
]

_KERNEL_CACHE_MAXSIZE = 8
_KERNEL_CACHE = OrderedDict()
_KERNEL_CACHE_LOCK = Lock()


def precompute_kernel(
    lmax, sampling, nside=None, spin=0, reality=False, forward=True
):
    """
    Build and cache the Wigner-d kernel for one transform configuration.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    spin : int
        Spin weight of the field.
    reality : bool
        Whether the kernel is for a real field. This is a BUILD
        parameter, not only an apply-time one: with ``reality=True``
        s2fft's precompute path slices ``ftm`` to ``m >= 0`` and expects
        a kernel whose last axis is ``L`` rather than ``2L - 1``.
        Building with one value and applying with the other raises
        ``ValueError: Size of label 'm' ... does not match previous
        terms``, so it is part of the cache key and callers must pass the
        same value here and at apply time.
    forward : bool
        Build the analysis kernel if True, the synthesis kernel if
        False. The synthesis kernel is only needed for iterative
        refinement.

    Returns
    -------
    jax.Array
        Kernel of shape ``(ntheta, L, L)`` when ``reality`` is True and
        ``(ntheta, L, 2L - 1)`` otherwise, where ``L`` is
        ``transform_lmax(...) + 1``.

    """
    key = (
        int(lmax),
        str(sampling),
        None if nside is None else int(nside),
        int(spin),
        bool(reality),
        bool(forward),
    )
    with _KERNEL_CACHE_LOCK:
        if key in _KERNEL_CACHE:
            _KERNEL_CACHE.move_to_end(key)
            return _KERNEL_CACHE[key]

    # NOTE: the numpy builder, deliberately. See the module docstring.
    built = s2fft.precompute_transforms.construct.spin_spherical_kernel(
        L=transform_lmax(lmax, sampling, nside=nside) + 1,
        spin=int(spin),
        reality=bool(reality),
        sampling=sampling,
        nside=nside,
        forward=bool(forward),
    )
    array = jnp.asarray(built)

    with _KERNEL_CACHE_LOCK:
        _KERNEL_CACHE[key] = array
        _KERNEL_CACHE.move_to_end(key)
        while len(_KERNEL_CACHE) > _KERNEL_CACHE_MAXSIZE:
            _KERNEL_CACHE.popitem(last=False)
        return _KERNEL_CACHE[key]


def clear_kernel_cache():
    """Release all cached kernels held by croissant."""
    with _KERNEL_CACHE_LOCK:
        _KERNEL_CACHE.clear()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_kernel_engine.py -v --no-cov`
Expected: PASS (5 tests: 2 cache/shape + 3 spins)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/croissant/kernel.py tests/test_kernel_engine.py
uv run ruff format src/croissant/kernel.py tests/test_kernel_engine.py
git add src/croissant/kernel.py tests/test_kernel_engine.py
git commit -m "feat: cache s2fft Wigner-d kernels for a new SHT engine"
```

---

## Task 3: Apply the kernel at niter = 0

**Files:**
- Modify: `src/croissant/kernel.py`
- Test: `tests/test_kernel_engine.py`

**Interfaces:**
- Consumes: `precompute_kernel` from Task 2.
- Produces: `kernel_compute_alm(data, lmax, sampling, nside=None, niter=0,
  spin=0, reality=True) -> jax.Array`, returning shape
  `batch_shape + (lmax + 1, 2 * lmax + 1)` — the same layout
  `sphere.compute_alm` returns.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kernel_engine.py`:

```python
def test_kernel_compute_alm_matches_s2fft_engine_scalar():
    """The kernel engine reproduces the s2fft engine for real scalars,
    including the batch axis and the returned layout."""
    from croissant import sphere

    rng = np.random.default_rng(4)
    data = rng.normal(size=(3, 12 * NSIDE**2))
    kwargs = dict(lmax=LMAX, sampling="healpix", nside=NSIDE, niter=0)
    expected = np.asarray(
        sphere.compute_alm(data, engine="s2fft", **kwargs)
    )
    got = np.asarray(kernel.kernel_compute_alm(data, **kwargs))
    assert got.shape == expected.shape == (3, LMAX + 1, 2 * LMAX + 1)
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )


def test_kernel_engine_follows_the_dtype_contract():
    """The engines share a dtype policy, so the kernel engine must too.

    Per ``sphere._dense_dtypes``, croissant's engines reproduce
    ``s2fft.forward``: complex128 out on an x64 runtime even for float32
    maps. A kernel engine that instead inherited the input dtype would
    silently change precision downstream, which the alm-value
    equivalence tests would not catch because they compare at x64.
    """
    from croissant import sphere

    rng = np.random.default_rng(13)
    for input_dtype in (np.float32, np.float64):
        data = rng.normal(size=(2, 12 * NSIDE**2)).astype(input_dtype)
        kwargs = dict(
            lmax=LMAX, sampling="healpix", nside=NSIDE, niter=0
        )
        expected = sphere.compute_alm(data, engine="s2fft", **kwargs)
        got = kernel.kernel_compute_alm(data, **kwargs)
        assert got.dtype == expected.dtype, (
            f"kernel engine returned {got.dtype} for {input_dtype} input, "
            f"s2fft engine returned {expected.dtype}"
        )


@pytest.mark.parametrize("spin", [2, -2])
def test_kernel_compute_alm_matches_s2fft_engine_spin(spin):
    """The kernel engine reproduces the s2fft engine for spin fields."""
    from croissant import sphere

    rng = np.random.default_rng(5)
    npix = 12 * NSIDE**2
    data = rng.normal(size=(2, npix)) + 1j * rng.normal(size=(2, npix))
    kwargs = dict(
        lmax=LMAX,
        sampling="healpix",
        nside=NSIDE,
        niter=0,
        spin=spin,
        reality=False,
    )
    expected = np.asarray(
        sphere.compute_alm(data, engine="s2fft", **kwargs)
    )
    got = np.asarray(kernel.kernel_compute_alm(data, **kwargs))
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_kernel_engine.py -k compute_alm -v --no-cov`
Expected: FAIL with `AttributeError: module 'croissant.kernel' has no
attribute 'kernel_compute_alm'`

- [ ] **Step 3: Write the minimal implementation**

Add to `src/croissant/kernel.py` (imports first: add `import jax` and
`from functools import partial` to the existing import block):

```python
def _spatial_ndim(sampling):
    """Number of trailing axes that hold the field's spatial samples."""
    return 1 if sampling == "healpix" else 2


def kernel_compute_alm(
    data,
    lmax,
    sampling,
    nside=None,
    niter=0,
    spin=0,
    reality=True,
):
    """
    Compute alm by contracting a cached Wigner-d kernel.

    Every axis before the spatial axes is treated as a batch axis, and
    the returned layout matches :func:`croissant.sphere.compute_alm`.

    Parameters
    ----------
    data : array_like
        Field samples, with spatial axes trailing.
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    niter : int
        Number of iterative refinement steps. Refinement is run by
        croissant, not by s2fft; see the module docstring.
    spin : int
        Spin weight of the field.
    reality : bool
        Whether the field is real. Forced False for nonzero spin, which
        s2fft's precompute path requires.

    Returns
    -------
    jax.Array
        Coefficients of shape ``batch + (lmax + 1, 2 * lmax + 1)``.

    """
    if niter < 0:
        raise ValueError(f"niter must be non-negative, got {niter}.")
    floor = transform_lmax(lmax, sampling, nside=nside)
    if floor != lmax:
        raise ValueError(
            f"The kernel engine needs lmax >= {floor} for "
            f"nside={nside} (s2fft's HEALPix FFT requires "
            "L >= 2 * nside), but lmax="
            f"{lmax} was requested. Use engine='dense', which builds at "
            "the required band-limit and keeps only the low-ell rows, or "
            "engine='s2fft'."
        )
    # Croissant's engines share a dtype contract, owned and documented by
    # sphere._dense_dtypes: they reproduce s2fft.forward, which returns
    # complex128 on an x64 runtime even for float32 maps. s2fft's
    # PRECOMPUTE path instead inherits the input dtype, so a float32 map
    # would come back complex64 with ~1e-7 relative error. Promote the
    # input rather than casting the result: casting the result would keep
    # that error. Imported lazily because sphere imports this module.
    from .sphere import _dense_dtypes

    real_dtype, _ = _dense_dtypes()
    data = jnp.asarray(data)
    if data.dtype.kind == "c":
        data = data.astype(jnp.result_type(real_dtype, 1j))
    else:
        data = data.astype(real_dtype)
    spatial_ndim = _spatial_ndim(sampling)
    spatial_shape = data.shape[-spatial_ndim:]
    batch_shape = data.shape[:-spatial_ndim]
    flat = data.reshape((-1,) + spatial_shape)

    reality = bool(reality) and spin == 0
    L = lmax + 1
    forward_kernel = precompute_kernel(
        lmax, sampling, nside=nside, spin=spin, reality=reality,
        forward=True
    )
    analyse = partial(
        s2fft.precompute_transforms.spherical.forward,
        L=L,
        spin=spin,
        kernel=forward_kernel,
        sampling=sampling,
        reality=reality,
        method="jax",
        nside=nside,
        iter=0,  # never delegate refinement; see the module docstring
    )
    flat_alm = jax.vmap(analyse)(flat)
    return flat_alm.reshape(batch_shape + (lmax + 1, 2 * lmax + 1))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_kernel_engine.py -v --no-cov`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/croissant/kernel.py tests/test_kernel_engine.py
uv run ruff format src/croissant/kernel.py tests/test_kernel_engine.py
git add src/croissant/kernel.py tests/test_kernel_engine.py
git commit -m "feat: analyse fields by contracting the cached kernel"
```

---

## Task 4: Croissant-side iterative refinement

**Files:**
- Modify: `src/croissant/kernel.py`
- Test: `tests/test_kernel_engine.py`

**Interfaces:**
- Consumes: `precompute_kernel`, `kernel_compute_alm` from Tasks 2–3.
- Produces: `kernel_compute_alm` honouring `niter > 0`. No new public names.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kernel_engine.py`:

```python
def test_refinement_converges_towards_a_band_limited_signal():
    """Refinement must actually converge, monotonically.

    HEALPix has no exact quadrature, so analysing a synthesised
    band-limited field does not recover its coefficients exactly. Each
    refinement step should shrink that error. s2fft's own precompute
    refinement fails this for spin != 0 -- it diverges -- which is why
    croissant runs the iteration itself.
    """
    L = LMAX + 1
    spin = 2
    rng = np.random.default_rng(6)
    flm = np.asarray(
        s2fft.utils.signal_generator.generate_flm(
            rng, L, spin=spin, reality=False
        )
    )
    field = np.asarray(
        s2fft.inverse(
            flm,
            L=L,
            spin=spin,
            nside=NSIDE,
            sampling="healpix",
            method="jax",
            reality=False,
        )
    )
    errors = []
    for niter in (0, 1, 2, 3):
        got = np.asarray(
            kernel.kernel_compute_alm(
                field[None],
                lmax=LMAX,
                sampling="healpix",
                nside=NSIDE,
                niter=niter,
                spin=spin,
                reality=False,
            )
        )[0]
        errors.append(np.abs(got - flm).max())
    for previous, current in zip(errors, errors[1:]):
        assert current < previous, f"refinement not converging: {errors}"
    assert errors[-1] < errors[0] / 10


@pytest.mark.parametrize("niter", [1, 3])
@pytest.mark.parametrize("spin", [0, 2])
def test_refined_kernel_engine_matches_s2fft_engine(niter, spin):
    """With refinement on, the kernel engine still matches s2fft's."""
    from croissant import sphere

    rng = np.random.default_rng(7)
    npix = 12 * NSIDE**2
    if spin == 0:
        data = rng.normal(size=(2, npix))
        reality = True
    else:
        data = rng.normal(size=(2, npix)) + 1j * rng.normal(size=(2, npix))
        reality = False
    kwargs = dict(
        lmax=LMAX,
        sampling="healpix",
        nside=NSIDE,
        niter=niter,
        spin=spin,
        reality=reality,
    )
    expected = np.asarray(
        sphere.compute_alm(data, engine="s2fft", **kwargs)
    )
    got = np.asarray(kernel.kernel_compute_alm(data, **kwargs))
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )


def test_inverse_kernel_is_not_built_when_niter_is_zero():
    """niter=0 must not pay for the synthesis kernel."""
    kernel.clear_kernel_cache()
    rng = np.random.default_rng(8)
    data = rng.normal(size=(1, 12 * NSIDE**2))
    kernel.kernel_compute_alm(
        data, lmax=LMAX, sampling="healpix", nside=NSIDE, niter=0
    )
    forwards = [key[-1] for key in kernel._KERNEL_CACHE]
    assert forwards == [True]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_kernel_engine.py -k refin -v --no-cov`
Expected: FAIL — `niter` is currently ignored, so the errors list is flat and
`assert current < previous` fails on the first comparison.

- [ ] **Step 3: Write the minimal implementation**

In `src/croissant/kernel.py`, replace the last two statements of
`kernel_compute_alm` (`flat_alm = jax.vmap(analyse)(flat)` and the `return`)
with:

```python
    if niter == 0:
        flat_alm = jax.vmap(analyse)(flat)
        return flat_alm.reshape(
            batch_shape + (lmax + 1, 2 * lmax + 1)
        )

    # Iterative refinement, run here rather than delegated to s2fft:
    # its precompute refinement branch builds an inverse kernel with the
    # broken jax builder and diverges for spin != 0. The iteration is
    # flm <- flm + F(f - I(flm)), the same one sphere.py applies to the
    # scalar dense matrix in gram form.
    inverse_kernel = precompute_kernel(
        lmax, sampling, nside=nside, spin=spin, reality=reality,
        forward=False
    )
    synthesise = partial(
        s2fft.precompute_transforms.spherical.inverse,
        L=L,
        spin=spin,
        kernel=inverse_kernel,
        sampling=sampling,
        reality=reality,
        method="jax",
        nside=nside,
    )

    def refine(field):
        alm = analyse(field)
        for _ in range(niter):
            alm = alm + analyse(field - synthesise(alm))
        return alm

    flat_alm = jax.vmap(refine)(flat)
    return flat_alm.reshape(batch_shape + (lmax + 1, 2 * lmax + 1))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_kernel_engine.py -v --no-cov`
Expected: PASS (all 13 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/croissant/kernel.py tests/test_kernel_engine.py
uv run ruff format src/croissant/kernel.py tests/test_kernel_engine.py
git add src/croissant/kernel.py tests/test_kernel_engine.py
git commit -m "feat: run kernel-engine refinement in croissant"
```

---

## Task 5: Wire `engine="kernel"` into the public API

**Files:**
- Modify: `src/croissant/sphere.py` (validation at ~432 and ~547)
- Modify: `src/croissant/beam.py` (docstring at 69)
- Modify: `src/croissant/__init__.py` (line 13 area)
- Test: `tests/test_engine_equivalence.py`, `tests/test_kernel_engine.py`

**Interfaces:**
- Consumes: `kernel.kernel_compute_alm`, `kernel.clear_kernel_cache`.
- Produces: `"kernel"` accepted by `sphere.compute_alm(engine=...)`,
  `SphBase.__init__(engine=...)`, `Beam`, `Sky`; `croissant.precompute_kernel`
  and `croissant.clear_kernel_cache` exported.

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_equivalence.py`, change the engine list:

```python
ENGINES = ["s2fft", "dense", "kernel"]
```

Append to `tests/test_kernel_engine.py`:

```python
def test_sphbase_accepts_the_kernel_engine():
    """Beam construction works with engine="kernel" and reports it."""
    from croissant import Beam

    rng = np.random.default_rng(9)
    data = rng.normal(size=(2, 12 * NSIDE**2)) ** 2
    beam = Beam(
        data,
        freqs=np.array([50.0, 60.0]),
        sampling="healpix",
        engine="kernel",
        niter=0,
    )
    assert beam.engine == "kernel"
    assert beam.alm.shape[-2:] == (beam.lmax + 1, 2 * beam.lmax + 1)


def test_full_pipeline_visibilities_agree_across_engines():
    """One end-to-end check that the engines are interchangeable.

    The transform-level equivalence tests already pin the alm values, and
    everything downstream of the transform -- the convolve einsum, the
    phase rotation, the beam-integral normalisation -- is
    engine-independent, so this cannot fail on alm values alone. It is
    here to catch what those tests structurally cannot: dtype
    promotion through the pipeline, tracer/jit interaction (the dense
    engine raises inside jit unless precomputed, the kernel engine
    builds lazily), and cache side effects across repeated calls.

    Deliberately ONE test rather than parametrising
    ``tests/test_physics.py`` over engines: the physics file is
    ground-truth and is not to be modified, and tripling it would cost
    ~90 s to re-test a corollary of the equivalence theorem.
    """
    import jax.numpy as jnp
    from astropy.time import Time as AstroTime

    from croissant import Beam, Simulator, Sky

    nside, npix = 8, 12 * 8**2
    freqs = jnp.linspace(50.0, 250.0, 3)
    t0 = AstroTime("2022-01-01 00:00:00")
    times_jd = jnp.linspace(t0.jd, t0.jd + 0.5, 4, endpoint=False)
    beam_data = jnp.ones((len(freqs), npix))
    tsky = 1e4 * (freqs / 150.0) ** (-2.5)
    sky_data = tsky[:, None] * jnp.ones((npix,))

    visibilities = {}
    for engine in ("s2fft", "kernel", "dense"):
        beam = Beam(
            beam_data,
            freqs,
            sampling="healpix",
            niter=0,
            engine=engine,
        )
        sky = Sky(
            sky_data,
            freqs,
            sampling="healpix",
            coord="galactic",
            niter=0,
            engine=engine,
        )
        sim = Simulator(
            beam, sky, times_jd, freqs, 0.0, 0.0, world="earth"
        )
        visibilities[engine] = np.asarray(sim.sim())

    reference = visibilities["s2fft"]
    for engine, got in visibilities.items():
        assert got.dtype == reference.dtype, (
            f"engine {engine!r} changed the visibility dtype: "
            f"{got.dtype} vs {reference.dtype}"
        )
        np.testing.assert_allclose(
            got,
            reference,
            rtol=0,
            atol=1e-9 * np.abs(reference).max(),
            err_msg=f"engine {engine!r} changed the visibilities",
        )


def test_unknown_engine_is_rejected_with_the_full_list():
    from croissant import sphere

    rng = np.random.default_rng(10)
    data = rng.normal(size=(1, 12 * NSIDE**2))
    with pytest.raises(ValueError, match="kernel"):
        sphere.compute_alm(
            data,
            lmax=LMAX,
            sampling="healpix",
            nside=NSIDE,
            engine="nonsense",
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_equivalence.py tests/test_kernel_engine.py -v --no-cov`
Expected: FAIL with `ValueError: Unsupported SHT engine 'kernel'.`

- [ ] **Step 3: Write the minimal implementation**

In `src/croissant/sphere.py`, in `compute_alm`, insert a `kernel` branch
immediately before the existing `if engine != "dense":` guard (~line 432):

```python
    if engine == "kernel":
        from . import kernel as _kernel

        return _kernel.kernel_compute_alm(
            data,
            lmax,
            sampling,
            nside=nside,
            niter=niter,
            spin=spin,
            reality=reality,
        )
    if engine != "dense":
        raise ValueError(
            f"Unsupported SHT engine {engine!r}. Supported engines are "
            "{'s2fft', 'kernel', 'dense'}."
        )
```

In `SphBase.__init__`, widen the validation (~line 547):

```python
        if engine not in {"s2fft", "kernel", "dense"}:
            raise ValueError(
                f"Unsupported SHT engine {engine!r}. Supported engines are "
                "{'s2fft', 'kernel', 'dense'}."
            )
```

The existing `if self._engine == "dense":` block that precomputes
`_dense_matrix` needs no change: `"kernel"` falls through to
`self._dense_matrix = None`, and `kernel_compute_alm` builds and caches its own
kernel on first use.

In `src/croissant/beam.py`, extend the `engine` docstring at line 69:

```python
        engine : {"s2fft", "kernel", "dense"}
            Spherical harmonic transform engine. Default is ``"s2fft"``.
```

In `src/croissant/__init__.py`, next to the existing dense import (line 13):

```python
from .kernel import clear_kernel_cache, precompute_kernel
```

and add `kernel,` to the submodule import list at line 5.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine_equivalence.py tests/test_kernel_engine.py -v --no-cov`
Expected: PASS. The equivalence tests now cover `kernel` at `niter` 0, 1 and 3
for scalar and both spins.

- [ ] **Step 5: Run the full suite, lint, commit**

```bash
uv run pytest -q
uv run ruff check
uv run ruff format --check
git add -A
git commit -m "feat: expose engine=\"kernel\" on compute_alm, Beam and Sky"
```

Expected: 501 existing tests plus the new ones, all passing. If any physics test
fails, stop — the change is wrong, not the test.

---

## Task 6: Benchmark the three engines

This task exists because the policy in Task 7 and the README table in Task 8
both make quantitative claims, and neither should rest on the FLOP argument
alone. It runs **before** the policy so `_AMORTISATION_THRESHOLD` and the
`niter` branch are calibrated from measurement.

It also settles the one open question in this plan: whether the kernel engine
really beats dense per call at `niter > 0`. The FLOP count says yes for any
nside > 4, but a dense matmul is a single BLAS call with ideal arithmetic
intensity while the kernel path is an FFT plus einsum with `2*niter+1`
sequential round trips through a synthesis step.

**Files:**
- Create: `benchmarks/benchmark_engines.py`
- Create: `benchmarks/results/engines-<YYYY-MM-DD>.md` (committed output)

**Interfaces:**
- Consumes: `kernel.kernel_compute_alm`, `kernel.kernel_nbytes`,
  `kernel.clear_kernel_cache` (Tasks 2–4); `sphere.compute_alm` with
  `engine="kernel"` (Task 5).
- Produces: measured numbers that Task 7's thresholds and Task 8's README table
  must both cite. No importable API.

- [ ] **Step 1: Write the benchmark**

Create `benchmarks/benchmark_engines.py`, following the conventions in
`benchmark_dense_healpix.py` (x64 via the environment before importing jax,
imports inside `main`, `key=value` output, `block_until_ready` around timings):

```python
"""
Benchmark croissant's three spherical harmonic engines.

Reports setup cost, resident size of the precomputed object, and per-call
cost for engine="s2fft", "kernel" and "dense" across HEALPix
resolutions, and checks that all three agree numerically at each
configuration so the timings cannot be read as a speed/accuracy
trade-off.

Also answers the niter question: dense folds refinement into its cached
matrix, so its per-call cost is independent of niter, while the kernel
engine pays 2*niter+1 applications. Per-call cost is roughly one pass
over the precomputed object, and dense's is ~1.5*nside times larger, so
the FLOP count says the kernel still wins for nside > 4 -- this measures
whether wall-clock agrees.

Dense configurations whose predicted footprint exceeds MEMORY_CAP_MIB are
skipped rather than built, and reported as skipped.
"""

import os
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "1")

MEMORY_CAP_MIB = 1024
SCALAR_NSIDES = (8, 16, 32)
SPIN_NSIDES = (8, 16)
NITERS = (0, 3)
REPEATS = 3


def _time_call(fn, *args, **kwargs):
    """Return (first_call_seconds, best_of_REPEATS_cached_seconds)."""
    started = perf_counter()
    result = fn(*args, **kwargs)
    result.block_until_ready()
    first = perf_counter() - started
    best = float("inf")
    for _ in range(REPEATS):
        started = perf_counter()
        fn(*args, **kwargs).block_until_ready()
        best = min(best, perf_counter() - started)
    return first, best


def main():
    import jax
    import numpy as np

    import jax.numpy as jnp

    from croissant import footprints, kernel, sphere

    jax.config.update("jax_enable_x64", True)
    rows = []

    for spin, nsides in ((0, SCALAR_NSIDES), (2, SPIN_NSIDES)):
        reality = spin == 0
        for nside in nsides:
            lmax = 2 * nside - 1
            npix = 12 * nside**2
            rng = np.random.default_rng(0)
            data = rng.normal(size=(4, npix))
            if not reality:
                data = data + 1j * rng.normal(size=(4, npix))
            data = jnp.asarray(data)

            kernel_mib = kernel.kernel_nbytes(
                lmax, "healpix", nside=nside
            ) / 2**20
            dense_mib = footprints.dense_nbytes(
                lmax, "healpix", nside=nside, spin=spin, reality=reality
            ) / 2**20

            for niter in NITERS:
                reference = None
                for engine in ("s2fft", "kernel", "dense"):
                    if engine == "dense" and dense_mib > MEMORY_CAP_MIB:
                        print(
                            f"spin={spin:+d} nside={nside} niter={niter} "
                            f"engine=dense SKIPPED "
                            f"predicted_mib={dense_mib:.1f}"
                        )
                        continue
                    kernel.clear_kernel_cache()
                    sphere.clear_dense_matrix_cache()

                    def run():
                        return sphere.compute_alm(
                            data,
                            lmax,
                            "healpix",
                            nside=nside,
                            niter=niter,
                            spin=spin,
                            reality=reality,
                            engine=engine,
                        )

                    setup_and_first, cached = _time_call(run)
                    got = np.asarray(run())
                    if reference is None:
                        reference = got
                        agreement = 0.0
                    else:
                        agreement = np.abs(got - reference).max() / max(
                            np.abs(reference).max(), 1e-300
                        )
                    mib = {
                        "s2fft": 0.0,
                        "kernel": kernel_mib,
                        "dense": dense_mib,
                    }[engine]
                    print(
                        f"spin={spin:+d} nside={nside} niter={niter} "
                        f"engine={engine} "
                        f"setup_plus_first_seconds={setup_and_first:.4f} "
                        f"cached_apply_seconds={cached:.6f} "
                        f"precompute_mib={mib:.2f} "
                        f"rel_vs_s2fft={agreement:.2e}"
                    )
                    rows.append(
                        (spin, nside, niter, engine, setup_and_first,
                         cached, mib, agreement)
                    )

    print()
    print("| spin | nside | niter | engine | setup+first (s) | "
          "cached apply (s) | precompute (MiB) | rel vs s2fft |")
    print("|---:|---:|---:|:--|---:|---:|---:|---:|")
    for spin, nside, niter, engine, setup, cached, mib, agree in rows:
        print(
            f"| {spin:+d} | {nside} | {niter} | {engine} | {setup:.3f} | "
            f"{cached:.6f} | {mib:.2f} | {agree:.1e} |"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and capture the output**

```bash
mkdir -p benchmarks/results
uv run python benchmarks/benchmark_engines.py \
  | tee benchmarks/results/engines-$(date +%F).md
```

Expected: every `rel_vs_s2fft` at or below ~1e-10, confirming the timings are
not a speed/accuracy trade-off. If any configuration disagrees more than that,
**stop** — a real cross-engine discrepancy outranks this whole plan, and Task 1's
equivalence tests should have caught it.

- [ ] **Step 3: Record the three conclusions the later tasks depend on**

Read the table and write down, at the top of the results file:

1. The batch size at which `kernel` setup+first beats `s2fft` — this calibrates
   `_AMORTISATION_THRESHOLD` in Task 7.
2. Whether `kernel` cached-apply beats `dense` cached-apply at `niter=3`. If it
   does, Task 7's policy stands as written. If `dense` wins on wall-clock
   despite the FLOP argument, add the `niter > 0 and dense_fits -> dense` branch
   back and cite these numbers in the comment.
3. The measured kernel-vs-dense setup and memory ratios, for the README table in
   Task 8, replacing the nside=16/32 figures quoted in the Background section
   (which were measured before the engine existed, via `dense._build_analysis_matrix`
   directly).

- [ ] **Step 4: Commit**

```bash
uv run ruff check benchmarks/benchmark_engines.py
uv run ruff format benchmarks/benchmark_engines.py
git add benchmarks/benchmark_engines.py benchmarks/results/
git commit -m "bench: compare the three SHT engines on cost and agreement"
```

---

## Task 7: The `auto` selection policy

Pure decision logic, isolated so it can be tested as a table without building
any transforms.

**Files:**
- Create: `src/croissant/engine_select.py`
- Test: `tests/test_engine_select.py` (create)

**Interfaces:**
- Consumes: `kernel.kernel_nbytes` and `kernel.transform_lmax` from Task 2; the
  measured conclusions recorded in Task 6 Step 3, which set
  `_AMORTISATION_THRESHOLD` and decide whether the `niter` branch is needed.
- Produces:
  - `DEFAULT_MEMORY_CAP_BYTES` (int, 512 MiB)
  - `dense_nbytes(lmax, sampling, nside=None, spin=0, reality=True) -> int`
  - `resolve_engine(lmax, sampling, nside=None, spin=0, niter=0,
    reality=True, batch_size=1, memory_cap=None) -> tuple[str, str]`
    returning `(engine_name, reason)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_select.py`:

```python
"""
Tests for automatic SHT engine selection.

The engines agree numerically, so choosing between them is a resource
decision and croissant can own it. These tests pin the policy as a
table: what matters is that the choice is predictable, never exceeds the
memory cap, and degrades to the matrix-free engine when precomputing
cannot pay for itself.
"""

import pytest

from croissant import engine_select, footprints


def test_single_transform_uses_the_matrix_free_engine():
    """One transform has nothing to amortise a precompute against."""
    engine, reason = engine_select.resolve_engine(
        lmax=63, sampling="healpix", nside=32, batch_size=1
    )
    assert engine == "s2fft"
    assert reason


def test_repeated_transforms_prefer_the_kernel():
    """A batch amortises the kernel build."""
    engine, _ = engine_select.resolve_engine(
        lmax=63, sampling="healpix", nside=32, batch_size=64
    )
    assert engine == "kernel"


@pytest.mark.parametrize("niter", [1, 3])
def test_refinement_selects_dense_when_the_operator_fits(niter):
    """At niter>0 dense folds refinement into its cached matrix.

    Measured in Task 6: dense beats the kernel engine's per-call cost by
    1.29x-6.65x at niter=3 across every benchmarked configuration. A flop
    count predicts the opposite, because dense moves ~1.5*nside times
    more data per pass, but constant factors dominate at these sizes.
    """
    engine, reason = engine_select.resolve_engine(
        lmax=31,
        sampling="healpix",
        nside=16,
        niter=niter,
        batch_size=64,
    )
    assert engine == "dense"
    assert "niter" in reason


def test_refinement_falls_back_to_kernel_when_dense_will_not_fit():
    """The niter>0 preference for dense is gated on affordability."""
    engine, _ = engine_select.resolve_engine(
        lmax=127,
        sampling="healpix",
        nside=64,
        niter=3,
        batch_size=64,
        memory_cap=200 * 1024**2,
    )
    assert engine == "kernel"


def test_amortisation_threshold_scales_with_kernel_size():
    """The batch a kernel needs grows with the kernel, not a constant.

    Measured crossovers: batch 1 at nside=8 and nside=16 (kernels of 0.12
    and 0.98 MiB), batch 8 at nside=32 (7.94 MiB). A fixed threshold
    cannot serve both, so a batch of 4 must be enough at nside=16 and not
    enough at nside=32.
    """
    small, _ = engine_select.resolve_engine(
        lmax=31, sampling="healpix", nside=16, batch_size=4
    )
    large, reason = engine_select.resolve_engine(
        lmax=63, sampling="healpix", nside=32, batch_size=4
    )
    assert small == "kernel"
    assert large == "s2fft"
    assert "amortise" in reason


def test_low_bandlimit_on_a_high_resolution_map_selects_dense():
    """Dense's clearest remaining advantage: it can build at the HEALPix
    L >= 2*nside floor and keep only the requested low-ell rows, which
    the kernel engine cannot do."""
    engine, reason = engine_select.resolve_engine(
        lmax=15, sampling="healpix", nside=64, batch_size=64
    )
    assert engine == "dense"
    assert "floor" in reason


def test_low_bandlimit_falls_back_when_dense_will_not_fit():
    engine, _ = engine_select.resolve_engine(
        lmax=15,
        sampling="healpix",
        nside=512,
        batch_size=64,
    )
    assert engine == "s2fft"


def test_nothing_exceeds_the_memory_cap():
    """No configuration may auto-select a precomputing engine whose
    footprint exceeds the cap."""
    for nside in (8, 16, 32, 64, 128):
        lmax = 2 * nside - 1
        engine, _ = engine_select.resolve_engine(
            lmax=lmax,
            sampling="healpix",
            nside=nside,
            niter=3,
            batch_size=1024,
        )
        if engine == "dense":
            assert (
                footprints.dense_nbytes(lmax, "healpix", nside=nside)
                <= engine_select.DEFAULT_MEMORY_CAP_BYTES
            )
        elif engine == "kernel":
            from croissant import kernel

            assert (
                kernel.kernel_nbytes(lmax, "healpix", nside=nside)
                <= engine_select.DEFAULT_MEMORY_CAP_BYTES
            )


def test_a_tiny_cap_forces_the_matrix_free_engine():
    engine, _ = engine_select.resolve_engine(
        lmax=63,
        sampling="healpix",
        nside=32,
        niter=3,
        batch_size=1024,
        memory_cap=1024,
    )
    assert engine == "s2fft"


def test_dense_footprint_beats_kernel_only_at_small_nside():
    """The O(nside**4) vs O(nside**3) crossover the policy relies on."""
    from croissant import kernel

    small = footprints.dense_nbytes(
        15, "healpix", nside=8
    ) / kernel.kernel_nbytes(15, "healpix", nside=8)
    large = footprints.dense_nbytes(
        127, "healpix", nside=64
    ) / kernel.kernel_nbytes(127, "healpix", nside=64)
    assert large > small


@pytest.mark.parametrize("engine", ["s2fft", "kernel", "dense"])
def test_explicit_choices_are_returned_unchanged(engine):
    got, reason = engine_select.resolve_engine(
        lmax=63,
        sampling="healpix",
        nside=32,
        requested=engine,
    )
    assert got == engine
    assert "explicit" in reason
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_engine_select.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'croissant.engine_select'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/croissant/engine_select.py`:

```python
"""
Automatic spherical harmonic engine selection.

Croissant's engines compute the same linear map to within ~1e-15, so
which one to use is a question about memory and reuse rather than about
results. That makes it croissant's decision to own: this module encodes
the rule that used to live in prose in the README.

The policy is deliberately conservative. It never selects a precomputing
engine whose footprint exceeds ``memory_cap``, and it falls back to the
matrix-free engine whenever precomputing cannot pay for itself. It is a
policy, not a promise: pin ``engine=`` explicitly to freeze behaviour.

Inputs are only quantities croissant can know at construction time. The
one thing it cannot know is whether an object will be reused across many
later calls, so ``batch_size`` stands in for the amortisation factor --
for ``Beam`` and ``Sky`` that is the number of frequencies.
"""

import math

from .footprints import dense_nbytes, kernel_nbytes, transform_lmax

__all__ = ["DEFAULT_MEMORY_CAP_BYTES", "ENGINES", "resolve_engine"]

DEFAULT_MEMORY_CAP_BYTES = 512 * 1024**2

#: Kernel megabytes that one batched transform can amortise. MEASURED in
#: Task 6, not reasoned: the batch size at which the kernel engine's
#: setup-plus-first call overtakes the matrix-free engine tracks the
#: kernel's own size almost 1:1 -- crossover 1 at nside=8 and nside=16,
#: where the kernel is 0.12 and 0.98 MiB, and crossover 8 at nside=32,
#: where it is 7.94 MiB. Both quantities grow as roughly nside**3.
#: A single fixed batch threshold cannot express this: it would have to be
#: 1 to serve nside=16 and 8 to serve nside=32. Only three resolutions
#: were fitted, so treat the 1:1 coefficient as order-of-magnitude, and
#: re-measure with benchmarks/benchmark_engines.py before changing it.
_MIB_PER_BATCHED_TRANSFORM = 1.0


def _amortisation_threshold(kernel_bytes):
    """
    Smallest batch size that can pay for a kernel of this size.

    Parameters
    ----------
    kernel_bytes : int
        Predicted size of the kernel that would be built.

    Returns
    -------
    int
        Minimum batch size, never below 1.

    """
    mib = kernel_bytes / 1024**2
    return max(1, math.ceil(mib / _MIB_PER_BATCHED_TRANSFORM))

ENGINES = ("s2fft", "kernel", "dense")


def resolve_engine(
    lmax,
    sampling,
    nside=None,
    spin=0,
    niter=0,
    reality=True,
    batch_size=1,
    memory_cap=None,
    requested=None,
):
    """
    Choose a spherical harmonic engine for one configuration.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Sampling scheme understood by s2fft.
    nside : int or None
        HEALPix resolution parameter, required for ``"healpix"``.
    spin : int
        Spin weight of the field.
    niter : int
        Number of iterative refinement steps requested.
    reality : bool
        Whether the field is real.
    batch_size : int
        Number of transforms the choice will be amortised over, i.e. the
        product of the leading batch axes.
    memory_cap : int or None
        Largest precomputed footprint to consider, in bytes. Defaults to
        :data:`DEFAULT_MEMORY_CAP_BYTES`.
    requested : str or None
        An explicit engine name, returned unchanged. ``None`` or
        ``"auto"`` selects automatically.

    Returns
    -------
    tuple of (str, str)
        The engine name and a short human-readable reason, suitable for
        logging or for reporting which mechanism an object chose.

    """
    if requested is not None and requested != "auto":
        if requested not in ENGINES:
            raise ValueError(
                f"Unsupported SHT engine {requested!r}. Supported "
                f"engines are {set(ENGINES)}."
            )
        return requested, f"explicit request for {requested!r}"

    cap = (
        DEFAULT_MEMORY_CAP_BYTES if memory_cap is None else int(memory_cap)
    )
    kernel_bytes = kernel_nbytes(lmax, sampling, nside=nside, reality=reality)
    dense_bytes = dense_nbytes(
        lmax, sampling, nside=nside, spin=spin, reality=reality
    )
    kernel_fits = kernel_bytes <= cap
    dense_fits = dense_bytes <= cap

    # The kernel engine cannot serve a band-limit below the HEALPix
    # L >= 2*nside floor; the dense engine can, by building at the floor
    # and keeping only the requested low-ell rows. That row selection is
    # dense's clearest remaining advantage, so it is checked first.
    needs_row_selection = transform_lmax(
        lmax, sampling, nside=nside
    ) != int(lmax)
    if needs_row_selection:
        if dense_fits:
            return (
                "dense",
                f"lmax={lmax} is below the HEALPix floor for "
                f"nside={nside}; only the dense engine can low-pass in "
                "one step",
            )
        return (
            "s2fft",
            f"lmax={lmax} is below the HEALPix floor for nside={nside} "
            f"and the dense operator needs {dense_bytes >> 20} MiB",
        )

    threshold = _amortisation_threshold(kernel_bytes)
    if batch_size < threshold:
        return (
            "s2fft",
            f"batch of {batch_size} cannot amortise a "
            f"{kernel_bytes / 1024**2:.1f} MiB kernel "
            f"(needs {threshold})",
        )

    # MEASURED, and it contradicts the flop count. Dense wins per call at
    # niter > 0 by 1.29x-6.65x across every configuration benchmarked in
    # Task 6, because its refinement is folded into the cached matrix
    # while the kernel engine pays 2*niter+1 passes. A flop count says
    # the opposite -- dense moves ~1.5*nside times more data per pass --
    # but at nside <= 32 that is swamped by constant factors: dense is one
    # BLAS call with ideal arithmetic intensity, whereas the kernel path
    # is an FFT plus an einsum plus 2*niter+1 sequential round trips
    # through a synthesis step. See conclusion 2 of
    # benchmarks/results/engines-2026-08-13.md. At large nside flops
    # should eventually dominate, but dense does not fit there anyway
    # (~12 GiB at nside=64), so the memory cap excludes it first.
    if niter > 0 and dense_fits:
        return (
            "dense",
            f"niter={niter} folds into a "
            f"{dense_bytes / 1024**2:.1f} MiB operator, measured "
            "cheapest per call",
        )

    if kernel_fits:
        return (
            "kernel",
            f"{kernel_bytes >> 20} MiB kernel amortises over "
            f"{batch_size} transforms"
            + (
                f"; dense would need {dense_bytes >> 20} MiB"
                if not dense_fits
                else ""
            ),
        )

    return (
        "s2fft",
        f"no precomputed engine fits under {cap >> 20} MiB "
        f"(kernel {kernel_bytes >> 20} MiB, dense "
        f"{dense_bytes >> 20} MiB)",
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_engine_select.py -v --no-cov`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/croissant/engine_select.py tests/test_engine_select.py
uv run ruff format src/croissant/engine_select.py tests/test_engine_select.py
git add src/croissant/engine_select.py tests/test_engine_select.py
git commit -m "feat: add an automatic SHT engine selection policy"
```

---

## Task 8: Accept `engine="auto"`, report the resolved engine, update the README

Note the deliberate choice here: `"auto"` is *accepted* but the default stays
`"s2fft"`. Flipping the default changes performance characteristics for every
existing caller, so it wants its own change with its own release note — see
"Deferred" below.

**Files:**
- Modify: `src/croissant/sphere.py` (`compute_alm` signature at 353,
  `SphBase.__init__` validation at ~547 and the `engine` property at ~490)
- Modify: `src/croissant/beam.py` (docstring at 69)
- Modify: `src/croissant/__init__.py`
- Modify: `README.md` (lines 51–55)
- Test: `tests/test_engine_select.py`

**Interfaces:**
- Consumes: `engine_select.resolve_engine` from Task 7.
- Produces: `engine="auto"` accepted everywhere `engine=` is; `SphBase.engine`
  reports the *resolved* name; `SphBase.engine_reason` returns the reason
  string; `croissant.resolve_engine` exported.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_engine_select.py`:

```python
def test_auto_resolves_to_a_concrete_engine_on_the_object():
    """A Beam built with engine="auto" reports the mechanism it chose,
    not the word "auto" -- otherwise performance questions and bug
    reports cannot say which path ran."""
    import numpy as np

    from croissant import Beam

    nside = 8
    rng = np.random.default_rng(11)
    data = rng.normal(size=(2, 12 * nside**2)) ** 2
    beam = Beam(
        data,
        freqs=np.array([50.0, 60.0]),
        sampling="healpix",
        engine="auto",
    )
    assert beam.engine in {"s2fft", "kernel", "dense"}
    assert isinstance(beam.engine_reason, str) and beam.engine_reason


def test_auto_agrees_with_every_explicit_engine():
    """auto cannot change results, only cost."""
    import numpy as np

    from croissant import sphere

    nside, lmax = 8, 15
    rng = np.random.default_rng(12)
    data = rng.normal(size=(4, 12 * nside**2))
    kwargs = dict(lmax=lmax, sampling="healpix", nside=nside, niter=0)
    auto = np.asarray(sphere.compute_alm(data, engine="auto", **kwargs))
    for engine in ("s2fft", "kernel", "dense"):
        got = np.asarray(
            sphere.compute_alm(data, engine=engine, **kwargs)
        )
        np.testing.assert_allclose(
            auto, got, rtol=0, atol=1e-10 * np.abs(got).max()
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_select.py -k auto -v --no-cov`
Expected: FAIL with `ValueError: Unsupported SHT engine 'auto'.`

- [ ] **Step 3: Write the minimal implementation**

In `src/croissant/sphere.py`, at the top of `compute_alm`, before the existing
engine dispatch:

```python
    if engine == "auto":
        from .engine_select import resolve_engine

        spatial_ndim = 1 if sampling == "healpix" else 2
        batch_size = int(
            np.prod(jnp.asarray(data).shape[:-spatial_ndim], dtype=int)
        )
        engine, _ = resolve_engine(
            lmax,
            sampling,
            nside=nside,
            spin=spin,
            niter=niter,
            reality=reality,
            batch_size=batch_size,
        )
```

In `SphBase`, add a static field beside `_engine` and set it during `__init__`,
replacing the current validation block:

```python
    _engine_reason: str = eqx.field(static=True)
```

```python
        from .engine_select import resolve_engine

        batch_size = int(np.prod(jnp.asarray(data).shape[:1], dtype=int))
        engine, engine_reason = resolve_engine(
            lmax if lmax is not None else 0,
            sampling,
            niter=niter,
            batch_size=batch_size,
            requested=engine,
        )
        self._engine = engine
        self._engine_reason = engine_reason
```

Note `resolve_engine` raises on unknown names, so the previous explicit
`if engine not in {...}` check is now redundant and should be removed. Because
`lmax` and `nside` are only known after the existing shape inference, move this
block to just after `self.lmax` and `self.nside` are assigned, and pass the real
values rather than the placeholder above:

```python
        engine, engine_reason = resolve_engine(
            self.lmax,
            self.sampling,
            nside=self.nside,
            niter=self._niter,
            batch_size=int(self.data.shape[0]),
            requested=engine,
        )
        self._engine = engine
        self._engine_reason = engine_reason
```

Add the reporting property beside the existing `engine` property:

```python
    @property
    def engine_reason(self):
        """Why the configured engine was chosen (see engine_select)."""
        return self._engine_reason
```

In `src/croissant/beam.py`, update the docstring at line 69:

```python
        engine : {"auto", "s2fft", "kernel", "dense"}
            Spherical harmonic transform engine. Default is ``"s2fft"``.
            ``"auto"`` lets croissant choose from the band-limit,
            sampling, niter and batch size; the choice is reported by the
            ``engine`` and ``engine_reason`` attributes.
```

In `src/croissant/__init__.py`:

```python
from .engine_select import resolve_engine
```

and add `engine_select,` to the submodule import list.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine_select.py -v --no-cov`
Expected: PASS (12 tests)

- [ ] **Step 5: Replace the prose rule in the README**

In `README.md`, replace lines 51–55 (the "In CPU benchmarks…" paragraph, whose
`engine="dense"` recommendation for `niter > 0` is superseded — see below) with
the following. Paste the measured table from
`benchmarks/results/engines-<date>.md` into the marked block; do not ship the
placeholder comment.

````markdown
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

<!-- Replace this block with the table produced by Task 6. -->

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
print(beam.engine)         # e.g. "kernel"
print(beam.engine_reason)  # e.g. "16 MiB kernel amortises over 64 transforms"
```

`"auto"` is a policy, not a promise: it may change between versions. Pin
`engine=` explicitly to freeze behaviour, and prefer an explicit engine when you
want the dense operator itself (for a Fisher or gram matrix, or an explicit
Jacobian), when you have a memory budget croissant cannot see, or when you know
about reuse it cannot see — for example the same `Beam` driving many thousands of
likelihood evaluations, where the batch size understates the amortisation.
````

- [ ] **Step 6: Run the full suite, lint, commit**

```bash
uv run pytest -q
uv run ruff check
uv run ruff format --check
git add -A
git commit -m "feat: accept engine=\"auto\" and report the resolved engine"
```

Expected: all tests pass, including `tests/test_physics.py` unchanged.

---

## Testing strategy: why the physics tests are not parametrised

`tests/test_physics.py` is deliberately left alone, and engine coverage is
factored differently.

Engine choice affects exactly one function, `compute_alm`. Everything
downstream — the `"flm,tm,flm->tf"` convolve einsum, the `exp(-i m phi(t))`
phase rotation, the beam-integral normalisation — is engine-independent. So once
the alm agree (Task 1, and Task 5 for `kernel`), every physics invariant must
agree too: parametrising the physics tests would verify a corollary of the
equivalence theorem, not an independent property.

The cost is real: 22 physics tests take 46.7 s, so three engines would add ~93 s
to a ~6 min suite. And CLAUDE.md makes the physics file ground-truth that is not
to be modified — adding `parametrize` decorators is a modification, and churn in
the invariant anchor is itself a cost.

What transform-level equivalence structurally *cannot* catch is covered by two
targeted tests instead:

- `test_kernel_engine_follows_the_dtype_contract` (Task 3) — the engines share
  the dtype policy documented in `sphere._dense_dtypes`, and the alm-value tests
  all compare at x64, so a dtype divergence would slip through.
- `test_full_pipeline_visibilities_agree_across_engines` (Task 5) — one
  end-to-end `Beam`/`Sky`/`Simulator` run per engine, catching tracer/jit
  interaction (the dense engine raises inside `jax.jit` unless precomputed,
  while the kernel engine builds lazily) and cache side effects.

Known gap, pre-existing and out of scope: `conftest.py` enables x64 globally, so
no test exercises the complex64 path that `_dense_dtypes` describes for
x64-disabled runtimes. The dtype test above pins the contract's *shape* but only
at x64.

## Deferred (not in this plan)

- **Making `"auto"` the default.** Every caller's performance profile would
  change, so it deserves its own commit, its own release note, and a decision on
  whether it is minor or breaking. The machinery lands here; flipping the
  default is a one-line follow-up.
- **`kernel` support for the low-lmax/high-nside case.** Handled by routing,
  not by capability: `kernel_compute_alm` raises below the floor and
  `resolve_engine` sends those configurations to `dense`. Teaching the kernel
  engine to build at the floor and slice the `ell` block afterwards would remove
  dense's last structural advantage, but it is not needed for this plan.
- **Dense engine unification (item 2 of the follow-up queue).** Still deferred,
  and this plan *shrinks* it. The headline piece was replacing the 173 s
  VJP-pullback builder with direct spin-harmonic construction to make polarized
  HEALPix affordable — but `auto` will now route that case to the kernel engine
  at 2.4 s and 15.75 MiB, so the slow builder stops being on the hot path. What
  remains is hygiene rather than performance: one bounded cache instead of
  `sphere.py`'s unbounded dict plus `dense.py`'s `lru_cache(6)`;
  `polarization.py:_analysis_alm` routing spin-0 real blocks through the full
  complex operator (2x waste); the build-loop `.at[].set()` churn; and
  `jax.jit` vs house-style `eqx.filter_jit`. Worth doing, no longer urgent.
- **Wall-clock validation of the policy.** The FLOP argument above says the
  kernel engine should beat dense per call at any realistic nside, including at
  `niter > 0`. FLOPs are not wall time: a dense matmul is one BLAS call with
  ideal arithmetic intensity, while the kernel path is an FFT plus an einsum with
  worse access patterns and, at `niter > 0`, `2*niter+1` sequential round trips
  through a synthesis step. Benchmark the two before trusting the policy at
  `niter > 0`, and adjust `_AMORTISATION_THRESHOLD` and the niter branch from
  measurement rather than from the FLOP count.
- **Reporting the choice through `Sky` and the polarized classes.** Task 8 adds
  `engine_reason` to `SphBase`, which they inherit, but the polarized wrappers
  may want to surface it explicitly.

## Self-Review

**Spec coverage.** The requested pieces each have tasks: cross-engine `niter>0`
equivalence (Task 1, extended in Task 5), the kernel engine (Tasks 2–5),
benchmarks proving the cost claims (Task 6), `auto` (Tasks 7–8), and the README
engine guide (Task 8 Step 5). The Background's two hard constraints —
numpy builder only, never delegate `iter>0` — appear in Global Constraints, in
`kernel.py`'s module docstring, as an inline comment at the `iter=0` call site,
and as an executable guard in
`test_kernel_matches_on_the_fly_transform`.

**Placeholder scan.** No TBDs. Every code step carries the literal code; every
test step names the command and the expected outcome. Task 1 is the one step
expected to pass rather than fail, and says so along with what to do if it does
not.

**Type consistency.** `precompute_kernel(lmax, sampling, nside, spin, reality, forward)`
is called with exactly those keywords in Tasks 3 and 4. `kernel_compute_alm`'s
signature matches `sphere.compute_alm`'s parameter names and its
`batch + (lmax+1, 2*lmax+1)` return layout, checked explicitly in Task 3.
`resolve_engine` returns `(str, str)` everywhere, and its `requested=` path is
the single validator for engine names after Task 8 removes the duplicated set
literal in `SphBase`.
