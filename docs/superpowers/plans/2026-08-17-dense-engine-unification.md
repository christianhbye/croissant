# Dense Engine Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `croissant.dense` sole ownership of the dense SHT engine — both
builders, one cache, one key schema — and return `croissant.sphere` to engine
dispatch.

**Architecture:** A pure relocation first, then policy. `sphere.py`'s
packed-real builder, cache, key helper and apply function move into `dense.py`
alongside the existing VJP builder. The two caches then collapse into one
unbounded dict whose key carries `spin` and `packed`, so the two operator
flavours cannot collide. No numerical behaviour changes anywhere.

**Tech Stack:** Python 3.11–3.13, JAX (`jnp`), Equinox (`eqx.Module`,
`eqx.filter_jit`), s2fft, SciPy (`special.sph_harm_y`), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-17-dense-engine-unification-design.md`

## Global Constraints

- Line length 79 characters, ruff enforced (`uv run ruff check`, `uv run ruff format`).
- Ruff lint rules: E, F, W, I. Imports must be isort-ordered.
- NumPy-style docstrings on every public function.
- Use `jnp` for array operations, never `numpy`, except in host-side builders
  that deliberately run on the CPU before device transfer.
- `eqx.field(static=True)` for non-traced fields on `eqx.Module` classes.
- Float comparisons in tests use `np.testing.assert_allclose`.
- Test timeout is 120 s per test. Keep new tests at `nside=2`, `lmax<=4`.
- `tests/test_physics.py` is ground truth. It must end at **0 diff vs main**.
  If it fails, the change is wrong — fix the change, never the test.
- Every task ends green and committed, so the branch stays bisectable.
- Branch is `dense-engine-unification`, already created, spec committed at
  `82868c2`.

**Note on commit granularity:** the spec's risk table proposed two commits
(relocation, then policy). This plan refines that into five, each independently
green. Same intent — a large move must not obscure a real change — at finer
grain.

---

## Baseline

Before Task 1, record the starting point. Later tasks compare against it.

- [ ] **Step 0: Record the baseline**

```bash
git status                      # expect: clean, on dense-engine-unification
uv run pytest -q 2>&1 | tail -5 # record the "N passed" number
uv run ruff check && uv run ruff format --check
```

Write the passing count into the PR notes. Every task must match or exceed it
(tasks add tests, never remove them).

---

## File Structure

| File | Responsibility after this plan |
| --- | --- |
| `src/croissant/utils.py` | gains `engine_dtypes()` — the engine-wide output dtype contract |
| `src/croissant/dense.py` | the whole dense engine: both builders, one cache, both public entry points, the packed apply |
| `src/croissant/sphere.py` | engine dispatch only: `_compute_alm_s2fft`, `compute_alm`, `SphBase` |
| `src/croissant/kernel.py` | unchanged except its dtype import moves to module level |
| `src/croissant/polarization.py` | one call site retargeted |
| `src/croissant/__init__.py` | re-points two public names to `.dense` |
| `tests/test_dense.py` | **new** — cache policy and builder tests added here |
| `tests/test_sphere.py` | unchanged except its imports; the existing dense tests stay put so the move's diff stays readable |
| `README.md` | `clear_dense_matrix_cache` gets a real paragraph |

---

## Task 1: Move the dtype contract to `utils.engine_dtypes`

`sphere._dense_dtypes` has three consumers, and `kernel.py:265` imports it
lazily with the comment "Imported lazily because sphere imports this module".
Moving it to `utils.py` retires that workaround and unblocks Task 2, which
would otherwise make the kernel engine import from the dense engine.

**Files:**
- Modify: `src/croissant/utils.py` (add `engine_dtypes`, add `import jax`)
- Modify: `src/croissant/sphere.py:16-26` (delete `_dense_dtypes`), `:31`, `:102`, `:200` (call sites)
- Modify: `src/croissant/kernel.py:259-268` (module-level import, update comment)
- Modify: `tests/test_kernel_engine.py:110`, `:217` (docstring references)
- Test: `tests/test_utils.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `utils.engine_dtypes() -> tuple[jnp.dtype, jnp.dtype]`, returning
  `(real_dtype, complex_dtype)`. Exact behaviour is unchanged from
  `sphere._dense_dtypes`: `(float64, complex128)` when `jax.config.x64_enabled`
  is True, `(float32, complex64)` otherwise.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_utils.py`:

```python
def test_engine_dtypes_matches_s2fft_output():
    """The dtype contract must track what s2fft actually returns.

    This is the whole reason the helper exists: the dense matrix
    precision follows JAX's x64 setting rather than the dtype of the
    maps it is applied to, because that is what s2fft.forward does.
    """
    real_dtype, complex_dtype = utils.engine_dtypes()
    nside = 2
    maps = jnp.zeros((12 * nside**2,), dtype=real_dtype)
    alm = s2fft.forward(
        maps,
        L=5,
        nside=nside,
        sampling="healpix",
        method="jax",
        reality=True,
    )
    assert alm.dtype == np.dtype(complex_dtype)
```

`tests/test_utils.py` must import `jax.numpy as jnp`, `numpy as np`, `s2fft`
and `from croissant import utils` if it does not already.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_utils.py::test_engine_dtypes_matches_s2fft_output -v`
Expected: FAIL with `AttributeError: module 'croissant.utils' has no attribute 'engine_dtypes'`

- [ ] **Step 3: Add `engine_dtypes` to `utils.py`**

Add `import jax` to the import block (`utils.py` currently imports
`jax.numpy as jnp` but not `jax` itself). Then add:

```python
def engine_dtypes():
    """
    Return the real and complex dtypes croissant's SHT engines produce.

    Every engine reproduces ``s2fft.forward``, which returns complex128
    alms on an x64-enabled runtime (float32 maps included) and complex64
    otherwise. Engine precision therefore follows JAX's x64 setting
    rather than the dtype of the input maps, and a cached transform built
    before ``jax.config.update("jax_enable_x64", True)`` must not be
    reused afterwards at the earlier, reduced precision — which is why
    both the dense and kernel caches stamp this into their keys.

    Returns
    -------
    real_dtype : jnp.dtype
        Real dtype matching the engines' precision.
    complex_dtype : jnp.dtype
        Complex dtype the engines' coefficients carry.

    """
    if jax.config.x64_enabled:
        return jnp.float64, jnp.complex128
    return jnp.float32, jnp.complex64
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_utils.py::test_engine_dtypes_matches_s2fft_output -v`
Expected: PASS

- [ ] **Step 5: Delete `_dense_dtypes` and update all three consumers**

In `sphere.py`, delete the `_dense_dtypes` function (lines 16-26) and replace
its three uses:

- `:31` in `_dense_matrix_key`: `_, complex_dtype = _dense_dtypes()`
  becomes `_, complex_dtype = utils.engine_dtypes()`
- `:102` in `_build_dense_matrix_healpix`: same substitution
- `:200` in `_build_dense_matrix_from_pixels`:
  `real_dtype, _ = _dense_dtypes()` becomes
  `real_dtype, _ = utils.engine_dtypes()`

`sphere.py` already has `from . import utils` at module level.

In `kernel.py`, replace the lazy import (lines 265-268):

```python
    # Croissant's engines share a dtype contract, owned and documented by
    # utils.engine_dtypes: they reproduce s2fft.forward, which returns
    # complex128 on an x64 runtime regardless of the input map dtype.
    real_dtype, _ = utils.engine_dtypes()
```

`kernel.py` already has `from . import utils` at module level, so delete the
`from .sphere import _dense_dtypes` line and the comment explaining why it was
lazy.

In `tests/test_kernel_engine.py`, update the two docstring references:
`:110` `sphere._dense_matrix_key` stays as-is (that helper has not moved yet;
Task 2 handles it), and `:217` `sphere._dense_dtypes` becomes
`utils.engine_dtypes`.

- [ ] **Step 6: Verify nothing else references the old name**

Run: `grep -rn "_dense_dtypes" src/ tests/ benchmarks/`
Expected: no output.

- [ ] **Step 7: Run the full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -5 && uv run ruff check && uv run ruff format --check`
Expected: baseline count + 1 passed, 0 failed, ruff clean.

- [ ] **Step 8: Confirm the physics tests are untouched**

Run: `git diff --stat main -- tests/test_physics.py`
Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add src/croissant/utils.py src/croissant/sphere.py \
        src/croissant/kernel.py tests/test_utils.py \
        tests/test_kernel_engine.py
git commit -m "refactor: move the engine dtype contract into utils

sphere._dense_dtypes had three consumers, not two: kernel.py imported it
lazily with a comment explaining the import had to dodge a cycle, since
sphere imports kernel. The name also misdescribed a helper the kernel
engine depends on.

utils.engine_dtypes is the same function in a module both consumers
already import at module level, so the lazy import goes away. Rejected
footprints.py, which promises pure arithmetic over transform geometry
and imports no jax; this reads jax.config.x64_enabled."
```

---

## Task 2: Relocate the dense engine into `dense.py`

The pure move. No policy change, no numerical change. The only rewrite is
forced by the move: the equiangular builder currently calls
`sphere._compute_alm_s2fft`, and importing that from `dense.py` would invert
the module dependency.

**Files:**
- Modify: `src/croissant/dense.py` (receives ~365 lines)
- Modify: `src/croissant/sphere.py:12-13`, `:29-51`, `:86-428` (removals), `:605`, `:614`, `:802` (call sites)
- Modify: `src/croissant/polarization.py:380`
- Modify: `src/croissant/__init__.py:21`
- Modify: `tests/test_sphere.py:11-17`
- Modify: `benchmarks/benchmark_dense_healpix.py:14-15`, `benchmarks/benchmark_engines.py:143`
- Test: `tests/test_dense.py` (new)

**Interfaces:**
- Consumes: `utils.engine_dtypes()` from Task 1.
- Produces, all in `croissant.dense`:
  - `precompute_dense_matrix(spatial_shape, lmax, sampling, nside=None, niter=0, chunk_size=None) -> jax.Array`
  - `dense_matrix_for(spatial_shape, lmax, sampling, nside=None, niter=0, *, tracing, explicit) -> jax.Array`
  - `clear_dense_matrix_cache() -> None`
  - `apply_packed_matrix(data, matrix, lmax, spatial_ndim=None) -> jax.Array`
    (renamed from `sphere._apply_dense_matrix`; D4 — symbols crossing module
    boundaries lose the leading underscore)
  - `_DENSE_MATRIX_CACHE`, `_dense_matrix_key`, `_positive_lm_indices`,
    `_build_dense_matrix`, `_build_dense_matrix_healpix`,
    `_build_dense_matrix_from_pixels` stay private to `dense.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dense.py`. This test pins the one behaviour the move could
silently break — the equiangular builder's `reality=True` packing, which is
what makes its output the packed operator by construction:

```python
"""Tests for the dense SHT engine: builders, cache and apply."""

import jax.numpy as jnp
import numpy as np
import s2fft

from croissant import dense
from croissant.footprints import spatial_shape

rng = np.random.default_rng(seed=0)


def test_equiangular_builder_produces_the_packed_operator():
    """The one-hot builder must keep s2fft's reality=True packing.

    The builder pushes one-hot pixel maps through a real forward
    transform and keeps only m >= 0. Losing reality=True would still
    produce a matrix of the right shape, so the guard has to compare
    values against s2fft directly.
    """
    lmax, sampling = 4, "dh"
    shape = spatial_shape(lmax, sampling, None)
    matrix = dense.precompute_dense_matrix(shape, lmax, sampling)

    ncoeff = (lmax + 1) * (lmax + 2) // 2
    assert matrix.shape == (ncoeff, int(np.prod(shape)))

    maps = jnp.asarray(rng.standard_normal(shape))
    expected = s2fft.forward(
        maps,
        L=lmax + 1,
        sampling=sampling,
        method="jax",
        reality=True,
    )
    ell, emm = dense._positive_lm_indices(lmax)
    packed = matrix @ maps.reshape(-1)
    np.testing.assert_allclose(
        packed, expected[ell, lmax + emm], rtol=1e-12, atol=1e-12
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dense.py -v`
Expected: FAIL with `AttributeError: module 'croissant.dense' has no attribute 'precompute_dense_matrix'`

- [ ] **Step 3: Move the code**

Cut from `sphere.py` and paste into `dense.py`, in this order, after the
existing imports:

| From `sphere.py` | Notes |
| --- | --- |
| `:12-13` cache globals | needs `from threading import RLock` in `dense.py` |
| `:29-40` `_dense_matrix_key` | unchanged |
| `:43-51` `_positive_lm_indices` | unchanged |
| `:86-187` `_build_dense_matrix_healpix` | unchanged |
| `:190-234` `_build_dense_matrix_from_pixels` | see Step 4 |
| `:237-261` `_build_dense_matrix` | unchanged |
| `:264-318` `precompute_dense_matrix` | unchanged |
| `:321-397` `dense_matrix_for` | update the docstring's `croissant.dense.DenseSphericalTransform` cross-reference, now same-module |
| `:400-403` `clear_dense_matrix_cache` | unchanged |
| `:406-428` `_apply_dense_matrix` | rename to `apply_packed_matrix` |

`dense.py`'s import block becomes:

```python
from functools import partial
from threading import RLock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import s2fft

from . import utils
from .footprints import spatial_shape as _spatial_shape
from .footprints import transform_lmax
```

`from functools import lru_cache` becomes `from functools import partial` —
`lru_cache` is still used by `_build_analysis_matrix` at this point, so keep
**both** imports until Task 3 removes it:
`from functools import lru_cache, partial`.

- [ ] **Step 4: Replace the `sphere._compute_alm_s2fft` call**

`_build_dense_matrix_from_pixels` calls `_compute_alm_s2fft`, which stays in
`sphere.py` as the s2fft engine. Add this private helper to `dense.py` instead:

```python
@partial(eqx.filter_jit, inline=True)
def _forward_real_chunk(basis, lmax, sampling, nside, niter):
    """Forward-transform a batch of real maps with the s2fft engine.

    A local copy of sphere._compute_alm_s2fft's inner call, specialised
    to what the builder needs: one leading batch axis, spin 0, real
    input. Duplicated rather than imported because dense.py must not
    import sphere.py -- sphere dispatches to this module, and a
    module-level import back would be a cycle.
    """
    m2alm = partial(
        s2fft.forward,
        L=lmax + 1,
        spin=0,
        nside=nside,
        sampling=sampling,
        method="jax",
        # The one-hot basis maps are real and the caller keeps only
        # m >= 0, so this is what makes the result the packed operator
        # by construction. See test_equiangular_builder_produces_the_
        # packed_operator.
        reality=True,
        precomps=None,
        spmd=False,
        L_lower=0,
        iter=niter,
    )
    return jax.vmap(m2alm)(basis)
```

Then in `_build_dense_matrix_from_pixels`, replace:

```python
        dense = _compute_alm_s2fft(
            basis,
            lmax,
            sampling,
            nside=nside,
            niter=niter,
            reality=True,
        )
```

with:

```python
        dense = _forward_real_chunk(basis, lmax, sampling, nside, niter)
```

and delete the four-line comment above the old `reality=True` argument, which
now lives in the helper.

- [ ] **Step 5: Update call sites**

- `sphere.py:605`: `dense_matrix_for(` becomes `dense.dense_matrix_for(`
- `sphere.py:614`: `_apply_dense_matrix(data, dense_matrix, lmax, spatial_ndim)`
  becomes `dense.apply_packed_matrix(data, dense_matrix, lmax, spatial_ndim)`
- `sphere.py:802`: `dense_matrix_for(` becomes `dense.dense_matrix_for(`
- `sphere.py`: add `from . import dense` to the module-level import block, and
  delete the lazy `from . import dense as _dense` at `:593`, replacing
  `_dense.dense_compute_alm(` at `:595` with `dense.dense_compute_alm(`. After
  Step 4 `dense.py` no longer imports `sphere.py`, so there is no cycle left to
  dodge.
- `polarization.py:380`: `sphere.dense_matrix_for(` becomes
  `dense.dense_matrix_for(`. `polarization.py` already imports `dense`.
- `__init__.py:21`: `from .sphere import clear_dense_matrix_cache, precompute_dense_matrix`
  becomes `from .dense import clear_dense_matrix_cache, precompute_dense_matrix`,
  merged into the existing `from .dense import ...` line and isort-ordered.

- [ ] **Step 6: Update tests and benchmarks**

`tests/test_sphere.py:11-17` — drop the moved names from the `croissant.sphere`
import and add a `croissant.dense` import:

```python
from croissant.dense import (
    _DENSE_MATRIX_CACHE,
    clear_dense_matrix_cache,
    precompute_dense_matrix,
)
from croissant.sphere import SphBase, compute_alm
```

The dense tests already in `tests/test_sphere.py` (`:339-347`, `:408-423`,
`:426-...`) stay where they are. Moving them as well would double this task's
diff and make the relocation harder to review; new tests go in
`tests/test_dense.py`.

`tests/test_kernel_engine.py:110` refers to `sphere._dense_matrix_key` in a
docstring. Update it to `dense._dense_matrix_key`.

`benchmarks/benchmark_dense_healpix.py:13-16` — both names already come from
`croissant.dense`, so this task leaves it alone. (Its `cache_info()` call at
`:46` breaks in Task 3, which handles it there.)

`benchmarks/benchmark_engines.py:143` — `sphere.clear_dense_matrix_cache()`
becomes `dense.clear_dense_matrix_cache()`; add the import.

Nothing imports `benchmarks/`, so the suite cannot catch a mistake here — check
by inspection, and run each touched benchmark's `--help` at minimum.

- [ ] **Step 7: Run the new test**

Run: `uv run pytest tests/test_dense.py -v`
Expected: PASS

- [ ] **Step 8: Verify the move is complete and the cycle is gone**

```bash
grep -n "_DENSE_MATRIX_CACHE\|_build_dense_matrix\|precompute_dense_matrix" src/croissant/sphere.py
grep -n "^from .sphere\|^from croissant.sphere\|import sphere" src/croissant/dense.py
```
Expected: no output from either.

- [ ] **Step 9: Run the full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -5 && uv run ruff check && uv run ruff format --check`
Expected: baseline + 2 passed, 0 failed, ruff clean.

This is the step that matters most in this task. `tests/test_engine_equivalence.py`
and `tests/test_polarization.py:575-580` (the warm-then-jit `RuntimeError`
contract) are the real regression gates for a move this size.

- [ ] **Step 10: Confirm the physics tests are untouched**

Run: `git diff --stat main -- tests/test_physics.py`
Expected: no output.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "refactor: give dense.py sole ownership of the dense engine

sphere.py owned a packed-real builder and cache because #123 added the
dense engine there for scalar real HEALPix; dense.py appeared in #124
only because spin +-2 needed a builder the packed one structurally could
not provide. The result was a dispatcher that also owned half an engine.

Pure relocation: both builders, the cache, the key helper and the packed
apply move to dense.py. sphere.py keeps _compute_alm_s2fft, compute_alm
and SphBase. No numerical change.

One rewrite was forced by the move. The equiangular builder called
sphere._compute_alm_s2fft, and importing that back would recreate the
cycle this removes, so it now calls s2fft.forward through a local
helper. reality=True is what makes its output the packed operator by
construction, so a test pins it against s2fft directly."
```

---

## Task 3: Unify the two caches

Replace `lru_cache(maxsize=6)` with the relocated dict, and extend the key so
the two operator flavours cannot collide.

**Files:**
- Modify: `src/croissant/dense.py`
- Test: `tests/test_dense.py`

**Interfaces:**
- Consumes: everything Task 2 produced.
- Produces:
  - `_dense_matrix_key(spatial_shape, lmax, sampling, nside, spin, packed, niter, complex_dtype)` — extended signature
  - `_build_analysis_matrix(lmax, sampling, nside, spin, niter, complex_dtype_name) -> jax.Array`
    — no longer `lru_cache`d, no `device_key`, returns **only** the matrix
  - `_full_matrix_for(lmax, sampling, nside, spin, niter, complex_dtype) -> jax.Array`
    — cache lookup wrapper for the VJP flavour

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_dense.py`:

```python
def test_clear_releases_both_operator_flavours():
    """One clear function must empty the whole engine's cache.

    Before unification clear_dense_matrix_cache reached only the packed
    half; the VJP half was reachable only through an lru_cache's own
    cache_clear, which no public name exposed.
    """
    lmax, nside, npix = 4, 2, 48
    dense.clear_dense_matrix_cache()
    dense.precompute_dense_matrix((npix,), lmax, "healpix", nside=nside)
    dense.dense_compute_alm(
        jnp.zeros((1, npix)), lmax, "healpix", nside=nside, spin=2
    )
    assert len(dense._DENSE_MATRIX_CACHE) == 2

    dense.clear_dense_matrix_cache()
    assert len(dense._DENSE_MATRIX_CACHE) == 0


def test_packed_and_full_operators_do_not_collide():
    """Identical geometry, two flavours, two entries.

    Both are spin 0 at the same lmax, sampling, nside and niter. Only
    the packed flag separates them, so a key that omitted it would
    return the m >= 0 operator to a caller expecting the full one.
    """
    lmax, nside, npix = 4, 2, 48
    dense.clear_dense_matrix_cache()
    packed = dense.precompute_dense_matrix(
        (npix,), lmax, "healpix", nside=nside
    )
    dense.dense_compute_alm(
        jnp.zeros((1, npix)), lmax, "healpix", nside=nside, spin=0
    )

    assert len(dense._DENSE_MATRIX_CACHE) == 2
    shapes = {m.shape for m in dense._DENSE_MATRIX_CACHE.values()}
    ncoeff_packed = (lmax + 1) * (lmax + 2) // 2
    assert packed.shape == (ncoeff_packed, npix)
    assert shapes == {(ncoeff_packed, npix), ((lmax + 1) ** 2, npix)}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dense.py -v -k "collide or releases"`
Expected: FAIL — `test_clear_releases_both_operator_flavours` asserts 2 entries
but finds 1, because the VJP matrix still lives in the `lru_cache`.

- [ ] **Step 3: Extend the key**

```python
def _dense_matrix_key(
    spatial_shape, lmax, sampling, nside, spin, packed, niter, complex_dtype
):
    """Return a hashable key for a cached dense SHT analysis matrix.

    ``packed`` separates the m >= 0 real operator from the full complex
    one. Both flavours exist at spin 0 and identical geometry, so
    without it a caller asking for one could be handed the other.

    Keys on ``jax.default_backend()`` rather than a device string, which
    is what sphere's half and ``kernel.precompute_kernel`` already do.
    Two devices on one backend therefore share an entry, costing a
    transfer rather than correctness.
    """
    return (
        tuple(spatial_shape),
        int(lmax),
        str(sampling),
        None if nside is None else int(nside),
        int(spin),
        bool(packed),
        int(niter),
        np.dtype(complex_dtype).str,
        jax.default_backend(),
    )
```

Update the two packed call sites in `precompute_dense_matrix` and
`dense_matrix_for` to pass `spin=0, packed=True` and
`complex_dtype` from `utils.engine_dtypes()[1]`.

- [ ] **Step 4: Un-cache the VJP builder and route it through the dict**

Delete the `@lru_cache(maxsize=6)` decorator, the `device_key=None` parameter
and the `del device_key` line. Change the return to the matrix alone:

```python
def _build_analysis_matrix(
    lmax, sampling, nside, spin, niter, complex_dtype_name
):
    """Materialize selected rows of corrected s2fft's linear operator."""
```

...body unchanged through the build loop...

```python
    # JAX's holomorphic VJP uses the complex transpose convention, so each
    # pulled-back coefficient basis vector is already one analysis row.
    return matrix
```

`ell_indices`, `m_indices` and `spatial_shape` are pure functions of
`(lmax, spin)` and `(lmax, sampling, nside)`, so callers recompute them rather
than caching them alongside a matrix that may be gigabytes. Add the lookup
wrapper:

```python
def _full_matrix_for(lmax, sampling, nside, spin, niter, complex_dtype):
    """Fetch the full complex operator for one configuration."""
    shape = _spatial_shape(lmax, sampling, nside)
    key = _dense_matrix_key(
        shape, lmax, sampling, nside, spin, False, niter, complex_dtype
    )
    with _DENSE_MATRIX_CACHE_LOCK:
        matrix = _DENSE_MATRIX_CACHE.get(key)
        if matrix is None:
            matrix = _build_analysis_matrix(
                lmax, sampling, nside, spin, niter,
                np.dtype(complex_dtype).name,
            )
            _DENSE_MATRIX_CACHE[key] = matrix
    return matrix
```

- [ ] **Step 5: Update `DenseSphericalTransform.__init__`**

Replace the `_build_analysis_matrix` call block (`dense.py:125-137`) with:

```python
        with jax.ensure_compile_time_eval():
            matrix = _full_matrix_for(
                int(lmax),
                str(sampling),
                None if nside is None else int(nside),
                int(spin),
                int(niter),
                dtype,
            )
        ell_indices, m_indices = _valid_lm_indices(int(lmax), int(spin))
        spatial_shape = _spatial_shape(
            int(lmax), str(sampling), None if nside is None else int(nside)
        )
```

The `device_key = str(jnp.empty((), dtype=jnp.uint8).device)` line goes away
with the `lru_cache` it existed to salt.

- [ ] **Step 6: Fix the benchmark that used the `lru_cache` API**

`benchmarks/benchmark_dense_healpix.py:46` prints
`_build_analysis_matrix.cache_info()`, which no longer exists. Replace with:

```python
    from croissant.dense import _DENSE_MATRIX_CACHE

    print(f"cache_entries={len(_DENSE_MATRIX_CACHE)}")
```

and drop `_build_analysis_matrix` from the import block at `:13-16`, which
leaves only `DenseSphericalTransform`. Nothing imports `benchmarks/`, so verify
by running it rather than by relying on the suite:
`uv run python benchmarks/benchmark_dense_healpix.py` — note this builds the
nside=32 operators and takes several minutes.

- [ ] **Step 7: Run the new tests**

Run: `uv run pytest tests/test_dense.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Run the full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -5 && uv run ruff check && uv run ruff format --check`
Expected: baseline + 4 passed, 0 failed, ruff clean.

Watch `tests/test_sphere.py::test_dense_general_operator_accepts_single_precision`
specifically: it pins that the full operator's precision follows the **map**
dtype while the packed one follows x64. The unified key must carry the dtype
actually used for each flavour, not one global answer.

- [ ] **Step 9: Confirm the physics tests are untouched**

Run: `git diff --stat main -- tests/test_physics.py`
Expected: no output.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: one cache for both dense operator flavours

The engine held two caches with policies neither chose against the
other: an unbounded dict for the small, ~20s-to-rebuild packed
matrices, and lru_cache(maxsize=6) for the ~197s VJP ones. The policy
was inverted relative to cost, and clear_dense_matrix_cache reached
only the first.

One dict now serves both, with spin and packed in the key so the two
flavours cannot collide at identical geometry -- both exist at spin 0.
Keys on jax.default_backend() rather than a device string, matching
the packed half and kernel.precompute_kernel; the device_key parameter
that only existed to salt the lru_cache goes with it.

Retention stays unbounded by decision: unbounded is what makes the
documented warm-then-jit recipe unconditional. The accepted cost is
that the VJP half stops evicting."
```

---

## Task 4: Make retention observable

Task 3's accepted cost is that the VJP half no longer evicts. This does not add
a policy — it makes what is held inspectable, and gives the release valve real
documentation.

**Files:**
- Modify: `src/croissant/dense.py`, `src/croissant/__init__.py`
- Modify: `README.md:179-186`
- Test: `tests/test_dense.py`

**Interfaces:**
- Consumes: `_DENSE_MATRIX_CACHE`, `_DENSE_MATRIX_CACHE_LOCK` from Task 3.
- Produces: `dense.dense_cache_nbytes() -> int`, also exported as
  `croissant.dense_cache_nbytes`.

- [ ] **Step 1: Write the failing test**

```python
def test_dense_cache_nbytes_tracks_both_flavours():
    """Retention is unbounded by design, so it must be inspectable."""
    lmax, nside, npix = 4, 2, 48
    dense.clear_dense_matrix_cache()
    assert dense.dense_cache_nbytes() == 0

    packed = dense.precompute_dense_matrix(
        (npix,), lmax, "healpix", nside=nside
    )
    assert dense.dense_cache_nbytes() == packed.nbytes

    dense.dense_compute_alm(
        jnp.zeros((1, npix)), lmax, "healpix", nside=nside, spin=2
    )
    assert dense.dense_cache_nbytes() > packed.nbytes

    dense.clear_dense_matrix_cache()
    assert dense.dense_cache_nbytes() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dense.py::test_dense_cache_nbytes_tracks_both_flavours -v`
Expected: FAIL with `AttributeError: module 'croissant.dense' has no attribute 'dense_cache_nbytes'`

- [ ] **Step 3: Implement**

```python
def dense_cache_nbytes():
    """
    Total bytes of dense analysis operators croissant currently holds.

    The cache is deliberately unbounded: an eviction policy would make
    the documented precompute-then-jit recipe conditional, since a later
    unrelated build could drop a warmed matrix and leave the next jitted
    explicit-dense call raising. The tradeoff is that retention grows
    with the number of distinct configurations touched -- a band-limit
    sweep at nside=32 over seven values of lmax retains about 904 MiB.
    This reports that figure so it can be watched, and
    :func:`clear_dense_matrix_cache` releases it.

    Returns
    -------
    int
        Size in bytes of every cached operator, both the packed real
        and the full complex flavour.

    """
    with _DENSE_MATRIX_CACHE_LOCK:
        return sum(int(matrix.nbytes) for matrix in
                   _DENSE_MATRIX_CACHE.values())
```

Add `dense_cache_nbytes` to the `from .dense import ...` line in `__init__.py`.

Then upgrade the benchmark line Task 3 patched — a dense-cache benchmark wants
bytes, not an entry count. In `benchmarks/benchmark_dense_healpix.py`:

```python
    from croissant.dense import dense_cache_nbytes

    print(f"cache_mib={dense_cache_nbytes() / 2**20:.3f}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dense.py::test_dense_cache_nbytes_tracks_both_flavours -v`
Expected: PASS

- [ ] **Step 5: Update the README**

`README.md:186` currently mentions `croissant.clear_dense_matrix_cache()` in
passing. Replace that sentence with a paragraph covering both facts:

```markdown
Dense operators are cached for the life of the process and are never
evicted, so that a matrix warmed with `croissant.precompute_dense_matrix`
is always still there when a later `jax.jit`-ed call needs it. The cost is
that retention grows with the number of distinct configurations you touch:
each `lmax`, `niter`, `nside` and sampling combination is a separate entry,
and a band-limit sweep at `nside=32` over seven values of `lmax` retains
about 904 MiB. Call `croissant.dense_cache_nbytes()` to see the current
total and `croissant.clear_dense_matrix_cache()` to release all of it.
```

- [ ] **Step 6: Run the full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -5 && uv run ruff check && uv run ruff format --check`
Expected: baseline + 5 passed, 0 failed, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: report dense cache retention with dense_cache_nbytes

The unified cache is unbounded by decision, so what it holds should at
least be visible. dense_cache_nbytes() sums both operator flavours, and
the README now states the tradeoff outright -- warmed matrices are never
evicted, and a seven-point band-limit sweep at nside=32 retains ~904 MiB
until cleared."
```

---

## Task 5: Build-loop assembly and the jit comments

**Files:**
- Modify: `src/croissant/dense.py`
- Test: `tests/test_dense.py`

**Interfaces:**
- Consumes: `_build_analysis_matrix` from Task 3.
- Produces: `_build_analysis_matrix(..., chunk_size=32)` — new trailing
  keyword parameter, matching `_build_dense_matrix_from_pixels`'s existing
  `chunk_size` convention.

- [ ] **Step 1: Write the failing test**

```python
def test_full_operator_assembly_is_chunk_size_independent():
    """Row batching must not change the assembled operator.

    The builder pulls back coefficient basis vectors in chunks. If
    assembly and chunking are correctly separated, a one-row-at-a-time
    build and a batched one are bitwise identical.
    """
    lmax, spin, nside = 3, 2, 2
    args = (lmax, "healpix", nside, spin, 0, "complex128")
    batched = dense._build_analysis_matrix(*args, chunk_size=32)
    one_at_a_time = dense._build_analysis_matrix(*args, chunk_size=1)
    np.testing.assert_array_equal(batched, one_at_a_time)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dense.py::test_full_operator_assembly_is_chunk_size_independent -v`
Expected: FAIL with `TypeError: _build_analysis_matrix() got an unexpected keyword argument 'chunk_size'`

- [ ] **Step 3: Replace the preallocate-and-set loop**

In `_build_analysis_matrix`, add `chunk_size=32` to the signature and replace
the `jnp.empty` preallocation plus `.at[start:stop].set(...)` loop with a list
and one concatenate — the pattern `_build_dense_matrix_from_pixels` already
uses:

```python
    blocks = []
    for start in range(0, ncoeff, chunk_size):
        stop = min(start + chunk_size, ncoeff)
        coefficient_basis = jax.nn.one_hot(
            jnp.arange(start, stop),
            ncoeff,
            dtype=cotangent_dtype,
        )
        rows = jax.vmap(lambda cotangent: pullback(cotangent)[0])(
            coefficient_basis
        ).reshape(stop - start, -1)
        blocks.append(rows.astype(complex_dtype))
    # JAX's holomorphic VJP uses the complex transpose convention, so each
    # pulled-back coefficient basis vector is already one analysis row.
    return jnp.concatenate(blocks, axis=0)
```

Delete the now-unused `matrix = jnp.empty(...)` block.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dense.py::test_full_operator_assembly_is_chunk_size_independent -v`
Expected: PASS

- [ ] **Step 5: Document why the jit decorators differ**

Per D7 this is a comment, not a conversion — `jax.jit` here would trace
arguments used to build shapes and fail. Add above
`DenseSphericalTransform.__call__`:

```python
    # jax.jit, not eqx.filter_jit: every non-array this method needs is a
    # static eqx.field on self, so the pytree already carries them as
    # static. apply_packed_matrix below cannot do the same -- it takes
    # lmax as a plain int and builds a shape from it.
    @jax.jit
```

and above `apply_packed_matrix`:

```python
# eqx.filter_jit, not jax.jit: lmax arrives as a plain Python int and is
# used to build the output shape, so plain jit would trace it and fail.
# Same reason sphere._compute_alm_s2fft uses filter_jit -- it takes
# sampling as a string.
@partial(eqx.filter_jit, inline=True)
```

- [ ] **Step 6: Run the full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -5 && uv run ruff check && uv run ruff format --check`
Expected: baseline + 6 passed, 0 failed, ruff clean.

- [ ] **Step 7: Confirm the physics tests are untouched**

Run: `git diff --stat main -- tests/test_physics.py`
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: assemble the VJP operator by concatenation

The builder preallocated with jnp.empty and filled by repeated
.at[start:stop].set(), where the sibling builder in the same module
already collected blocks and concatenated once. chunk_size becomes a
parameter so a test can pin that assembly is independent of batching.

Also records why the two jit decorators in this module differ. The
follow-up queue listed that as an inconsistency to resolve; it is
load-bearing in both directions, and converting apply_packed_matrix to
plain jax.jit would trace the lmax it builds a shape from."
```

---

## Final verification

- [ ] **Step 1: Full suite, timed**

Run: `uv run pytest -q 2>&1 | tail -5`
Expected: baseline + 6 passed, 0 failed.

- [ ] **Step 2: Physics tests byte-identical to main**

Run: `git diff --stat main -- tests/test_physics.py`
Expected: no output.

- [ ] **Step 3: Lint clean**

Run: `uv run ruff check && uv run ruff format --check`

- [ ] **Step 4: Acceptance criteria from the spec**

```bash
# 1. sphere.py holds no dense builder, cache or key helper
grep -n "_DENSE_MATRIX_CACHE\|_build_dense_matrix\|_dense_matrix_key" \
     src/croissant/sphere.py            # expect: no output
# 2. one cache, one clear function
grep -rn "lru_cache" src/croissant/dense.py   # expect: no output
# 4. kernel.py imports no code from sphere.py
grep -n "from .sphere import" src/croissant/kernel.py  # expect: no output
```

- [ ] **Step 5: Public API unchanged from the top level**

```bash
uv run python -c "
import croissant
for name in ('precompute_dense_matrix', 'clear_dense_matrix_cache',
             'dense_cache_nbytes', 'DenseSphericalTransform',
             'dense_compute_alm'):
    assert hasattr(croissant, name), name
print('public dense API intact')
"
```

- [ ] **Step 6: Each commit is independently green**

```bash
git rebase --exec 'uv run pytest -q 2>&1 | tail -2' main
```
Expected: every commit reports 0 failed. This is the bisectability gate the
spec's risk table requires.

- [ ] **Step 7: Report to Christian**

Summarise: passing count vs baseline, physics diff, ruff status, and the two
behaviour changes that need his eye before any push —
`clear_dense_matrix_cache` now clears both halves, and the VJP operators no
longer evict. Do not push; outward actions are approved individually.
