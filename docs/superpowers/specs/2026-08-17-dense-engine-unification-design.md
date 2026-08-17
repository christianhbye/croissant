# Dense engine unification

**Date:** 2026-08-17
**Status:** approved, not yet implemented
**Scope:** PR B of the kernel-engine follow-up plan (queue item 3)

## Motivation

Croissant's dense SHT engine lives in two modules that never agreed on a
policy. `sphere.py` owns a packed-real builder and an unbounded cache;
`dense.py` owns a general spin-capable builder and an `lru_cache(maxsize=6)`.
The split is not arbitrary, but the divergence in policy, naming and public
surface is.

### How the split arose

`4de2852` (#123, "add dense spherical harmonic engine") built the dense engine
inside `sphere.py` for one workload: scalar, real, low-band-limit HEALPix. Both
of its builders assume that workload.

- HEALPix: `scipy.special.sph_harm_y` evaluated at pixel centers, pixel-area
  quadrature, explicit Gram refinement. No SHT is involved.
- Equiangular: one-hot pixel maps through `s2fft.forward(reality=True)`, packed
  to `m >= 0`.

`d4736db` (#124, full-Stokes polarization) then needed spin +-2 analysis, which
neither builder can serve:

1. The `m >= 0` packing is a real-field identity. A cross-baseline pair
   response is genuinely complex, so the discarded half cannot be recovered by
   conjugation. `docs/polarization.md:218` states this.
2. `sph_harm_y` is a spin-0 function. SciPy has no spin-weighted equivalent,
   and hand-rolling one was later ruled out: the analytic oracle in
   `tests/harmonics_reference.py` overflows at `ell >= 58`, precisely the
   regime that matters.

So #124 added a builder that needs no closed form at all — take `jax.vjp` of
`s2fft.forward` at a zero map and read off rows (`dense.py:56-97`). It works at
any spin because it transposes whatever s2fft does. The price is the full
`(L**2 - s**2, npix)` complex layout and `ncoeff` pullbacks.

**The split is therefore a builder split (real-and-scalar vs complex-or-spin),
and the two caches are downstream of it.** Collapsing the builders is not the
goal and would be a regression: scalar HEALPix would move from a measured 20.8s
direct build to a 196s VJP build for an operator the two agree on to 3.06e-15.

### What is actually wrong

Measured footprints on this codebase (HEALPix, x64):

| config | packed-real | full complex (spin -2) |
| --- | --- | --- |
| nside 16, lmax 31 | 24.8 MiB | 47.8 MiB |
| nside 32, lmax 63 | 390.0 MiB | 767.2 MiB |
| nside 64, lmax 127 | 6.0 GiB | 12.0 GiB |

Against that:

- **The policy is inverted relative to cost.** The small, cheap-to-rebuild
  packed-real matrices (~20.8s) are the unbounded ones. The huge,
  expensive-to-rebuild VJP matrices (~197s) are the ones capped at 6 entries.
  The half that would benefit most from retention is the half that evicts.
- **Half the engine has no clear function.** `precompute_dense_matrix` and
  `clear_dense_matrix_cache` are exported and documented in the README; the VJP
  half is reachable only through `dense._build_analysis_matrix.cache_clear()`.
- **Two key schemas, reasoned about independently.** `sphere` keys on
  `spatial_shape` plus `jax.default_backend()`; `dense` keys on `spin` plus a
  `device_key` it immediately `del`s.
- **The naming is inverted.** `sphere.py` is the dispatcher — `compute_alm`
  resolves an engine and routes — yet it also owns a builder and a cache.
  `dense.py` is named for the engine but owns half of it.

## Decisions

Recorded with their rationale so they are not re-litigated.

### D1. Move, do not merge

`dense.py` owns the dense engine end to end. `sphere.py` returns to dispatch.
Both builders keep their internals unchanged; this PR relocates and unifies
policy, and makes no numerical change.

Rejected: unifying onto one builder. Measured 9.4x regression for scalar
HEALPix, for an operator that already agrees to 3.06e-15.

Deferred: replacing the VJP builder with a kernel-based one. Plausible — the
kernel is built at the floor and row-selected, so it could serve the sub-floor
case — but it needs its own equivalence proof and benchmark table, and belongs
in its own PR.

### D2. The unified cache stays unbounded

One `dict` plus `RLock` in `dense.py` replaces both the existing dict and the
`lru_cache(maxsize=6)`.

Rationale: unbounded is what makes the documented warm-then-jit recipe
unconditional. Any eviction policy makes it conditional — a later unrelated
build can drop a warmed matrix, and the next jitted explicit-dense call raises
`RuntimeError` (today `sphere.py:388`). Bounding was considered in three forms
(byte budget with pinned warms, plain byte budget, count bound) and all were
judged to buy less than the contract they weaken, at this stage.

**Known consequence, accepted:** the VJP half loses its `maxsize=6` bound, so
worst-case retention on the polarized path increases. Today those 767 MiB
matrices evict; afterwards they do not. Mitigated by D6, not by a policy.

Cost of the accumulation is real and should be recorded: the cache key includes
`lmax`, so a band-limit convergence check at nside=32 over
`lmax in [8, 16, 24, 32, 40, 48, 56]` produces 7 keys and retains 904.3 MiB for
the life of the process. Bounded retention remains a legitimate future item;
this PR deliberately does not take it.

### D3. `engine_dtypes` moves to `footprints.py`

`_dense_dtypes` has three consumers, not two: `sphere.py`, and `kernel.py:266`
via `from .sphere import _dense_dtypes`. Its own comment at `kernel.py:259`
calls it "the dtype contract croissant's engines share". Leaving it in
`dense.py` would make the kernel engine import from the dense engine.

It moves to `footprints.py` — the module that already owns cross-engine
geometry and size prediction (`transform_lmax`, `spatial_shape`,
`_kernel_itemsize`, `_COMPLEX_ITEMSIZE`) — and is renamed `engine_dtypes`,
since the name `_dense_dtypes` already misdescribes a helper the kernel engine
imports.

This also removes a workaround: `kernel.py:265` currently notes "Imported
lazily because sphere imports this module", so the helper's present home forces
a function-body import to dodge a cycle. From `footprints.py` it becomes an
ordinary module-level import, alongside the `transform_lmax` that module
already provides.

### D4. Symbols that cross module boundaries lose the leading underscore

`_apply_dense_matrix` is called by `sphere.compute_alm` and becomes
`dense.apply_packed_matrix`. `_dense_dtypes` becomes `footprints.engine_dtypes`
per D3. Symbols that stay internal to `dense.py` keep their underscore.

This makes the cross-module surface explicit rather than relying on the
private-access precedent `kernel.py` currently sets.

### D5. No re-export shim in `sphere.py`

`croissant.precompute_dense_matrix` and `croissant.clear_dense_matrix_cache`
keep working: `__init__.py` imports them from `.dense` instead of `.sphere`.
No shim is left behind in `sphere.py`.

Rationale: `CLAUDE.md` treats `alm.py` and `croissant.jax` as deprecated shims
not to be extended, so adding a fresh one runs against project direction. The
only importers of the `croissant.sphere.*` path are our own tests and
`benchmarks/benchmark_dense_healpix.py`, both updated here. The README
advertises only the top-level names, which are unchanged.

Accepted breakage: a direct `croissant.sphere.precompute_dense_matrix` import.
Unreleased surface — 5.3.0 is still parked.

### D6. Retention becomes observable

Add a read-only `dense_cache_nbytes()` reporting total bytes held across both
flavours, and give `clear_dense_matrix_cache()` a real README paragraph rather
than the single line it has today. This is not a policy — it turns D2's
accepted cost from invisible into inspectable.

### D7. The jit item is a comment, not a conversion

Queue item 3 lists "`jax.jit` vs house-style `eqx.filter_jit`". On inspection
the framing is backwards and the conversion would be a bug.

`jax.jit` is the dominant style (`beam.py`, `sky.py`, `simulator.py`,
`multipair.py`, `polarization.py`, `rotations.py`). `eqx.filter_jit` appears
twice, both in `sphere.py`, and both are load-bearing:

- `_compute_alm_s2fft` takes `sampling` as a string.
- `_apply_dense_matrix` uses `lmax` to construct a shape (`sphere.py:418`).

Plain `jax.jit` would trace both and fail. `DenseSphericalTransform.__call__`
may keep `@jax.jit` because its ints are static `eqx.field` members on a pytree.
The change is a comment at each site explaining why the decorator differs.

## Design

### Module layout after the move

`dense.py` (191 -> ~560 lines) gains, in order:

- the cache globals and lock
- `_dense_matrix_key`, extended per the schema below
- `_positive_lm_indices`
- `_build_dense_matrix_healpix`, `_build_dense_matrix_from_pixels`,
  `_build_dense_matrix`
- `precompute_dense_matrix`, `dense_matrix_for`, `clear_dense_matrix_cache`,
  `dense_cache_nbytes`
- `apply_packed_matrix` (was `sphere._apply_dense_matrix`)

and keeps `_valid_lm_indices`, `_build_analysis_matrix`,
`DenseSphericalTransform`, `dense_compute_alm`.

`sphere.py` (854 -> ~490 lines) keeps `_compute_alm_s2fft`, `compute_alm` and
`SphBase`, and imports `dense` at module level. The existing lazy
`from . import dense as _dense` calls inside function bodies are no longer
needed: after the move `dense.py` does not import `sphere`, so there is no
cycle to avoid.

### Dependency wrinkle

`_build_dense_matrix_from_pixels` calls `sphere._compute_alm_s2fft`. Importing
that from `dense.py` would invert the module dependency and reintroduce the
cycle D1 removes. The builder instead calls `s2fft.forward` directly under a
local jitted helper, vmapped over the chunk axis. It already pins every
argument the wrapper sets, including `reality=True`, which must be preserved —
the one-hot basis maps are real and the packing keeps only `m >= 0`, so that
flag is what makes the result the packed operator by construction.

### Unified key schema

```
(spatial_shape, lmax, sampling, nside, spin, packed, niter,
 complex_dtype, backend)
```

`packed` distinguishes the `m >= 0` real operator from the full complex one, so
the two flavours cannot collide at identical geometry. `spatial_shape` is
formally derivable from `(sampling, nside, lmax)` via `footprints.spatial_shape`
but is retained: it is the first positional parameter of the public
`precompute_dense_matrix`, and keying on it preserves today's behaviour for a
caller who passes a shape inconsistent with the rest.

Backend rather than device: the VJP half currently keys on
`str(jnp.empty((), dtype=jnp.uint8).device)` while `sphere` and `kernel.py` both
key on `jax.default_backend()`. Standardizing on backend makes all three engine
caches agree. Accepted consequence: two devices on one backend share an entry,
which costs a transfer, not correctness.

### Build-loop cleanup

`_build_analysis_matrix` (`dense.py:79-94`) preallocates with `jnp.empty` and
fills by repeated `matrix.at[start:stop].set(rows)`. It becomes a list of
blocks plus one `jnp.concatenate`, matching the pattern
`_build_dense_matrix_from_pixels` already uses. Output is expected to be
bitwise identical.

### Call-site updates

- `sphere.compute_alm`: `dense_matrix_for` and `_apply_dense_matrix` become
  `dense.dense_matrix_for` and `dense.apply_packed_matrix`.
- `sphere.SphBase.__init__`: `dense_matrix_for` becomes `dense.dense_matrix_for`.
- `polarization.py:380`: `sphere.dense_matrix_for` becomes
  `dense.dense_matrix_for`. `polarization.py` already imports `dense`.
- `kernel.py:266`: `from .sphere import _dense_dtypes` becomes
  `from .footprints import engine_dtypes`.
- Docstring and comment references to `sphere._dense_dtypes` and
  `sphere._dense_matrix_key` (`kernel.py:79`, `kernel.py:259-260`,
  `tests/test_kernel_engine.py:110`, `tests/test_kernel_engine.py:217`) are
  updated to the new homes.
- `tests/test_sphere.py:12-16` imports `_DENSE_MATRIX_CACHE`,
  `clear_dense_matrix_cache` and `precompute_dense_matrix` from
  `croissant.sphere`; these move to `croissant.dense`. Whether the
  cache-behaviour tests themselves move to a `tests/test_dense.py` is an
  implementation call, not a design one.
- `benchmarks/benchmark_dense_healpix.py:14-15` and
  `benchmarks/benchmark_engines.py:143` update to the new import paths.
  Nothing imports `benchmarks/`, so these cannot break the suite and must be
  updated by inspection.

## Testing

This is a refactor with no intended numerical change, so the primary bar is
that nothing moves:

- Full suite green, with zero net change in passing tests. The baseline count
  is established by running the suite on `1c4b59e` at implementation start, not
  assumed from an earlier branch's figure.
- `tests/test_physics.py` 0 diff vs main. These are ground truth and must not
  be edited; a failure here means the refactor is wrong.
- `ruff check` and `ruff format` clean.

New tests for what the PR actually changes:

1. `clear_dense_matrix_cache()` releases **both** flavours. Today it releases
   one; this is the behaviour change most likely to be silently wrong.
2. Packed and full operators built at identical geometry occupy distinct cache
   entries and do not collide.
3. `dense_cache_nbytes()` tracks both flavours and returns to zero after a
   clear.
4. The equiangular builder still produces the packed operator after dropping
   the `_compute_alm_s2fft` wrapper — i.e. `reality=True` survived the
   rewrite. A direct comparison against `s2fft.forward` output at one small
   configuration pins it.
5. `_build_analysis_matrix`'s concatenate rewrite is bitwise identical to the
   `.at[].set()` version at one small configuration.

Existing coverage that must keep passing unchanged, as the real regression
gate: `tests/test_engine_equivalence.py` (all engines agree),
`tests/test_polarization.py:575-580` (the warm-then-jit `RuntimeError`
contract), `tests/test_sphere.py:410-432` (cache identity and entry count).

## Out of scope

- Bounded retention for either the dense or the kernel cache. Queue item 6
  covers the kernel side and is PR C.
- Replacing the VJP builder with a kernel-based build.
- The sub-floor HEALPix branch (`polarization.py:198`), which still builds a
  full complex operator for a real spin-0 sky. It is the last remnant of the
  "2x waste" item and is a behaviour change, not a move.
- Queue items 4, 5, 6 and 7, which are PR C.

## Risks

| Risk | Mitigation |
| --- | --- |
| VJP half becomes unbounded (D2) | Accepted and documented; D6 makes it observable |
| `clear_dense_matrix_cache` semantics widen silently | Test 1 above; called out in the PR body |
| Equiangular builder loses `reality=True` in the rewrite | Test 4 above |
| Backend-vs-device key change alters caching on multi-device hosts | Costs a transfer, not correctness; noted in the PR body |
| Large move obscures a real change in review | Structure as two commits: a pure relocation with no behaviour change, then the policy and cleanup commit. Each must pass the suite alone so the branch stays bisectable |

## Acceptance criteria

1. `sphere.py` contains no dense builder, cache or key helper.
2. One cache, one lock, one key schema, one clear function, covering both
   operator flavours.
3. `croissant.precompute_dense_matrix` and `croissant.clear_dense_matrix_cache`
   work unchanged from the top-level namespace.
4. `kernel.py` imports no code from `sphere.py`, and its lazy
   `from .sphere import _dense_dtypes` becomes a module-level import from
   `footprints`. Docstring cross-references to `croissant.sphere.compute_alm`
   and `SphBase` are unaffected and stay.
5. Full suite green, `tests/test_physics.py` 0 diff vs main, ruff clean.
6. Both commits pass the suite independently.
