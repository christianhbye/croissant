# Spin-transport discriminating test — design

**Date:** 2026-08-11
**Status:** awaiting review
**Context:** Post-#124 follow-up. The conventions/code-equivalence review
of the full-Stokes polarization work verified the einsum identity, spin
labels vs s2fft, dual pairing, and phase conventions, but left one gap:
no test discriminates correct spin transport from a naive
scalar-rotation of Q/U. Every existing rotation test passes under both.

## What is being tested

`rotations.rotate_alm` applies one scalar-form Wigner-D action to all
components of the polarized stack, on the premise (stated in its
docstring) that spin-weighted fields rotate through the same Wigner-D
action *once their coefficients have been formed in a consistent spin
basis*. The object under test is therefore the **composition** of
croissant's spin-±2 analysis (`_compute_sky_dual_alm`: Q ∓ iU analyzed
at spin ∓2) with that rotation, cross-validated against an independent
transport implementation (healpy's polarized machinery).

A spin-sign swap, conjugation error, or dual-pairing mistake anywhere
in the chain corrupts the m-structure of the coefficients. A rotation
with β ≠ 0 mixes m's and does not commute with such errors — which is
what makes gal→FK5 (β ≈ 62°) and gal→MEPA discriminating, and why
z-rotations (time evolution) are structurally immune.

Why current tests are blind:

- `test_polarized_mepa_rotation_matches_scalar_sky` sets Q = U = 0; the
  spin-±2 blocks have never been rotated in any test.
- No scalar healpy-vs-croissant *alm rotation* cross-check exists
  either — only the rotation matrices are cross-checked, not the
  active/passive sense in which each library applies them to alm.

## Out of scope

- Beam-side `compute_alm_in_frame` (same `rotate_alm` machinery; the
  response-dual pairing was verified in the #124 review).
- Topo→MEPA time evolution (z-rotations, transport-free by
  construction).
- Polarized HEALPix analysis accuracy (owned by `test_s2fft_pin.py`;
  this design uses MWSS throughout precisely so it survives the future
  s2fft unpin untouched).
- Polarized ground loss, normalization, temperature units (separate
  queue items).

## Section 0 — Fix the analytic-harmonic helpers in `test_s2fft_pin.py`

**Diagnosis (verified empirically 2026-08-11).** `wigner_d(ell, m, n,
beta)` computes the transposed element d^ℓ_{n,m} (Wikipedia
convention) rather than d^ℓ_{m,n} as its signature suggests.
`spin_spherical_harmonic`'s prefactor `(-1)**(spin + abs(m))` cancels
the transpose's (−1)^{m−n} sign only up to a leftover global factor
**(−1)^spin**: invisible at the even spins the pin tests use, a sign
error at odd spin. Evidence (MWSS forward delta test against s2fft,
`method="numpy"`, L=8): the old helper yields peak amplitude −1 at
spins ±1 and ±3 and +1 at ±2, with machine-precision off-diagonals;
the corrected form yields +1 at every spin; `old == (−1)^s × new`
holds pointwise to 0.0.

**Changes:**

1. Rewrite `wigner_d` so it computes d^ℓ_{m,n} (Wikipedia/Varshalovich
   explicit sum) as documented: sum index k over
   `max(0, n-m) … min(ell+n, ell-m)`, denominator
   `(ell+n-k)! k! (ell-m-k)! (m-n+k)!`, sign `(-1)**(m-n+k)`,
   `cos(β/2)**(2ℓ+n-m-2k)`, `sin(β/2)**(m-n+2k)`.
2. Rewrite `spin_spherical_harmonic` in the Goldberg/McEwen–Wiaux form
   actually used by s2fft:
   `(-1)**spin * sqrt((2ℓ+1)/4π) * wigner_d(ell, m, -spin, θ) * exp(imφ)`.
3. Add an odd-spin regression test: analytic sYℓm sampled on the MWSS
   grid → `s2fft.forward(spin=±1, sampling="mwss", reality=False)` →
   exact delta with amplitude +1, parametrized over modes including
   (1, 1), atol 1e-10. MWSS keeps this certification independent of
   the HEALPix pin.

**Acceptance:** existing HEALPix pin tests pass unchanged (guaranteed
by the even-spin identity; verified by running them), and the new
odd-spin test passes.

## Section 1 — File layout

One new self-contained file `tests/test_spin_transport.py`, mirroring
the `test_s2fft_pin.py` pattern of a convention-certifying module with
a docstring explaining what it certifies. It imports `wigner_d` and
`spin_spherical_harmonic` from `tests/test_s2fft_pin.py` (tests/ is a
package). Three tests plus one shared helper. healpy is already a dev
dependency; no packaging changes.

Promotion of the discriminator into `test_physics.py` (under the
never-modify policy) is deliberately deferred until it has proven
itself, to be decided together with the queued promotion of the
unpolarized-reduction invariant.

## Section 2 — Convention bridge (test 1)

**Helper** `teb_to_duals(alm_t, alm_e, alm_b, lmax)`: converts
healpy-packed (m ≥ 0, COSMO-convention) T/E/B alm into croissant's
component-stack format — scalar block in s2fft 2D (ℓ, m) layout plus
P∓ duals — handling:

- healpy m ≥ 0 packing → full 2D (ℓ, m) via the reality relations
  (T, E, B are real fields);
- the COSMO→IAU U-sign flip (U_IAU = −U_COSMO);
- the E/B → spin-±2 combination. Candidate form (standard HEALPix
  relation): a^{±2}_{ℓm} = −(E_{ℓm} ± i B_{ℓm}), paired with
  croissant's P∓ = Q ∓ iU_IAU at spin ∓2.

**Bridge test:** for single unit modes — E-only and B-only,
parametrized over (ℓ, m) ∈ {(2, 0), (3, 1), (5, 3)} so m = 0, m ≠ 0,
and negative-m reconstruction are all exercised — synthesize Q/U with
`hp.alm2map(..., pol=True)` at nside=16 (exact evaluation at pixel
centers, zero interpolation) and compare pointwise against the
analytic sum over `spin_spherical_harmonic` applied to the claimed
duals, atol 1e-10.

The bridge test involves only healpy and the analytic formula — never
croissant — so a failure always means the bridge formula is wrong and
is fixed in the *bridge helper*, never in croissant. The exact sign
pairing above is a candidate; the bridge test is the arbiter.

## Section 3 — Discriminator (test 2)

Fixed-seed random band-limited T/E/B (+V) spectra at lmax=8, one
frequency.

**Input maps:** vetted bridge → duals → s2fft MWSS spin-∓2 synthesis →
Q/U maps (assert imaginary parts vanish at ~1e-12 — consistency of the
reality relations); T and V via spin-0 synthesis. Data enters
`PolarizedSky` as IAU (the U flip happens in the bridge; croissant's
`cosmo_to_iau` has its own test and stays out of this chain).

**Croissant path:** `PolarizedSky(..., sampling="mwss",
coord="galactic")` → `compute_alm_eq(world="earth")` and
`compute_alm_eq(world="moon", et=0.0)` — the genuine
map → analysis → rotation pipeline.

**Reference path:** `eul = rotmat_to_eulerZYX(get_rot_mat("galactic",
target))` → `hp.Rotator(rot=eul, eulertype="ZYX")` → `rotate_alm`
applied to T, E, B, V *each as a scalar* (exact in harmonic space; E
and B are rotation scalars, so all transport physics lives in the
already-certified bridge) → bridge → expected (I, V, P−, P+) stack.
The `rotmat_to_eulerZYX` ↔ healpy correspondence is already pinned by
`test_rotmat_to_euler` for the gal→FK5 and gal→MEPA matrices
specifically.

**Assertions, ordered for diagnosis:** spin-0 blocks (I, V) first — a
failure there indicates a harness/orientation mismatch between the
libraries, not a transport bug — then the P∓ blocks. Tolerance
rtol = atol = 1e-8 (every step is an exact-sampling harmonic
operation; observed per-mode precision is ~1e-15).

## Section 4 — Non-vacuity guard (test 3)

Same inputs. Mutant path: analyze Q and U each at spin 0
(`reality=False`), rotate with croissant's own `rotate_alm`, recombine
into mutant duals in alm space. Assert the mutant misses the reference
P∓ blocks by O(1): max abs difference > 0.1 × max abs of the reference
P∓ coefficients. This proves the test discriminates the exact failure
mode it
was built for (and documents that β ≠ 0 is what gives it teeth — a
z-rotation would pass the mutant).

## Section 5 — Parameters and tolerances

| Item | Value |
| --- | --- |
| lmax (discriminator) | 8 |
| nside (bridge) | 16 |
| Sampling (croissant path) | mwss |
| Frequencies | 1 |
| Seed | 2026 |
| Bridge tolerance | atol 1e-10 |
| Discriminator tolerance | rtol = atol = 1e-8 |
| Guard threshold | > 0.1 × field scale |
| Odd-spin helper test | atol 1e-10 |

64-bit precision comes from `tests/conftest.py`. Expected added
runtime: well under a second.

## Implementation order

1. Section 0 (helper fix + odd-spin regression) — everything else
   imports these helpers.
2. Bridge helper + bridge test (section 2).
3. Discriminator (section 3).
4. Non-vacuity guard (section 4).

Each step lands with its tests passing before the next begins
(TDD: for section 0, the odd-spin test is written first and fails
against the old helper).

## Acceptance criteria

- Full suite passes, including unchanged HEALPix pin tests.
- The discriminator fails if the P∓ rotation is replaced by scalar
  rotation of Q/U (demonstrated by the guard).
- No changes to `src/` — this work certifies existing behavior. If the
  discriminator *fails* against croissant as-is, that is a physics bug
  finding: stop, diagnose, and report before touching src.
