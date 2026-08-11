# Spin-transport test and fix — design (as built)

**Date:** 2026-08-11
**Status:** implemented on this branch (commits `968a37c`, `5d51329`)
**Context:** Post-#124 follow-up. The conventions/code-equivalence
review of the full-Stokes polarization work verified the einsum
identity, spin labels vs s2fft, dual pairing, and phase conventions,
but left one gap: no test discriminated correct spin transport from a
mislabeled one — every existing rotation test passed under both. An
earlier revision of this spec designed a healpy cross-validation
harness for that gap; pre-implementation verification of its constants
falsified its central premise and located a real transport bug in
croissant, so the harness collapsed into two minimal regression tests
plus a source fix. This document records what was built and why it is
trusted.

## The bug

Croissant stores Stokes parameters internally in the IAU convention
(`U_IAU = -U_COSMO`). In s2fft's Goldberg / McEwen–Wiaux basis this
makes `Q + iU` the spin −2 object and `Q - iU` the spin +2 object —
but `_compute_sky_dual_alm` and `_compute_response_dual_alm` analyzed
them at the opposite labels. `docs/polarization.md` shows the origin:
its spin derivation is carried out in the COSMO convention (and is
correct there), but its conclusion was applied to the internal IAU U.

Consequences of the mismatched labels:

- **Statics: unaffected.** A fixed-spin harmonic contraction is
  complete for any spin, and sky and beam used matching labels — so
  the quadrature identity genuinely held (and still does).
- **z-rotations (time evolution): unaffected.** Zero transport phase.
- **Frame rotations with β ≠ 0 (gal→FK5/MEPA skies, topo→MEPA
  beams): corrupted.** Wigner-D rotation of opposite-labeled duals
  applies the complex conjugate `e^{+2iψ}` of the physical transport
  phase `e^{-2iψ}`: order-unity Q/U errors (measured: 140% of field
  scale for a gal→FK5 E-only sky).
- **Band limitation: degraded.** Duals of band-limited E/B skies were
  not band-limited (O(1) out-of-band power), an lmax-truncation loss
  even for statics.

## The fix (commit `5d51329`)

Swap the `Q ∓ iU` combinations between the spin ∓2 analyses in
`_compute_sky_dual_alm`, and the `(BQ ∓ iBU)/2` combinations likewise
in `_compute_response_dual_alm`. The response slots swap in lockstep
with the sky slots, so the conjugated einsum still reproduces
`∫ (BQ·Q + BU·U)` exactly — verified by the pre-existing quadrature
test passing unchanged. `docs/polarization.md` now states the spin
assignments in both conventions and warns that the U sign and the spin
labels must flip together.

## Evidence base

Every link is empirical at machine precision or a mathematical
identity; no step rests on convention-from-memory:

1. **s2fft basis certification.** The analytic helpers in
   `test_s2fft_pin.py` reproduce `s2fft.forward` deltas exactly. The
   helpers themselves had an even/odd-spin bug (below), fixed first.
2. **healpy pairing.** `hp.alm2map(pol=True)` implements
   `(Q + iU_COSMO) = Σ −(E+iB)ₗₘ ₂Yₗₘ` in that basis, certified to
   1e-15 (the swapped pairing fails at O(1)).
3. **Scalar rotation orientation.** croissant `gal2eq` ↔ healpy
   `Rotator.rotate_alm` agree to 1e-14, so scalar-rotation references
   do not depend on trusting one library over the other.
4. **Transport sign.** For an E-only sky the true rotated field needs
   only scalar rotation of E. Pre-fix croissant matched the conjugate
   transport formula to 7.6e-14 on input band-limited in its own bases
   (zero truncation), while the correct formula was independently
   certified at 5.7e-14 — a clean implementation of exactly the wrong
   sign, not an approximation error.

## As-built deliverables

**Helper fix + odd-spin certification (`968a37c`,
`tests/test_s2fft_pin.py`).** `wigner_d` computed the transposed
d^ℓ_{n,m}; the `(-1)^(spin+|m|)` prefactor cancelled the transpose
only up to a global `(-1)^spin` — invisible at even spin, a sign error
at odd spin. Rewritten as the documented d^ℓ_{m,n} (with float casts;
the factorial products exceed int64 beyond ℓ + |m| ≥ 21) and the
Goldberg form `(-1)^s √((2ℓ+1)/4π) d^ℓ_{m,-s} e^{imφ}`. Even-spin
values are bit-identical, so the HEALPix pin certification is
untouched; a new MWSS forward-delta test (spins ±1, modes including
(1,1), atol 1e-10) certifies odd spin independently of the pin.

**Regression tests (`5d51329`, `tests/test_spin_transport.py`).** Both
fail at O(1) on pre-fix src; both pass post-fix:

1. `test_polarized_duals_of_single_e_mode_are_band_limited` — a single
   E(3,1) mode through `PolarizedSky.compute_alm()` must yield duals
   supported only at (3, ±1), atol 1e-8. Mismatched labels smear O(1)
   power over every ℓ.
2. `test_gal_to_fk5_rotation_transports_polarization` — gal→FK5
   rotation of a random E-only sky (lmax 8, seed 2026) must match the
   reference built from healpy scalar rotation of E plus the certified
   pairing, compared as maps at 1e-8 × field scale.

Design properties: MWSS sampling throughout (independent of the s2fft
HEALPix pin, survives the future unpin untouched); the reference path
contains no spin-transport machinery of its own (E-modes rotate as
scalars); the one convention relation used is certified against healpy
at machine precision.

## Deliberately not covered / follow-ups

- Beam-side topo→MEPA transport test: same `rotate_alm` machinery and
  now-matched labels; optional future hardening.
- gal→MEPA variant: identical code path to gal→FK5 (only the rotation
  matrix differs).
- Promotion into `test_physics.py`: decide together with the queued
  promotion of the unpolarized-reduction invariant.
- luseepy pins `v5.3.0.dev0`, which predates the fix: tag a new dev
  release after merge and bump the pin.

## Acceptance (met)

- Full suite: 486 passed. Physics tests and the quadrature identity
  unchanged, as the completeness argument requires.
- HEALPix pin tests pass bit-identically at even spin.
- Both regression tests demonstrated failing against pre-fix src
  before the fix was applied.
