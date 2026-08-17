# Amortisation crossover, measured by batch ladder (2026-08-14)

Recalibration of `croissant.engine_select._MIB_PER_BATCHED_TRANSFORM`
and of how the threshold treats `niter`. Supersedes conclusion 1 of
[`engines-2026-08-13.md`](engines-2026-08-13.md); conclusions 2 and 3
there are unaffected.

Reproduce with:

    uv run python -u benchmarks/benchmark_engines.py --sections ladder

## What was wrong

The shipped threshold was `ceil(resident_mib / 1.0)`, where
`resident_mib` counts BOTH kernels when `niter > 0`, because both really
are held. The memory accounting is right and is unchanged here. What was
wrong is using that same doubled figure to size the batch: refinement
makes a kernel *easier* to justify, not harder, because s2fft repeats
its entire Wigner-d recursion on each of the `2*niter+1` passes while
the kernel engine re-contracts a table it already has. The threshold
doubled exactly where the true crossover falls.

## Ground truth

Crossover = smallest batch at which the kernel engine's cold
setup-plus-first call beats the matrix-free engine's. Cold means every
cache cleared, including JAX's compilation cache; each point is the
median of 3 from-cold repeats. CPU, x64, scalar unless noted.

| lmax | nside | niter | measured crossover | old policy | new policy |
|---:|---:|---:|:--|---:|---:|
| 31 | 16 | 0 | at or below 1 | 1 | 1 |
| 31 | 16 | 3 | at or below 1 | 2 | 1 |
| 63 | 32 | 0 | in (8, 12] | 8 | 10 |
| 63 | 32 | 0 (spin 2) | in (4, 8] | 16 | 10 |
| 63 | 32 | 3 | at or below 1 | 16 | 1 |
| 127 | 64 | 0 | in (64, 96] | 64 | 79 |
| 127 | 64 | 3 | in (12, 16] | 128 | 11 |

The `niter=0` rows divided by kernel size give 1.02, 1.26 and 1.22 batch
elements per MiB — near enough constant that the old law's *form* was
right at `niter=0`, and its coefficient only about 25% too aggressive.
The defect was confined to `niter`, exactly as the follow-up note that
prompted this work predicted.

The cells marked "at or below 1" come from the confirmation pass of the
`fit` section rather than a full ladder: the kernel wins at batch 1, so
there is no smaller batch to bracket against.

## Two findings that shaped the policy

**Refinement lowers the crossover, by a lot.** At lmax=127 it falls from
the (64, 96] band to (12, 16]. The old policy asked for 128 there.
Against a realistic frequency axis that means running the matrix-free
engine at roughly 131 s more than the kernel would have cost — the
single worst case this recalibration fixes, and the reason the threshold
now divides by `2*niter+1`.

**The crossover is spin-independent, so the threshold is sized from
geometry.** A spin-2 kernel is twice the bytes of a scalar one at the
same geometry, but the two take the same time to build (10.7 s scalar
against 10.2 s at spin 2, lmax=127) because both run the same recursion
over the same rings. Measured crossovers agree accordingly — at lmax=63
the spin-2 crossover, (4, 8], is if anything *lower* than the scalar
(8, 12], since s2fft's spin transform is also dearer per map. The
threshold is therefore computed from the scalar `reality=True` kernel
size as a proxy for build work. Resident bytes still govern the memory
cap; that is a separate question and keeps its own doubled accounting.

## Method notes, including a wrong turn

The ladder is a direct measurement. Two cheaper approaches were tried
first and are recorded here so the choice is auditable:

- **`sweep`** (2026-08-13) timed each batch point once. Its answer at
  nside=32 was 8, against this ladder's (8, 12] — roughly right.
- **`fit`** modelled the cold cost as `a + m*B` and solved for the
  crossover. It went wrong twice. First it mixed a cold intercept with a
  *warm* slope, which understates s2fft's marginal cost and biased every
  answer high (nside=32/niter=0 came out at 38.8). Refitting both terms
  from cold calls moved that to 22.5 — still more than twice the ladder
  value. The residual error is the model itself: at lmax=63 both cold
  curves are nearly flat beyond a batch of ~12 (s2fft 3.09 → 3.39 from
  batch 12 to 32) because compilation dominates, so a straight line
  through B=1 and B=32 misdescribes the middle. The fit's own
  confirmation pass is what caught this, reporting MISMATCH at batch 11.

So `sweep` was closer than `fit` at the resolution where they disagree,
and the initial diagnosis that `sweep` was "noise-limited" overstated
the case: the curves are close because both engines are
compilation-bound at that size, not because the timing was unreliable.
`fit` retains one use — it is cheap, and it aimed the ladders at the
right ranges.

## Caveats

- **HEALPix only.** The policy applies to `mw`, `mwss`, `dh` and `gl`
  too, and none is measured here. Inherited, not new — the previous
  calibration was HEALPix-only as well — but the extrapolation is
  unevidenced.
- **One machine, CPU, x64.** Absolute seconds are hardware-specific. The
  crossover is a ratio and travels better than the timings, but a GPU
  would shift compilation and per-map costs differently.
- **Absolute times drift within a long run.** Build times for identical
  configurations differed by up to 2x between a 45-minute run and a
  short one, with later rows slower — consistent with thermal
  throttling. Crossovers are unaffected, because a uniform slowdown
  cancels in the comparison, but do not read the seconds below as
  reproducible constants.
- **`niter=1` and `niter=2` are not measured**, only 0 and 3. The
  `2*niter+1` divisor interpolates them.
- **Above lmax=127 is extrapolation.** At nside=128 the scalar kernel is
  511 MiB and the policy asks for a batch of about 631; the spin kernel
  is 1020 MiB and exceeds the 512 MiB cap, so it resolves to the
  matrix-free engine on memory grounds whatever the threshold says.

## Raw ladder output

```
# ladder: spin=+0 nside=32 lmax=63 niter=0
batch=  4 s2fft=1.9622 kernel=2.3252 winner=s2fft  margin=18.5%
batch=  8 s2fft=2.4508 kernel=2.9813 winner=s2fft  margin=21.6%
batch= 12 s2fft=3.0847 kernel=2.9319 winner=kernel margin=5.2%
batch= 16 s2fft=3.1202 kernel=2.8447 winner=kernel margin=9.7%
batch= 24 s2fft=3.0790 kernel=2.8524 winner=kernel margin=7.9%
batch= 32 s2fft=3.3875 kernel=2.8467 winner=kernel margin=19.0%
crossover=IN (8, 12]

# ladder: spin=+2 nside=32 lmax=63 niter=0
batch=  4 s2fft=2.7808 kernel=3.1014 winner=s2fft  margin=11.5%
batch=  8 s2fft=3.2377 kernel=3.1995 winner=kernel margin=1.2%
batch= 12 s2fft=3.2011 kernel=3.1756 winner=kernel margin=0.8%
batch= 16 s2fft=3.4244 kernel=3.1444 winner=kernel margin=8.9%
batch= 24 s2fft=3.2862 kernel=3.1061 winner=kernel margin=5.8%
batch= 32 s2fft=3.6774 kernel=3.1368 winner=kernel margin=17.2%
crossover=IN (4, 8]

# ladder: spin=+0 nside=64 lmax=127 niter=0
batch= 16 s2fft=4.3225 kernel=10.8259 winner=s2fft  margin=150.5%
batch= 24 s2fft=5.8717 kernel=10.9267 winner=s2fft  margin=86.1%
batch= 32 s2fft=6.5809 kernel=10.6014 winner=s2fft  margin=61.1%
batch= 48 s2fft=7.8748 kernel=10.6978 winner=s2fft  margin=35.8%
batch= 64 s2fft=9.4018 kernel=10.8329 winner=s2fft  margin=15.2%
batch= 96 s2fft=12.1084 kernel=11.0261 winner=kernel margin=9.8%
crossover=IN (64, 96]

# ladder: spin=+0 nside=64 lmax=127 niter=3
batch=  4 s2fft=15.4849 kernel=23.2042 winner=s2fft  margin=49.9%
batch=  8 s2fft=20.1321 kernel=23.3577 winner=s2fft  margin=16.0%
batch= 12 s2fft=21.9379 kernel=23.4704 winner=s2fft  margin=7.0%
batch= 16 s2fft=24.1520 kernel=23.4076 winner=kernel margin=3.2%
batch= 24 s2fft=27.1182 kernel=22.8307 winner=kernel margin=18.8%
crossover=IN (12, 16]
```
