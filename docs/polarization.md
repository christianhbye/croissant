# Full-Stokes conventions

Croissant uses public Stokes order `(I, Q, U, V)` and the IAU tangent-basis
convention. Pixel values for `PolarizedSky` are real. Pair-response samples
may be complex and use layout
`(pair, frequency, stokes, spatial...)`.

`PolarizedSky` accepts `galactic`, `equatorial`, `mepa`, and `topo`
coordinates. A topocentric map can be analyzed with `compute_alm()` and stays
in its local frame. `compute_alm_eq()` rejects topocentric input because a
bare sky object does not carry the concrete observer location and reference
epoch needed to transport it into a global frame.

## Why precomputed pair responses

Croissant consumes a per-baseline pair-Stokes response rather than per-antenna
Jones patterns. This is a deliberate architectural constraint: time must keep
entering the simulation only through the diagonal azimuthal rotation

```text
p_m(t) = exp(-i m phi(t)).
```

Given separate antenna Jones patterns the measurement equation contains a
product of three fields,

```text
V_ab = int h_a(n) B_E(n, t) h_b^dagger(n) dn,
```

and keeping those three factors separate in harmonic space would introduce
Wigner-3j mode coupling. That is outside the design. A supplied pair response
has already contracted the two Jones patterns, so the measurement is linear in
the Stokes sky,

```text
V_p(t) = sum_A int M_{p,A}(n) S_A(n, t) dn,   A in (I, Q, U, V),
```

and after both sides are transformed into matching harmonic bases it remains a
sum of diagonal sky-response overlaps. The per-time cost is

```text
O(N_pair * N_freq * N_component * lmax^2),
```

so polarization adds a small component axis but no harmonic mode coupling and
no per-time pixel-space operation. Baseline fringe, antenna cross products,
and any feed leakage are incorporated by the response producer, not by
Croissant.

## Data model

```text
PolarizedSky.data.shape   == (frequency, stokes, spatial...)   real
PairStokesBeam.data.shape == (pair, frequency, stokes, spatial...)  complex
polarized_convolve(...)   -> (time, pair, frequency)           complex
```

Stokes ordering is explicit and validated; all four components are required,
so no shape inference is ever performed. Sky pixels are real physical Stokes
values, while harmonic coefficients are complex and differentiable with
respect to those pixels.

One `pair` entry is one complex visibility product. There is no separate
output-product axis: the reversed pair is the complex conjugate and is not
stored, so a four-port instrument supplies the 10 unique pairs with `a <= b`.
Several feed products (`XX`, `XY`, `YX`, `YY`, or `RR`, `RL`, `LR`, `LL`) are
separate entries along the same pair axis. Croissant never reinterprets the
pair axis as a Stokes output axis.

Conventions that would otherwise be implicit are carried as metadata rather
than assumed: `pairs`, `frame`, `tangent_basis`, `baseline_direction`,
`visibility_definition`, `units`, and `convention`.

`polarized_convolve` is a public composable primitive in the same sense as
`multi_convolve`; there is no polarized simulator orchestration class. A
consumer calls Croissant's transforms, rotations, and convolution directly
from its own simulator.

## Tangent basis and IAU/COSMO conversion

The tangent basis is ordered `(e_theta, e_phi)`, with theta the colatitude,
so that `e_theta x e_phi = n` with `n` the outward radial direction. Under
the HEALPix/COSMO basis-rotation convention,

```text
Q' =  cos(2 psi) Q + sin(2 psi) U
U' = -sin(2 psi) Q + cos(2 psi) U,
```

where `psi > 0` is a right-handed rotation of the tangent basis about the
outward radial direction `n` (rotating `e_theta` towards `e_phi`) and the
primed Stokes parameters are measured in the rotated basis. With this
orientation a spin-`s` quantity acquires `exp(-i s psi)`, matching the
McEwen & Wiaux spin convention implemented by s2fft.

Therefore, with `U` in the COSMO convention used by this derivation,

```text
Q + i U_COSMO    has spin +2
Q - i U_COSMO    has spin -2.
```

IAU and COSMO inputs differ only by

```text
U_IAU = -U_COSMO.
```

Croissant stores Stokes parameters internally in the IAU convention, so
in terms of the internal `U` the spin assignments swap:

```text
Q - i U_IAU    has spin +2
Q + i U_IAU    has spin -2.
```

Everywhere else in this document an unsubscripted `U` denotes this
internal IAU `U`.

The spin labels and the `U` sign convention must flip together: an
opposite-spin analysis still inverts (each fixed-spin harmonic family is
complete), but it is not band-limited for band-limited `E`/`B` skies,
and Wigner-D frame rotation then applies the complex conjugate of the
physical transport phase.

The same sign change is applied to a Q/U response vector. This is the
contragredient response conversion and leaves the physical contraction
unchanged. Stokes V is not changed by this conversion.

## Circular-polarization sign

The first producer, luseepy, uses RMS phasors with time dependence
`exp(+i omega t)` and an arrival-direction `(e_theta, e_phi)` basis. Croissant
uses the coherency convention

```text
[[I + Q, U - i V],
 [U + i V, I - Q]].
```

Thus a pure positive-V deterministic fixture has a Jones vector proportional
to `(1, +i)` in this basis. When viewed toward the arriving wave source, the
real electric vector rotates from `+e_theta` toward `-e_phi` as time
increases. This fixture, rather than an informal right/left label, is the
normative sign definition used by both packages.

In IEEE/IAU labels the fixture is right-hand circular: with arrival
direction `n` (observer towards source) the wave propagates along `-n`, and
the fixture's electric vector `e_theta cos(wt) - e_phi sin(wt)` rotates
right-handedly about `-n`, i.e. about the propagation direction. Positive
Stokes V therefore denotes IEEE/IAU right-hand circular polarization,

```text
V = RCP - LCP,   equivalently   V = (RR - LL) / 2
```

for circular feed products, matching the IAU radio convention used by
common interferometry software.

## Harmonic dual

The desired Q/U part of one pair measurement is

```text
BQ Q + BU U
  = 1/2 (BQ - i BU) (Q + i U)
  + 1/2 (BQ + i BU) (Q - i U).
```

Croissant stores the sky contraction dual

```text
(I, V, P-, P+)
```

with spins `(0, 0, -2, +2)`, where `P-` is the spin `-2` analysis of
`Q + i U` and `P+` the spin `+2` analysis of `Q - i U` (internal IAU
`U`; each combination analyzed at its physical spin per the previous
section), and the response dual

```text
(BI, BV, 1/2 (BQ + i BU), 1/2 (BQ - i BU))
```

with the same spin labels. Since the sky pixels are physical real Stokes
values, the two polarized sky fields are complex conjugates of each
other. The harmonic contraction

```python
jnp.einsum("fclm,tm,pfclm->tpf", sky_alm.conj(), phases, beam_alm)
```

is then exactly the pixel-space IQUV integral for arbitrary complex pair
responses. Tests compare this identity against direct mwss quadrature and
differentiate it with respect to all sky and response components.

Modes with `ell < 2` carry no spin-2 content and are zero by construction in
the `P-` and `P+` blocks. Croissant does not use an internal E/B
representation: reality identities that hold for a real sky do not hold for
complex cross-baseline responses, so the `P+/P-` dual is carried through
unreduced on both sides.

Both `PolarizedSky.compute_alm(lmax=...)` and
`PairStokesBeam.compute_alm(lmax=...)` accept a requested output bandlimit.
For low-bandlimit HEALPix analysis they use the cached dense transform
directly, avoiding a full transform at the pixel grid's inferred bandlimit.

## Dense transforms for spin-weighted fields

The scalar dense HEALPix path exploits real-field `m >= 0` conjugate symmetry,
which is not valid for arbitrary complex pair responses. Spin-weighted and
complex analysis therefore goes through `croissant.dense`, which stores the
full 2D harmonic layout. Matrices are keyed on `lmax`, sampling, `nside`,
spin, the iterative-refinement count `niter`, complex precision, and the JAX
device; the packed real spin-0 cache additionally keys on the spatial shape
and backend. Matrix construction happens outside JIT and is cached; matrix
application is a native JAX matmul and is differentiable with respect to the
input maps. Gradients with respect to the cached matrix itself are not an API
goal.

s2fft's HEALPix FFT requires `L >= 2 * nside` even when only lower modes are
wanted. Croissant builds that supported operator and selects the requested
low-`ell` rows, so the low-`lmax`, high-`nside` case is preserved rather than
lost.

On the 2026-07-23 CPU development benchmark
(`benchmarks/benchmark_dense_healpix.py`), `nside=32`, `lmax=30`, and the
three required spins `(0,-2,+2)` retained 539.1 MiB of device matrices.
Chunked matrix construction took 21.9 s total, with a 1.26 GiB macOS peak
memory footprint (2.32 GiB maximum RSS including compiler/runtime memory).
First applications took 45--47 ms per spin and cached applications took
7.2--7.5 ms. This is a construction-time cache tradeoff, not per-time
simulation work; release GPU benchmarks must enforce an appropriate device
memory budget.

## Rotations and time phases

Coordinate rotations act on already formed spin coefficients through one
Wigner-D operator for every batch/component. Q and U are never rotated as two
independent scalar maps. A z rotation retains Croissant's phase convention

```text
exp(-i m phi(t)).
```

`jd_to_et` accepts a TDB Julian date. For Moon simulations, `mepa` is the
inertial frame obtained by freezing SPICE's `MOON_ME` orientation at the
reference epoch. It is not the body-fixed MCMF frame: retaining the reference
epoch is what preserves the Moon's initial spin phase.

Positive `beam_rot` follows the astronomical azimuth convention, measured
from local North towards East, and is given in degrees (matching the
scalar `Beam`). It acts on every response component through the same
`exp(+i m beam_rot)` harmonic phase.

## Calibration and ground

`PairStokesBeam` multiplies its response by a horizon mask before every
analysis (default: the upper hemisphere, `theta <= pi/2`; pass `horizon=`
to override), so below-horizon response is excluded from all four
components' coefficients alike.

`PairStokesBeam` performs no physical normalization. Its first luseepy
consumer supplies open-circuit effective-length products in `m^2`, then
applies the frequency-dependent physical scale after the native transforms.
Cross-pair normalization is never inferred from a complex Stokes response; an
explicit per-pair or per-pair-per-frequency `normalization` argument to
`polarized_convolve` remains available for users who want it.

The polarized primitive does not apply the scalar `fgnd` ground correction,
which is not meaningful applied independently to Q, U, and V; luseepy adds the
Moon covariance algebraically upstream. Unpolarized ground would be
`(T_gnd, 0, 0, 0)` seen through the pair response's I component, but polarized
ground pickup is deferred: `PairStokesBeam` has no `compute_fgnd` analogue and
`correct_ground_loss` is scalar-only, so there is no path that silently
applies scalar-ground assumptions to Q, U, or V.

## s2fft dependency

The pinned s2fft revision fixes HEALPix spin recursion at exact HEALPix nodes.
The failure mode involves zero Wigner-d recursion entries and produces
percent-level spin-2 inaccuracies, so it is not optional for the polarized
path. `tests/test_s2fft_pin.py` certifies the pin: the forward transform is
checked against analytically sampled spin-weighted harmonics and the inverse
against s2fft's independent Turok-recursion base implementation. Neither is
a forward/inverse round trip (related forward and inverse errors partially
cancel). The file fails on s2fft 1.4.0 and passes at the pinned revision;
it doubles as the acceptance test for switching to the upstream release
once astro-informatics/s2fft#387 is merged.

The git pin in `pyproject.toml` is a development expedient, not a proposal for
release. It should be replaced with an upstream s2fft release containing the
fix once one is available, re-verifying the same regression test. Until then,
co-developed consumers must pin the identical revision.

## Scalar API compatibility

The polarized layer is additive. Scalar data shapes remain valid,
`sphere.compute_alm()` keeps its scalar defaults, `Sky.compute_alm()` and
`Beam.compute_alm()` keep their return shapes, scalar `convolve()` keeps its
signature and numerics, `Simulator.sim()` keeps returning real
`(time, frequency)` arrays, and existing multipair callers keep receiving
complex `(time, pair, frequency)` arrays. New component axes are introduced
only by the polarized types and functions.
