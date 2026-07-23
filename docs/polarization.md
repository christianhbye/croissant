# Full-Stokes conventions

Croissant 6 uses public Stokes order `(I, Q, U, V)` and the IAU tangent-basis
convention. Pixel values for `PolarizedSky` are real. Pair-response samples
may be complex and use layout
`(pair, frequency, stokes, spatial...)`.

## Tangent basis and IAU/COSMO conversion

The tangent basis is ordered `(e_theta, e_phi)`, with theta the colatitude.
Under the HEALPix/COSMO basis-rotation convention,

```text
Q' =  cos(2 psi) Q + sin(2 psi) U
U' = -sin(2 psi) Q + cos(2 psi) U.
```

Therefore

```text
P+ = Q + i U    has spin +2
P- = Q - i U    has spin -2.
```

IAU and COSMO inputs differ only by

```text
U_IAU = -U_COSMO.
```

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

with spins `(0, 0, -2, +2)`, and the response dual

```text
(BI, BV, 1/2 (BQ - i BU), 1/2 (BQ + i BU))
```

with the same spin labels. Since the sky pixels are physical real Stokes
values, `P- = conj(P+)`. The harmonic contraction

```python
jnp.einsum("fclm,tm,pfclm->tpf",
           sky_alm.conj(), phases, beam_alm)
```

is then exactly the pixel-space IQUV integral for arbitrary complex pair
responses. Tests compare this identity against direct mwss quadrature and
differentiate it with respect to all sky and response components.

Both `PolarizedSky.compute_alm(lmax=...)` and
`PairStokesBeam.compute_alm(lmax=...)` accept a requested output bandlimit.
For low-bandlimit HEALPix analysis they use the cached dense transform
directly, avoiding a full transform at the pixel grid's inferred bandlimit.

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
from local North towards East. It acts on every response component through
the same `exp(+i m beam_rot)` harmonic phase.

## Calibration and ground

`PairStokesBeam` performs no physical normalization. Its first luseepy
consumer supplies open-circuit effective-length products in `m^2`, then
applies the frequency-dependent physical scale after the native transforms.
The polarized Croissant primitive does not apply scalar ground correction;
luseepy adds the Moon covariance algebraically.
