# Full-polarization plan

## Status

This document records the proposed design for adding differentiable full-Stokes
polarization support to Croissant. Implementation has not started.

Agreed decisions:

- Support standard Stokes **I, Q, U, V**.
- Accept precomputed per-baseline pair–Stokes response beams, rather than
  per-antenna Jones beams.
- Preserve Croissant's diagonal-in-\(m\) time evolution and single-einsum hot
  path.
- Use IAU polarization conventions at the public API and provide an explicit
  HEALPix/COSMO converter.
- Use the corrected S2FFT branch
  [`fix/healpix-spin-recursion-node`](https://github.com/slosar/s2fft/tree/fix/healpix-spin-recursion-node),
  pinned to commit
  [`cefdf468ec2540818bafb37ed60d7b1fbba2f21f`](https://github.com/slosar/s2fft/commit/cefdf468ec2540818bafb37ed60d7b1fbba2f21f).
- Keep the existing scalar `Sky`, `Beam`, `Simulator`, and multipair APIs
  working without numerical or shape changes.
- Co-develop with the first producer/consumer: **luseepy** (the LuSEE-Night
  instrument-response refactor). In the co-development checkout the two
  repositories are siblings and the consumer-side plans are `../PLAN.md` and
  `../AGENTS.md`. Pair–Stokes beams are produced by luseepy's
  `beam_conversion` / `Beam.pair_stokes_maps` in the bare (open-circuit)
  antenna basis.

The input format is settled against the first producer:

- Layout `(pair, frequency, stokes, theta, phi)`, complex, Stokes order
  IQUV.
- One `pair` entry is one complex visibility product; there is no separate
  output-product axis. The producer supplies the 10 unique port pairs
  (a ≤ b) of a 4-port instrument; the reversed pair is the complex
  conjugate and is not stored. Splitting cross-correlations into real and
  imaginary output channels is a consumer-side concern.
- The producer's native grid (1° equiangular including the poles, with the
  wraparound phi bin dropped: 181 × 360) is exactly the s2fft "mwss"
  sampling at L = 180, so the first consumer needs no regridding path.

## Architectural invariant

Croissant must not evaluate polarized visibilities in pixel space at every
time. Time must continue to enter only through the diagonal azimuthal rotation

\[
    p_m(t) = \exp[-i m \phi(t)].
\]

If Croissant received separate antenna Jones patterns, the measurement
equation would contain a product of three fields,

\[
    V_{ab} = \int h_a(\Omega)\,B_E(\Omega,t)\,
                   h_b^\dagger(\Omega)\,d\Omega ,
\]

and keeping those factors separate in harmonic space would introduce
Wigner-3j mode coupling. This is deliberately outside the proposed design.

The supplied pair–Stokes beam has already contracted the two Jones patterns.
For visibility product \(p\), let its four directional response functions be
\(M_{p,A}(\Omega)\), with \(A \in \{I,Q,U,V\}\). The measurement is linear in
the Stokes sky:

\[
    V_p(t) =
    \sum_A \int M_{p,A}(\Omega)\,S_A(\Omega,t)\,d\Omega .
\]

After transforming the scalar and spin fields into matching harmonic bases,
this remains a sum of diagonal sky–beam overlaps. Schematically,

\[
    V_p(t) =
    \sum_{c,\ell,m}
    a^{c*}_{\ell m}\,
    e^{-i m\phi(t)}\,
    b^{p,c}_{\ell m},
\]

where \(c\) labels the chosen internal polarization basis. The runtime kernel
therefore remains equivalent to

```python
jnp.einsum("fclm,tm,pfclm->tpf", sky_alm.conj(), phases, beam_alm)
```

up to the final, convention-dependent definition of the harmonic dual basis.
The exact conjugations and factors are not to be inferred from this schematic
equation: they will be derived explicitly and locked down with analytic sign
tests before becoming API.

The expected per-time complexity is
\(O(N_\mathrm{pair} N_\mathrm{freq} N_\mathrm{component} L^2)\). Polarization
adds a small component dimension but does not introduce harmonic mode coupling
or a per-time pixel-space operation.

## Proposed public data model

### Polarized sky

Add a polarized sky type while retaining `Sky` as the scalar, I-only type:

```text
PolarizedSky.data.shape == (frequency, stokes, spatial...)
PolarizedSky.stokes      == ("I", "Q", "U", "V")
PolarizedSky.convention  == "IAU"
```

Using a distinct type avoids interpreting an ordinary scalar array
ambiguously and prevents silent changes to `Sky.compute_alm()` return shapes.
The implementation may share most spatial-grid and dense-engine machinery with
`SphBase`.

Required behavior:

- Stokes ordering is explicit and validated.
- Missing Stokes components may eventually be supported through named
  construction helpers, but the first implementation should require all four
  components to avoid shape inference.
- IAU is the default public convention.
- COSMO data requires an explicit convention argument or conversion call.
- Sky pixel data remain real-valued physical Stokes parameters. Harmonic
  coefficients are complex and fully differentiable with respect to those
  pixels.

### Precomputed pair–Stokes beam

Add a provisional `PairStokesBeam`/`MuellerBeam` type:

```text
beam.data.shape == (pair, frequency, stokes, spatial...)
beam.stokes     == ("I", "Q", "U", "V")
beam.convention == "IAU"
```

Resolved: the producer (luseepy) calls these pair–Stokes responses, so the
class name is **`PairStokesBeam`**. Strictly, one complex visibility product
has a four-element Stokes response vector (one row of a Mueller-like
operator), rather than necessarily a complete \(4\times4\) Mueller matrix.

Required behavior:

- Pair-beam samples may be complex.
- A pair entry produces one complex visibility.
- Multiple feed products such as XX, XY, YX, YY or RR, RL, LR, LL are separate
  pair/product entries, or are flattened into that axis by an adapter.
- Baseline fringe, antenna cross-products, leakage, and any desired feed
  response are assumed to have been incorporated by the beam producer.
- The coordinate frame, tangent-basis convention, pair ordering, baseline
  direction, and visibility definition must be metadata rather than implicit
  assumptions.
- Conversion between IAU and COSMO must transform the Q/U response
  contragrediently so the physical visibility is unchanged.

### Simulation output

For polarized pair beams, the natural output is

```text
(time, pair, frequency), complex
```

This agrees with the current multipair orientation. Croissant should not
silently reinterpret the pair axis as Stokes output. Convenience metadata or
adapters may label the pair entries as linear/circular feed products.

The scalar path remains:

```text
Simulator.sim() -> (time, frequency), real
```

An additive `PolarizedSimulator` is initially safer than changing
`Simulator.sim()` dispatch and return types. Sharing an internal simulator base
can avoid duplication. This choice should be revisited after the first
end-to-end prototype.

Independently of `PolarizedSimulator`, the component-aware convolution must
remain a public, composable primitive (as `multi_convolve` is today): the
first consumer calls Croissant's transforms, rotations, and convolution
directly from its own simulator classes and does not use the
`PolarizedSimulator` orchestration.

## Polarization and sign conventions

The final implementation must have one normative convention document; these
rules must also appear in public docstrings. No sign may be determined merely
by matching a visually plausible map.

### Tangent basis and Q/U

The user-facing convention is IAU. HEALPix/COSMO input is supported explicitly.
According to the
[HEALPix conventions](https://healpix.sourceforge.io/html/intro_HEALPix_conventions.htm),
IAU and HEALPix/COSMO differ by

\[
    U_\mathrm{IAU} = -U_\mathrm{COSMO},
\]

with I and Q unchanged. V conversion is handled separately because its sign is
tied to the adopted circular-polarization convention.

For the HEALPix/COSMO tangent-basis rotation convention,

\[
\begin{aligned}
Q' &= \cos(2\psi)Q + \sin(2\psi)U,\\
U' &= -\sin(2\psi)Q + \cos(2\psi)U.
\end{aligned}
\]

Consequently,

\[
    P_+ = Q+iU,\qquad P_- = Q-iU
\]

transform as spin \(+2\) and spin \(-2\), respectively, under the HEALPix spin
labeling. S2FFT's implemented spin convention must be checked against these
equations with a map-space basis-rotation test; matching an API argument named
`spin=2` is not sufficient evidence by itself.

Under the same convention, the intended E/B definitions are

\[
    a^E_{\ell m}
      = -\frac{a^{+2}_{\ell m}+a^{-2}_{\ell m}}{2},
    \qquad
    a^B_{\ell m}
      = -\frac{a^{+2}_{\ell m}-a^{-2}_{\ell m}}{2i}.
\]

Modes with \(\ell<2\) have no spin-2 content and must be zero by construction.

### Stokes V

The public convention will be IAU/IEEE. Before implementing the coherency or
feed-product adapters, the following must be quoted from a normative reference
and fixed in tests:

- the viewing direction used to define right-hand circular polarization;
- whether positive V is RCP minus LCP;
- the time/Fourier convention for the electric field;
- the corresponding sign in the XY/YX and RL/LR coherency entries.

This is a known source of incompatible conventions. The implementation must
not include a bare `+iV` or `-iV` without documenting all four choices above.
Pure positive-V fixtures for both linear and circular feeds will be required.

The first producer constructs pair beams from fields defined with the
engineering \(e^{+j\omega t}\) time convention (luseepy AGENTS.md); this is
one of the four choices above and must be recorded in the normative
convention document. The producer's V-sign freeze (a circular point-source
test, luseepy PLAN Phase 8.3 / caveat C4) and Croissant's positive-V
fixtures must share the same fixture definitions so both packages lock the
same sign.

### Rotations and phases

Croissant currently uses

\[
    \exp[-im\phi(t)]
\]

for the time-dependent z rotation. Polarized components must preserve this
sign. Spin affects how Q/U and their response kernels are represented, but a
z-axis rotation of properly defined spin-harmonic coefficients remains
diagonal in \(m\).

General coordinate transformations must rotate spin-weighted fields with
their tangent bases. Applying a scalar map rotation independently to Q and U is
incorrect. If an internal I/E/B/V coefficient representation is used, its
equivalence to direct spin-\(\pm2\) rotation must be verified numerically.

The beam and sky transformations form a dual pair. A sign or conjugation
change on only one side can preserve some auto-correlation tests while giving
incorrect cross-hands, so rotation tests must use complex pair beams.

## Harmonic representation

I and V are spin-0 fields. Q and U are not independent scalar fields under
coordinate rotation.

Two implementation representations will be evaluated:

1. Store \(I\), \(V\), \(P_+\), and \(P_-\) coefficients and construct
   corresponding opposite-spin/dual pair-beam coefficients.
2. Convert the polarization block to E/B coefficients and derive pair-beam
   E/B duals such that the final overlap is the same Hermitian-looking
   contraction used by the scalar path.

The selected representation must satisfy all of the following:

- one componentwise, diagonal-in-\((\ell,m)\) overlap at simulation time;
- the same \(e^{-im\phi(t)}\) phase array for every component;
- no loss of information for complex pair beams;
- straightforward full-sky coordinate rotations;
- reverse-mode JAX gradients through sky and beam pixels;
- an explicit derivation of every factor of 2, complex conjugation, and
  \((-1)^m\);
- analytic agreement in pixel and harmonic space.

The E/B form is likely easier for rotation and for presenting a four-component
internal axis, but it must not be chosen until the dual beam transform has been
derived for arbitrary complex \(M_Q,M_U\). Reality identities valid for a real
sky cannot be applied to complex cross-baseline beam responses.

## S2FFT dependency

The pinned S2FFT commit fixes HEALPix spin recursion at exact HEALPix nodes.
The failure mode involves zero Wigner-d recursion entries and can produce
percent-level spin-2 inaccuracies.

Development dependency:

```text
s2fft @ git+https://github.com/slosar/s2fft.git@cefdf468ec2540818bafb37ed60d7b1fbba2f21f
```

The exact revision must appear in the uv lock file. A regression test in
Croissant must fail with the affected release and pass at the pinned commit.
The test should not be only an S2FFT round trip, since related forward and
inverse errors can partially cancel; it should compare against an independent
construction or a known analytic spin field.

Caveat: a Git dependency may be undesirable for a final PyPI release. Before
release, prefer an upstream S2FFT release containing the fix and replace the
Git pin after verifying the same regression test.

## Dense-engine extension

The current dense HEALPix implementation is scalar and exploits real-field
\(m\geq0\) conjugate symmetry. That representation is insufficient for
arbitrary complex pair–Stokes beams.

The cache key must be extended to include at least:

- spatial geometry and sampling;
- `nside` and `lmax`;
- spin;
- reality/complex-input mode;
- iterative-refinement count;
- real/complex precision;
- JAX backend/device as appropriate;
- S2FFT/convention version if it affects generated coefficients.

Required operators:

- real spin-0 analysis for sky I and V;
- real spin-2 sky analysis from Q/U, either directly or through \(P_\pm\);
- complex spin-0 analysis for pair-beam I and V responses;
- complex spin-\(\pm2\) analysis for arbitrary pair-beam Q/U responses.

At `nside=32`, `lmax=30`, one full valid-mode dense matrix has
\(961\times12288\) complex entries: approximately 90 MiB in complex64 or
180 MiB in complex128. Multiple spin operators therefore fit comfortably on
an A100, but cache residency and host/device duplication must be measured.
Relations between spin operators may reduce storage, but only if they remain
valid for complex beam maps.

The direct SciPy scalar spherical-harmonic construction cannot generate
spin-weighted matrices. The spin matrices should be built from the corrected
S2FFT operator or a validated Wigner-d/spin-harmonic construction. In
particular, the implementation must investigate any S2FFT HEALPix restriction
relating \(L\) to `nside`; the dense low-`lmax`, `nside=32` use case must not be
lost.

Matrix creation happens outside JIT and is cached. Matrix application is a
native JAX matmul/einsum and remains differentiable with respect to input maps.
Gradients with respect to the cached transform matrix are not an API goal.

## Beam normalization and ground

Pair–Stokes beams do not necessarily have the same normalization semantics as
the existing scalar power beam.

Initial behavior should therefore be explicit:

- accept a supplied per-pair/per-frequency normalization, or declare that the
  response beam is already calibrated;
- do not infer cross-pair normalization from a complex Stokes response;
- retain current antenna-power normalization helpers for compatible scalar
  workflows;
- define unpolarized ground as \((T_\mathrm{gnd},0,0,0)\);
- compute polarized ground pickup through the pair beam's I response;
- do not apply the scalar `fgnd` correction independently to Q, U, and V.

Ground behavior can be deferred from the first polarized visibility prototype,
but the API must fail clearly rather than silently apply scalar assumptions.

Resolved for the first consumer: luseepy pair–Stokes beams are calibrated
physical responses (products of effective lengths, units of m²). Croissant
applies no normalization and returns the raw complex overlap integrals; the
consumer applies its own physical prefactors (\(k_B\eta_0/\lambda^2\), etc.)
downstream. Ground pickup is deferred: the first consumer computes its
below-horizon (Moon) term algebraically outside Croissant, so the polarized
path only needs to raise clearly if scalar ground machinery is requested.

## Backward compatibility

The following are hard requirements:

- Existing scalar data shapes remain valid.
- `sphere.compute_alm()` retains its scalar defaults.
- `Sky.compute_alm()` and `Beam.compute_alm()` retain their current return
  shapes.
- Scalar `convolve()` keeps its signature and numerical behavior.
- Scalar `Simulator.sim()` keeps returning real `(time, frequency)` arrays.
- Existing multipair callers continue to receive
  `(time, pair, frequency)` complex arrays.
- New component axes are introduced only by explicit polarized types or
  polarized functions.
- Existing tests are not weakened or rewritten merely to accommodate the new
  path.

Where practical, polarized convolution should be implemented as a generalized
primitive and the scalar function should either remain untouched or call it
with a singleton component axis only after strict equivalence testing.

## Implementation sequence

### Phase 1: conventions and transform primitives

1. Pin the corrected S2FFT revision with uv.
2. Add a focused regression for the HEALPix spin-recursion bug.
3. Define public constants/enums for IQUV ordering and IAU/COSMO convention.
4. Implement and test IAU/COSMO sky and response-beam conversion.
5. Generalize low-level S2FFT transforms to accept `spin` and `reality`
   without changing scalar defaults.
6. Implement spin-\(\pm2\), E/B, and inverse-conversion helpers.
7. Derive and test the harmonic dual of a complex pair Q/U response.

Phase 1 is complete only when an analytic polarized pixel-space integral
matches the proposed diagonal harmonic contraction, including complex beams.

### Phase 2: dense transforms

1. Extend dense cache keys and matrix construction for spin and complex maps.
2. Support the low-`lmax`, high-`nside` HEALPix case.
3. Compare dense coefficients and gradients against corrected S2FFT wherever
   the two engines overlap.
4. Benchmark construction time, cached transform time, memory, and JIT
   behavior at `nside=32`, `lmax=30`.

### Phase 3: polarized objects

1. Add `PolarizedSky`.
2. Add the precomputed pair–Stokes beam type.
3. Validate frequency, spatial geometry, Stokes order, coordinate convention,
   and pair metadata.
4. Add precomputation methods analogous to scalar sky/beam ALM helpers.
5. Ensure transforms remain inside the JAX graph when gradients are requested.

### Phase 4: simulator and rotations

1. Implement the component-aware diagonal convolution.
2. Preserve `(time, pair, frequency)` output.
3. Add spin-correct topocentric/equatorial/galactic/MEPA rotations.
4. Verify that the existing z-rotation phase convention is unchanged.
5. Add optional explicit normalization.
6. Add or clearly defer polarized ground pickup.

### Phase 5: documentation and integration

1. Add a normative `docs/polarization.md`.
2. Cross-link it from `docs/math.md`, README, and relevant API docstrings.
3. Include complete examples for IAU and COSMO maps.
4. Include an example with several complex feed products represented along
   the pair axis.
5. Run the complete test suite in supported precisions/backends.
6. Record benchmarks showing that time scaling remains the diagonal-einsum
   path.

## Test plan

### Transform correctness

- Analytic spin-\(\pm2\) harmonic maps.
- Forward/inverse and adjoint checks.
- Correct zeros for \(\ell<2\).
- E/B to Q/U round trips.
- Corrected S2FFT regression at exact HEALPix nodes.
- Dense versus corrected-S2FFT coefficient agreement.
- Float32/complex64 and float64/complex128 tolerances.

### Differentiability

- Reverse-mode gradients with respect to I, Q, U, and V sky pixels.
- Reverse-mode gradients with respect to all complex pair-beam components,
  treating real and imaginary parts consistently.
- Dense and S2FFT gradient agreement.
- Selected finite-difference or complex-step checks outside JIT.
- Gradients through coordinate rotations and the final time contraction.

### Sign and convention fixtures

- Pure positive I, Q, U, and V skies.
- A 45-degree tangent-basis rotation.
- IAU to COSMO conversion and back.
- E-only and B-only patterns.
- Known spin-\(+2\) and spin-\(-2\) modes.
- Positive-V linear-feed and circular-feed correlations.
- Baseline reversal and the expected complex conjugation/Hermiticity.
- Complex leakage response, so tests cannot pass accidentally through
  real-field symmetry.

### Architecture and compatibility

- Pixel-space polarized reference integral versus harmonic result.
- Cross-package oracle: the consumer's direct native-grid quadrature of a
  smooth low-lmax IQUV sky against the Croissant harmonic path (luseepy
  PLAN Phase 8.7), independent of shared transform code. The consumer also
  maintains a second, independent polarized harmonic kernel
  (`TopoJaxSimulator`: per-time Wigner rotations) that consumes the same
  pair–Stokes coefficients — agreement between it and Croissant's
  convolution is part of the consumer's acceptance testing and doubles as
  an external check on this implementation.
- Ingestion of a producer-shaped `(10, N_freq, 4, 181, 360)` complex
  pair–Stokes beam on the mwss grid at L = 180.
- Multiple pair/product entries.
- Arbitrary time arrays and existing \(e^{-im\phi}\) phases.
- No pixel/time tensor is created by the production simulator.
- Scalar regression suite unchanged.
- Existing multipair normalization and gradient tests unchanged.
- JIT cache stability for repeated simulations.

## Acceptance criteria

The feature is complete when:

- all four Stokes components affect complex pair visibilities with documented
  signs;
- IAU is the public default and COSMO conversion is explicit and tested;
- arbitrary complex precomputed pair–Stokes beams are supported;
- the per-time computation remains a diagonal harmonic einsum;
- both corrected S2FFT and dense engines are differentiable;
- `nside=32`, `lmax=30` dense operation is practical on the target GPU;
- coordinate rotations are spin-correct;
- the scalar API and complete existing test suite remain unchanged;
- polarization conventions and caveats are documented from pixels through
  output visibilities.

## Co-development decisions

Resolved against the first producer/consumer, luseepy (2026-07-23):

1. Pair-beam layout: `(pair, frequency, stokes, theta, phi)`, complex, IQUV
   order. Metadata carries the port order, the pair index list, coordinate
   frame, tangent-basis convention, and the frequency grid.
2. One stored pair is one complex correlation; no separate output-product
   axis. (10 unique pairs of a 4-port instrument; conjugate pairs are not
   stored.)
3. Beam class name: `PairStokesBeam`.
4. Response beams arrive calibrated; Croissant applies no normalization and
   returns raw complex overlap integrals. The optional explicit
   normalization argument remains available for other users.
5. Polarized ground pickup is deferred; the first consumer handles its
   ground term algebraically upstream. The polarized path must raise rather
   than apply scalar-ground assumptions.

Still open (to be resolved before the corresponding APIs are finalized):

6. Select the internal \(P_\pm\) or E/B dual representation after the
   analytic complex-beam overlap test (Phase 1, step 7). This is
   Croissant-internal; the consumer imposes no constraint.
7. Replace the temporary Git dependency with an upstream S2FFT release when
   the fix becomes available. Until then, co-developed consumers (including
   luseepy CI) must pin the identical revision.
8. Freeze the joint sign conventions (V sign, \(e^{+j\omega t}\) producer
   convention, IAU tangent basis) in `docs/polarization.md` with fixtures
   shared with the producer's test suite.

API-freeze gate: decisions above marked resolved should be re-validated
against the first representative beam files exported by luseepy's converter
(its Phase 1) before the `PairStokesBeam` constructor signature is frozen in
Phase 3.

