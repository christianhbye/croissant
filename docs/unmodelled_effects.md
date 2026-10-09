# Known unmodelled effects

Croissant leaves some physics out and approximates some of what it keeps.
When a simulation disagrees with data or with another simulator, check this
page before suspecting a bug. Each entry gives the size of the effect, how it
shows up, and its status. Sizes come from simulations or exact frame
calculations unless marked *estimate*.

A PR that changes what croissant models updates this page.

| Symptom | First suspects |
|---|---|
| Offset from an astropy-based reference that scales with sky temperature | [aberration](#annual-aberration-of-the-sky), [Doppler boost](#doppler-boost-of-the-sky-brightness) |
| Terms at 4, 8, ... cycles per sidereal day | [HEALPix aliasing](#healpix-azimuthal-aliasing) |
| Error that grows through a long call | [frozen Earth frame](#earth-frame-frozen-at-the-first-time-sample), [lunar pole drift](#lunar-pole-drift-and-rate-within-a-call) |
| Pointing or timing off by up to ~15″ or ~1 s for future dates | [IERS fallbacks](#earth-orientation-data) |
| Residuals that follow the Sun, the Moon or a planet rather than the stars | [moving sources](#sources-that-move-against-the-stars) |
| Frequency structure that tracks the horizon | [terrain diffraction](#terrain-diffraction) |
| A bump at a bright compact source's transit | [sky model](#sky-model-content) |

## Earth frame and kinematics

### Annual aberration of the sky

The Earth's orbital motion, ~30 km/s, shifts apparent positions toward its
direction of motion (the apex) by up to 20.5″. The Earth's rotation adds up to
0.3″ more (diurnal aberration). Aberration squeezes the sky toward the apex,
which no rotation can represent, and croissant's frames are rotations. Moon
simulations leave it out too; the Moon shares the Earth's orbital motion.

- **Size:** in the MIST comparison of #163 (FEKO dipole, Haslam sky, two
  near-polar sites), an offset of 0.5 K at 40 MHz and 0.04 K at 125 MHz from
  a reference built on astropy's `AltAz`, steady over LST. At mid-latitudes
  the beam passes the apex differently through the day, so expect LST
  structure there (not measured).
- **Signature:** an offset from astropy-based references, which include
  aberration.
- **Status:** not modelled, by decision (#163). Modelling it means remapping
  the sky to apparent positions before the harmonic transform. The apex moves
  ~1° per day, which changes the shift by under 0.4″, so one remap per
  simulated day is enough.

### Doppler boost of the sky brightness

The same velocity brightens the sky toward the apex. For a sky with
`T ∝ ν^β` the change is a dipole of amplitude `(1 − β) v/c`: 3.5e-4 for
β = −2.55 (*estimate*). Like aberration it is fixed on the sky over a day and
applies on the Moon.

- **Signature:** a dipole fixed on the sky whose size scales with the sky
  temperature. Astropy-based references leave it out as well.
- **Status:** not modelled.

### Solar light deflection

The Sun bends light by 1.75″ at its limb, falling to 4 mas at 90° from it
(*estimate*). Astropy's `AltAz` includes it.

- **Status:** not modelled.

### FK5 and GCRS

Skies enter the Earth frame through astropy's FK5 J2000 and are then treated
as GCRS. With #164 the beam lands in GCRS itself. The two frames differ by
33 mas (1.6e-7 rad), so beam and sky are offset by that much: of order 1 mK
at 40 MHz, scaling #163's numbers (*estimate*).

- **Status:** not corrected. The fix routes Earth skies through ICRS, which
  means deciding whether `coord="equatorial"` is FK5 J2000 or ICRS; the two
  differ by these 33 mas.

### Earth frame frozen at the first time sample

The simulation frame is CIRS at `times_jd[0]`, and time turns the sky about
its pole at the rate of the Earth rotation angle. The true pole moves by
precession and nutation: by 44–140 mas after one day and 0.12–0.42″ after
three, for four start dates from 2014 to 2026 (erfa, polar motion held
fixed). Precession alone moves the pole 55 mas per day, and the fortnightly
nutation terms add or subtract about as much. Polar motion, also frozen at
`times_jd[0]`, changes by a few mas per day (*estimate*).

- **Signature:** an error that grows through a call. Splitting the call into
  shorter ones, each with its own `times_jd[0]`, bounds it by the drift over
  one piece.
- **Status:** not modelled.

### Earth orientation data

The frame at `times_jd[0]` takes UT1 and polar motion from astropy's IERS
tables. For dates outside the table (beyond its predictions, about a year
ahead; offline, beyond the copy bundled with astropy) astropy:

- holds UT1 − UTC at its last value, without a warning. While leap seconds
  keep |UT1 − UTC| below 0.9 s, the error is under ~1 s of Earth rotation:
  ~15″ about the pole, the same as shifting every time by ~1 s. The CGPM
  plans to relax that 0.9 s limit by 2035.
- uses the 50-year mean polar motion, with a warning. The error is up to
  ~0.5″.

For dates years ahead, ERFA also warns of a "dubious year", because future
leap seconds are unknown.

## Moon

### Lunar pole drift and rate within a call

The simulation frame is MEPA at `times_jd[0]`, and time turns the sky about
its pole at a constant rate. Over one lunar sidereal day (27.3 days) the real
pole moves 2.2–2.5′ net and up to 2.7–4.4′ along the way, and the turn departs
from a constant rate by up to 0.85′ (SPICE, six start dates from 2020 to
2032). Restarting the frame every 6 h, 1 day or 3 days bounds the pole drift
at 0.13′, 0.52′ or 1.54′ (worst case over start dates in 2026).

- **Size in visibilities:** not measured.
- **Status:** open, #149, which compares three fixes.

Aberration and the Doppler boost (above) apply on the Moon as well.

## Sources that move against the stars

Croissant turns a fixed sky map. The Sun (~1° per day against the stars), the
Moon (~13° per day) and the planets move, so a sky map cannot carry them
through a call: a body painted into the map at `times_jd[0]` drifts from its
true position by that much. Add them outside the harmonic convolution
instead ([Adding point sources](#adding-point-sources)). The sizes below are
for a beam of directivity 6 with the body near its peak (*estimates*).

- **Sun:** the quiet corona, of order 1e6 K over a disk ~0.6° wide, gives of
  order 1e4 Jy at 100 MHz, ~15 K. Solar bursts are orders of magnitude
  brighter.
- **Moon (Earth sites):** it blocks the sky behind it. Away from the Galactic
  plane the 40 MHz sky is ~7,000–22,000 K (Haslam scaled with β = −2.55) and
  the Moon ~230 K, so it removes 0.2–0.7 K. It also reflects terrestrial RFI.
- **Planets:** Jupiter emits decametric bursts below ~40 MHz.
- **Earth (lunar sites):** from the near side the Earth is radio-bright (RFI)
  and stays near one point of the local sky, so it turns with neither the
  stars nor croissant's sky.

## Adding point sources

Add point sources, moving or fixed, outside the harmonic convolution. A
source of flux density `S` at apparent topocentric direction `n_s(t)` (from
astropy's `AltAz`, for example, which includes aberration) adds

```text
dT_ant(t) = h(n_s) B(n_s) S λ² / (2 k N)
```

to the antenna temperature. `B` is the beam, `h` its horizon weight at the
source (0 below the horizon) and `N = Beam.compute_norm()`, the full-sphere
beam integral that `Simulator` divides by. Add it to `sim.sim()` before any
`correct_ground_loss`. A disk much smaller than the beam, such as the Sun or
the Moon, counts as a point. A body that hides brighter sky enters with
negative flux: for the Moon, `S = 2k (T_moon − T_sky) Ω_moon / λ²`.

Evaluated this way, each source sits at its exact position at every time
sample, which a sky map cannot offer:

- a map puts a source at a pixel centre, up to about half a pixel off
  (~0.5° at nside 64);
- the harmonic sum evaluates the beam truncated at `lmax` at the source,
  which departs from the beam near sharp features such as the horizon edge;
- a moving source has no fixed place in a map.

If the diffuse map already contains the source (Cas A and Cyg A are in
Haslam), remove it from the map first.

Croissant has no helper for this yet. #77 (`alm2points`) evaluates a
harmonic series at arbitrary points, which removes the pixel and motion
problems. It keeps the `lmax` truncation, though, since summing the beam's
alm at a point is what the harmonic convolution already does; near the
horizon edge, interpolate the beam's own grid instead.

## Physics outside croissant's scope

### Atmosphere

- **Ionosphere** (Earth): refraction, absorption and emission, each scaling
  roughly as ν⁻² and varying with time. Not modelled; size not measured here.
- **Tropospheric refraction** (Earth): about 0.5° at the horizon and zero at
  the zenith (*estimate*). Not modelled. Astropy's `AltAz` leaves it out too
  unless given a pressure.

### Ground

A uniform temperature `Tgnd` behind a horizon mask, with fractional weights
at the edge (#155) and terrain through `horizon_theta` (#161) or `horizon`.
No ground reflection and no emissivity structure.

### Terrain diffraction

Croissant's horizon is geometric: sky above it counts fully and sky below it
not at all, with fractional weights only for cells the horizon crosses. A
real ridge diffracts. In the knife-edge model (ITU-R P.526), a ridge at
distance `d` sets the angular scale `θ_F = sqrt(λ / 2d)`: 3.5° at 40 MHz and
2.2° at 100 MHz for a ridge 1 km away, 1.1° and 0.7° at 10 km (*estimate*).
At the geometric edge the sky is down 6 dB, and θ_F below it 14 dB. Above the
edge the sky ripples by up to +1.4 dB out to ~2θ_F.

- **Signature:** frequency structure that tracks the horizon. θ_F scales as
  ν^(−1/2), so the effective horizon moves with frequency, which a geometric
  mask cannot do. That matters for global-signal work.
- **Status:** not modelled.

### Sky model content

Bright compact sources such as Cas A and Cyg A are in maps like Haslam only
at the map's resolution and with the map's spectral model, which can differ
from their own spectra; Cas A also fades over time. A residual at such a
source's transit points at the sky model, not at croissant. Replacing the
map's copy with a [point source](#adding-point-sources) fixes the position
and lets the source carry its own spectrum.

## Numerical effects that can look physical

### HEALPix azimuthal aliasing

HEALPix's polar rings have 4 pixels, so a function of θ alone picks up
m = 4k modes, which then turn with the sky. A beam symmetric about the zenith,
at either pole on an nside 8 grid, varied by 1.1e-5 to 3.0e-5 (`1 + cos θ`)
and 7.7e-5 to 1.1e-4 (`cos⁴ θ`, upper hemisphere) of its visibility over a
day. The physics test for this case,
`TestBeamProperties::test_zenith_symmetric_beam_at_a_pole_sees_a_constant_sky`,
uses an MWSS grid and sees 3e-8 to 7e-8, the polar-motion level.

- **Signature:** terms at 4, 8, ... cycles per sidereal day.
- **Avoid:** use MWSS or another equiangular sampling when a result depends
  on azimuthal symmetry.

### Band limit and transform accuracy

- Band-limiting at `lmax` smooths sharp features such as a horizon edge and
  makes them ring.
- HEALPix forward transforms are approximate quadratures; `niter`
  iterations reduce their error.
