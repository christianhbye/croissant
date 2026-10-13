Contributions are welcome!

- Create an [issue](https://github.com/christianhbye/croissant/issues) if you discover a bug, have a feature request, or a question.
- If you want to fix an issue, clone this repository and install it in developer mode (with `pip install .[dev]`). Then create a [pull request](https://github.com/christianhbye/croissant/pulls) with your changes.

Questions, comments, and suggestions can also be sent to me through email at chbye@berkeley.edu.

## Independent pixel integration tests

`tests/pixel_reference.py` is a test-only scalar Earth simulator. It uses
Astropy to transform equal-area celestial pixels to unrefracted `AltAz`
at each UTC timestamp, evaluates an analytic beam at those directions, and
integrates with NumPy. Its beam integrals and ground fraction are independent
of Croissant. It imports no Croissant code and calls no harmonic transforms.
The synthetic sky and beam need no downloaded maps or instrument files;
Earth orientation comes from Astropy's bundled historical IERS-B table.

Run the integration tests with:

```bash
uv run pytest tests/test_pixel_integration.py
```

They carry the `slow` marker and a 600-second per-test timeout. The default
test suite and CI include them; `uv run pytest -m 'not slow'` excludes them.

Coverage includes three frequencies, two historical epochs, northern and
southern sites, both geographic poles, irregular times across a day,
positive and negative beam rotations, and flat and asymmetric terrain
horizons. Terrain is tested both fixed to the ground and attached to the
antenna. Both the full-beam antenna temperature with frequency-dependent
ground emission and the ground-corrected sky temperature are checked.
A monopole case retains beam response at the horizon to expose boundary-row
normalization errors. Every time/frequency cell must pass; JUnit properties
retain maximum absolute residuals in K and relative residuals per frequency.

| Check | Relative tolerance |
| --- | ---: |
| Flat horizon and unmasked sky | `2e-4` |
| Terrain at harmonic band limit `L=64` | `2e-3` |
| Pixel refinement, flat (`nside=128` to `256`) | `1e-5` |
| Pixel refinement, terrain (`nside=256` to `512`) | `5e-5` |
| Harmonic refinement, flat (`L=32` to `64`) | `1e-5` |
| Harmonic refinement, terrain (`L=64` to `128`) | `6e-4` |

Here `L = lmax + 1`. The terrain edge is not band-limited: its coarser
harmonic grid needs a larger tolerance than the frame tests. Refining that
grid must also reduce its worst relative temperature residual against the
pixel sum, or reach the frame mismatch tolerance of `2e-4`. Once that
tolerance is reached, further refinement need not reduce the residual from
the physical differences between the simulators.

The comparison allows the physical differences listed in
[`docs/unmodelled_effects.md`](docs/unmodelled_effects.md): Astropy includes
aberration and light deflection, while Croissant uses a rotation and freezes
its Earth frame at the first timestamp. The tests separately double pixel
resolution and harmonic band limit to check numerical convergence. These
tolerances apply to the synthetic cases, not arbitrary sky maps or long
observations. Keep the stricter coordinate unit tests: effects smaller than
the apparent-sky mismatch need those tests.

Lunar, polarized, and finite integration-window comparisons remain outside
this reference's scope. A lunar extension can reuse the pixel integrator,
but needs a LunarTopo/SPICE coordinate path and a separate convergence and
frame-drift budget.
