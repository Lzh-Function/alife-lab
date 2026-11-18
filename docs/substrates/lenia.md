# Substrate contract — `lenia`

- State: `cells` float64[H,W] in [0, 1], `tick`, centroid tracker (`last`, `dx`, `dy`).
  The kernel FFT is derived from the spec on load and is not stored in checkpoints.
- Modes: reference (LeniaND default CPU path, see `docs/reference_fidelity/lenia.md`) and
  exploratory. Clock unit: tick. Backend: `numpy` (float64 FFT).
- Initialisation: `pattern` (catalogue code, currently `O2u` = Orbium unicaudatus, centred),
  `random` (seeded PCG64 `size`² patch with `density`), `empty`.
- Interventions: `paint {x, y, radius, value}`, `erase {x, y, radius}` (disc on the torus).
- Inspection `{x, y, radius}`: value, potential U, growth G(U), next value, patch mass/max.
- Metrics: `mass`, `alive_fraction` (A > 0.01), `max_value`, `centroid_displacement`.
- Display codec `field_u8`: header `{width, height}`, payload uint8 `round(A·255)` row-major.
- Comparison: `max_abs_difference` with `exact_equal`.
