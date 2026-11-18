# Substrate contract — `flow_lenia`

- State: `cells` float64[H,W,C] ≥ 0 (not bounded by 1), `tick`, `mass_baseline`, centroid
  tracker. The rule and kernel FFTs are derived from the spec on load; the realised rule is
  stored in checkpoint metadata and must match.
- Modes: reference (official JAX semantics, see `docs/reference_fidelity/flow_lenia.md`) and
  exploratory. Clock unit: tick. Backend: `numpy` (float64).
- Rule: explicit `kernels` (source, target, r, m, s, h, a[3], b[3], w[3]) with `R`, or sampled
  from the seed with a C×C `connectivity` count matrix (default `[[10]]` for one channel).
- Interventions: `paint {x, y, radius, value, channel?}`, `erase {x, y, radius, channel?}`.
- Inspection `{x, y, radius}`: per channel value, affinity U, α, flow and transport offset;
  patch mass; the realised rule.
- Metrics: `mass`, `mass_drift`, `alive_fraction`, `max_value`, `centroid_displacement`.
- Display codec `field_u8`: header `{width, height, channels}`, payload uint8 HWC.
- Comparison: `max_abs_difference` with `exact_equal`.
