# ADR 0011 — Second substrate (Boids) and substrate-neutral platform paths

- Status: accepted (Phase 2, slice 3)
- Date: 2026-11-18

## Decision
- `ExperimentSpec.experiment` is a pydantic discriminated union on `substrate`
  (`ConwayExperiment | BoidsExperiment`). Conway's JSON form is unchanged, so existing spec
  hashes are unchanged (pinned by `test_conway_spec_hashes_are_stable_across_schema_changes`).
- No shared state class: each adapter owns its state. The adapter contract gained
  `render(state) -> (header, bytes)` (codec-specific display payload) and
  `state_distance(a, b)` (exact comparison with a substrate-defined metric). Worker previews,
  checkpoint frames, replay frames, replay verification and comparisons use these instead of
  grid-specific code.
- Intervention kinds are validated against the substrate manifest
  (`conway: brush`, `boids: spawn, remove`); the API accepts any well-formed kind name.
- Frontend: per-substrate descriptors (`frontend/src/lib/substrates.ts`) define tools and
  create-form fields; `WorldView` selects the grid or particle renderer.
- Comparison reports expose `metric` and `distance_to_first`; `hamming_to_first` is kept only
  for Conway for compatibility.

## Consequences
- Adding a substrate requires: spec model, adapter (incl. render/state_distance), manifest,
  presets, UI descriptors, reference tests and a fidelity record.
- Bugs found and fixed while generalising: the replay API overwrote the frame codec, and
  replay provenance over-counted interventions after the frame tick.
