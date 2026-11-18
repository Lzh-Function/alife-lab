# ADR 0015 — Read-only inspector and Particle Life

- Status: accepted (Phase 2, slice D)
- Date: 2026-11-18

## Inspector
- Adapter contract gains `inspect(state, spec, query)`; it must not mutate the state (tested
  for grid, tracker, positions, velocities, ids and RNG state).
- `GET /experiments/{id}/inspect?x&y&radius[&tick]`: without `tick` on a live run the worker
  answers at its next safe boundary (`source.method = live_worker`); otherwise the state is
  reconstructed by deterministic replay (`deterministic_replay`). Inspection commands are not
  stored in `commands`, not broadcast and create no events. Live and replayed inspections of
  the same tick are identical (tested).
- Conway: cell, live-neighbour count, next-tick outcome (checked against the actual step),
  neighbourhood patch. Boids: boids in radius and, for the nearest, the weighted
  separation/alignment/cohesion contributions, computed by a separate function whose sum
  equals the acceleration used by the dynamics (rtol 1e-9). `steering()` is untouched so
  existing runs replay bit-identically.

## Particle Life
- Third substrate; shares the cell-list neighbour search (moved to
  `substrates/particles/neighbors.py`). See `docs/reference_fidelity/particle_life.md`.
- Sweep factor levels may now be structured (e.g. matrices) when labelled.

## Other
- pyarrow is imported lazily (only for chunk I/O): worker RSS for short interactive runs is
  back to ≈54 MB (was ≈113 MB with an eager import).
