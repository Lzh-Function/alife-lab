# ADR 0032 — Rough terrain, knee reflexes and gait visualisation for creatures

Status: accepted (2026-11-21)

## Decision
- `CreatureExperiment.terrain` (optional, kind `rough`) is a MuJoCo height field generated
  from the experiment seed (`SeedSequence([seed, 7])`), so search seeds and holdout seeds are
  different terrains. It is part of the scientific spec; specs without it serialise exactly as
  before (digests unchanged).
- `CreatureDynamics.reflex` (optional): two gains added to the CPG knee commands (tilt and
  foot-in-the-air). Searchable as `dynamics.reflex.tilt_gain|contact_gain` only when the base
  target has a reflex. The reflex is part of the substrate's controller, not a reward.
- Sensing uses the orientation and contacts from the start of the previous physics step
  (one-step delay). These are explicit state (`sensed_rot`, `contacts`) and are checkpointed,
  so restored runs continue bitwise; older checkpoints (no terrain/reflex) remain loadable.
- Visualisation is passive: the `scene3d_f32` frame header carries per-foot contacts, reflex
  commands, tilt, path length and a short torso trail; static terrain heights are fetched once
  via `GET /experiments/{id}/inspect?scene=true` (uint16, ≤ 241² samples). The browser draws
  the terrain with a draped 1 m grid, green feet on contact, the trail, a footfall (gait)
  diagram, a tilt plot and per-leg reflex bars. Preview frames may still be dropped.

## Consequences
- H3 (reflex helps on rough terrain) was pre-registered and not supported; see
  docs/reference_fidelity/creature.md.
- Harder terrains (steps, slopes) or other sensors would be new terrain kinds / spec fields.
