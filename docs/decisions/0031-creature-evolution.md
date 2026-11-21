# ADR 0031 — Evolving creature bodies and gaits with the search plugin

Status: accepted (2026-11-21)

## Decision
- `creature` is a search target: searchable paths `dynamics.controller.frequency_hz`,
  `.amplitudes.<j>`, `.phases.<j>`, `dynamics.body.legs.<i>.(angle_deg|upper|lower|radius)`,
  `dynamics.body.torso_radius`, `dynamics.gear`, `dynamics.joint_stiffness`; the number of
  legs (the genotype's topology) is fixed per search.
- Objective `creature_distance` v1: planar displacement after `eval_steps` ticks × fraction of
  ticks with uprightness > 0.5 (flipping over is penalised); viable = upright ≥ 90 %. Archive
  columns reused: speed = displacement / time, mass = body mass, alive_fraction = upright
  fraction. Evaluations use MuJoCo on the CPU inside the search worker (~12 ms per 20 s
  rollout of a quadruped), no GPU.
- Presets: CPG only (CMA-ES), body + CPG (CMA-ES, random), MAP-Elites over speed × body mass
  (0–1.5 m/s × 0–90 kg) whose parents give a body lineage.

## Consequences
- See docs/reference_fidelity/creature.md for the pre-registered comparisons and the
  "walks in a circle" finding; a straightness-aware objective would be a new version.
