# ADR 0030 — Embodied creatures on MuJoCo (CPU); MJX measured and not adopted yet

Status: accepted (2026-11-21)

## Context
Phase 6 asks for embodied evolution (morphology and control co-evolution) with MJX-JAX /
MJX-Warp and a 3D viewer. The project owner deferred the remote/batch compute pillar.

## Decision
- Optional extra `embodied` = MuJoCo 3.17 (Apache-2.0); new substrate `creature`: a spherical
  torso with legs (hip swing about the vertical, knee lift about the tangent), CPG control
  `A_j sin(2 pi (f t + phi_j))`, passive joint springs, MJCF generated from the spec,
  MuJoCo C on the CPU in float64 (deterministic, single-threaded).
- MJX was measured on the RTX 5070 with a 4-legged creature: 1024 × 1000 steps in 8.6 s
  (0.12 M steps/s), 4096 × 1000 in 11.1 s (0.37 M steps/s); one CPU core runs 0.18 M steps/s
  and the machine has 24 cores. For creatures of this size MJX is therefore not the faster
  path; it is not installed. It can return for large models or populations once measured
  again.
- 3D view: codec `scene3d_f32` (geom list in the header; per geom position and rotation
  matrix as float32) rendered by three.js 0.186.1 (MIT), loaded lazily, camera following the
  torso. Intervention `push` (an instantaneous torso velocity change).
- Joint damping, armature and springs are attached to the leg joints only (class `limb`); the
  free root joint is undamped (asserted by a test).

## Consequences
- Search over morphology and controller parameters uses CPU MuJoCo runs (next slice).
- Momentum is not exactly conserved while legs move because of the joints' armature
  (reflected rotor inertia); free fall agrees with -g t exactly without motor torque and to
  < 1 % with moving legs. Documented in the model card.
