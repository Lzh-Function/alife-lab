# Model card — Embodied creature (`creature`, plugin 0.1.0, exploratory only)

Project model (no reference implementation): MuJoCo 3.17 is the physics engine; morphology,
controller and metrics are this repository's. Inspired by Sims (1994) virtual creatures.

## Model
Torso sphere (radius `torso_radius`); leg i at angle a on the equator: thigh `upper` outwards,
hip hinge about z (±45°); shin `lower` outwards/down (0.2/0.8 of its length), knee hinge about
the tangent (±60°). Same density for all parts; collisions only with the floor (condim 3).
Leg joints: damping 1, armature 0.05, springs `joint_stiffness` to the rest pose. Motors
`gear * ctrl`, ctrl ∈ [−1, 1] from a CPG evaluated at every physics step (10 ms; 5 per tick),
integrator implicitfast. Negative control: `controller.kind = off`.

## Verified (tests/reference/test_creature.py, tests/integration/test_creature_platform.py)
| Property | Result |
|---|---|
| Free fall before contact: centre-of-mass velocity = −g t (no motor torque) | rel. 1e-9 |
| Same with moving legs (armature breaks exact momentum conservation) | < 1 % |
| Free root joint has no damping or armature | pass |
| Joints, actuators and geoms generated per leg; CPG formula | pass |
| Negative control (off): displacement < 0.1 m in 20 s, zero motor work | pass |
| Determinism; checkpoint (mjSTATE_INTEGRATION) ⇒ bitwise continuation; push intervention | pass |
| Worker restart + push + replay verification; past inspection by replay | `verified` |

## Measured (CPU, one core)
A 4-legged creature: 1000 physics steps ≈ 5.6 ms. Hand-designed gaits over 20 s:
wave 0.99 m, upright ≥ 0.93; trot 3.3 m but tips over (upright −1); hexapod tripod 0.13 m;
off 0.00 m. MJX comparison in ADR 0030.

## Evolution of body and gait (pre-registered before the sweeps, 2026-11-21)
H1: with CPG and body searched together, CMA-ES reaches a higher final best_score than
uniform random search (sweep `search_creature_coevolve_cma_vs_random`, seeds 1–5, 640
evaluations each). H2: co-evolution CMA-ES reaches a higher final best_score than CPG-only
CMA-ES (sweeps `search_creature_coevolve_cma_vs_random` and `search_creature_gait_cma_vs_random`,
same seeds and budget; exact permutation test on the 5 + 5 values, one-sided direction as
stated). Pilot seeds 101–102 are excluded.

### Results (sweeps run 2026-11-21 after the pre-registration above)
| Condition (640 evaluations, seeds 1–5) | final best_score (m, 20 s) |
|---|---|
| Co-evolution, CMA-ES | 16.1, 23.0, 25.1, 15.9, 22.0 (mean 20.4) |
| Co-evolution, random | 12.7, 11.5, 9.2, 12.0, 10.3 (mean 11.1) |
| CPG only, CMA-ES | 9.9, 16.6, 30.3, 12.5, 21.0 (mean 18.0) |
| CPG only, random | 11.5, 8.6, 10.7, 11.5, 10.1 (mean 10.5) |

H1 holds (Cliff's δ = 1.0, exact p = 0.0079). H2 is not supported: +2.4 m, one-sided exact
p = 0.29 (δ = 0.28). Hand-designed wave gait: ~1 m.
Not pre-registered (descriptive): every co-evolved best body is lighter than the fixed one
(12.5–19.2 kg vs 26.8 kg). Re-running each CMA-ES best for 60 s: co-evolution 22.6, 67.9,
30.6, 25.6, 10.1 m; CPG only 8.7, 47.2, 92.0, 6.6, 51.1 m (all upright). Several bests walk in
curves or circles: the objective measures the distance at 20 s only (the pilot's CPG best
covers 53 m of path in 60 s but ends 3.3 m from the start). A claim such as "moves far away"
needs a longer re-test, which the score alone does not provide.

## Limitations
Search keeps the number of legs fixed (4); objective v1 rewards distance at 20 s, not
straightness. Flat floor only; legs only on the torso's equator; no sensors or feedback control (open-loop
CPG); no self-collision. Development note: joint defaults were moved to a `limb` class; this
was hygiene, not a bug fix (the free joint had not been damped before either).
