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

Optional terrain (`experiment.terrain`, kind `rough`): a MuJoCo height field of smooth random
bumps — uniform noise on a grid with spacing `wavelength`, bilinearly interpolated to
`resolution`, scaled to [0, `amplitude`], ramped to 0 inside `flat_radius` of the start. The
noise comes from `SeedSequence([seed, 7])`, so every seed is a different terrain.
Optional reflex (`dynamics.reflex`): knee command += `tilt_gain` · lean_i + `contact_gain` ·
[foot i in the air], where lean_i is the horizontal component of the torso's up axis along
leg i (positive when the body leans towards that leg; a positive knee command pushes the
foot down). Sensing has a one-step (10 ms) delay: it uses the orientation and contacts from
the start of the previous physics step; both are part of the checkpointed state.

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
| Terrain deterministic per seed, different across seeds, flat start, equals the hfield (float32) | pass |
| Reflex sign (legs on the leaning side and feet in the air push down); gains 0 = no reflex, bitwise | pass |
| Terrain + reflex: checkpoint ⇒ bitwise continuation (incl. contacts and sensed orientation) | pass |

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

## Rough terrain and reflexes (pre-registered before the sweeps, 2026-11-21)
H3: on rough terrain (height field, bumps ≤ 0.2 m, wavelength 1 m, a new terrain per seed),
CMA-ES evolving body + CPG + reflex gains (tilt 0–2, contact 0–1) reaches a higher final
holdout_best (mean score on 3 unseen terrains) than CMA-ES evolving body + CPG without
feedback. Sweeps `search_creature_rough_reflex_sweep` vs `search_creature_rough_plain_sweep`,
seeds 1–5, 16 × 80 generations × 2 search terrains each; one-sided exact permutation test on
the 5 + 5 values. Pilot seeds 101–102 excluded. Background: CPG + reflex controllers are known
to be more robust on uneven ground (Taga 1991; Kimura et al.; Ijspeert 2008); this is a
reproduction attempt in this model, not a new claim.

### Results (sweeps run 2026-11-21 after the pre-registration above)
| Condition (1280 evaluations, seeds 1–5) | final holdout_best (m, 20 s, 3 unseen terrains) |
|---|---|
| body + CPG + reflex | 33.4, 24.0, 27.5, 24.1, 18.7 (mean 25.5) |
| body + CPG, no feedback | 34.4, 13.3, 22.0, 18.0, 22.0 (mean 22.0) |

H3 is not supported: +3.6 m, one-sided exact p = 0.19 (Cliff's δ = 0.44; per seed −1.1,
+10.7, +5.5, +6.1, −3.3 m). Re-scoring each best with the final code reproduces every
recorded holdout_best exactly. Not pre-registered (descriptive): every evolved reflex keeps a
clear tilt gain (0.55–1.53; contact gain 0–0.19), and switching the reflex off in each reflex
best lowers its holdout score in 5/5 runs (33.4→30.3, 24.0→9.9, 27.5→21.9, 24.1→21.9,
18.7→17.7 m): evolution builds gaits that rely on the reflex, but lineages without it find
equally good gaits on these bumps. All bests of both sweeps stay upright on 100 % of the ticks,
so 0.2 m bumps may not be hard enough to need feedback; a harder terrain is untested.

## Limitations
Search keeps the number of legs fixed (4); objective v1 rewards distance at 20 s, not
straightness. Terrain is a single kind (smooth random bumps); legs only on the torso's equator;
the only feedback is the two-gain knee reflex (no learning, no other sensors); no
self-collision. The 3D view's terrain is downsampled to ≤ 241² samples (display only). Development note: joint defaults were moved to a `limb` class; this
was hygiene, not a bug fix (the free joint had not been damped before either).
