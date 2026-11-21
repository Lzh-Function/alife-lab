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

## Limitations
Flat floor only; legs only on the torso's equator; no sensors or feedback control (open-loop
CPG); no self-collision. Development note: joint defaults were moved to a `limb` class; this
was hygiene, not a bug fix (the free joint had not been damped before either).
