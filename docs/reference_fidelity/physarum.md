# Reference fidelity — Physarum-inspired agents (`physarum`, plugin 0.1.0)

## Reference
J. Jones, "Characteristics of pattern formation and evolution in approximations of Physarum
transport networks", Artificial Life 16(2):127–153 (2010). Implemented from the paper's model
description; no code reused. This is a **phenomenological** transport-network model; it is not
a biological simulation of *Physarum polycephalum* and results say nothing about real slime
moulds.

## Model as implemented (order per tick, documented)
1. Motor stage in a random order (one permutation per tick): move `step_size` along the heading;
   with `exclusion`, a move into a lattice cell occupied by another agent fails and the agent
   takes a uniformly random heading; a successful move deposits `deposit` into the new cell.
2. Sensory stage (reads only): sensors at `sensor_offset` straight ahead and at ±SA; F > L, R:
   keep; F < L, R: rotate ±RA at random; L > R: +RA; R > L: −RA; then random heading with
   probability `p_random_turn`.
3. Food sources add their amount to a disc of the trail.
4. 3 × 3 mean-filter diffusion (periodic), then `trail *= 1 − decay`.

Deviations / choices: positions are continuous and agents start at a uniformly random point
inside distinct lattice cells; the motor stage precedes the sensory stage (Jones describes both
stages but the exact interleaving is a scheduling choice); the sensitivity threshold and
sensor width > 1 are not implemented; `collisions: none` (no exclusion) is exploratory only.
All random numbers of a tick are drawn before the sequential loop, so the Numba-compiled loop
and the Python reference loop are bit-identical (tested).

## Verified (tests/reference/test_physarum.py, tests/integration/test_physarum_platform.py)
| Property | Result |
|---|---|
| Numba motor loop == Python reference loop (30 ticks: positions, headings, trail, occupancy) | bit-identical |
| Exclusion invariant: one agent per lattice cell, occupancy grid consistent (200 ticks) | pass |
| Sensory rule table (keep / left / right / random side) | pass |
| Diffusion conserves the trail before decay; decay removes exactly the fraction | rel 1e-12 |
| Network forms (contrast > 1.0); no-sensing control stays uniform (< 0.4); no-deposit control has no trail | pass |
| Checkpoint round trip incl. RNG state ⇒ bitwise continuation | pass |
| Worker restart + attractant intervention + replay verification across a checkpoint | `verified` |

## Negative controls (environment-mediated interaction removed), 200², 15 %, seed 0
| condition | tick 100: coverage / contrast | tick 1000: coverage / contrast |
|---|---|---|
| Jones model | 0.78 / 0.87 | 0.45 / 1.40 |
| `sensing: false` | 1.00 / 0.21 | 1.00 / 0.21 |
| `deposit: 0` | 0 / — | 0 / — |
The same initial state (same seed) gives a network only when agents both write and read the
field — the DoD contrast "behaviour changes when the environment field is removed".

## Measured (2026-11-19, 200², 6000 agents, Numba motor loop)
1.45 ms/tick in-process (1000 ticks in 1.45 s); GUI at max speed 465 ticks/s incl. metrics and
previews; worker RSS ~157 MB (Numba loaded). Without Numba the Python reference loop is used
(much slower; same results).
