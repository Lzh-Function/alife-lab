# Model card — AgentWorld (`agent_world`, plugin 0.1.0, exploratory only)

A project model for spec §6.12 (lightweight learning agents). No reference implementation; the
spec rejects `mode: reference`. **Autonomous run**: there is no external fitness or objective;
agents live, eat, signal, die and reproduce. The only reward is internal (an agent's own energy
change), used by the `lifetime` learning mode.

## Model (see `substrates/agent_world/model.py` for the fixed order)
Periodic continuous world with a food grid (linear regrowth to capacity) and a signal grid
(diffusion, decay). Inputs: food and signal at left/front/right sensors, energy / threshold,
bias. Actions: turn left, forward, turn right (turns move half a step), emit signal.
Controllers: hand-coded `baseline` (turn towards the largest food reading), `random`,
`mlp` (8 → hidden tanh → 4 softmax with temperature). Adaptation modes for `mlp`: `fixed`,
`evolution` (offspring inherit genetic weights + N(0, σ)), `lifetime` (REINFORCE on the output
layer with a running reward baseline; learned deltas are **not** inherited). Feeding shares
contested cells proportionally; reproduction at an energy threshold (child gets half).
Controls: `communication: false`, `shuffle_policies: true`, `kind: random`, `mode: fixed`.
Not implemented (planned): GRU / NCA policies, evolution of learning rules, memory ablation,
attack/defend actions, other scenarios (mazes, cooperative transport).

## Verified (tests/reference/test_agent_world.py, tests/integration/test_agent_world_platform.py)
| Property | Result |
|---|---|
| Action probabilities sum to 1; without communication signal inputs are zero | pass |
| Feeding never takes more than a cell holds; food stays within [0, capacity] | pass |
| `fixed` and `lifetime` offspring carry founder genetic weights exactly; `evolution` mutates | pass |
| `lifetime`: older agents have learned deltas, newborns start with none (not inherited) | pass |
| Policy shuffle moves controllers between agents; `random` gives uniform probabilities | pass |
| Checkpoint (agents, histories, grids, RNG) ⇒ bitwise continuation; inspector history | pass |
| Worker restart + intervention + replay verification; past inspection by replay | `verified` |

## Observations (E1; 100 × 100, 100 agents, seed 0, tick 3000)
| condition | population | standing food | forward rate | signal rate |
|---|---|---|---|---|
| evolution | 324 | 387 | 0.52 | 0.13 |
| lifetime learning | 334 | 380 | 0.61 | 0.11 |
| fixed (control) | 320 | 460 | 0.32 | 0.20 |
| baseline | 325 | 425 | 0.59 | 0 |
| random (control) | 319 | 501 | 0.25 | 0.27 |
| policy shuffle (control) | 318 | 496 | 0.26 | 0.20 |
| no communication | 325 | 407 | 0.47 | 0.15 |
The population saturates at the food-limited carrying capacity in all conditions; adaptive
conditions harvest more (less standing food) and move straight more often. Single seed only:
these are observations, not statistically established effects (use sweeps for that).

## Measured
0.46–0.61 ms per tick with ~320 agents (NumPy, one process).
