# Reference fidelity — Abstract reaction network (`reaction_network`, plugin 0.1.0)

## Reference
D. T. Gillespie, "Exact stochastic simulation of coupled chemical reactions", J. Phys. Chem.
81(25):2340–2361 (1977) — the direct method. Implemented from the paper; no code reused.
This is an **abstract** artificial chemistry: species are labels, reactions are declared
rules with stochastic rate constants; no physico-chemical validity is claimed.

## Model as implemented
- Propensity `a_μ = c_μ · Π_i C(x_i, s_i)` (binomial coefficients = Gillespie's `h_μ`).
- Direct method: `r1 ∈ (0, 1]`, `τ = −ln(r1) / a0`; reaction chosen by cumulative propensities
  against `r2 · a0`; propensities of reactions that depend on changed species are recomputed.
- One tick = `time_per_tick` simulated time; if the next event would cross the tick end the
  drawn τ is discarded (exact by memorylessness) and time is set to the tick end.
- Buffered species (`constant: true`, Gillespie's X̄) keep their count.
- Disabled reactions (spec `disabled_reactions`, or knock-out interventions) never fire.
- More than `max_events_per_tick` events in one tick raises an error (no silent truncation).
- Cycle **candidates**: directed cycles (length ≤ 6) in the species graph built from reactions
  that fired at least once, with edges from reactants to net-produced species (returned
  catalysts are not products), excluding buffered species; direct autocatalysis listed apart.
  Candidates are observations from the reaction history, not proofs of self-maintenance.

## Verified (tests/reference/test_reaction_network.py, tests/integration/test_reaction_network_platform.py)
| Property | Result |
|---|---|
| Propensities use binomial combinatorics (A + B, 2A, 3A) | exact |
| A + B ⇌ C conserves A + C and B + C every tick (300 ticks, > 1000 events) | exact |
| A → ∅: mean count at t = 1 over 200 seeds matches 100 e⁻¹ (Binomial) | within 4 SE |
| ∅ → A, A → ∅ (100, 1): stationary mean 100 ± 1.5 and variance/mean in (0.85, 1.15) — Poisson | pass |
| Buffered species constant; cannot be changed by interventions | pass |
| Event budget exceeded ⇒ error | pass |
| Catalytic cycle preset: cycle I1→I2→I3→I4 found; knock-out control: replicators extinct, no cycle | pass |
| Lotka preset: X̄ fixed, Y1/Y2 vary, direct autocatalysis {Y1, Y2} | pass |
| Checkpoint round trip incl. RNG, knock-outs and firing counts ⇒ bitwise continuation | pass |
| Worker restart + injection + knock-out + replay verification | `verified` |

## Observations (E0/E1, seed 0)
- Lotka reactions: oscillating Y1/Y2 around ~10³ over the first 10 time units (≈ 200–300
  events per tick of 0.01).
- Catalytic cycle: the four replicators persist with sequential oscillations (each rises after
  its catalyst), counts ~0–550; with catalysis knocked out they are extinct by t ≈ 25–50; with
  raw material only nothing but food in/outflow happens.

## Measured (2026-11-19, pure Python direct method)
0.36–0.62 million events/s including per-tick metrics (5 species, 3–10 reactions); GUI
6.8·10³ ticks/s for the cycle preset; worker RSS ~63 MB.
