# Model card — Protocells (`protocell`, plugin 0.1.0, exploratory only)

An abstract compartment model written for this project (no reference implementation; the spec
rejects `mode: reference`). It makes no claim about membrane biophysics or real chemistry. Its
purpose is the spec's protocell DoD: an exact matter/energy ledger, division with recorded
parent–child identity, and controls that change lifespan and inheritance.

## Model (see `substrates/protocell/model.py` for the fixed order)
Integer molecule counts everywhere (field food/waste; per cell food, catalyst, membrane,
waste); inflow (Poisson), decay and diffusion (binomial hops); uptake through the membrane
(binomial with 1 − exp(−permeability)); tau-leaped binomial chemistry with catalyst
concentration C / area: lipid synthesis F + C → C + M, replication F + C → 2C, maintenance
F → W proportional to the membrane (the energy expenditure), membrane and catalyst decay;
waste export; rupture by internal pressure; death below a minimum membrane; division at a
membrane threshold with binomial (p = ½) partition of every content and log-normal mutation of
heritable traits (k_lipid, k_replicate, permeability); Brownian motion and overlap repulsion.

## Ledger
`field food + field waste + Σ cells (food + catalyst + membrane + waste)
 = initial + inflow − degraded`, exactly (integers). Interventions that add matter (`add_food`,
`add_cell`) are booked as inflow; `kill` releases contents into the field. Every process count
(uptake, released, lipid, replication, maintenance, decays, divisions, ruptures, deaths) is in
the ledger, so energy expenditure (maintenance events) is accounted alongside matter.

## Verified (tests/reference/test_protocell.py, tests/integration/test_protocell_platform.py)
| Property | Result |
|---|---|
| Ledger residual is exactly 0 after every tick and every intervention | pass |
| Binomial hopping conserves molecules exactly | pass |
| Division records: parent ends at the division tick, children carry parent and birth tick; every living cell traces back to an initial cell | pass |
| No membrane / no metabolism ⇒ extinction, no division | pass |
| Growth-only control never divides, cells enlarge | pass |
| Random-fraction partition is > 3× more asymmetric than binomial | pass |
| Fixed traits: no mutation | pass |
| Checkpoint ⇒ bitwise continuation; a checkpoint violating the ledger is rejected | pass |
| Worker restart + food intervention + replay verification | `verified` |

## Observations (E1, 64², 20 initial cells, seed 0, 1500 ticks)
| condition | population | divisions | deaths | partition asymmetry |
|---|---|---|---|---|
| default | 20 → 204 | 184 | 0 | 0.066 |
| no membrane | 0 (by tick 300) | 0 | 20 | — |
| no metabolism | 0 (by tick 300) | 0 | 20 | — |
| growth only | 20 (radius 1.8 → 4.6) | 0 | 0 | — |
| random partition | 209 | 271 | 80 | 0.54 |
Mean heritable traits barely moved within 1500 ticks (k_replicate 0.020, permeability
0.050 → 0.053); selection on traits is not claimed.

## Measured
6.2 ms/tick with ~200 cells on 64² (NumPy, one process); ~1.4 ms/tick without cells.
