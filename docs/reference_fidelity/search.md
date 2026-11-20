# Model card — Search (`search`, plugin 0.1.0, exploratory only)

An **optimisation run**: it proposes parameter sets for a target world (Lenia), runs each
candidate headless with the unchanged Lenia adapter and scores it with a pre-declared
objective. The Lenia rule has no reward; the score is computed outside it.

## Algorithms
- `random`: uniform samples of the normalised box (baseline).
- `cma_es`: CMA-ES in the normalised box, mean started at the centre, step size `sigma0`.
  Update equations of Hansen's `purecma` (positive log-rank weights, rank-one + rank-mu, `hsig`,
  squared-norm CSA). Out-of-box proposals are evaluated at the clipped point and receive the
  penalty `10 · |u − clip(u)|²` in the CMA-ES f-value.

- `map_elites` (with a `qd` section): uniform choice among filled grid cells, child =
  clip(parent + N(0, mutation_sigma)) in the normalised box; uniform samples while the grid is
  empty. The grid (two descriptors, fixed bounds/bins) is also measured passively for the other
  algorithms. Entry needs viability on every search seed; quality = score; ties keep the
  earlier elite. `parent` gives the lineage of mutated children.

## Objective `lenia_moving_localized` v1
Checked every 10 ticks and at the end of `eval_steps`: alive (a cell > 0.01), alive fraction
≤ `max_alive_fraction`, and ≥ 90 % of the mass within 2R of the circular centroid. Viable ⇒
score 0.1 + speed (centroid displacement over the second half / ticks); otherwise
0.1 × (last passing check) / eval_steps. Development finding: a first draft without the
concentration test was won by a world-filling lattice of rings whose shared centroid jitters
(reward hacking); the test that rejects it is kept as a regression test.

## Verified (tests/reference/test_cmaes.py, test_search.py, tests/integration/test_search_platform.py)
| Property | Result |
|---|---|
| CMA-ES strategy parameters equal purecma (pycma 4.5.0) | rel. 1e-15 |
| Mean, C, sigma equal purecma for 40 generations when told the same candidates (N = 2, 3, 4; purecma lazy gap 0) | ≤ 1e-10 rel. |
| Convergence on the sphere; invariance to monotone f transforms; JSON state round trip | pass |
| Catalogued Orbium (m 0.15, s 0.015) viable, score > 0.5; m = 0.17 not viable; ring lattice rejected | pass |
| Candidates run the unchanged Lenia adapter (bitwise equal final field) | pass |
| Holdout seeds never change proposals or scores | pass |
| Deterministic; checkpoint ⇒ bitwise continuation (both algorithms) | pass |
| Worker restart + replay verification; past inspection by replay; algorithm sweep | `verified` |
| QD grid: each elite is the earliest best eligible candidate of its cell; incremental = rebuilt | pass |
| MAP-Elites parents are elites of earlier generations; QD metrics consistent; continuation | pass |
| Spec without `qd` keeps its digest; serialize v1 checkpoints load | pass |

## Measured (CPU, single core, 64 × 64 Lenia, 300 ticks per evaluation)
One evaluation ≈ 30 ms; one CMA-ES generation of 12 ≈ 0.3 s (failing candidates stop early).
Development run, 25 generations × 12, seeds 1–3: CMA-ES best 0.869–0.871 (m ≈ 0.136–0.139,
s ≈ 0.018–0.019), 57–66 % of samples viable; random search 0.832–0.849, 3–8 % viable. The
catalogued Orbium parameters score 0.72. These are E1 observations on one objective.

## QD comparison (sweep `search_qd_comparison`, confirmatory, seeds 1–5)
Lenia (m, s, R ∈ [8, 20]) from the Orbium pattern, grid speed 0–1 (10 bins) × mass 0–160
(16 bins), 2000 evaluations per run. Final coverage: MAP-Elites 0.153 ± 0.011, random
0.089 ± 0.011, CMA-ES 0.119 ± 0.032. MAP-Elites vs random: exact permutation p = 0.0079
(Holm 0.016), Cliff's δ = 1.0 — the pre-declared hypothesis holds. CMA-ES vs random:
p = 0.10 (Holm), δ = 0.6. MAP-Elites vs CMA-ES was not pre-declared (descriptive only).
CPU time per run (single core): MAP-Elites ≈ 19 s, random ≈ 6 s, CMA-ES ≈ 54 s (viable
candidates run all ticks). Pilot seeds 99, 101, 102 were used to choose bounds and budget.
Caveat: quality (0.1 + speed) and the speed descriptor are correlated by construction.

## Limitations
Lenia is the only target. The centroid speed is not a creature tracker (separate blobs share a
centroid). With the seed-free pattern initialisation, extra search/holdout seeds repeat the same
evaluation. No novelty search or CVT grid yet (planned).
