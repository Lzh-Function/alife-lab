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

## Measured (CPU, single core, 64 × 64 Lenia, 300 ticks per evaluation)
One evaluation ≈ 30 ms; one CMA-ES generation of 12 ≈ 0.3 s (failing candidates stop early).
Development run, 25 generations × 12, seeds 1–3: CMA-ES best 0.869–0.871 (m ≈ 0.136–0.139,
s ≈ 0.018–0.019), 57–66 % of samples viable; random search 0.832–0.849, 3–8 % viable. The
catalogued Orbium parameters score 0.72. These are E1 observations on one objective.

## Limitations
Lenia is the only target. The centroid speed is not a creature tracker (separate blobs share a
centroid). With the seed-free pattern initialisation, extra search/holdout seeds repeat the same
evaluation. No QD archive, novelty or lineage yet (planned).
