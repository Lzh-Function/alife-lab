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

## Discovery ledger (ADR 0025)
Assay plan v1 (tests/reference/test_assays.py, integration test_discovery_ledger): the found
variant (m 0.1358, s 0.018) passes all checks under `faster_than_base` (speed 0.772 vs base
0.623 in a 128² world over 3000 ticks) → validated, E2. Negative controls: a dissolving
candidate (m 0.17) fails every check → rejected; the base parameters claiming to be faster
than themselves fail `baseline` → rejected. On the development DB, the best candidate of
random search seed 1 (m 0.1309, s 0.0162, speed 0.750) is also validated, so faster variants
form a region rather than a single point.

## ASAL objectives (ADR 0026, tests/reference/test_asal.py)
Ports of the ASAL scores are checked on hand-computed cases (prompt tiling t mod P, zero-masked
lower triangle of open-endedness, illumination). CLIP ViT-B/32 (OpenAI weights, SHA-256
pinned) on a synthetic disc / stripes / noise set ranks each image's own description first.
A CLIP search is deterministic and continues bitwise from a checkpoint; on the GPU a worker
restart passes replay verification. Measured (RTX 5070, 128² Lenia, 200 ticks, 12 candidates
per generation): ≈ 0.8–0.9 s per generation, peak torch pool 690 MiB, worker RSS 1.7 GB,
16 CLIP images in 14 ms.

Findings (exploratory, E0–E1):
- The empty world scores 0.254 against "a biological cell"; best candidates reach only
  0.26–0.29. Before `blank_score` existed, CMA-ES started in a region where every candidate
  died and stayed on this blank plateau.
- With random initial patches the best scores did not survive holdout seeds (0.27–0.285 →
  0.247–0.258): they were lucky initial states. The presets therefore start from Orbium.
- Prompt specificity: the best pattern of a pilot was as similar to "a car" (0.246) as to "a
  biological cell" (0.245) and most similar to "a microscope image of cells" (0.274).
- Sweep `search_asal_cma_vs_random` (seeds 1–3, 240 evaluations each): random 0.281 vs
  CMA-ES 0.261 (Cliff's δ = −1, exact p = 0.10, the minimum for 3 vs 3); one CMA-ES run
  stayed at the blank level. On this weak, rugged objective uniform sampling did better.

## ASAL on multichannel Lenia (ADR 0027)
Configured like `SakanaAI/asal` (5N7KKM, 128², 256 ticks, prompts "a biological cell" /
"two biological cells" on 2 frames, 64 × 64 centred RGB crop, 45 kernel parameters within
±1 logit, population 16, step 0.1 logit). Pilot seed 101 (excluded from the sweep): CMA-ES
0.293 vs random 0.285 after 40 generations; the creature itself scores 0.244, an empty
crop 0.253. Prompt specificity of the pilot's best CMA-ES pattern: cell prompts 0.290–0.301
vs "a car" 0.212, "a city at night" 0.195, "a cat" 0.208 (single-channel Lenia: 0.245 vs
0.246). Confirmatory sweep `search_asal_lenia_mc_cma_vs_random` (seeds 1–5, 640 evaluations
each, GPU): CMA-ES 0.2948 ± 0.0035 vs random 0.2869 ± 0.0056, Cliff's δ = 0.84, exact
p = 0.024 — the pre-declared hypothesis holds. Per run ≈ 158 s with 4 runs sharing the GPU,
peak torch pool 884 MiB, worker RSS 2.1 GB. Evidence: E1 for "CLIP rates these patterns as
more cell-like"; nothing here says they are cells.

## Limitations
Lenia is the only target. The centroid speed is not a creature tracker (separate blobs share a
centroid). With the seed-free pattern initialisation, extra search/holdout seeds repeat the same
evaluation. No novelty search or CVT grid yet (planned).
