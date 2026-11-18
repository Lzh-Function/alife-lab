# ADR 0013 — Sweeps and seed-level statistics

- Status: accepted (Phase 2, slice B)
- Date: 2026-11-18

## Decision
- `SweepSpec` (hashed, `extra="forbid"`): base preset or spec, overrides, `seeds` (unique,
  ≤ 100), up to two `factors` (paths under `dynamics|world|initialization|runner|observations`),
  `reference_condition`, a pre-declared `endpoint {metric, at: final|tick}`, `design`
  (`exploratory` | `confirmatory`; confirmatory requires a `hypothesis`), `profile_id`.
  At most 200 runs. Every member spec is validated before anything is written.
- Members are ordinary runs (label `name | condition | seed N`, `ticks_per_second = 0`) that go
  through the normal queue and `ALIFE_MAX_ACTIVE_JOBS` (parallel across runs, one core each).
- Statistics (`alife_lab.analysis.stats`, NumPy only): one endpoint value per seeded run;
  describe; treatment − reference mean difference with a fixed-seed percentile bootstrap CI;
  Cliff's δ; two-sided permutation test of the mean difference (exact enumeration when
  C(n₁+n₂, n₁) ≤ 200 000, otherwise 20 000 seeded permutations with the (k+1)/(m+1)
  estimator); Holm adjustment across comparisons. Verified against brute force.
- Runs that did not complete or have a null endpoint are excluded and listed. Notes are
  returned as codes (+ English text) and translated in the GUI.
- The report does not assign evidence levels.

## Validation
- `configs/sweeps/boids_alignment_ablation.yaml` reproduces, through the platform, exactly
  the per-seed values computed directly with the adapter (bitwise), and the exact p = 2/924
  for complete separation of 6 vs 6 seeds.
- `configs/sweeps/conway_density_survey.yaml` (exploratory) is an example with no detectable
  difference (Holm-adjusted p = 0.66, n = 8 per condition) — null results are reported in the
  same format.

## Limitations
- Interactive runs share the FIFO queue with sweep members (no priorities yet).
- Reports are recomputed on request (deterministic), not stored as artifacts.
- Only comparisons against one reference condition; no factorial/interaction analysis.
