# ADR 0052: `cultural_world` Stage 3 (P4–P5) — Compositional Tech DSL, Externalized Artifacts, Cumulative Culture 5-Criterion Assay & Discovery Ledger E2/E3 Integration

- **Status:** Accepted
- **Date:** 2026-11-24
- **Related:** ADR 0050 (`cultural_world` P0–P3), ADR 0025 (Discovery Ledger), `docs/plans/cultural_world_plan.md` (§ 0 Stage 3, § 3 P4–P5)

## Context

`cultural_world` P0–P3 separated genetic, lifetime, and cultural states over a conserved 3-resource ecology using a 6-op linear skill sequence. Extending to Stage 3 (P4–P5) requires:
1. Compositional techniques built cleanly without pre-baked tech trees or `eval`/`exec`, where order and dependency matter (`combine` requires dual material preparation `process_a` + `process_b`; `transform` requires prior processing).
2. Externalization of procedural knowledge into physical spatial artifacts (`PlacedArtifact`) on the torus arena that conserve material mass in `ResourceLedger`, survive creator death until natural decay or explicit `destroy_objects` intervention, and test Hypothesis H6.
3. Separation of five distinct technological evaluation dimensions (`tech_novelty`, `tech_validity`, `tech_utility`, `tech_transfer`, `tech_improvement`) plus pre-registered 5-criterion cumulative culture validation (`cumulative_culture`) wired directly into the SQLite Discovery Ledger up to **E3**.

## Decisions

1. **Whitelisted Order-Sensitive 10-Op Compositional DSL (`technology.py`):**
   - `ALLOWED_OPS` extends P3's 6 ops (`locate`, `gather`, `process_a`, `process_b`, `extract`, `store`) with composition ops (`combine`, `transform`) and externalization ops (`place_object`, `use_object`).
   - All skills are finite sequences (`<= max_program_length <= 12`) interpreted strictly by `evaluate_pipeline(ops, target_resource)` (zero `eval`/`exec`).
   - Order matters: `combine` yields synergy (`target_resource = 3`, dual `material_a + material_b` crafting) ONLY if both `process_a` and `process_b` precede it; `transform` unlocks catalysis ONLY after preparation (`process_a`, `process_b`, or `combine`). Misordered ops incur penalties.
   - `SkillProcedure` adds `secondary_parent_id` and `root_efficiency` so multi-parent compositions form a true dual-parent Knowledge DAG.
   - `allow_composition=False` strictly forbids `combine`, `transform`, and two-skill composition as a clean negative control.

2. **Physical Externalized Artifacts & Strict Conservation (`PlacedArtifact`):**
   - Creating a `PlacedArtifact` deducts `artifact_material_cost` cleanly from the creator's `inv_a` / `inv_b` inventory into the artifact's `mass_a` / `mass_b`.
   - `ResourceLedger` includes active artifact mass in `observed` inventories so `resource_ledger_residual <= 1e-9` holds across creation, decay (`decay_artifacts`), and `destroy_objects` interventions (which return all artifact mass to local grid cells without mutating agent genomes, `dw2`, or in-head `skills`).
   - Nearby agents within `observation_range` can acquire the encoded procedure asynchronously even after creator death (`mode="externalized"`) and receive a workbench boost when practicing near an active artifact.

3. **Five Separated Technological Metrics & Discovery Ledger Integration (P4–P5):**
   - `compute_metrics` exposes `tech_novelty`, `tech_validity`, `tech_utility`, `tech_transfer`, `tech_improvement`, `active_artifacts`, `externalized_learning_fraction`, and `compositional_skill_fraction`.
   - `src/alife_lab/evolution/cultural_assays.py` defines the pre-registered `cumulative_culture` plan (`plan_hash` locked prior to execution) with five independent-seed checks:
     1. `post_inventor_retention` (survival, deaths > 0, post-inventor retention >= 0.15)
     2. `compositional_refinement` (depth >= 2.0 & improvement over paired `allow_composition=False`, Cliff's delta > 0)
     3. `transmission_ablation` (superior retention and utility/energy over paired `transmission_regime="none"`)
     4. `erase_culture_causal` (midpoint paired fork: `intact` vs `erase_culture + destroy_objects` vs `mute_tx` + optional H6 `erase_only_keep_artifacts` diagnostic)
     5. `heldout_transfer` (unseen faster `shifting_patches` regime + `shock_environment` phase shift vs `transmission_regime="none"`)
   - Judgement promotes discoveries registered in the SQLite Discovery Ledger to **E3** if all 5 checks pass, **E2** if checks 1/2/3/5 pass and only `erase_culture_causal` fails, and **E1 (`rejected`)** otherwise (except **`inconclusive`** on runtime errors).

4. **Two-Level Parameter Sweeps (`core/sweep.py` / `service/sweeps.py`):**
   - Factor paths accept 1 or 2 dot levels (e.g. `dynamics.technology.allow_composition`, `dynamics.transmission.transmission_cost`) via recursive `_set_path` so Stage 3 ablations and phase diagrams validate and execute through the standard sweep service.
