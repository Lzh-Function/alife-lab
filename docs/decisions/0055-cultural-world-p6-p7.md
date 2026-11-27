# ADR 0055: `cultural_world` Stage 5 (P6–P7) — Atomic Local Barter, Division of Labor, Graph Inspection, and CPU/GPU Scaling Boundary

- **Status:** Accepted
- **Date:** 2026-11-27
- **Related:** ADR 0050 (`cultural_world` P0–P3), ADR 0052 (`cultural_world` P4–P5), ADR 0054 (GUI–CUI Parity), `docs/plans/cultural_world_plan.md` (§ 0 Stage 5, § 3 P6–P7)

## Context

`cultural_world` P0–P5 separated genotype, lifetime learning, and compositional/externalized cultural skills (`L0–L3`), verified up to evidence level E3. Advancing to Stage 5 (`L4–L5` primitive social structure & P7 scaling) requires:
1. Interdependent spatially segregated materials (`material_a` west, `material_b` east) that make bilateral local barter functionally meaningful without adding external civilization fitness rewards.
2. Strict mass-conserving, all-or-nothing atomic exchange (`apply_local_exchange`) and an explicit causal intervention (`forbid_exchange`) plus a role-shuffled control (`shuffle_roles_control`) to separate unilateral geographic specialization from genuine bilateral division of labor.
3. Read-only graph inspection (`knowledge_dag` + `trade_network`), deterministic checkpoint v3 persistence, end-to-end GUI/CLI parity (`ResearchHub`, `CulturalGraphPanel`, `InspectorPanel`, `LiveLab`), and empirical CPU hotspot + CUDA MLP scaling measurement before committing to any GPU backend migration.

## Decisions

1. **Opt-In Social Config & Segregated Ecology (`core/spec.py`, `ecology.py`):**
   - `CulturalWorldSocial` adds `allow_exchange=False`, `interdependent_resources=False`, `exchange_range=5.5`, `exchange_lot=0.25`, `exchange_energy_cost=0.005`, `specialization_bonus=0.35`, and `shuffle_roles_control=False`. Keeping both primary flags `False` by default preserves P0–P5 config hashes, RNG streams (`env, birth, policy, learn, transmit, tech`), and tests unchanged.
   - When `interdependent_resources=True`, `init_ecology` and `step_ecology` strongly segregate `material_a` (west cosine peak near `x=0.25`) and `material_b` (east cosine peak near `x=0.75`) with floor `0.03` while maintaining exact `ResourceLedger` accounting (`resource_ledger_residual <= 1e-9`).
   - Presets `cultural_world_exchange` and `cultural_world_no_exchange_control` match across all ecology, composition, and externalization settings, differing solely in `allow_exchange` (`True` vs `False`).

2. **Atomic Local Barter, Specialization Multiplier, and Matched Controls (`social.py`, `model.py`):**
   - `apply_local_exchange(state, exp)` executes after skill/artifact transmission and before metabolism using `state.rngs["transmit"]` (returning immediately with zero RNG draws when exchange is disabled).
   - Under normal operation (`shuffle_roles_control=False`), A-surplus agents (`inv_a > inv_b + 0.04`) pair locally within `exchange_range` on the periodic torus with B-surplus agents (`inv_b > inv_a + 0.04`), preferring complementary roles (`A`/`B`/`dual`) and repeat partners. Transfers move `qty = min(exchange_lot, 0.5 * inv_a[i], 0.5 * inv_b[j])` (`>= 0.03`) atomically, deduct `exchange_energy_cost`, and leave `inv_a.sum()` and `inv_b.sum()` exactly conserved and non-negative.
   - `specialization_multiplier` boosts skill execution efficiency (`1 + specialization_bonus * agent_spec` for `material_a`/`material_b` specialists and trade-backed `dual_a_b` recipes) when `interdependent_resources=True` and `not shuffle_roles_control`.
   - `shuffle_roles_control=True` randomly shuffles candidate pairings and trade directions, rejects un-coverable trades, and disables the specialization multiplier.

3. **Separating Specialization from Division of Labor (`metrics.py`, `social.py`):**
   - Six new `CulturalWorldMetric` literals are added: `trade_count`, `trade_volume`, `specialization_index`, `division_of_labor_index`, `interdependence_index`, `repeat_partner_rate`.
   - `specialization_index` measures mean per-agent concentration across A/B executions (`|ua - ub| / (ua + ub)`), whereas `division_of_labor_index = sqrt(pop_comp * interdependence_index)` requires **both** population-wide complementary A/B balance (`4 * p_a * p_b * specialization_index`) **and** active recent bilateral exchange (`interdependence_index`). Unilateral isolated specialists therefore score high on `specialization_index` (`~0.98`) and `0.0` on `division_of_labor_index`.

4. **Causal `forbid_exchange` Intervention, Checkpoint v3, & Graph Inspection (`interventions.py`, `codecs.py`, `adapter.py`, `app.py`):**
   - `forbid_exchange` supports `{"forbid": bool}` or empty `{}` toggle, updating `state.exchange_override` and returning `{"exchange_enabled": bool}` without mutating genotypes, `dw2`, `energy`, inventories, `skills`, `artifacts`, or ecology fields.
   - `SERIALIZE_VERSION = 3` persists `exchange_override`, bounded `trade_events` (`<= 400`), bounded `partner_pairs` (`<= 2000`), `trade_count`, `trade_volume`, `repeat_trades`, `agent_role_uses` (pruned on agent death), and `agent_trades` (pruned on agent death), while backward-compatibly loading `serialize_version` 1 and 2 checkpoints.
   - `extract_cultural_graph` exports bounded (`max_nodes=80, max_edges=200`) `knowledge_dag` and `trade_network` JSON structures exposed via `GET /api/v1/experiments/{run_id}/inspect?view=graph`.

5. **H4 Assay, P7 Hotspot Benchmark, Honest GPU Decision, & GUI-CUI Parity (`cultural_social_assay.py`, `research_tools.py`, `cli.py`, `frontend/`):**
   - `run_h4_exchange_assay` compares 4 paired-seed arms (`exchange_intact`, `no_exchange_control`, `role_shuffled_control`, `forbid_exchange_midpoint`) using `SeedSequence([seed, 6])`, Cliff's $\delta$, percentile bootstrap 95% CIs, and exact/Monte-Carlo paired sign-flip permutation tests.
   - `run_cultural_scaling_benchmark` measures wall time, `ms_per_tick`, `agent_ticks_per_sec`, peak RSS, `max_ledger_residual`, 6-phase CPU hotspots, and an optional lazy `torch_cuda` / `cupy` batched sensor+MLP probe inside worker/CLI processes (never in the API process).
   - **Empirical GPU Decision:** Measured on NVIDIA GeForce RTX 5070 (`N=400`, `iters=120`), batched CUDA sensor+MLP reproduces CPU `float64` within `2.78e-16` (`float32` within `6.32e-08`) but runs at **`0.169×`** speed (`17.36 ms` CPU vs `102.64 ms` GPU), because `sensors_policy` accounts for only `~3.9%` of step runtime while pairwise `exchange` (`~46%`) and `transmission` (`~39%`) dominate. Accordingly, `cultural_world` keeps pure-NumPy CPU execution by default.
   - Both `cultural_social_assay` and `cultural_benchmark` are registered in `RESEARCH_TOOLS` (23 total tools) and rendered in `ResearchHub.tsx`, while `LiveLab.tsx`, `InspectorPanel.tsx`, and `CulturalGraphPanel.tsx` provide live `forbid_exchange`, role inspection, and interactive SVG DAG / Trade Network views in JA and EN.

## Consequences

- **Positive:** Closed mass conservation (`residual <= 1.62e-11` across 5-seed 500-tick H4 assays and `<= 1.31e-12` across P7 scaling sweeps), strict causal isolation under `forbid_exchange`, exact checkpoint v3 determinism, and 100% browser-to-CLI parity.
- **Negative / Limitations:** Under fixed shared food inflow, exchange increases `tech_utility` (`4.003` vs `3.278` no-exchange) and demographic turnover rather than increasing per-capita standing `mean_energy` (`5.097` vs `5.162`, `Cliff's δ = -0.28`, `exact_paired p = 0.75`). Additionally, with `n=5` seeds, exact paired sign-flip permutation has a discrete minimum two-sided p-value floor of `2/32 = 0.0625`.
- **Reversibility:** All P6–P7 engine behavior is gated behind opt-in `CulturalWorldSocial` fields and additive API/UI views, leaving all P0–P5 presets, tests, and non-cultural substrates untouched.
