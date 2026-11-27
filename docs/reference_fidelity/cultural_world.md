# CulturalWorld (`cultural_world`) — Phase P0–P5 Benchmark & Integrity Report

> **Scientific mode:** `exploratory` only (`mode="reference"` is rejected at `ExperimentSpec` validation; no external biological or historical literature oracle is claimed).
>
> **Architecture:** [ADR 0050](../decisions/0050-cultural-world.md) · [ADR 0052](../decisions/0052-cultural-world-p4-p5.md) · [ADR 0055](../decisions/0055-cultural-world-p6-p7.md) · **Unified Roadmap:** [cultural_world_plan.md](../plans/cultural_world_plan.md)

## 1. Scope & Structural Invariants Verified

`cultural_world` implements a pure-NumPy `float64` autonomous artificial-life substrate designed to isolate **three inheritance/learning mechanisms** under closed mass-conserved ecology and paired causal interventions:

1. **Genotype layer (`w1, b1, w2, b2, gene_lr, gene_social`)** — inherited at birth with configurable Gaussian mutation when `genetic_evolution=True`.
2. **Lifetime plasticity layer (`dw2, rbase, energy, inv_a, inv_b, private_best_*`)** — updated during life via REINFORCE on per-tick energy change (`action = float64` policy over 12 sensors × 6 actions) and private spatial foraging; **zeroed on every newborn** (`make_offspring`).
3. **Cultural layer (`skills: list[SkillSlot]` over `SkillProcedure` DSL)** — whitelisted 6-op procedural recipes (`locate`, `gather`, `process_a`, `process_b`, `extract`, `store`; Python `eval`/`exec` excluded), starting empty (`forbid_preinstalled_recipes=True`), invented/refined through resource trial or acquired via `vertical`, `horizontal`, or `oblique` channels.

Verified invariants (see `tests/reference/test_cultural_world.py`):
- **Conservation ledger (`resource_ledger_residual <= 1e-9`):** exact accounting `observed (field + agent inventory) == initial + inflow - consumed - decayed` for `food`, `material_a`, and `material_b` across `stationary`, `seasonal`, `shifting_patches`, agent deaths (inventories returned to local cell), and `add_resource` interventions. Maximum residual observed across 26,000 benchmark ticks (including delayed checkpoint restores): **`5.695e-10`**.
- **Causal intervention isolation (`erase_culture`):** clears `skills` only while keeping `ids, w1, b1, w2, b2, dw2, energy, inv_a, inv_b, x, y` and `food, material_a, material_b` bit-identical (`np.array_equal`).
- **Deterministic snapshot & split RNG (`rng_env`):** `serialize_state` copies all arrays safely without aliasing; delayed `deserialize_state` reproduces uninterrupted trajectories bit-for-bit.

---

## 2. Measured 5-Seed A–E Ablation Benchmark (Ticks 0–800)

- **Command configuration:** `cultural_world_baseline` preset defaults (`world = 64.0 × 64.0`, `resolution = 32`, `environment_regime = "seasonal"`, `regime_period = 200`, `count = 80`, `max_population = 400`, `controller = "mlp"`, `invention_prob = 0.015`, `max_age = 300`, `reproduce_threshold = 9.5`), **5 seeds** (`0..4`), **800 ticks** per arm.
- **Measured CPU compute (across all 26,000 ticks including midpoint forks):**
  - Wall-clock elapsed: **`44.040 s`**
  - Mean step cost: **`1.694 ms/tick`** (CPU `float64` single-process)
  - Peak RSS: **`103.0 MiB`**
  - Max `resource_ledger_residual`: **`5.695e-10`** (`0.0000` reported at tick-800 endpoints when rounded to 4 decimals)

| Arm | Regime (`V / H / O`, cost) | Pop (t=800) | Births / Deaths | Mean Energy | Skills / Agent | Knowledge Div. | Retention After Death | Social Learn Frac. | Max Depth | Innovation Utility |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A: `genetic_only`** (`lifetime=False`, `V=H=O=False`, `inv=False`) | none | `0.00 ± 0.00` | `0.4 ± 0.5 / 80.4 ± 0.5` | `NaN` (extinct) | `0.00 ± 0.00` | `0.0 ± 0.0` | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | `0.00 ± 0.00` | `0.0000 ± 0.0000` |
| **B: `lifetime_only`** (`lifetime=True`, `V=H=O=False`, `inv=True`) | none | `119.00 ± 24.71` | `274.8 ± 35.4 / 235.8 ± 12.6` | `3.83 ± 0.16` | `0.44 ± 0.10` | `52.0 ± 15.8` | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | `1.20 ± 0.45` | `0.8064 ± 0.0182` |
| **C: `vertical_only`** (`V=True, H=False, O=False`, `cost=0.01`) | V only | `365.00 ± 28.38` | `705.4 ± 77.0 / 420.4 ± 49.9` | `4.67 ± 0.53` | `1.13 ± 0.04` | `249.6 ± 33.8` | `0.3749 ± 0.0304` | `0.5340 ± 0.0316` | `3.00 ± 0.71` | `0.8277 ± 0.0109` |
| **D: `full_free`** (`V=H=O=True`, `cost=0.0`) | V/H/O free | `400.00 ± 0.00` | `1014.2 ± 30.2 / 694.2 ± 30.2` | `15.32 ± 1.58` | `3.44 ± 0.08` | `92.2 ± 12.8` | `0.3473 ± 0.0596` | `0.9795 ± 0.0017` | `10.20 ± 2.28` | `1.2170 ± 0.0114` |
| **E: `baseline_costly`** (`V=H=True, O=False`, `cost=0.01`) | V/H costly | `400.00 ± 0.00` | `992.0 ± 56.3 / 672.0 ± 56.3` | `13.65 ± 3.79` | `3.56 ± 0.20` | `93.2 ± 10.9` | `0.3694 ± 0.0487` | `0.9769 ± 0.0050` | `7.00 ± 1.41` | `1.2341 ± 0.0187` |

*(Mean ± sample standard deviation over 5 seeds.)*

---

## 3. Midpoint Causal Fork Benchmark (Arm E Checkpointed at Tick 400 -> Stepped to Tick 800)

Paired checkpoints saved at `t=400` on Arm E (`seeds = 0..4`) were restored into three causal branches:
1. **`intact`**: control continuation (`V=H=True`).
2. **`erase_culture`**: one-time `erase_culture` intervention at `t=400` (skills wiped to empty; transmission stays enabled; genotypes/`dw2`/energy/resources unchanged).
3. **`mute_tx`**: `set_transmission` intervention (`V=H=O=False`) at `t=400` (existing skills stay in adults until death, blocking further copying).

| Branch & Checkpoint | Pop | Mean Energy | Skills / Agent | Knowledge Div. | Retention After Death | Max Depth | Innovation Utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`intact` (`t=450`)** | `400.00 ± 0.00` | `10.83 ± 1.05` | `3.72 ± 0.09` | `118.2 ± 12.4` | `0.1299 ± 0.0493` | `6.00 ± 1.22` | `1.1560 ± 0.0355` |
| **`erase_culture` (`t=450`)** | `400.00 ± 0.00` | `10.53 ± 0.19` | `3.14 ± 0.16` | `125.4 ± 5.5` | `0.0968 ± 0.0338` | `3.60 ± 0.55` | `1.1272 ± 0.0390` |
| **`mute_tx` (`t=450`)** | `400.00 ± 0.00` | `10.46 ± 0.90` | `3.33 ± 0.19` | `129.2 ± 14.5` | `0.1323 ± 0.0453` | `6.00 ± 1.22` | `1.1273 ± 0.0287` |
| **`intact` (`t=800`)** | `400.00 ± 0.00` | `13.65 ± 3.79` | `3.56 ± 0.20` | `93.2 ± 10.9` | `0.3694 ± 0.0487` | `7.00 ± 1.41` | `1.2341 ± 0.0187` |
| **`erase_culture` (`t=800`)** | `400.00 ± 0.00` | `13.34 ± 1.26` | `3.61 ± 0.17` | `105.0 ± 8.5` | `0.2748 ± 0.0434` | `7.60 ± 1.14` | `1.1401 ± 0.0317` |
| **`mute_tx` (`t=800`)** | `214.80 ± 36.15` | `3.49 ± 0.19` | `0.67 ± 0.05` | `143.4 ± 30.3` | `0.1867 ± 0.0353` | `1.60 ± 0.55` | `1.0666 ± 0.0231` |

---

## 4. Honest Scientific Interpretation, Negative/Null Findings & Evidence Level

- **Evidence Level:** **E3 (intervention-supported within exploratory model)** for channel/transmission causation and **E1–E2** for cumulative refinement depth vs. lifetime plasticity.
- **What succeeded:**
  - Separating the 3 layers prevented hidden Lamarckian inheritance (verified in unit tests and `A_genetic_only` / `B_lifetime_only`).
  - Under the default seasonal-scarcity ecology (`N_0 = 80`), `A_genetic_only` went extinct in `5/5` seeds, whereas lifetime learning + private invention sustained `N ≈ 119`, vertical transmission reached `N ≈ 365`, and peer horizontal/oblique sharing hit carrying capacity (`400`) with deeper skill refinement (`depth 7.0–10.2` vs `1.2–3.0`) and higher utility (`1.22–1.23` vs `0.81–0.83`).
  - Muting cultural channels (`mute_tx`) at `t=400` caused carrying capacity and utility to collapse (`pop 400 -> 215`, `mean_energy 13.65 -> 3.49`, `depth 7.00 -> 1.60`) by `t=800` as original cohorts turned over.
- **Honest negative/null findings & caveats:**
  1. **Single-shot `erase_culture` partially recovers when `N=400` and `invention_prob=0.015`:** at `t=450` (+50 ticks), `improvement_depth` drops sharply (`6.0 -> 3.6`), and at `t=800` `innovation_utility` (`1.14` vs `1.23`) and `skill_retention_after_death` (`0.27` vs `0.37`) stay lower, yet population (`400`) and raw skill count (`3.61`) recover via abundant de novo invention + horizontal spread. True irreversible hysteresis likely requires either rare foundational inventions (`invention_prob << 0.015`), bottlenecked population size, or multi-step prerequisite tech DAGs (planned for Phase P4).
  2. **`C_vertical_only` showed higher `knowledge_diversity` (`249.6`) than `D/E` (`92.2 / 93.2`):** without peer horizontal copying, isolated family lineages retain idiosyncratic variants rather than converging onto high-utility shared recipes.
  3. **`D_full_free` (`cost=0.0`) vs `E_baseline_costly` (`cost=0.01`)** showed higher `improvement_depth` (`10.2` vs `7.0`) in `D`, while `mean_energy` (`15.32` vs `13.65`) overlapped within 1 SD at `N=400` saturation (`max_population = 400` caps demographic separation in `D` vs `E`).
  4. **Scope boundary (P0–P7 vs future stages):** Compositional DSL + conserved `PlacedArtifact` external memory (P4) and the pre-registered 5-criterion E2/E3 cumulative-culture assay & 2D phase diagram (P5) are documented in §§5–8 below; atomic barter, division of labor, `forbid_exchange`, and CPU/GPU scaling benchmarks (P6–P7) are implemented and documented in §§9–12 below. Multi-species Connectome × Culture hybrids remain outside the current implemented scope.

---

## 5. Stage 3 / P4–P5 Extensions & Conservation Invariants

Phase P4–P5 (`docs/decisions/0052-cultural-world-p4-p5.md`) extends `cultural_world` opt-in (`allow_composition=False`, `allow_externalization=False` by default so P0–P3 hashes and behaviors remain invariant; both enabled in `cultural_world_compositional`):

- **Safe order-sensitive 10-op DSL (`src/alife_lab/substrates/cultural_world/technology.py`):** extends P3's 6 ops with `combine`, `transform`, `place_object`, `use_object`. No `eval`/`exec` and no fixed tech tree; `evaluate_pipeline` rewards valid operator ordering (`process_a + process_b` before `combine`/`transform` before `extract`/`store`) and penalizes misordered ops.
- **Dual-parent Knowledge DAG (`secondary_parent_id`, `root_efficiency`):** `compose_skills` synthesizes two owned skills into a child procedure with `depth = max(d_a, d_b) + 1`.
- **Conserved `PlacedArtifact` external memory:** placing an artifact debits `artifact_material_cost` equally from local `material_a`/`material_b` cells; decaying after `artifact_decay_ticks` or clearing via `destroy_objects` returns exact mass to the local cell (`resource_ledger_residual <= 1e-9`). Nearby artifacts boost same-target execution by `+18%` and allow asynchronous acquisition after creator death (`externalized_learning_fraction`).
- **8 new P4 metrics:** `tech_novelty`, `tech_validity`, `tech_utility` (net after `skill_energy_cost`), `tech_transfer`, `tech_improvement` (vs `root_efficiency`), `active_artifacts`, `externalized_learning_fraction`, `compositional_skill_fraction`.

---

## 6. P5 5-Criterion Pre-Registered Assay (`cultural_world_compositional` vs `cultural_world_no_composition`)

Commands run (`5` independent seeds from `SeedSequence([seed, 5])`, `600` ticks/check):
- `uv run --no-sync alife cultural assay cultural_world_compositional --seeds 5 --steps 600 --json`
- `uv run --no-sync alife cultural assay cultural_world_no_composition --seeds 5 --steps 600 --json`

| Pre-Registered Check | `cultural_world_compositional` (`84.711 s` wall) | `cultural_world_no_composition` Control (`66.283 s` wall) |
|---|---|---|
| 1. `post_inventor_retention` | **PASS** · `viable=100%`, `deaths=579.4`, `ret=0.542` | **PASS** · `viable=100%`, `deaths=512.2`, `ret=0.360` |
| 2. `compositional_refinement` | **PASS** · `depth=9.00 vs 10.20 (Cliff's δ=-0.44)`, `net_util=4.969 vs 2.598 (Cliff's δ=+1.00)`, `comp=0.999` | **FAIL** · `depth=10.20 vs 10.20 (δ=0.00)`, `net_util=2.598 vs 2.598 (δ=0.00)`, `comp=0.000` |
| 3. `transmission_ablation` (vs `none`) | **PASS** · `ret=0.542 vs 0.000`, `net_util=4.969 vs 1.364`, `E=17.58 vs 5.41` | **PASS** · `ret=0.360 vs 0.000`, `net_util=2.598 vs 1.400`, `E=5.32 vs 4.78` |
| 4. `erase_culture_causal` (`+60` / mute) | **PASS** · `d60=7.20 vs wiped=5.40`, `mute net_util=4.969 vs 3.751`, `E=17.58 vs 3.84` | **PASS** · `d60=7.60 vs wiped=3.60`, `mute net_util=2.598 vs 2.099`, `E=5.32 vs 3.47` |
| 5. `heldout_transfer` (period×0.65 + shock) | **PASS** · `viable=100%`, `ret=0.485`, `net_util=4.778 vs 1.341` | **PASS** · `viable=100%`, `ret=0.355`, `net_util=2.636 vs 1.395` |
| **Overall Verdict & Evidence Level** | **`validated` (`E3`)** | **`rejected` (`E1`)** |

**Notable empirical finding on Check 2:** Single-parent chain mutation in `cultural_world_no_composition` reaches slightly deeper linear revision chains (`10.20` vs `9.00`, `δ = -0.44`), because two-parent composition synthesizes shorter specialized paths cleanly (`max(d_a, d_b) + 1`), yet compositional synergy nearly doubles net energetic return per material unit (`tech_utility = 4.969` vs `2.598`, **`Cliff's δ = +1.00`**) and raises mean agent energy (`17.58` vs `5.32`).

---

## 7. H6 Externalized Artifact Recovery Benchmark (5 Seeds × Warmup 300 -> +120 Ticks)

Measured on `seeds = 0..4` (`alife-lab` single CPU process): peak RSS **`107.54 MiB`**, `cultural_world_compositional` step throughput **`2.504 ms/tick`** (`300`-tick warmup), and **maximum `resource_ledger_residual = 0.0`** across all sampled ticks and interventions (`<= 1e-9`).

| Arm at `t=300` -> `t=420` | `ext_learn` Frac. | `improvement_depth` | `tech_utility` | `mean_energy` | `active_artifacts` | `ret` After Death |
|---|---:|---:|---:|---:|---:|---:|
| **`ext_intact`** | `0.3041 ± 0.0323` | `7.80 ± 1.72` | `4.526 ± 0.154` | `12.74 ± 0.33` | `96.8` | `0.359` |
| **`ext_keep_artifacts` (`erase_culture` only)** | **`0.2848 ± 0.0350`** | **`7.80 ± 0.75`** | **`4.367 ± 0.177`** | **`12.07 ± 0.50`** | **`101.8`** | **`0.370`** |
| **`ext_erase_and_destroy` (`+destroy_objects`)** | `0.0788 ± 0.0076` | `6.80 ± 1.17` | `3.581 ± 0.101` | `8.49 ± 0.42` | `33.2` | `0.213` |
| **`no_ext_erase` (`no_externalization`)** | `0.0000 ± 0.0000` | `7.20 ± 0.75` | `3.360 ± 0.126` | `7.25 ± 0.54` | `0.0` | `0.208` |

Paired `Cliff's δ` (`ext_keep_artifacts` vs `ext_erase_and_destroy` / vs `no_ext_erase`): `ext_learn = +1.00 / +1.00`, `tech_utility = +1.00 / +1.00`, `mean_energy = +1.00 / +1.00`, `improvement_depth = +0.44 / +0.36`. Physical `PlacedArtifact` external storage buffers populations against abrupt agent-memory erasure (`E=12.07` vs `8.49` / `7.25`).

---

## 8. 2D P5 Phase Diagram (`regime_period` × `transmission_cost`, 3 Seeds × 500 Ticks)

Command: `uv run --no-sync alife cultural phase-diagram cultural_world_compositional --periods 80,200,500 --costs 0.0,0.015,0.04 --seeds 0,1,2 --steps 500 --json` (`55.95 s` wall).

| `regime_period` | `tx_cost = 0.000` (`depth / ret / util / E`) | `tx_cost = 0.015` (`depth / ret / util / E`) | `tx_cost = 0.040` (`depth / ret / util / E`) |
|---:|---|---|---|
| **`80`** (fast) | `6.67 / 0.451 / 4.855 / 14.93` | `8.33 / 0.495 / 4.844 / 12.83` | `7.33 / 0.511 / 4.778 / 9.65` |
| **`200`** (mid) | `9.67 / 0.480 / 4.756 / 15.31` | `7.67 / 0.454 / 4.817 / 12.92` | `7.00 / 0.450 / 4.808 / 9.16` |
| **`500`** (slow) | `16.33 / 0.167 / 4.840 / 13.63` | `17.67 / 0.115 / 4.838 / 11.06` | `16.67 / 0.130 / 4.755 / 5.61` |

- Higher `transmission_cost` monotonically reduces equilibrium `mean_energy` across every period (`15.31 -> 12.92 -> 9.16` at `period=200`; `13.63 -> 11.06 -> 5.61` at `period=500`) while `compositional_skill_fraction` remains `>= 0.9968`.
- **H5 regime trade-off observed:** Slower turnover (`period=500`) accumulates deeper refinement chains (`16.3–17.7` vs `6.7–9.7`), yet shows lower `post-inventor retention` (`0.12–0.17` vs `0.45–0.51`) as fewer environmental phase resets prompt broad cross-lineage horizontal sharing compared with faster cycling (`period=80–200`).


---

## 9. Stage 5 / P6 Extensions — Atomic Barter, Biomes & Causal Isolation

Phase P6–P7 ([ADR 0055](../decisions/0055-cultural-world-p6-p7.md)) extends `cultural_world` opt-in (`allow_exchange=False`, `interdependent_resources=False` by default; both enabled in `cultural_world_exchange`, with `allow_exchange=False` in `cultural_world_no_exchange_control`):

- **Biome Segregation (`ecology.py`):** West `material_a` (`x ≈ 0.25`) and east `material_b` (`x ≈ 0.75`) cosine biomes under exact `ResourceLedger` accounting (`resource_ledger_residual <= 1e-9`).
- **Atomic Local Barter (`social.py`):** Co-located agents within `exchange_range = 5.5` swap `qty = min(exchange_lot, 0.5 * inv_a[i], 0.5 * inv_b[j])` (`>= 0.03`) all-or-nothing using `rngs["transmit"]`, preserving `inv_a.sum()` and `inv_b.sum()` identically with zero negative inventory.
- **Specialization vs Division of Labor (`metrics.py`):** `specialization_index` measures unilateral A/B focus, whereas `division_of_labor_index = sqrt(pop_comp * interdependence_index)` requires both balanced population A/B specialization and active bilateral exchange.
- **Causal Intervention (`forbid_exchange`) & Checkpoint v3 (`SERIALIZE_VERSION = 3`):** Toggles `exchange_override` while keeping genomes, `dw2`, `energy`, inventories, `skills`, `artifacts`, and resource grids bit-identical; v3 snapshots replay deterministically bit-for-bit.

---

## 10. P6 H4 5-Seed Social & Causal Assay (`cultural_world_exchange`)

Command run (`5` paired seeds from `SeedSequence([seed, 6])`, `500` ticks/arm; midpoint `forbid_exchange` forked at `t=250`):
- `uv run --no-sync alife cultural social-assay cultural_world_exchange --seeds 5 --steps 500 --json` (`wall_seconds = 19.095 s`, **`all_checks_passed = true`**, 7/7 checks passed).

| Arm (`n = 5` seeds, `t = 500`) | Viable | Mean Energy | `tech_utility` | Trade Count | Trade Volume | `specialization_index` | `division_of_labor_index` | `interdependence_index` | `repeat_partner_rate` | Max Residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`exchange_intact`** | `1.00` | `5.0968` | **`4.0031`** | **`3380.8`** | **`717.44`** | `0.9861` | **`0.8269`** | **`0.7237`** | `0.4549` | `1.623e-11` |
| **`no_exchange_control`** | `1.00` | `5.1621` | `3.2775` | `0.0` | `0.00` | `0.9770` | `0.0000` | `0.0000` | `0.0000` | `1.472e-11` |
| **`role_shuffled_control`** | `1.00` | `5.3867` | `2.8153` | `1065.0` | `225.77` | `0.9884` | `0.5522` | `0.3975` | `0.2569` | `1.529e-11` |
| **`forbid_exchange_midpoint`** | `1.00` | `5.0229` | `3.5741` | `1330.6` | `316.53` | `0.9774` | `0.2516` | `0.0712` | `0.5181` | `1.617e-11` |

| Paired Comparison (`exchange_intact` vs Control) | Mean Diff | 95% Bootstrap CI | Cliff's $\delta$ | Exact Paired $p$ (`2^5 = 32`) | Exact Perm. $p$ (`252`) |
|---|---:|---|---:|---:|---:|
| `division_of_labor_index` vs `no_exchange_control` | `+0.8269` | `[0.7904, 0.8629]` | **`+1.00`** | `0.0625` | `0.00794` |
| `division_of_labor_index` vs `role_shuffled_control` | `+0.2747` | `[0.2024, 0.3605]` | **`+1.00`** | `0.0625` | `0.00794` |
| `interdependence_index` vs `no_exchange_control` | `+0.7237` | `[0.6581, 0.7893]` | **`+1.00`** | `0.0625` | `0.00794` |
| `interdependence_index` vs `role_shuffled_control` | `+0.3262` | `[0.2486, 0.4011]` | **`+1.00`** | `0.0625` | `0.00794` |
| `interdependence_index` vs `forbid_exchange_midpoint` | `+0.6525` | `[0.5869, 0.7181]` | **`+1.00`** | `0.0625` | `0.00794` |
| `post_midpoint_trades` vs `forbid_exchange_midpoint` | `+2050.2` | `[1690.2, 2328.8]` | **`+1.00`** | `0.0625` | `0.00794` |
| `tech_utility` vs `no_exchange_control` | `+0.7256` | `[0.1203, 1.3075]` | `+0.68` | `0.125` | `0.07937` |
| `mean_energy` vs `no_exchange_control` *(null)* | `-0.0653` | `[-0.3412, 0.2168]` | `-0.28` | `0.75` | `0.73016` |

All 7 pre-registered assay checks (`mass_conservation`, `positive_trades`, `no_exchange_zero_trades`, `division_of_labor_gt_no_exchange`, `division_of_labor_gt_role_shuffled`, `forbid_exchange_halts_subsequent_trades`, `forbid_exchange_reduces_interdependence`) passed (`post_midpoint_trades = 0.0` in `forbid_exchange_midpoint` vs `+2050.2` in `exchange_intact`).

---

## 11. P7 CPU Scaling Grid, Hotspot Breakdown & GPU Probe

Command run (`seed = 1`, `steps = 120` per cell, `--include-gpu`):
- `uv run --no-sync alife cultural benchmark --populations 80,200,400 --skills 2,4,8 --steps 120 --include-gpu --json` (`wall_seconds = 5.206 s`).

| Pop | Max Skills | Wall (s) | ms / tick | Agent-ticks / s | Peak RSS (MiB) | Max Residual | `ecology` (ms) | `sensors_policy` (ms) | `skills_tech` (ms) | `transmission` (ms) | `exchange` (ms) | `mortality_repro` (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `80` | `2` | `0.1155` | `0.963` | `99,318.7` | `96.34` | `5.684e-13` | `7.207` | `12.983` | `15.615` | `39.380` | `30.393` | `8.416` |
| `80` | `4` | `0.0973` | `0.811` | `113,763.9` | `96.34` | `4.832e-13` | `6.750` | `12.070` | `13.905` | `28.468` | `27.116` | `7.664` |
| `80` | `8` | `0.0953` | `0.794` | `116,229.5` | `96.34` | `4.832e-13` | `6.652` | `11.572` | `13.965` | `27.708` | `26.624` | `7.373` |
| `200` | `2` | `0.3587` | `2.989` | `80,665.2` | `100.38` | `1.307e-12` | `7.865` | `20.100` | `25.016` | `142.682` | `145.106` | `16.097` |
| `200` | `4` | `0.3531` | `2.942` | `82,435.7` | `100.38` | `1.222e-12` | `7.839` | `20.098` | `26.100` | `144.957` | `136.686` | `15.640` |
| `200` | `8` | `0.3580` | `2.984` | `81,471.8` | `100.38` | `1.052e-12` | `7.885` | `20.918` | `29.094` | `147.867` | `133.539` | `16.871` |
| `400` | `2` | `0.7204` | `6.004` | `66,625.7` | `103.32` | `8.242e-13` | `8.453` | `27.793` | `31.998` | `273.074` | `358.385` | `18.768` |
| `400` | `4` | `0.6963` | `5.803` | `68,931.5` | `103.32` | `9.095e-13` | `8.222` | `28.469` | `34.555` | `270.425` | `335.018` | `17.803` |
| `400` | `8` | `0.7182` | `5.985` | `66,831.2` | `103.32` | `1.108e-12` | `8.237` | `27.783` | `37.948` | `278.534` | `332.299` | `31.558` |

- **GPU Batched Sensor + MLP Probe (`NVIDIA GeForce RTX 5070`, `torch_cuda`):**
  - `available = true`, `N = 400`, `iters = 120`: CPU `float64` = **`17.36 ms`**, GPU `float64` = **`102.64 ms`**, **`speedup_vs_cpu = 0.169×`** (`~5.9×` slower than CPU), `float64 max_abs_diff = 2.776e-16` (`float32 max_abs_diff = 6.32e-08`).

---

## 12. Stage 5 Honest Findings, Negative/Null Results & Caveats

1. **Null effect on standing `mean_energy`:** `exchange_intact` (`5.0968`) did not outperform `no_exchange_control` (`5.1621`, `Cliff's δ = -0.28`, `exact_paired p = 0.75`) or `role_shuffled_control` (`5.3867`) on per-agent standing energy. Barter + specialization boosts net skill utility (`tech_utility = 4.0031` vs `3.2775` / `2.8153`, `Cliff's δ = +0.68`) and supports higher demographic throughput over the shared, non-traded `food` field, keeping equilibrium per-capita energy Malthusian-bounded.
2. **Spatial specialization is not division of labor:** Segregated biomes alone drove `specialization_index = 0.9770` in `no_exchange_control` with `division_of_labor_index = 0.0000`, validating the `division_of_labor_index` formulation that requires active bilateral interdependence (`0.8269` intact vs `0.5522` role-shuffled vs `0.0000` no-exchange).
3. **Small-`n` paired permutation floor (`n = 5`):** With `5` paired seeds, exact sign-flipping has `2^5 = 32` assignments, setting a minimum achievable two-sided exact paired `p`-value of `2/32 = 0.0625` even when `5/5` pairs agree monotonically (`Cliff's δ = +1.00`; unpaired 5-vs-5 exact permutation `p = 0.00794`).
4. **Negative GPU speedup at `N = 400`:** `sensors_policy` is only `~3.9%` (`27.78 ms` / `~716 ms`) of CPU step time at `N=400`, whereas pairwise $O(N^2)$ `exchange` (`332.30 ms`, `~46%`) and `transmission` (`278.53 ms`, `~39%`) dominate. Moving the tiny 12→8→6 MLP to CUDA slows execution to `0.169×` CPU due to kernel launch overhead; P7 therefore keeps pure-NumPy CPU execution as the production path unless future populations scale by orders of magnitude with spatial hashing.
