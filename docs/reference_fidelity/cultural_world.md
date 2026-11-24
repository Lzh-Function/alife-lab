# CulturalWorld (`cultural_world`) — Phase P0–P3 Benchmark & Integrity Report

> **Scientific mode:** `exploratory` only (`mode="reference"` is rejected at `ExperimentSpec` validation; no external biological or historical literature oracle is claimed).
>
> **Architecture:** [ADR 0050](../decisions/0050-cultural-world.md) · **Unified Roadmap:** [cultural_world_plan.md](../plans/cultural_world_plan.md)

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
  4. **Scope limitations (P0–P3 vs P4–P7):** Multi-agent prerequisite dependency trees (Phase P4), novelty/ASAL trajectory comparison against random/baseline controls (Phase P5), and Connectome × Culture neuro-symbolic hybrids (`S1–S7` / `fly_world` bridge in Stage 3–6 of `docs/plans/cultural_world_plan.md`) are not yet implemented.
