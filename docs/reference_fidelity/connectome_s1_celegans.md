# Connectome Stage 2 / S1 (*C. elegans* 302-Neuron Graded + Gap-Junction Nervous System & Crawling Body) Report

> **Scientific mode:** `exploratory` (uses offline `synthetic_celegans_302` canonical wiring + simplified 12-segment Resistive Force Theory planar crawling mechanics per [ADR 0051](../decisions/0051-connectome-s1-celegans.md); no external empirical EM connectome or real calcium data has been fetched yet without explicit user approval).
>
> **Architecture:** [ADR 0051](../decisions/0051-connectome-s1-celegans.md) · **Unified Roadmap:** [cultural_world_plan.md](../plans/cultural_world_plan.md) · **Connectome Plan:** [connectome_plan.md](../plans/connectome_plan.md)

## 1. Scope, Isolation Guarantees & Physics/Solver Checks

1. **Zero Impact on FlyWire / Spiking LIF Paths:**
   - `src/alife_lab/connectome/lif.py` (`step_substep_cpu`, `step_tick_cpu`) was kept completely untouched; existing `test_connectome_lif.py`, `test_connectome_flywire_f0.py`, `test_connectome_f3.py`, `test_connectome_f4.py`, and `test_connectome_f8.py` remain 100% green and bit-identical.
2. **Conductance-Based Graded + Gap-Junction Oracle (`src/alife_lab/connectome/graded.py`):**
   - `MODEL_GRADED` + `SYNAPSE_GAP_JUNCTION` evaluated via vectorized diagonal-implicit exponential Euler (`dt_ms=0.25/0.5`, deterministic `float32`).
   - Verified 2-cell zero-leak symmetric ohmic conservation ($V_0 + V_1 = -70.0\text{ mV}$ to $< 10^{-4}\text{ mV}$ across 80 substeps), endpoint silencing isolation, excitatory/inhibitory equilibrium bounds, and heterogeneous `MODEL_SPIKING_LIF` + `MODEL_GRADED` signal propagation (`step_hybrid_tick_cpu`).
3. **302-Neuron *C. elegans* Species Pack & Tabular Loader (`src/alife_lab/connectome/celegans.py`):**
   - Built-in offline `synthetic_celegans_302` contains **302 unique canonical hermaphrodite somatic neurons** (282 somatic + 20 pharyngeal, 119 classes), 515 chemical synapses, and 434 symmetric ohmic gap-junction edges, passing all structural checks (`celegans_302_reference_checks`).
   - Pure-CSV loader `load_celegans_from_edgelist_csv` (zero pickles) verified on synthetic edgelists and ready for OpenWorm `herm_full_edgelist.csv` once approved.
4. **Planar 12-Segment Resistive Force Theory (`src/alife_lab/connectome/worm_body.py`):**
   - Exact $3 \times 3$ overdamped viscous balance ($c_\perp / c_\parallel = 3.0$) satisfies:
     - Scallop theorem standing wave (zero phase gradient): `< 1e-2 mm` displacement over 1 cycle.
     - Head-to-tail traveling wave: `+0.3701 mm` over `2.0 s` vs symmetric tail-to-head `-0.3701 mm`.

---

## 2. Measured Multi-Seed Benchmark (`synthetic_celegans_302`, `seeds = 0..4`)

- **Compute & Memory:** `4,850` ticks total across `worm_embodied` & `worm_activity`, wall-clock elapsed **`6.246 s`** (**`1.288 ms/tick`** CPU single-process), peak RSS **`97.5 MiB`**.
- Note: Under continuous deterministic ODE dynamics (`MODEL_GRADED` + `float64` RFT) with identical initial conditions, `intact` and `no_gap_junctions` yield zero seed-to-seed variance (`± 0.0000`), whereas `shuffled_control` varies across 5 independent random degree-preserving edge permutations (`seeds 0..4`).

### 2.1 Closed-Loop Crawling & Causal Controls (`t = 200` ticks = `2.0 s`, `n = 5` seeds)

| Condition | Distance (mm) | Signed Forward (mm) | Body Wave RMS (rad) | AVB Mean (mV) | AVA Mean (mV) | Reversals | Chemotaxis Index |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`intact` (`reference`, gap ON)** | `0.1775 ± 0.0000` | `+0.1150 ± 0.0000` | `0.3058 ± 0.0000` | `-9.19 ± 0.00` | `-62.04 ± 0.00` | `1.00 ± 0.00` | `0.5955 ± 0.0000` |
| **`no_gap_junctions` (`gap=False`)** | `0.0067 ± 0.0000` | `+0.0002 ± 0.0000` | `0.0341 ± 0.0000` | `-21.77 ± 0.00` | `-33.95 ± 0.00` | `0.00 ± 0.00` | `0.5877 ± 0.0000` |
| **`shuffled_control` (`wiring=shuffled`)** | `0.0136 ± 0.0062` | `-0.0015 ± 0.0041` | `0.1160 ± 0.1091` | `-32.34 ± 2.83` | `-34.17 ± 6.08` | `0.60 ± 0.55` | `0.5875 ± 0.0006` |

### 2.2 Sensory Switching, Extended Crawl (`t=800`) & Avoidance (`t=400`)

- **Nose-poke causal reversal (`poke_nose` 25 ticks at `t=100..130`, `n=5` seeds):**
  - `AVB` hyperpolarizes (`-9.2 -> -66.66 mV`), `AVA` depolarizes (`-62.0 -> -10.02 mV`), body reverses cleanly (`delta_signed_forward = -0.0305 mm`, cumulative reversals `1 -> 2`).
- **Extended crawl (`worm_embodied_crawl`, `t=800` = `8.0 s`):**
  - `distance_mm = 0.7825 mm`, `forward_speed_mm_s = +0.0968 mm/s`, `body_wave_power = 0.2961 rad`, `avb_mean_mv = -9.41 mV`, `ava_mean_mv = -62.00 mV`, `chemotaxis_index = 0.6261`.
- **Repellent avoidance (`worm_embodied_avoidance`, `t=400` = `4.0 s`):**
  - Approaching the repellent plume triggers nociceptive command reversal (`avb_mean_mv = -67.34 mV`, `ava_mean_mv = -8.24 mV`, `chemotaxis_index = 0.5434`).

### 2.3 Whole-Brain Calcium Activity Comparison (`run_worm_activity_benchmark`, 120 ticks × 218 held-out cells, `n = 5` seeds)

| Model Arm | Held-out $R^2$ | Held-out Pearson $r$ | RMSE ($\Delta F/F$) |
|---|---:|---:|---:|
| **`reference` (intact 302-cell)** | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.0000 ± 0.0000` |
| **`no_gap_junctions`** | `0.2937 ± 0.0004` | `0.6186 ± 0.0005` | `0.2073 ± 0.0000` |
| **`shuffled_control`** | `-0.0973 ± 0.0198` | `0.2325 ± 0.0621` | `0.2584 ± 0.0023` |

---

## 3. Honest Assessment, Limitations & Pending Real-Data Gate

- **Evidence Level:** **E3** within the offline canonical `synthetic_celegans_302` model (`intact` vs `no_gap_junctions`, `shuffled_control`, and `poke_nose` causal circuit reversals), **E1** (synthetic-to-synthetic oracle) for `worm_activity` until real empirical calcium traces are approved and tested.
- **Honest Limitations:**
  1. `synthetic_celegans_302` uses curated canonical biological pathway weights rather than real empirical Cook et al. (2019) / White et al. (1986) synapse counts (`celegans_302`), which await explicit user approval before download.
  2. Real calcium imaging traces (e.g., Kato et al. 2015 / Atanas & Kim et al. 2023 WormWideWeb) are not yet fitted; pure CSV/JSON ingestion in `worm_activity.py` is implemented and ready.
  3. Chemotaxis trajectories show mild improvement (`0.595 -> 0.626` over `8 s`) under baseline head steering gains; stronger klinotaxis tuning and sinusoidal clinical/behavioral benchmarks against real *C. elegans* tracking (velocity `~0.15-0.25 mm/s`, reversal latency) belong to empirical S1 validation after real connectome/activity acquisition.
