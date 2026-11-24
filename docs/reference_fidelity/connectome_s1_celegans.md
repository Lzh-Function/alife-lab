# Connectome Stage 2 / S1 (*C. elegans* 302-Neuron Graded + Gap-Junction Nervous System & Crawling Body) Report

> **Scientific mode:** `exploratory` (uses offline `synthetic_celegans_302` for embodied crawling + real empirical `celegans_302` EM wiring & `worm_data_short.parquet` whole-brain GCaMP recordings downloaded into `.alife/datasets/` after explicit user approval, per [ADR 0051](../decisions/0051-connectome-s1-celegans.md)).
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

## 3. Honest Assessment & Synthetic Baseline Limitations

- **Synthetic Evidence Level:** **E3** within the offline canonical `synthetic_celegans_302` model (`intact` vs `no_gap_junctions`, `shuffled_control`, and `poke_nose` causal circuit reversals), **E1** (synthetic-to-synthetic oracle) for offline `run_worm_activity_benchmark`.
- **Synthetic Crawling Baseline Notes:**
  1. `synthetic_celegans_302` uses curated canonical biological pathway weights rather than real empirical Cook et al. (2019) / White et al. (1986) synapse counts (`celegans_302`).
  2. Chemotaxis trajectories show mild improvement (`0.595 -> 0.626` over `8 s`) under baseline head steering gains; stronger klinotaxis tuning against real *C. elegans* tracking (`~0.15-0.25 mm/s`, reversal latency) remains open work.

---

## 4. Real Empirical *C. elegans* EM Connectome & Whole-Brain Calcium Validation

### 4.1 Pinned Empirical Dataset Provenance (Stored Outside Git in `.alife/datasets/`)

| Dataset ID | Source & Immutable Commit | File | Size (Bytes) | SHA-256 |
|---|---|---|---:|---|
| **`celegans_302`** | `openworm/CElegansNeuroML` (`1ca08b14f4e795a973ce447565e0c97b897ee4dc`) | `herm_full_edgelist.csv` | `245,463` | `142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca` |
| **`celegans_activity`** | HF `qsimeon/celegans_neural_data` (`a28aacad1a997ac7aee98f1c52993ab8c0599ee8`, MIT) | `worm_data_short.parquet` | `669,511,381` | `17701963149851d37d40955096de181d8c5378c256fe4ede93864380f9363ddc` |

- **`alife connectome summary celegans_302`:**
  - `cells=302`, `edges=5,764` (`chemical=3,604`, `gap_junction=2,160` deduplicated symmetric pairs), `synapses=29,114`, `isolated=2` (`CANL`/`CANR`, which lack chemical/electrical synapses in the adult somatic wiring diagram; all 148 other CSV tokens are verified muscles/end-organs).
  - VNC zero-pad normalization (`VA01 -> VA1`, `DB01 -> DB1`, etc.) raises canonical neuron coverage from `236/302` to `300/302` synaptic cells; all 4 `celegans_302_reference_checks` pass (`ok`).
  - `worm_data_short.parquet`: `42,798` rows across 12 papers (`32,879` labeled rows, `0` unmatched neuron strings after normalization, `341` worms with `>= 20` matched canonical neurons).

### 4.2 Unsigned Raw EM Connectome in `worm_embodied` (Honest Negative / Blocker Finding)

- Applying `celegans_302` (`herm_full_edgelist.csv`) directly inside `worm_embodied` with GABA-only inhibitory signing (`5,656` excitatory vs `108` inhibitory edges = `98.13%` E / `1.87%` I) fails to generate directional forward crawling or nose-poke reversal across global scalar conductance scaling (`g_chem` in `0.004..0.08`, `g_gap` in `0.05..0.14`):
  - At default `g_chem=0.08`, excitatory feedback saturates the brain near `0 mV` (`avb=-2.06 mV`, `ava=-1.43 mV`, `poke_nose delta_sf=+0.0008 mm`).
  - At `g_chem=0.008, g_gap=0.08`: `reference (gap=True)` yields `distance_mm=0.009`, `signed_forward=+0.003 mm`, `avb=-10.0 mV`, `ava=-8.2 mV` (`poke_nose delta_sf=+0.0002 mm`) while `gap=False` lets excitation escape into `ava=-1.8 mV` (`signed_forward=-0.535 mm`).
- **Biophysical cause:** `herm_full_edgelist.csv` lacks postsynaptic receptor polarity (`GluCl`/`AChCl`); sensory pools (`ALM/AVM/ASH`) project `97` excitatory chemical synapses onto `AVA/D/E` AND `97` onto `AVB/PVC`, while `AVB -> AVA` carries `47` excitatory synapses. Closed-loop locomotion switching on the real EM graph requires inferred/measured postsynaptic signs (e.g., Fenyves et al. 2020 *PLoS Comput. Biol.* `bank-fenyves/CeConn-SignPrediction`, not yet downloaded).

### 4.3 Real Whole-Brain Calcium Activity Fitting (`alife connectome fit-worm`, `horizon = 15` = `5.0 s`, `seeds = 1..5`, RTX 5070 CUDA)

- All arms evaluated strictly on held-out **final 30% frames** (zero leakage, zero teacher forcing):
  - **`spatial_unobs_r2 / r`**: 50% observed → 50% unobserved canonical neurons via EM submatrix `(W_chem + W_gap)[unobs, obs]`.
  - **`rollout15_r2 / r`**: 15-step (`5.0 s`, matching the dataset's `window_size=15` smoothing timescale) open-loop recurrent rollout.

| Cohort (`n_worms`, `mean_cells`, VRAM, wall) | Arm | `spatial_unobs_r2` | `spatial_unobs_r` | `rollout15_r2` | `rollout15_r` |
|---|---|---:|---:|---:|---:|
| **`Venkatachalam2024`** (`12`, `188.4`, `115.5 MiB`, `25.69 s`) | **`reference`** (EM chem + gap) | **`+0.0859`** | **`+0.3684`** | `+0.7076` | `+0.8623` |
| | `no_gap_junctions` | `-0.0043` | `+0.2657` | **`+0.7102`** | `+0.8631` |
| | `raw_em_gaba_only` | `+0.0547` | `+0.2745` | `+0.3197` | `+0.6145` |
| | `shuffled_control` | `-0.0066` | `+0.2153` | `+0.6951` | `+0.8608` |
| | `unconnected` | `-0.0533` | `-0.0090` | `+0.7042` | `+0.8839` |
| **`Flavell2023`** (`10`, `101.9`, `101.3 MiB`, `25.00 s`) | **`reference`** (EM chem + gap) | **`+0.1069`** | **`+0.3717`** | `+0.6598` | `+0.8311` |
| | `no_gap_junctions` | `+0.0622` | `+0.3026` | **`+0.6605`** | `+0.8313` |
| | `raw_em_gaba_only` | `+0.0555` | `+0.2857` | `+0.5752` | `+0.7903` |
| | `shuffled_control` | `-0.0016` | `+0.2102` | `+0.6531` | `+0.8329` |
| | `unconnected` | `-0.0298` | `-0.0042` | `+0.6577` | `+0.8448` |
| **`Kato2015`** (`10`, `44.6`, `99.5 MiB`, `25.30 s`) | **`reference`** (EM chem + gap) | **`+0.2618`** | **`+0.5185`** | `+0.7063` | `+0.8546` |
| | `no_gap_junctions` | `+0.1575` | `+0.4327` | **`+0.7104`** | `+0.8565` |
| | `raw_em_gaba_only` | `+0.1994` | `+0.4616` | `+0.6840` | `+0.8548` |
| | `shuffled_control` | `+0.0587` | `+0.2652` | `+0.6935` | `+0.8529` |
| | `unconnected` | `-0.0153` | `+0.0070` | `+0.6852` | `+0.8622` |

- **Statistical Contrasts (`n = 5` seeds per cohort, two-sided exact permutation test & Cliff's $\delta$):**
  - `spatial_unobs_r2`:
    - `reference` vs `shuffled_control`: `d = +1.00, p = 0.0079` in **all 3 cohorts** (`Venkatachalam2024`, `Flavell2023`, `Kato2015`).
    - `reference` vs `no_gap_junctions`: `d = +1.00, p = 0.0079` in `Venkatachalam2024` & `Kato2015`; `d = +0.52, p = 0.2063` (not statistically significant at `n=5`) in `Flavell2023`.
    - `reference` vs `raw_em_gaba_only`: `d = +1.00, p = 0.0079` in `Kato2015`; `d = +0.60, p = 0.0873` (`Venkatachalam2024`); `d = +0.68, p = 0.0714` (`Flavell2023`).
  - `temporal_rollout_r2` (`15` steps):
    - `reference` vs `shuffled_control`: `d = +1.00, p = 0.0079` in all 3 cohorts (`+0.0067..+0.0128` gain).
    - `reference` vs `raw_em_gaba_only`: `d = +1.00, p = 0.0079` in all 3 cohorts (most pronounced in `Venkatachalam2024`: `+0.7076 vs +0.3197`, showing raw GABA-only E/I signs distort multi-step recurrent dynamics).
    - `reference` vs `no_gap_junctions`: `d = -1.00, p = 0.0079` across all 3 cohorts (gap junctions slightly hurt `rollout15_r2` by `0.0007..0.0041`), and `unconnected` autonomous per-cell decay already scores `+0.6577..+0.7042` because moving-average smoothing (`window_size=15` in `worm_data_short.parquet`) creates high intrinsic autocorrelation across 5-second open-loop rollouts without external stimuli.

### 4.4 Updated Evidence Assessment

- **Empirical Structural Verification (`celegans_302`):** Verified against OpenWorm `herm_full_edgelist.csv` (`300/302` synaptic neurons + `2` isolated `CANL/R`, `5,764` edges, `29,114` synapses).
- **Empirical Calcium Activity Fit (`fit-worm`):** **E2/E3 observational + negative-control support** across 3 independent real cohorts for **simultaneous unobserved-neuron EM connectivity** (`reference > shuffled_control`, `d=+1.00, p=0.0079` in 3/3 cohorts; electrical gap-junction contribution clearly positive in `Venkatachalam2024` and `Kato2015`), alongside honest null/weak gain over autonomous `unconnected` baselines in open-loop 15-step temporal rollouts on smoothed unstimulated recordings.
- **Embodied Crawling (`worm_embodied`):** **E3** on offline canonical `synthetic_celegans_302`, and **blocked at raw unsigned EM** on `celegans_302` without postsynaptic receptor polarity annotations.
