# Reference fidelity — Adult *Drosophila* connectome (`flywire_783`, F0–F6)

Status: **F0–F6 complete** (2026-11-23); F7+ not yet implemented.

> **Honest framing:** This dataset and the upcoming model are a hypothesis-testing
> simplification derived from one adult female *Drosophila melanogaster* wiring diagram, not a
> biological fly brain. True synaptic conductances, cell-specific biophysics, electrical gap
> junctions, neuromodulation, plasticity and inter-individual variability are absent from the
> static connectivity table.

## References and pinned upstream files
All tables are read in tabular formats only (`.parquet`, `.csv`, `.tsv`, `.csv.gz`). Upstream
pickle files (e.g. `sez_neurons.pickle`) are explicitly rejected by `alife_lab.connectome.registry`.
Data files are stored outside Git in `.alife/datasets/flywire_783/raw/` and verified by SHA-256
before loading.

- Dorkenwald et al. (2024), *Neuronal wiring diagram of an adult brain*, Nature
  (FlyWire v783 whole-brain connectome: 139,255 proofread neurons, ~54.5M chemical synapses).
- Schlegel et al. (2024), *Whole-brain annotation and multi-connectome cell typing of
  Drosophila*, Nature (`flyconnectome/flywire_annotations`, commit
  `a83b2776d60d5764cef36b927f5f9679c16c47a2`).
- Shiu et al. (2024), *A Drosophila computational brain model reveals sensorimotor processing*,
  Nature 634:210–219 (`philshiu/Drosophila_brain_model`, MIT code, commit
  `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`).
- FlyWire Codex v783 tables (`https://storage.googleapis.com/flywire-data/codex/data/fafb/783/`).

| File | Size (bytes) | SHA-256 | Source |
|---|---:|---|---|
| `Completeness_783.csv` | 3,327,347 | `bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311` | `philshiu/Drosophila_brain_model` |
| `Connectivity_783.parquet` | 100,804,642 | `efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347` | `philshiu/Drosophila_brain_model` |
| `Supplemental_file1_neuron_annotations.tsv` | 31,720,298 | `b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd` | `flyconnectome/flywire_annotations` |
| `classification.csv.gz` | 934,402 | `e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663` | FlyWire Codex v783 |
| `coordinates.csv.gz` | 5,314,546 | `14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05` | FlyWire Codex v783 |
| `neurons.csv.gz` | 1,679,884 | `6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719` | FlyWire Codex v783 |
| `neuropil_synapse_table.csv.gz` | 4,674,663 | `e525bdea7bc2fe585cf8ab8f9fc76bea5e29f2c3e5240588b6add518bcda5ab1` | FlyWire Codex v783 |

Total raw dataset size: **148,455,782 bytes (141.6 MiB)**.
Canonical normalized `pack_digest`:
`0cc1ff1d5579c29eeaa6e7afb4aac4c01b1bf958be067aa8390443aa5cf81862`.

## F0 verification summary (measured 2026-11-21)

### Structural counts and comparison against published values
| Quantity | Measured (`flywire_783`) | Published / reference | Match / note |
|---|---:|---:|---|
| Neurons in simulation graph | 138,639 | 138,639 (Shiu v783) / 139,255 (Dorkenwald all proofread) | Exact match with Shiu v783. The 616-neuron difference corresponds to neurons without connected edges in the Shiu v783 graph (all 138,639 have ≥1 edge; isolated = 0). |
| Directed neuron-to-neuron edges | 15,091,983 | 15,091,983 (Shiu v783) | Exact match; 0 duplicate pairs, 0 self-loops. |
| Total chemical synapses | 54,492,922 | ~54.5M (Dorkenwald 2024) | Consistent (sum of `Connectivity` over all edges). |
| Excitatory edges / synapses | 9,059,302 (60.03%) / 32,494,717 (59.63%) | Sign = +1 (Shiu table) | 96,672 excitatory presynaptic neurons. |
| Inhibitory edges / synapses | 6,032,681 (39.97%) / 21,998,205 (40.37%) | Sign = −1 (Shiu table) | 41,333 inhibitory presynaptic neurons. |
| Dale's law consistency (mixed-sign pre neurons) | 0 | 0 | Every presynaptic neuron with outgoing edges has a single sign. |
| Out-degree (median / mean / max / zero) | 79 / 108.86 / 9,783 / 634 | — | Strong hub skew; informs F1 CUDA kernel load balancing. |
| In-degree (median / mean / max / zero) | 69 / 108.86 / 10,356 / 1,549 | — | Strong hub skew. |
| 3D coordinate coverage (nm) | 138,639 / 138,639 (100.00%) | — | `soma`: 118,086 (85.18%), `anchor`: 20,539 (14.81%), `codex`: 14 (0.01%), `missing`: 0. |
| Named cell types | 137,149 / 138,639 (98.93%) | — | From `cell_type` / `hemibrain_type`. |
| Primary neuropil regions | 78 regions (+ `UNASGD`: 4,458) | 78 FlyWire neuropils | Derived from `neuropil_synapse_table.csv.gz` with Codex `group` fallback. |

### Key biological populations (IO ports resolved for F1+)
| Port ID | Direction | Count | Selection rule |
|---|---|---:|---|
| `sugar_grn` | sensory | 32 | `cell_class == 'gustatory' and cell_sub_class == 'sugar'` |
| `sugar_and_low_salt_grn` | sensory | 57 | `cell_class == 'gustatory' and cell_sub_class in ('sugar', 'sugar/low_salt')` |
| `water_grn` | sensory | 30 | `cell_class == 'gustatory' and cell_sub_class == 'water'` |
| `bitter_grn` | sensory | 42 | `cell_class == 'gustatory' and cell_sub_class == 'bitter'` |
| `antennal_grooming_mechanosensory` | sensory | 187 | `cell_class == 'mechanosensory' and cell_sub_class == 'grooming'` |
| `jo_mechanosensory` | sensory | 1,091 | `cell_type.startswith('JO-')` |
| `olfactory_sensory` | sensory | 2,279 | `cell_class == 'olfactory'` |
| `mn9_proboscis_extension` | motor | 2 | `cell_type == 'CB0701'` (`720575940660219265` right, `720575940618238523` left) |
| `proboscis_motor` | motor | 24 | `super_class == 'motor' and cell_sub_class == 'proboscis_motor_neuron'` |
| `mushroom_body_kc` | circuit | 5,177 | `cell_class == 'Kenyon_Cell'` |
| `mushroom_body_mbon` | circuit | 96 | `cell_class == 'MBON'` |
| `mushroom_body_dan` | circuit | 331 | `cell_class == 'DAN'` |

### Repeatability and performance (CPU, Intel Core Ultra 7 270K Plus)
- Loading `flywire_783` twice yields identical `pack_digest` and canonical JSON summary
  (`tests/reference/test_connectome_flywire_f0.py`).
- Full load + SHA-256 verification of all 7 files: **3.17 s** (CLI summary: **3.31 s**).
- Canonical CSR construction (`(pre, post)` ascending sort over 15,091,983 edges): **0.582 s**.
- Canonical CSR memory footprint (`indptr + post + signed_weight + kind`): **136,382,407 bytes
  (130.1 MiB)**.
- Peak process RSS during load + CSR build: **2,173.8 MiB**.

## Known limitations (F0)
- 14 of 138,639 neurons in `Completeness_783.csv` are absent from
  `Supplemental_file1_neuron_annotations.tsv`; their classification, neurotransmitter and nm
  coordinates are populated from the FlyWire Codex v783 tables (`classification.csv.gz`,
  `neurons.csv.gz`, `coordinates.csv.gz`).
- 20,553 neurons (14.82%) lack a soma coordinate in `Supplemental_file1_neuron_annotations.tsv`
  and fall back to backbone anchor coordinates (`pos_x/y/z`) or Codex coordinates. The
  `position_source` array records this per cell for honest 3D visualization in F2.
- 4,458 neurons are absent from `neuropil_synapse_table.csv.gz` and fall back to `neurons.csv.gz`
  `group` or `UNASGD`.
- Shiu et al.'s published figures used FlyWire v630 (127,400 neurons), while their repository
  and this pack use FlyWire v783 (138,639 neurons). F1 reproduction results below document
  both matches and non-matches explicitly.

---

## F1 LIF simulation engine & Shiu et al. (2024) reproduction (measured 2026-11-21)

### Model equations and defaults
Exact linear integration over `dt = 0.1 ms` (10 substeps = 1 ms tick) of Shiu et al. (2024):
- `dv/dt = (v_0 - v + g) / t_mbr` (outside refractory period), `dg/dt = -g / tau`
- Spike when `v > v_th` and `refrac == 0` -> reset `v = v_rst`, `g = 0`, `refrac = t_rfc`
  (or `0` for externally Poisson-stimulated neurons, matching `philshiu/Drosophila_brain_model`)
- Defaults: `v_0 = -52 mV`, `v_rst = -52 mV`, `v_th = -45 mV`, `t_mbr = 20 ms`, `tau = 5 ms`,
  `t_rfc = 2.2 ms` (22 substeps), `t_dly = 1.8 ms` (18 substeps), `w_syn = 0.275 mV`,
  `f_poi = 250` (`v += 68.75 mV` on Poisson hits), default stimulus rate `150 Hz`.
- Implementation: `alife_lab.connectome.lif` (CPU `float32` oracle),
  `alife_lab.connectome.lif_cupy` (CUDA C++ `--fmad=false` + CUDA graphs + batched runner),
  `alife_lab.substrates.connectome.adapter` (`connectome` substrate plugin 0.1.0).

### Verification gates (`tests/reference/test_connectome_lif.py`, `tests/integration/test_connectome_platform.py`)
| Gate | Result |
|---|---|
| Single-neuron passive decay & synaptic pulse vs continuous analytic solution (10 ms) | pass (`|err| < 1e-4 mV`) |
| 1.8 ms synaptic delay, 2.2 ms refractory period, exact +/− cancellation, silencing | pass |
| Checkpoint save → restore bitwise continuation (NPZ round-trip) | pass (`exact_equal=True`, `max|dv|=0`) |
| CPU (`numpy`) vs GPU (`cupy` eager & CUDA graph) on full FlyWire v783 (138,639 neurons, 50 ms) | **pass: bit-identical (`max|dv|=0.0`, 0 differing bits)** |
| GPU repeat determinism & `CupyBatchedLIFRunner` per-brain match against CPU oracle | **pass: bit-identical** |
| Worker restart + `stimulate`/`silence` interventions + deterministic replay verification | pass (`status="verified"`) |

### Shiu et al. (2024) sensorimotor reproduction (FlyWire v783, 5 seeds [1..5], 1000 ms each)
| Condition (150 Hz unless noted) | MN9 mean rate (Hz, 5-seed mean [per-seed]) | Descending spikes/s (5-seed mean) | Total brain spikes/s (mean) | Assessment |
|---|---:|---:|---:|---|
| Sugar GRNs (`sugar_grn`, 32 cells) | **76.7** `[75.0, 83.5, 75.0, 76.0, 74.0]` | 1,804.6 | 18,650.2 | **Reproduced:** strong MN9 proboscis extension drive across all 5 seeds |
| Sugar (32) + Bitter (`bitter_grn`, 42 cells) | **0.4** `[0.5, 0.5, 0.5, 0.0, 0.5]` | 646.6 | 18,295.0 | **Reproduced:** 99.5% suppression of MN9 by bitter pathway |
| Sugar + low-salt (`sugar_and_low_salt_grn`, 57) | **15.4** `[16.0, 23.0, 12.5, 14.0, 11.5]` | 7,503.0 | 706,859.0 | **Reproduced** (lower MN9 rate than pure sugar; strong network-wide recruitment) |
| Sugar + low-salt (57) + Bitter (42) | **0.0** `[0, 0, 0, 0, 0]` | 7,214.2 | 669,960.6 | **Reproduced:** 100% MN9 suppression |
| Sugar (32) with `sugar_grn` silenced (intervention) | **0.0** `[0, 0, 0, 0, 0]` | 0.0 | 4,736.6 | **Control passes:** silencing blocks all downstream propagation |
| Shuffled control brain (`wiring="shuffled_control"`, sugar 32) | **0.0** `[0, 0, 0, 0, 0]` | — | ~9,739 | **Control passes:** MN9 response depends on biological wiring, not degree/weight statistics |
| Antennal grooming mechanosensory (187 cells) | **0.0** `[0, 0, 0, 0, 0]` | **478.2** `[473, 488, 480, 471, 479]` | 30,255.8 | **Reproduced (circuit-level):** activates grooming-linked descending neurons (`DNg84` ~100 Hz, `DNg35`, `DNg57`, `DNg29`) with 0 MN9 |
| Shuffled control brain (antennal grooming 187) | 0.0 | **0.4** `[0, 0, 0, 0, 2]` | — | **Control passes:** >1,000× drop in descending output |
| Johnston's organ (`jo_mechanosensory`, 1,091) | 0.0 | 3,114.2 | 188,142.2 | Strong descending recruitment, 0 MN9 |
| Water GRNs (`water_grn`, 30 cells, 150 Hz & 200 Hz) | **0.0** `[0, 0, 0, 0, 0]` | 385.8 / 502.4 | 7,070 / 9,434 | **Not reproduced here:** unlike Shiu et al. (v630 curated water GRNs), v783 `cell_sub_class == 'water'` alone does not drive MN9 above threshold at 150–200 Hz (recorded honestly without tuning) |

### Measured performance & VRAM (real FlyWire v783, RTX 5070 12 GiB / Core Ultra 7 270K Plus)
| Mode | Wall time per 1 s sim | Real-time factor | Brain-s / wall-s | GPU memory (MiB) |
|---|---:|---:|---:|---:|
| CPU oracle (`numpy`, 1 core) | ~6.16 s (extrapolated from 50 ms = 0.308 s) | 0.16× | 0.16 | — |
| GPU eager (`cupy`, no CUDA graph, 10-step) | 0.2349 s | 4.3× | 4.3 | 129.3 |
| GPU CUDA graph (10-step = 1 tick) | 0.0600 s | 16.7× | 16.7 | 129.3 |
| GPU CUDA graph (100-step = 10 ms batch) | 0.0465–0.0572 s | **17.5–21.5×** | **17.5–21.5** | 128.8–129.3 |
| Batched GPU (`B = 8`, shared CSR, CUDA graph) | 0.2162 s | 4.6× / brain | **37.0** | 220.4 (13.1 MiB/brain) |
| Batched GPU (`B = 32`, shared CSR, CUDA graph) | 2.5041 s | 0.40× / brain | 12.8 | 534.5 (13.1 MiB/brain) |
| Batched GPU (`B = 64`, shared CSR, CUDA graph) | 4.9812 s | 0.20× / brain | 12.8 | 953.4 (13.1 MiB/brain) |
| Batched GPU (`B = 128`, shared CSR, CUDA graph) | 9.8320 s | 0.10× / brain | 13.0 | 1,791.1 (13.1 MiB/brain) |

Notes on scaling:
- Per-brain GPU state is **13.1 MiB** on top of the shared **115.6 MiB** CSR graph, so ~500–700
  brains fit in memory on a 12 GiB GPU; however, aggregate throughput peaks around **B = 8**
  (**37.0 brain-s/s**) before L2 cache contention from random `atomicAdd` targets flattens
  throughput to ~13 brain-s/s at B = 32–128.
- High-activity regimes (e.g. `sugar_and_low_salt_grn` with ~700k spikes/s) still run at
  **10.4× real-time** (0.096 s per simulated second) with CUDA graphs.

---

## Phase F2: Browser Observation, Interactive Controls & 3D Brain View (2026-11-21)

### Implemented capabilities

- **CPU-only metadata, neuron search & 3D geometry API** (`src/alife_lab/connectome/query.py`,
  `src/alife_lab/api/app.py`):
  - `GET /api/v1/connectome/{dataset_id}/meta`
  - `GET /api/v1/connectome/{dataset_id}/cells?q=&region=&role=&port=&limit=&offset=`
  - `GET /api/v1/connectome/{dataset_id}/geometry` (quantized `int16` normalized FAFB 3D positions +
    `uint8` role + `uint8` region palette = `1.11 MB` for all 138,639 FlyWire v783 neurons; skips
    loading the 100 MiB connectivity table in the API process).
- **Deterministic preview codec (`connectome_v1`)** in `ConnectomeAdapter.render()`:
  - Emits per-port firing rates, top-16 neuropil firing intensities and exact sorted `int32` fired
    neuron indices (~1–8 KB/frame), preserving byte-identical keyframe verification during replay.
- **Browser UI (`ConnectomeView.tsx`, `Brain3D.tsx`, `ConnectomePanel.tsx`, `InspectorPanel.tsx`)**:
  - 2D measurement view: rolling spike raster plot, brain-region firing intensity bars, highlighted
    IO port rates (MN9, sugar, bitter, grooming).
  - 3D glowing brain view (lazy-loaded Three.js WebGL2 point cloud): places all 138,639 FlyWire
    neurons at their FAFB 3D coordinates and illuminates spiking neurons in real time (presentation
    only; never alters scientific state).
  - Live browser controls for `stimulate`, `silence`, `clear_stimuli`, neuron search & per-cell
    inspection, compatible with deterministic Timeline rewind/replay and fork/paired-fork.
  - Five catalog presets (`connectome_tiny_circuit`, `connectome_fly_sugar`,
    `connectome_fly_sugar_bitter`, `connectome_fly_grooming`, `connectome_fly_shuffled_control`),
    beginner guide section, glossary terms, parameter/metric `?` help and full `ja`/`en` i18n.

### F2 Verification (Playwright Chromium E2E)

- `e2e/connectome.spec.ts` (2/2 passed in 20.1 s):
  1. `connectome_tiny_circuit` (offline CPU): create → observe 2D raster & MN9 response → search
     `CB0701` & inspect → live `stimulate` + `silence` → switch to `brain3d` & back → pause →
     Timeline rewind/replay → replay verification (`ok-text`) → fork at tick 20 with parent lineage
     → beginner guide (6.8 s).
  2. `connectome_fly_sugar` (real FlyWire v783, 138,639 neurons): create → live simulation →
     search `CB0701` (exact 2 MN9 neurons) → inspect `mn9_proboscis_extension` → 3D glowing brain
     rendering all 138,639 neurons (12.5 s).

---

## Phase F3: Systematic Cell-Type Silencing Screen & Sensory Input Search (2026-11-21)

### 1. Systematic single-cell-type silencing screen on FlyWire v783 (RTX 5070, CuPy CUDA graphs, B=8)

Stimulus: `sugar_grn` at 150 Hz; target readout: `mn9_proboscis_extension` (`CB0701`, 2 neurons);
1000 ms per evaluation. Silencing follows Shiu et al. (2024)'s synaptic-output block semantics
(silenced neurons cannot scatter outgoing spikes).

- **Discovery screen (top 24 active non-target cell types × 5 seeds = 125 brain-seconds)**:
  - Wall time: **4.58 s** (**27.3 brain-s/s**); intact baseline MN9 = **76.70 ± 3.87 Hz**.
  - Shuffled-wiring negative control (top 8 active types × 5 seeds = 45 brain-seconds):
    **1.77 s** (**25.4 brain-s/s**); baseline and all silenced conditions = **0.00 ± 0.00 Hz**.
- **Confirmatory 10-seed re-test of top 5 causal hits (`seeds = 1..10`, 60 brain-seconds)**:
  - Wall time: **2.27 s** (**26.5 brain-s/s**); intact baseline MN9 = **74.50 ± 4.15 Hz**.

| Cell type | Neurons | Mean MN9 (Hz) | Δ vs intact (Hz) | Change | Cliff's δ | p (exact perm) | p (Holm, m=5) | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `CB0553` | 2 | **0.65** | **-73.85** | **-99.1%** | -1.00 | <0.0001 | **0.0001** | Essential bilateral excitatory relay to MN9 |
| `DNge059` | 2 | **47.40** | **-27.10** | **-36.4%** | -1.00 | <0.0001 | **0.0001** | Strong excitatory contributor |
| `CB0062` | 2 | **63.25** | **-11.25** | **-15.1%** | -0.94 | <0.0001 | **0.0001** | Moderate excitatory contributor |
| `CB0480` | 2 | **91.65** | **+17.15** | **+23.0%** | +0.98 | <0.0001 | **0.0001** | Inhibitory brake (disinhibition when silenced) |
| `DNge051` | 2 | **97.35** | **+22.85** | **+30.7%** | +1.00 | <0.0001 | **0.0001** | Strong inhibitory brake (disinhibition when silenced) |

### 2. Sensory input search on FlyWire v783 (4 sensory ports, wiring fixed)

Search space: `[sugar_grn, bitter_grn, water_grn, antennal_grooming_mechanosensory]` in `[0, 200]` Hz
(`stimuli.<idx>.rate_hz` only; wiring and LIF parameters frozen). Objective: `connectome_mn9_rate` (v1),
6 generations × population 8 × 2 search seeds + 3 disjoint holdout seeds, 500 ms per evaluation,
5 outer search seeds (`1..5`), GPU batched CUDA graphs (`cupy`, bitwise identical to CPU `numpy`).

| Condition | Holdout MN9 mean ± SD (Hz) | Median (Hz) | Wall time (5 seeds) | Notes |
|---|---:|---:|---:|---|
| `CMA-ES` (`reference` wiring) | **78.80 ± 8.27** | 81.33 | 13.47 s | Selected `sugar_grn = 200 Hz` in 5/5 seeds and `bitter_grn = 0 Hz` in 4/5 seeds |
| `Random` (`reference` wiring) | **68.00 ± 7.89** | 70.00 | 9.27 s | CMA-ES vs Random: Δ = +10.80 Hz, 95% CI [+1.93, +19.93], δ = +0.68, exact p = 0.0714 (honestly recorded: directional at 6 gens / 5 seeds, not p < 0.05) |
| `CMA-ES` (`shuffled_control` wiring) | **0.00 ± 0.00** | 0.00 | 23.05 s | CMA-ES (`reference`) vs Shuffled: Δ = +78.80 Hz, δ = +1.00, exact **p = 0.0079** |

---

## Phase F4: Olfactory Associative Learning — Mushroom Body `KC -> MBON` Plasticity (2026-11-21)

> **Honest framing:** Phase F4 uses an **explicitly simplified phenomenological three-factor
> eligibility-trace plasticity rule** inspired by *Drosophila* Mushroom Body literature
> (Hige et al. 2015; Cohn et al. 2015; Handler et al. 2019; Jiang & Litwin-Kumar 2021). It is
> **not** a biochemical cAMP/PKA/cGMP model, and it is strictly disabled in `mode="reference"`.

### 1. Mushroom Body anatomical scope & measured FlyWire v783 findings

- Plastic edges: **only excitatory `KC -> MBON` synapses** (**62,261** of 15,091,983 edges;
  256,719 synapses). All other 15,029,722 edges in FlyWire v783 remain strictly immutable.
- Anatomical DAN compartments (`>= 5` synapses): `PPL101` (2 bilateral cells) forms 894 synapses
  onto GABAergic `MBON11` (2 bilateral γ1pedc cells), and `MBON11` sends **358 direct inhibitory
  synapses back onto `PPL101`** (`-95, -102, -81, -80`).
- **Sparse KC odor ports (`odor_a_kc`, `odor_b_kc`):** In the uniform-weight Shiu et al. (2024)
  LIF model (which lacks APL divisive inhibition), stimulating raw ORNs (`ORN_DM1` vs `ORN_DL5`
  even at 10 Hz) activates ~3,790 / 5,177 KCs with Jaccard overlap **0.998**. F4 therefore drives
  two disjoint 250-cell sparse `KCg-m` ensembles (4.8% sparsity, 0% overlap) to isolate MB
  plasticity cleanly without modifying the reference LIF equations.
- **Measured `MBON11 -> PPL101` feedback loop:** During Odor A alone (600 ms), intact `MBON11`
  inhibits `PPL101` (`mbon11_dan <= 10.08`), whereas after `KC_A -> MBON11` depression, weakened
  `MBON11` disinhibits `PPL101` (`mbon11_dan = 17.11`), forming a self-sustaining memory loop
  (cf. Felsenberg et al. 2018). Setting the phasic dopamine burst threshold `dan_threshold = 18.0`
  cleanly separates external punishment bursts (`~27–30` at 220 Hz) from endogenous recurrence.

### 2. Pavlovian conditioning protocol on real FlyWire v783 (5 seeds `1..5`, CuPy GPU)

Command: `uv run alife connectome pavlov flywire_783 --seeds 1,2,3,4,5 --wiring <reference|shuffled_control> --backend cupy`
Parameters: `odor_hz = 150`, `dan_hz = 220`, `probe_ms = 300`, `trial_ms = 200`, `n_trials = 4`,
`ext_ms = 1200`, `unpaired_gap_ms = 1500`, `tau_elig = 200 ms`, `tau_dan = 150 ms`,
`dan_threshold = 18.0`, `lr_ltd = 0.00022`, `lr_ext = 0.03`.

| Condition / Readout | Real FlyWire v783 (mean ± SD, median) | Shuffled control (`wiring="shuffled_control"`) | Exact perm p vs Paired Post (Ref) | Cliff's δ (Ref) |
|---|---:|---:|---:|---:|
| Pre-training Odor A → `MBON11` (Hz) | **201.00 ± 4.18** (200.00) | 0.00 ± 0.00 | **0.0079** | -1.00 |
| After 1 paired trial Odor A → `MBON11` (Hz) | **101.00 ± 7.69** (96.67) | 0.00 ± 0.00 | — (dose-response: 5/5 seeds `post < 1-trial < pre`) | — |
| After 4 paired trials Odor A → `MBON11` (Hz) | **89.33 ± 9.17** (91.67) | 0.00 ± 0.00 | (reference arm) | — |
| Extinction (1200 ms Odor A alone) → `MBON11` (Hz) | **199.33 ± 4.65** (201.67) | 0.00 ± 0.00 | **0.0079** | +1.00 |
| Pre-training Odor B → `MBON11` (Hz) | **198.33 ± 1.67** (198.33) | 0.00 ± 0.00 | — | — |
| Post-training Odor B → `MBON11` (Hz, specificity) | **198.67 ± 0.75** (198.33) | 0.00 ± 0.00 | **0.0079** | -1.00 |
| Negative control: Unpaired (1500 ms shift) → `MBON11` (Hz) | **173.00 ± 65.06** (201.67) | 0.00 ± 0.00 | **0.0476** | -0.60 |
| Negative control: DAN silenced (`mushroom_body_dan`) → `MBON11` (Hz) | **200.00 ± 1.18** (200.00) | 0.00 ± 0.00 | **0.0079** | -1.00 |
| Changed `KC -> MBON` edges (Paired / Unpaired / DAN silenced) | **353.0 ± 6.7** / median **1.0** / **0.0 ± 0.0** | 0.0 / 0.0 / 0.0 (`2,494` plastic edges) | — | — |
| Downstream `MBON01` (Pre → Post Odor A, Hz) | 7.00 ± 2.98 → 5.67 ± 0.91 | 0.00 → 0.00 | — (see honest note below) | — |
| Wall time (5 seeds, full protocol, RTX 5070) | **35.22 s** | **24.76 s** | — | — |

### 3. Honest caveats & non-matches recorded for F4

1. **Unpaired control seed 5 stochastic cascade:** In 4/5 seeds (`1..4`), the 1500 ms shifted-timing
   unpaired control produced `205.0, 201.7, 200.0, 201.7 Hz` with `0–1` changed edges; in seed 5,
   a transient whole-brain recurrent burst during the `PPL101` pulse recruited ~27,084 KC→MBON
   edges (`56.7 Hz`), reflecting the absence of APL divisive inhibition in the uniform-weight
   Shiu et al. LIF model. Across all 5 seeds, paired vs unpaired still remained significant
   (`p = 0.0476`, median `91.67` vs `201.67 Hz`).
2. **Downstream `MBON01` disinhibition:** In the clean 8-neuron `synthetic_mb` circuit, depressing
   GABAergic `MBON11` robustly disinhibits avoidance `MBON01` (`~30 → ~67 Hz`). In the full
   138,639-neuron FlyWire v783 brain, `MBON01` stays low (`7.0 → 5.7 Hz`) because it receives
   additional convergent inputs beyond `MBON11`; downstream avoidance readout in the full brain
   therefore requires multi-MBON or behavioral-body integration (F8).
3. **Bitwise CPU/GPU & checkpoint equivalence:** Verified on `synthetic_mb` in
   `tests/reference/test_connectome_f4.py` (`exact_equal=True`, `max_plastic_scale_diff=0.0`),
   and verified in Chromium via Playwright E2E `e2e/connectome.spec.ts` (3/3 passed).

---

## Phase F5: Hidden-Truth System Identification on FlyWire v783 Olfactory–MB–LH Subcircuit (2026-11-21)

> **Honest framing:** Phase F5 is an **external gradient-based system identification benchmark**
> (`connectome_sysid` substrate / `alife connectome sysid`), using a differentiable firing-rate
> connectome in PyTorch. Recovering synthetic hidden mechanisms from simulated recordings tests
> *identifiability under partial, coarse, noisy observations* — it does not prove biological truth.

### 1. Subcircuit & measured GPU compute budget (RTX 5070 12 GiB)

- Extracted subcircuit (`olfactory_mb_lh`): **11,276 cells**, **1,105,291 internal edges**,
  **1,094 cell types**, **17 cell classes** (Antennal Lobe `AL_L/R` + `olfactory` ORNs +
  Mushroom Body `Kenyon_Cell`/`MBON`/`DAN` + Lateral Horn `LH_L/R`), driven by 12 disjoint
  glomerular ORN input groups (8 training odor mixtures, 4 unseen holdout odor mixtures).
- Single BPTT forward+backward epoch (`B = 8`, `T = 50` steps = `100 ms`, `dt = 2 ms`):
  **108.4 ms/epoch**, **184.4 MiB** (bare benchmark) / **340.3 MiB peak VRAM** during full
  3-restart + shuffled-control experiment (`58.23 s` total wall time).

### 2. Main hidden-truth recovery benchmark (25% cells recorded = 2,819 obs / 8,457 unobs, 10 ms bins, noise SD 0.015)

Command: `uv run alife connectome sysid flywire_783 --subcircuit olfactory_mb_lh --obs-fraction 0.25 --bin-ms 10 --noise-sd 0.015 --epochs 100 --restarts 3 --device cuda`
Hidden teacher mechanisms: heterogeneous per-type `tau_m` (1,094 types), 15% hidden sign flips
(164 active types), 17-class neuromodulation (gain + bias), 128 candidate electrical gap junctions
(32 active).

| Arm | Unseen `heldout_all_r2` | Unseen `heldout_unobs_r2` (8,457 unrecorded cells) | `tau_pearson_r` | `sign_flip_auroc` | `neuromod_pearson_r` | `gap_auroc` |
|---|---:|---:|---:|---:|---:|---:|
| `untrained` (uniform baseline) | 0.4934 | 0.4909 | n/a | 0.5000 | n/a | 0.5000 |
| **`reference` (biological FlyWire wiring, best of 3)** | **0.9694** | **0.9619** | **0.4895** | **0.8319** | **0.3567** | 0.5342 |
| `shuffled_control` (degree-preserving shuffled wiring) | 0.6254 | 0.5838 | 0.1979 | 0.4473 | — | — |

- Multi-restart identifiability (`n_restarts = 3` on identical recording): `verdict = well_identified`
  (`fit_r2_spread = 0.0012`, `heldout_unobs_r2_spread = 0.0031`, `tau_disagreement_ms = 1.784 ms`).

### 3. Observability curve & cell-class observability map (`alife connectome sysid-map`, 160.03 s)

#### Recording fraction curve ("how much recording is needed?")

| `obs_fraction` | Observed cells | Unseen `heldout_unobs_r2` | `tau_pearson_r` | `sign_flip_auroc` | Identifiability verdict |
|---:|---:|---:|---:|---:|---|
| 0.05 | 564 | **0.8946** | 0.2021 | 0.7263 | `well_identified` |
| 0.20 | 2,255 | **0.9421** | 0.4083 | 0.7861 | `well_identified` |
| 0.50 | 5,638 | **0.9633** | 0.5936 | 0.8049 | `well_identified` |
| 1.00 | 11,276 | **0.9773** | 0.6878 | 0.8125 | `well_identified` |

#### Single-class observability map ("which cells are most informative to record?")

| Recorded class only | Observed cells | Unseen `heldout_unobs_r2` | `tau_pearson_r` | `sign_flip_auroc` | Identifiability verdict |
|---|---:|---:|---:|---:|---|
| `LHLN` | 517 | **0.6978** | 0.0739 | 0.6187 | **`non_unique_sloppy`** |
| `ALLN` | 429 | **0.6358** | 0.1324 | 0.5691 | `well_identified` |
| `Kenyon_Cell` | 5,177 | **0.6118** | 0.0480 | 0.5974 | **`non_unique_sloppy`** |
| `MBON` | 96 | **0.5711** | 0.0536 | 0.5786 | `partially_identifiable` |
| `ALPN` | 685 | 0.2271 | **0.1499** | **0.6621** | **`non_unique_sloppy`** |

### 4. Key honest findings & non-identifiability limits (F5)

1. **Cross-layer stratified sampling dramatically beats single-layer recording:** Recording just
   **564 random cells across layers (5%)** yields `heldout_unobs_r2 = 0.8946` and `well_identified`
   solutions, outperforming **5,177 `Kenyon_Cell`s (46% of the circuit)** (`0.6118`,
   `non_unique_sloppy`) because single-layer recordings leave upstream/downstream mechanisms
   underdetermined.
2. **Sparse single-pair gap junctions are near-unidentifiable at low coverage in large circuits:**
   On the 40-cell `synthetic_olfactory` circuit at 50% coverage, gap-junction discovery reaches
   `AUROC = 0.8241`. On 11,276 FlyWire neurons at 25% coverage, only `0.25^2 = 6.25%` of random
   gap pairs have both endpoints observed, so `gap_auroc = 0.5342` (near chance) — an explicit,
   quantified limit of partial recording.
3. **Browser & offline CPU verification:** `tests/reference/test_connectome_f5.py` (4/4 passed on
   CPU) and Playwright E2E `e2e/connectome.spec.ts` (4/4 passed) verify `connectome_sysid`
   end-to-end without loading PyTorch in the FastAPI process.

---

## Phase F6: Fixed-Topology Task Learning — Biological Wiring vs Shuffled Wiring (2026-11-23)

> **Honest framing (mandatory):** Every model below is a **learning machine borrowing the fly
> connectome topology** (fixed graph structure with external PyTorch training), **NOT**
> biological learning by a fly.

### 1. Pre-registered protocol & decision rule

Before execution, `make_preregistration()` locks the hypothesis and evaluation rule
(hashed via SHA-256 into each report):

- **Subcircuit:** FlyWire v783 `olfactory_mb_lh` (**11,276 cells, 1,105,291 internal edges**),
  12 glomerular ORN input groups, 256 deterministic anatomical readout cells (`MBON`/`LHLN`/`LHCENT`).
- **Control:** Independent degree- and weight-preserving `shuffled_control` graphs generated per
  seed (`seeds = 1..5`), evaluated on identical unseen holdout test trials (`n_train = 64`,
  `n_test = 64`, leak-free training-set z-score readout standardization).
- **Pre-registered verdict thresholds:** `validated` if `mean_diff >= +0.05`,
  `cliffs_delta >= +0.50`, and exact permutation `p < 0.05`; `rejected` for the symmetric negative
  effect; otherwise `inconclusive`.

### 2. Full 3 tasks × 3 scopes × 5-seed benchmark on real FlyWire v783 (RTX 5070 CUDA)

| Task | Trainable scope | Biological `reference` test acc (mean ± SD) | `shuffled_control` test acc (mean ± SD) | Δ (`ref − shuf`) | Cliff's δ | Exact perm p | Pre-registered verdict | Wall time / Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---|---:|
| `odor_classification` (4-class, 25% chance) | `readout_only` | 0.950 ± 0.028 | **0.991 ± 0.009** | -0.041 | -1.00 | 0.0079 | `inconclusive` | 4.41 s / 110.6 MiB |
| `odor_classification` | `io_only` | **0.975 ± 0.018** | 0.969 ± 0.016 | +0.006 | +0.20 | 0.6905 | `inconclusive` | 39.96 s / 188.9 MiB |
| `odor_classification` | `partial_weights` | 0.953 ± 0.022 | **0.991 ± 0.009** | -0.037 | -1.00 | 0.0079 | `inconclusive` | 33.93 s / 258.4 MiB |
| `short_memory` (4-class, 48 ms delay) | `readout_only` | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.000 | +0.00 | 1.0000 | `inconclusive` | 1.99 s / 110.2 MiB |
| `short_memory` | `io_only` | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.000 | +0.00 | 1.0000 | `inconclusive` | 63.04 s / 230.7 MiB |
| `short_memory` | `partial_weights` | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.000 | +0.00 | 1.0000 | `inconclusive` | 51.70 s / 341.9 MiB |
| `nonlinear_xor` (2-class, 50% chance) | `readout_only` | 0.731 ± 0.107 | **0.812 ± 0.044** | -0.081 | -0.60 | 0.1508 | `inconclusive` | 1.71 s / 110.0 MiB |
| `nonlinear_xor` | `io_only` | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.000 | +0.00 | 1.0000 | `inconclusive` | 38.41 s / 187.3 MiB |
| `nonlinear_xor` | `partial_weights` | 0.609 ± 0.064 | 0.609 ± 0.142 | -0.000 | +0.20 | 1.0000 | `inconclusive` | 33.16 s / 257.6 MiB |

### 3. Honest interpretation of the null/negative F6 result

1. **No generic topology advantage over degree-matched shuffled reservoirs:** Under the
   pre-registered criteria, **0 of 9 conditions were validated** (all 9 were `inconclusive`).
   We record this negative result without modifying thresholds post-hoc.
2. **Shortcut mechanism in shuffled wiring:** In biological FlyWire v783, sensory ORNs pass
   through 3–4 specialized stages (`ORN -> ALPN/ALLN -> KC -> MBON/LHLN`) before reaching the
   readout neurons, whereas whole-subcircuit degree-preserving shuffling introduces direct 1-hop
   random edges from ORNs onto `MBON`/`LHLN` readout cells—creating a high-SNR random Echo State
   projection that slightly outperforms biological routing on `odor_classification` (`0.991` vs
   `0.950`, `p = 0.0079`). Adding learnable input projections (`io_only`) closes this gap
   (`0.975` vs `0.969` on odor classification; `1.000` vs `1.000` on XOR).
3. **Memory ceiling caveat:** With 256 standardized readout cells and `tau = 20 ms`, a `48 ms`
   delay saturates at `1.000` for both wirings; distinguishing biological vs shuffled recurrent
   memory persistence would require much longer delays (`> 200 ms`) or spiking noise.
4. **Offline & browser verification:** `tests/reference/test_connectome_f6.py` (5/5 passed on CPU)
   and Playwright E2E `e2e/connectome.spec.ts` (5/5 passed) verify `connectome_task` end-to-end.



