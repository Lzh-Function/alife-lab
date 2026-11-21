# Reference fidelity — Adult *Drosophila* connectome (`flywire_783`, F0–F1)

Status: **F0–F1 complete** (2026-11-21); F2 (dedicated 3D brain visualization, cell search and
raster UI) not yet implemented.

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


