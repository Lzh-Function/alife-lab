# Reference fidelity — Adult *Drosophila* connectome (`flywire_783`, F0)

Status: **F0 complete** (2026-11-21); F1 (CPU oracle + CUDA LIF engine and Shiu et al. 2024
sensorimotor reproduction) not yet implemented.

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
  and this pack use FlyWire v783 (138,639 neurons). F1 reproduction results will document any
  v630 vs v783 discrepancies explicitly.
