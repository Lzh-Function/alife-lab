# ADR 0040: Connectome Hidden-Truth System Identification (F5)

- Status: Accepted
- Date: 2026-11-21
- Related: ADR 0035 (species-pack), ADR 0036 (LIF engine), ADR 0038 (F3 screen & search), ADR 0039 (F4 plasticity), `docs/plans/connectome_plan.md`

## Context

Phase F5 tests a central scientific question before attempting to fit real animal brain recordings (F7): **given an anatomical connectome that omits biological mechanisms (heterogeneous cell-type time constants, neurotransmitter sign errors, neuromodulation, electrical gap junctions), how accurately can gradient-based system identification recover those hidden mechanisms and predict unseen stimuli from realistic limited recordings (partial cell coverage, coarse temporal bins, noise)?**

Per `AGENTS.md` and `docs/plans/connectome_plan.md` §3.2 / §4 F5:

1. **External optimization boundary:** Fitting and loss functions must stay strictly outside the core `connectome` simulation substrate.
2. **No GPU runtime in API process:** PyTorch must be imported lazily inside CLI commands and worker processes only.
3. **Honest identifiability & non-uniqueness:** When multiple distinct parameter sets fit the same partial recording equally well (sloppy / non-unique solutions), the system must measure and report that non-uniqueness rather than hiding it.
4. **Offline CPU-only CI viability:** Every test and browser E2E check must work offline without requiring the 141 MiB FlyWire dataset or a GPU.

## Decision

### 1. Subcircuit extraction & offline olfactory circuit (`src/alife_lab/connectome/subcircuit.py`)

- `extract_subcircuit()` extracts the real FlyWire v783 Antennal Lobe + Mushroom Body + Lateral Horn subcircuit (`subcircuit_id="olfactory_mb_lh"`: **11,276 cells, 1,105,291 internal edges, 1,094 cell types, 17 cell classes**) with 12 disjoint biological glomerular ORN input groups (`ORN_DA1`, `ORN_VA1d`, `ORN_VA1v`, ...), plus degree-preserving `shuffled_control` wiring via `make_shuffled_csr()`.
- `make_synthetic_olfactory_pack()` (`dataset_id="synthetic_olfactory"`) provides a built-in 40-neuron, 18-type, 6-class E/I olfactory microcircuit (`ORN -> ALPN / ALLN -> KC -> MBON / LHLN`) for fast offline CPU tests and interactive browser exploration.

### 2. Differentiable rate-based hidden-truth engine (`src/alife_lab/connectome/sysid.py`)

- Implements a sparse COO PyTorch firing-rate connectome (`dt = 2 ms`, `T = 50` steps = 100 ms, batch of 8 training odors + 4 disjoint unseen holdout odors). Presynaptic sign/gain multipliers multiply `r` prior to `torch.sparse.mm(W_base, r * m_cell)` so the sparse matrix remains coalesced and constant across steps.
- Models all four hidden biological mechanisms from `connectome_plan.md` §4 F5:
  1. Per-type membrane time constants `tau_c` (`[6, 50]` ms vs uniform `20` ms baseline).
  2. Per-type hidden excitatory/inhibitory sign flips (`sign_flip_fraction = 0.15` of active types).
  3. Per-class neuromodulatory gain `gamma_k` and tonic bias `b_k` (scored with separate gain and bias Pearson correlations to avoid artificial mean-offset correlation).
  4. Sparse undirected electrical gap junctions `g_ik * (r_k - r_i)` on non-chemical intra-class candidate pairs.
- Applies a realistic recording operator (`ObservationSpec`: partial cell mask by `obs_fraction` or `obs_classes`, temporal average pooling `bin_ms`, additive Gaussian noise `noise_sd`).
- Evaluates `untrained` baseline, `n_restarts` independent random initializations on `reference` wiring, and `shuffled_control` wiring on both training observations and **unseen holdout stimuli** (`heldout_all_r2`, `heldout_obs_r2`, `heldout_unobs_r2`), plus multi-restart identifiability classification (`well_identified` / `partially_identifiable` / `non_unique_sloppy`).
- `run_observability_study()` produces both the recording-fraction curve (`0.05..1.00`) and the cell-class observability map (`ALPN`, `ALLN`, `Kenyon_Cell`, `MBON`, `LHLN`).

### 3. External substrate `connectome_sysid`, CLI & browser integration

- Registers `connectome_sysid` (`src/alife_lab/substrates/connectome/sysid_adapter.py`, `backends=("torch", "torch_cuda")`, `mode="exploratory"` only) as a separate external-optimization substrate alongside `nca_train` and `search`.
- Renders a deterministic `160x120` RGB (`field_u8`) 4-quadrant diagnostic view (hidden teacher unseen-stimulus heatmap, student prediction, absolute error map, and 5 mechanism-recovery bars), supporting pause, step, inspect, checkpoint save/restore, and side-by-side comparison of biological vs shuffled wiring in the browser.
- Adds CLI subcommands `alife connectome sysid` and `alife connectome sysid-map`, 3 catalog presets, beginner guide, glossary entry, and full `ja`/`en` i18n.

## Consequences

- On RTX 5070, 1 forward+backward BPTT epoch over the 11,276-neuron, 1,105,291-edge FlyWire v783 olfactory–MB–LH subcircuit takes **108.4 ms** and **340.3 MiB peak VRAM** (well below the 1–2 GiB estimate in `connectome_plan.md` §7).
- From 25% noisy recordings, biological wiring achieves `heldout_unobs_r2 = 0.9619` and `sign_flip_auroc = 0.8319` (vs `0.5838` and `0.4473` for shuffled wiring), while honestly revealing that (a) single-pair gap junctions among 11,276 neurons are poorly identifiable at 25% coverage (`AUROC = 0.534`), and (b) recording a single cell class (even 5,177 `Kenyon_Cell`s) yields `non_unique_sloppy` fits and worse whole-circuit generalization (`0.612`) than recording just 564 stratified cells across layers (`0.895`).
