# ADR 0038: Connectome Systematic Cell-Type Silencing Screen and Sensory Input Search (F3)

- Status: Accepted
- Date: 2026-11-21
- Related: ADR 0035 (connectome species-pack), ADR 0036 (bit-identical CPU/CUDA LIF), ADR 0037 (F2 browser observation), `docs/plans/connectome_plan.md`

## Context

Phase F3 extends the FlyWire v783 connectome platform with two systematic circuit investigation capabilities:

1. **Single-cell-type silencing screens ("一つずつ黙らせる総当たり")**: silence neurons grouped by `cell_type` and test whether a target motor response (e.g. MN9 proboscis extension) drops or increases relative to the intact brain, paired with a degree-preserving shuffled-wiring negative control.
2. **Sensory input search ("入力の探索")**: optimize *only* sensory stimulation rates (`stimuli.<idx>.rate_hz`) using the platform's existing external search layer (CMA-ES, MAP-Elites, Novelty Search, IMGEP, Random baseline) without modifying biological wiring or injecting rewards into the connectome substrate.

## Decision

### 1. First-class `silenced_cell_types` and sweep integration

- Add optional `silenced_cell_types: tuple[str, ...] = ()` to `ConnectomeExperiment` (omitted from canonical JSON when empty so existing spec digests remain invariant).
- `ConnectomeAdapter.initialize()` resolves declared `cell_type`s to cell indices and marks `lif.silenced[idx] = 1`, matching Shiu et al. (2024)'s **synaptic-output silencing** semantics (silenced neurons cannot scatter outgoing spikes into the delay ring). Unknown `cell_type` names fail loudly with `ValueError`.
- Extend `SweepFactor` (`core/sweep.py`) and `SweepService` (`service/sweeps.py`) to accept top-level connectome factor paths (`wiring`, `silenced_ports`, `silenced_cell_types`) and digit-containing metric IDs (e.g. `mn9_rate_hz`), enabling standard `alife sweep` and browser Sweeps UI ablation studies.

### 2. High-throughput GPU-batched silencing screen (`src/alife_lab/connectome/ablation.py`, CLI `alife connectome screen`)

- For screens spanning dozens to hundreds of cell types × seeds, `run_silencing_screen()` executes all `(condition, seed)` pairs through a shared `ConnectomePack`/`CSRGraph` using `CupyBatchedLIFRunner` (`batch_size=8` CUDA graphs on GPU) or exact `step_tick_cpu` on CPU, yielding **bitwise-identical** spike counts across CPU and GPU.
- `discover_active_cell_types()` automatically selects the top-K most active non-input/non-target `cell_type`s from a baseline pass when an explicit `--cell-types` list is not supplied.
- Statistics reuse `alife_lab.analysis.stats` (`describe`, `compare`, `holm`) at the independent seed level (never treating ticks within a run as independent samples).

### 3. Sensory input search via existing `search` substrate

- Add `ConnectomeExperiment` to `SearchExperiment.target`, register versioned objective `connectome_mn9_rate` (v1) in `evolution/objectives.py`, and restrict `search_path_kind()` strictly to `stimuli.<idx>.rate_hz` (wiring and LIF parameters are non-searchable by contract).
- Add `numeric_policy: "float32_connectome"` to `SearchDynamics` and map `SEARCH_MANIFEST.backend_numeric_policies` so connectome searches run on both `numpy` (CPU) and `cupy` (GPU batched CUDA graphs) with bitwise-identical scores, while leaving Lenia/Creature (`float64_cpu`) and CLIP (`float32_clip_torch`) backend selection completely unchanged.
- Map connectome evaluation outputs into existing QD archive columns (`speed` = MN9 rate Hz, `mass` = total stimulus rate Hz, `alive_fraction` = fraction of active brain neurons) so MAP-Elites, Lineage, SearchInspector and "Open as run" work without special cases.

## Consequences

- On RTX 5070, a 24-cell-type × 5-seed FlyWire v783 screen (125 simulated brain-seconds) finishes in **4.58 s** (**27.3 brain-s/s**), and a 10-seed confirmation of the top 5 hits finishes in **2.27 s** (**26.5 brain-s/s**), automatically identifying both essential excitatory relays (`CB0553`, -99.1% MN9) and inhibitory brakes (`DNge051`, +30.7% MN9) at `p_holm = 0.0001`.
- Offline CPU-only CI remains fast and self-contained via `synthetic_tiny` presets, sweeps and tests.
