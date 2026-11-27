# ADR 0053: Multi-Species Connectome Packs (S2–S4), Cross-Species Comparison, and F13a–c Exploratory Chimeras

- **Status:** Accepted (2026-11-27)
- **Context:** Stage 4 of the unified roadmap ([`docs/plans/cultural_world_plan.md`](../plans/cultural_world_plan.md) §0, [`docs/plans/connectome_plan.md`](../plans/connectome_plan.md) §5 S2–S4 & F13) extends ALife Lab's `ConnectomePack` contract beyond adult *Drosophila* (FlyWire v783, BANC, MaleCNS) and *C. elegans* (S1) to three additional empirical whole-brain / whole-body connectomes across distant phyla, plus a uniform cross-species comparative layer and exploratory cross-species / intra-species chimera experiments.

## Decisions

### 1. Pure Tabular Species Packs (S2–S4) + Offline CPU Synthetic Counterparts

Three empirical datasets are pinned by exact URL and SHA-256 checksum in [`src/alife_lab/connectome/registry.py`](../../src/alife_lab/connectome/registry.py) and parsed via pure tabular readers (CSV / ZipSlip-checked `.csv.zip` / `.txt`; **zero pickles**, no external archive extraction to disk) in [`src/alife_lab/connectome/species_packs.py`](../../src/alife_lab/connectome/species_packs.py):

| Stage | `dataset_id` | Organism & Scope | Empirical Source | Cells | Directed Edges | Weight Unit | Cell Model |
|---|---|---|---|---:|---:|---|---|
| **S2** | `fly_larva_l1` | *Drosophila melanogaster* L1 larva whole brain | Winding et al. (2023) *Science* (Netzschleuder `fly_larva.csv.zip`) | 2,956 | 110,677 (collapsed from 4 multigraph types; 352,611 synapses) | synapse count | `MODEL_SPIKING_LIF` |
| **S3** | `platynereis_3d` | *Platynereis dumerilii* 3-day larva whole-body synaptic connectome | Verasztó, Jasek, Jékely et al. (2024/2025) *eLife* (JekelyLab commit `ab750290...`) | 1,720 connected cells (neurons + muscles + ciliary effectors) | 8,451 (19,991 synapses) | synapse count | `MODEL_GRADED` |
| **S4** | `ciona_larva` | *Ciona intestinalis* tadpole larva CNS | Ryan, Lu & Meinertzhagen (2016) *eLife* (Netzschleuder `cintestinalis.csv.zip`) | 205 | 2,903 (13,295 units) | 0.1 µm synaptic depth (`round(um * 10)`) | `MODEL_SPIKING_LIF` |

- **Honest scope note on `platynereis_3d`:** While the full EM volume contains ~9,162 somatic/epidermal cells, the official synaptic connectome adjacency matrix (`full_connectome_adjacency_matrix.csv`) represents the **1,720 synaptically connected cells** (neurons, muscles, ciliary band effectors). Both numbers are documented transparently.
- **Offline CPU CI parity:** Deterministic synthetic packs (`synthetic_fly_larva`, `synthetic_platynereis`, `synthetic_ciona`, `synthetic_chimera_worm_fly_mb`) preserve identical IO port names, cell models, and reference-check interfaces so CPU-only CI runs every loader, comparative assay, chimera, CLI command, and API endpoint with zero network downloads.

### 2. Cross-Species Comparative Statistics & Explicit Synaptic Scaling

[`src/alife_lab/connectome/cross_species.py`](../../src/alife_lab/connectome/cross_species.py) evaluates every species under one identical structural metric suite (`density`, `mean_out_degree`, `std_out_degree`, `max_in/out_degree`, `top1pct_hub_synapse_share`, `reciprocity`, `excitatory_chem_fraction`, `gap_junction_edge_fraction`, role fractions) and a paired degree-preserving `shuffled_control` sensorimotor probe.

- **Explicit small-brain synaptic weight scaling:** Applying Shiu et al. (2024) adult-fly `w_syn_mv = 0.275 mV` directly to 205-cell `ciona_larva` or 2,956-cell `fly_larva_l1` yields 0 Hz silence because small brains have far fewer converging presynaptic partners. `recommended_lif_params()` documents explicit per-scale defaults (`ciona_larva`: `1.45 mV`, `fly_larva_l1`: `0.85 mV`, adult fly unchanged at `0.275 mV`) and applies the **exact same value** to both `reference` and `shuffled_control` arms.

### 3. Hybrid Spiking + Graded Runtime Isolation

Cross-species circuit grafts (F13c) mix `MODEL_GRADED` host neurons (*C. elegans*, resting around `-35 mV`) and `MODEL_SPIKING_LIF` graft neurons (*Drosophila* Mushroom Body, threshold `-45 mV`) inside one `ConnectomePack`.
- `make_initial_hybrid_state()` allocates a full LIF `delay_ring` of length `lif_params.delay_steps + 1` while initializing graded cells to `e_leak_mv` and LIF cells to `v_0`.
- `step_hybrid_tick_cpu()` temporarily clamps graded-cell voltages/conductances to `(v_0, 0.0)` during the LIF substep so graded potentials (`-35..-5 mV > v_th`) never spuriously emit LIF spikes into `delay_ring`. Cross-model graded→LIF current is injected strictly when a graded cell depolarizes above `max(e_exc_mv, 0.0)` (active sensory/bridge drive), and inhibitory LIF→graded spikes inject bounded hyperpolarizing pulses.
- `ConnectomeAdapter._make_world()` preserves the experiment's `LIFParams` whenever a pack contains both `MODEL_GRADED` and `MODEL_SPIKING_LIF` cells, guaranteeing exact bitwise checkpoint round-trip across pause/resume/fork.

### 4. Exploratory Chimeras (F13a–c) Boundaries & Mandatory Provenance

[`src/alife_lab/connectome/chimera.py`](../../src/alife_lab/connectome/chimera.py) implements three distinct chimera paradigms:
1. **F13a (`build_intraspecies_hybrid_pack`, `run_f13a_intraspecies_hybrid_assay`):** Cell-type-aligned crossover between two packs sharing homologous cell types (e.g., `male_cns_v1 × banc_888` across 7,445 shared cell types).
2. **F13b (`run_f13b_brain_transplant_experiment`):** Cross-species brain transplant into the 12-segment planar RFT crawling/swimming body via an explicit 4-parameter bridge layer, compared under identical optimization budgets against `donor_shuffled` and `donor_silenced`.
3. **F13c (`build_mb_graft_chimera_pack`, `run_f13c_mb_graft_assay`):** Circuit graft transplanting *Drosophila* larva Mushroom Body cells (`KC`, `MBON`, `MBIN/DAN`) into the 302-cell *C. elegans* host via explicit `AWC/AWA -> KC`, `ASH -> DAN`, and `MBON -> AVB/AVA` bridge synapses, evaluated on Pavlovian odor-aversion conditioning against `unpaired_control`, `graft_silenced`, and `shuffled_graft`.

**Mandatory scientific guardrails:**
- `ConnectomeExperiment` validator in [`src/alife_lab/core/spec.py`](../../src/alife_lab/core/spec.py) **rejects `mode="reference"`** for chimera datasets (`chimera_worm_fly_mb`, `synthetic_chimera_worm_fly_mb`); chimeras are strictly `mode="exploratory"`.
- Every chimera pack and report carries `CHIMERA_FRAMING_NOTE`, exact `species_provenance` cell/edge fractions, and explicit bridge-edge share.
- The browser Inspector (`InspectorPanel.tsx`) renders per-cell `species_name` and multi-species `species_provenance` percentages automatically whenever a pack contains >1 species.

## Consequences

- **Positive:** Five empirical species spanning ~700× neuron scale (205 to 138,639 cells) share one data contract, one CLI, one Web GUI, and one checkpoint format.
- **Negative / Limitations:** Real empirical gap-junction lists are not published in the `platynereis_3d` synaptic matrix (only chemical synapses are recorded there; synthetic Platynereis includes gap edges for code-path testing). F13b results show honestly that a tuned 4-parameter bridge can extract thrust from shuffled graded/runaway spiking brains as well as intact ones unless body-specific topography is constrained (`ciona_larva` intact beats shuffled 2.48×, whereas `platynereis_3d` and `fly_larva_l1` do not).
- **Reversibility:** All Stage 4 logic lives in dedicated modules (`species_packs.py`, `cross_species.py`, `chimera.py`) plus additive dataset specs/presets; existing FlyWire, BANC, MaleCNS, C. elegans, and CulturalWorld hashes and tests remain untouched.
