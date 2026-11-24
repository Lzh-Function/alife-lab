# ADR 0051: Stage 2 Connectome S1 (*C. elegans* 302-Neuron Graded + Gap-Junction Nervous System & Crawling Body)

- **Status:** Accepted
- **Date:** 2026-11-24
- **Related:** ADR 0035 (Connectome Species Packs), `docs/plans/connectome_plan.md` (§ S1), `docs/plans/cultural_world_plan.md` (§ 0 Stage 2)

## Context

Prior connectome substrates (`connectome`, `fly_embodied`, `fly_swarm`, `fly_world`) model *Drosophila* spiking neurons using `MODEL_SPIKING_LIF` and `SYNAPSE_CHEMICAL` only. Extending the common `ConnectomePack` architecture to Stage 2 / S1 (*Caenorhabditis elegans*, 302 somatic neurons) requires:
1. Non-spiking graded membrane potentials (`MODEL_GRADED`, continuous sigmoidal neurotransmitter release) and bidirectional ohmic electrical synapses (`SYNAPSE_GAP_JUNCTION`).
2. Strict zero-impact isolation from all existing FlyWire / BANC / MaleCNS spiking LIF code paths and bit-exact tests.
3. Offline-viable 302-neuron species pack (`synthetic_celegans_302`) alongside a pure-CSV external loader (`celegans_302`) requiring user approval before external data download, zero pickles.
4. A closed-loop planar Resistive Force Theory (RFT) crawling body (`worm_embodied`) driven directly by connectome motor pools, with honest separation between connectome circuitry vs proprioceptive stretch coupling.

## Decisions

1. **Zero-Impact Separation from `lif.py` Spiking Oracle:**
   - `src/alife_lab/connectome/lif.py` (`step_substep_cpu`, `step_tick_cpu`) remains untouched so all existing `MODEL_SPIKING_LIF` FlyWire/BANC/MaleCNS runs are bit-for-bit invariant.
   - New graded + gap-junction solver lives in `src/alife_lab/connectome/graded.py` (`GradedParams`, `GradedCircuit`, `step_graded_substep_cpu`, `step_graded_tick_cpu`, `step_hybrid_tick_cpu`).
   - `ConnectomeAdapter` detects `has_graded_or_gap(pack, csr)` and dispatches graded packs to `step_graded_tick_cpu` (CPU `float32`) while leaving all spiking packs on the untouched LIF / CuPy path.

2. **Unconditionally Stable Conductance-Based Graded + Gap ODE:**
   - Chemical synapses are conductance-based with continuous sigmoidal presynaptic release $s_i = \sigma((V_i - V_{\text{half}}) / V_{\text{slope}}) \cdot (1 - \text{silenced}_i)$.
   - Electrical gap junctions (`SYNAPSE_GAP_JUNCTION`) contribute symmetric ohmic current $\sum_u g_{\text{gap}} w_{uj} (V_u - V_j)$ gated off if either endpoint is silenced or when `gap_junctions_enabled=False`.
   - Membrane potential evolves via diagonal-implicit exponential Euler in deterministic `float32` using vectorized `np.bincount` accumulation.

3. **302-Neuron *C. elegans* Pack & Tabular Real-Data Loader (`celegans.py`):**
   - `synthetic_celegans_302`: built-in offline 302-neuron hermaphrodite *C. elegans* wiring pack with canonical cell names (`ALML/R`, `AVM`, `PLML/R`, `PVM`, `ASEL/R`, `AWC*`, `ASH*`, `AVA*`, `AVB*`, `AVD*`, `AVE*`, `PVC*`, `DB1..7`, `VB1..11`, `DA1..9`, `VA1..12`, `DD1..6`, `VD1..13`, `SMD*`, `SMB*`, `RMD*`, `I1..I6/M1..M5/NSM*`, etc.), anatomical somatic 3D worm coordinates, 13 named `IOPort`s, chemical synapses, and symmetric gap junctions.
   - `celegans_302`: pure-CSV parser (`load_celegans_from_edgelist_csv`, zero pickles) pinned in `src/alife_lab/connectome/registry.py` (`CELEGANS_302_SPEC`) to OpenWorm `CElegansNeuroML/herm_full_edgelist.csv` at commit `1ca08b14f4e795a973ce447565e0c97b897ee4dc` (`245,463 B`, SHA-256 `142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca`), downloaded into `.alife/datasets/celegans_302/` (outside Git) only with explicit user approval.
   - Parser normalizes zero-padded single-digit VNC motor names (`VA01 -> VA1`, `DB01 -> DB1`, etc., raising matched canonical neurons from `236/302` to `300/302` with only `CANL`/`CANR` lacking synapses and all 148 other tokens verified non-neuronal muscles/end-organs) and deduplicates bidirectional `electrical` rows without doubling gap-junction weights (`5,764` edges: `3,604` chemical + `2,160` gap junctions = `29,114` total synapses).
   - **Unsigned Raw EM Receptivity Finding:** Because `herm_full_edgelist.csv` provides topology and synapse counts (`chemical` vs `electrical`) without postsynaptic receptor polarity (`GluCl`/`AChCl`), GABA-only signing (`DD/VD/RME/AVL/DVB/RIS`) leaves `98.13%` of edges excitatory (`ALM/AVM/ASH` send `97` chemical synapses to both `AVA/D/E` and `AVB/PVC`; `AVB -> AVA` sends `47` excitatory synapses). Under `worm_embodied` with unmodified raw EM signs, global scalar conductance scaling saturates near `0 mV` (`g_chem=0.08`) or deadlocks `AVA`/`AVB` together (`g_chem=0.008`, `AVB=-10.0 mV`, `AVA=-8.2 mV`, `signed_forward=+0.003 mm`) rather than switching cleanly; postsynaptic receptor polarity (e.g., Fenyves et al. 2020 `bank-fenyves/CeConn-SignPrediction`, pending separate approval if desired) or data-driven sign estimation is required for empirical whole-animal command switching.

4. **Planar 12-Segment Resistive Force Theory Crawling Body (`worm_body.py`):**
   - Solves exact $3 \times 3$ overdamped viscous drag balance $\sum_s D_s \mathbf{v}_s = \mathbf{0}$, $\sum_s \mathbf{q}_s \times (D_s \mathbf{v}_s) = 0$ with $c_\perp / c_\parallel = 3.0$ in `float64`.
   - Dorsal (`DB/DA - DD`) and ventral (`VB/VA - VD`) motor voltages drive curvature muscles along 11 body joints, while proprioceptive curvature feedback enters upstream/motor pools only in proportion to connectome command gating (`AVB` vs `AVA`)—ensuring `gap_junctions_enabled=False` and `wiring="shuffled_control"` genuinely degrade locomotion without hidden bypasses.

5. **Honest Framing & Empirical Activity Fitting (`worm_activity.py`):**
   - Synthetic-to-synthetic oracle (`run_worm_activity_benchmark`) evaluates offline sensitivity to gap-junction ablation and edge shuffling (`E1`).
   - Empirical whole-brain fitting (`load_celegans_activity_recordings`, `run_real_worm_activity_fit`, `alife connectome fit-worm`) ingests pure Apache Parquet (`worm_data_short.parquet` from Hugging Face `qsimeon/celegans_neural_data` at commit `a28aacad1a997ac7aee98f1c52993ab8c0599ee8`, `669,511,381 B`, SHA-256 `17701963149851d37d40955096de181d8c5378c256fe4ede93864380f9363ddc`, MIT license, ZERO pickles) and evaluates 5-arm connectome constraints (`reference`, `no_gap_junctions`, `raw_em_gaba_only`, `shuffled_control`, `unconnected`) on strictly held-out final 30% frames (`spatial_unobs_r2` + `temporal_rollout_r2`).
   - `worm_embodied` enforces `mode="exploratory"` (RFT kinematics and synthetic conductances are simplified mechanical/phenomenological models, not biophysical finite-element cuticle or patch-clamp models).
