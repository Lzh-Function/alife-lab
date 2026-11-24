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
   - `celegans_302`: pure-CSV parser (`load_celegans_from_edgelist_csv`, zero pickles) for real OpenWorm/WormWiring edgelists, kept strictly unloaded until the user approves downloading the external files listed in Stage 2 verification.

4. **Planar 12-Segment Resistive Force Theory Crawling Body (`worm_body.py`):**
   - Solves exact $3 \times 3$ overdamped viscous drag balance $\sum_s D_s \mathbf{v}_s = \mathbf{0}$, $\sum_s \mathbf{q}_s \times (D_s \mathbf{v}_s) = 0$ with $c_\perp / c_\parallel = 3.0$ in `float64`.
   - Dorsal (`DB/DA - DD`) and ventral (`VB/VA - VD`) motor voltages drive curvature muscles along 11 body joints, while proprioceptive curvature feedback enters upstream/motor pools only in proportion to connectome command gating (`AVB` vs `AVA`)—ensuring `gap_junctions_enabled=False` and `wiring="shuffled_control"` genuinely degrade locomotion without hidden bypasses.

5. **Honest Framing & Activity Comparison:**
   - Recorded calcium traces ($\Delta F/F$) can be loaded via pure CSV/JSON in `worm_activity.py`; before user approval for external downloads, `run_worm_activity_benchmark` evaluates synthetic ground-truth held-out $R^2$ / Pearson $r$ against `no_gap_junctions` and `shuffled_control`.
   - `worm_embodied` enforces `mode="exploratory"` (RFT kinematics and synthetic conductances are simplified mechanical/phenomenological models, not biophysical finite-element cuticle or patch-clamp models).
