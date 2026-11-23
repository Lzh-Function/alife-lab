# ADR 0044: Phase F8 Level 2 — Whole-CNS (Brain + Ventral Nerve Cord) Direct Leg Motor Control

- **Status:** Accepted & Implemented (2026-11-23)
- **Context:** Phase F8 Level 1 ([ADR 0043](0043-connectome-embodied-fly.md)) coupled the 138,639-cell FlyWire v783 brain to a 6-legged MuJoCo fly via a hand-crafted tripod CPG. As documented in `connectome_plan.md §4 (F8 Level 2)`, replacing the hand-crafted CPG with a biological Ventral Nerve Cord (VNC) connectome tests the core scientific question: *how much leg motor recruitment, proprioceptive modulation, and stepping emerges directly from a real Brain + VNC wiring diagram under uniform-weight spiking LIF physics?*

## Decision

1. **Two real whole-CNS species packs (pure Apache Arrow `.feather` + `.csv`, zero pickles):**
   - `banc_888` — **BANC v888** adult female Brain + VNC (Lee Lab / Bates et al., CC-BY 4.0, Harvard Dataverse DOI `10.7910/DVN/7WTH1N`): 156,808 proofread non-glia neurons, 13,376,835 directed edges, 41,867,711 chemical synapses.
   - `male_cns_v1` — **FlyEM Male CNS v1.0** adult male Brain + VNC (HHMI Janelia / Cambridge / Google Research, CC-BY 4.0): 165,122 traced neurons, 25,563,197 directed edges, 124,025,046 chemical synapses.
   - Both normalize into the shared `ConnectomePack` schema via `src/alife_lab/connectome/cns_packs.py`, resolving all **36/36** antagonist leg motor neuron pools (`mn_{LF..RH}_{coxa_pro,coxa_ret,fem_flex,fem_ext,tib_ext,tib_flex}`) and **6/6** leg proprioceptive feedback ports (`proprio_{LF..RH}`), plus deterministic SHA-256-keyed local binary caches (`pack_cache_v2.npz`, `allow_pickle=False`) loading in `0.15–0.21 s`.
   - A 72-neuron offline `synthetic_cns_embodied` Brain+VNC circuit keeps CPU-only CI 100% self-contained.

2. **Honest Level 2 physics (`control_mode: level2_vnc` / `level2_vnc_tuned`):**
   - `step_body_level2_vnc_1ms()` removes ALL hand-crafted sinusoidal tripod CPGs and ALL artificial planar velocity injection (`d.qvel[0..1, 5]`).
   - All 18 MuJoCo leg actuators (`6 legs × 3 joints`) are driven directly by antagonist rate differences of the 36 VNC motor neuron pools (`tanh((ext - flex) / norm_hz)`), with closed-loop proprioceptive load & joint-deflection feedback (`proprioceptive_feedback()`) routed into `proprio_{LF..RH}`.
   - Brain-only packs (such as `flywire_783`) explicitly fail validation if `control_mode != "level1_cpg"`.
   - `tune_vnc_locomotion()` (`alife connectome tune-vnc`) searches only global biophysical scaling (`w_syn_mv`, `vnc_proprio_gain_hz`, `explore_drive_hz`) while keeping the biological wiring topology 100% intact, always comparing against a degree-preserving shuffled-VNC control.

3. **2-Pane Brain+VNC 3D Visualization & Live 36-Channel VNC Telemetry:**
   - `Brain3D` renders the full 3D Brain + Cervical Connective + VNC point cloud at 60 FPS using GPU shader spike decay.
   - `EmbodiedFlyView` displays an explicit Level 2 honest framing banner and a live 6-leg × 6-muscle VNC motor neuron rate + proprioceptive load matrix.

## Honest Scientific Findings & Limitations

- Real GPU measurements on RTX 5070 + CPU float64 MuJoCo achieve **1.52×–2.18× real time** (`banc_888`: `2.18× RT`, `118.1 MiB VRAM`; `male_cns_v1`: `1.52× RT`, `211.9 MiB VRAM`), proving whole-CNS Brain + VNC + physics runs faster than real time on a single consumer GPU.
- Intact BANC v888 and MaleCNS v1.0 wiring genuinely transmit descending (`DNp09`) and proprioceptive signals into all 6 legs' motor neuron pools, and shuffling the wiring collapses VNC motor recruitment by **17×–33×** (`banc_888`: `2.7 → 0.0 Hz`; `male_cns_v1`: `13.4 → 0.4 Hz`).
- **Critical negative/honest finding:** Under uniform-weight LIF (`w_syn_mv = 0.275 mV`, no heterogeneous per-cell-type conductances, no intrinsic plateau/pacemaker currents, no neuromodulatory VNC gain fields), neither raw BANC v888 nor raw MaleCNS v1.0 spontaneously produces sustained high-speed tripod forward walking (`0.008–0.025 cm` in 400 ms vs `~0.9 cm` in Level 1 CPG). This matches `connectome_plan.md §8` and motivates future cell-type-specific VNC biophysical fitting (F5/F7 methods applied to VNC premotor circuits).

## Consequences

- Level 1 (`level1_cpg`) remains available for rich interactive chemotaxis/foraging/grooming demos, while Level 2 (`level2_vnc` / `level2_vnc_tuned`) provides an honest, zero-CPG scientific testbed for female (BANC) and male (MaleCNS) whole-CNS motor control.
- MaleCNS v1.0 (`pIP10` courtship song DNs already mapped) directly unlocks future Phase F11 (male ↔ female brain-to-brain courtship communication).
