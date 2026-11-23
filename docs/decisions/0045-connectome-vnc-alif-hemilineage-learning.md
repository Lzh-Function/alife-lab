# ADR 0045: Phase F8 Level 2 Deepening — Bit-Exact ALIF and Fixed-Topology Per-Hemilineage VNC Learning

- **Status:** Accepted
- **Date:** 2026-11-23
- **Related:** ADR 0043 (Embodied Fly Level 1), ADR 0044 (Whole-CNS Level 2), `docs/reference_fidelity/connectome_fly.md`, `docs/plans/connectome_plan.md`

## Context

In ADR 0044, zero-shot uniform-weight LIF on real adult *Drosophila* whole-CNS connectomes (`banc_888`, 156,808 cells; `male_cns_v1`, 165,122 cells) achieved faster-than-real-time direct 36-muscle leg actuation (`1.52×–2.18× RT`) and 17×–33× intact-vs-shuffled motor specificity, but produced only small leg twitches (`0.008–0.082 cm`) rather than sustained locomotion. Two root causes were identified:

1. **Physical mass/sign bug:** `density="350"` in cm-scale MuJoCo coordinates made the fly `12.212 kg` (preventing pure contact-friction walking even under an ideal tripod gait), and `femur_pitch` had an inverted lift sign.
2. **Missing intrinsic VNC biophysics:** Uniform-weight LIF lacks neuron-local spike-frequency adaptation (K+Ca channels), developmental hemilineage conductance diversity, motor-unit dendritic input resistance (size principle), and tonic subthreshold aminergic neuromodulation.

## Decisions

1. **Pure-MuJoCo contact-friction calibration (`fly_body.py`):**
   - Recalibrate body density (`density="14"`, `0.4885 kg` scaled mass), contact friction (`2.0 0.02 0.01`), joint PD gains (`kp=6.5/9.0/8.0`, `damping=0.12/0.32/0.25`), and `femur_pitch` lift sign (`-femur_pitch` lifts UP).
   - Keep **ZERO** `d.qvel` velocity injection in Level 2 (`dead_brain = -0.035 cm` over 1.5 s; ideal 8 Hz tripod = `+1.205 cm/s` on pure MuJoCo contact friction).

2. **Bit-exact CPU/GPU Adaptive LIF (`lif.py`, `lif_cupy.py`):**
   - Add optional per-neuron adaptation state `adapt`, tonic resting bias `bias_mv`, adaptation increment `adapt_inc_mv`, `tau_adapt_ms`, and fixed-point `weight_quant_div = 16.0` (`int32` scaled synaptic weights).
   - Non-ALIF runs remain zero-overhead (`adapt is None`); ALIF runs use explicit non-FMA IEEE-754 `float32` CUDA ops (`__fadd_rn`, `__fsub_rn`, `__fmul_rn`) and preserve exact bitwise CPU == GPU equivalence and exact checkpoint/replay determinism (serializing `adapt` in `FlyEmbodiedAdapter`).
   - Support in-place GPU weight/parameter updates (`CupyLIFRunner.upload_alif_weights_and_params()`) so CMA-ES rollouts reuse a single resident CUDA allocation without reallocating VRAM.

3. **Fixed-topology 16-hemilineage biophysical parameterization (`vnc_learn.py`):**
   - Map BANC v888 (female) and MaleCNS v1.0 (male) onto **16 identical developmental hemilineage & functional groups** (`hl_20A_22A`, `hl_03A`, `hl_19A`, `hl_04B`, `hl_13A`, `hl_12B`, `hl_21A`, `hl_16B`, `hl_13B`, `hl_08A_09A`, `hl_other_exc`, `hl_other_inh`, `descending`, `proprioceptive`, `leg_motor_swing`, `leg_motor_stance`).
   - Strictly freeze biological topology and Dale's-law signs (`max(1, round(|w| * scale * 16))` preserving exact edge mask and sign).
   - Keep `group_bias_mv` strictly subthreshold (`[-1.5, +2.4] mV` vs `7.0 mV` spike threshold) so zero external input yields exact `0` spikes.
   - Apply `leg_motor_swing` / `leg_motor_stance` scales postsynaptically (motor-unit input resistance) and interneuron/DN/proprio scales presynaptically.
   - Run CMA-ES on the exact closed VNC+DN+MN+proprio subcircuit (`26,182` cells BANC / `21,703` cells MaleCNS) and always verify on the **full whole-CNS connectome** (`156,808` / `165,122` cells) alongside **matched-budget CMA-ES trained from scratch on degree-preserving shuffled VNC**, transferred-to-shuffled, open-loop (`proprio=0`), `DNp09`-silenced, and cross-sex zero-shot transfer controls.

4. **Safe profile persistence (`configs/vnc_profiles/*.json`, `spec.py`, `embodied_adapter.py`):**
   - Store learned 37-float profiles as compact, human-readable JSON (no pickles) under `configs/vnc_profiles/`.
   - Validate `FlyEmbodiedDynamics.vnc_profile` strictly via basename regex (`^[a-z0-9][a-z0-9_-]{1,63}$`) both in Pydantic and in `resolve_vnc_profile_path()` to prevent path traversal.

## Consequences & Recorded Negative Findings

- **Positive:** Real BANC v888 (`+1.407 cm`, `yaw = -0.019 rad` = `1.1°`, `74.6 Hz`, `1.25× RT`) and MaleCNS v1.0 (`+0.897 cm`, `28.7 Hz`, `1.75× RT`) now walk forward on 100% pure MuJoCo contact friction with zero CPG, zero velocity injection, 8.6×–25× stronger motor recruitment and 37×–44× straighter heading than shuffled controls, and zero-shot cross-sex transfer (`MaleCNS -> BANC = +0.955 cm`, `1.4°` straight).
- **Honest limitations recorded:** (1) True anti-phase tripod gait (`tripod_corr < -0.3`) did not emerge from 16-hemilineage ALIF alone (BANC produced synchronous bursting `tripod_corr = +0.957`; MaleCNS `+0.026`). (2) Unconstrained postsynaptic `leg_motor_stance / leg_motor_swing` ratio (`~29×`) allows even a shuffled VNC firing at `3–9 Hz` to ratchet forward (`+1.13–1.32 cm`) while veering `41°–63°` sideways, underscoring why matched-budget shuffled-from-scratch controls are mandatory. (3) Resting tarsus ground-contact proprioception still drives the VNC when `DNp09` alone is silenced unless proprioceptive gain is descending-gated.
