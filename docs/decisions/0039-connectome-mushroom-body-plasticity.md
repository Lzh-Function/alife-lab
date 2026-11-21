# ADR 0039: Connectome Mushroom Body Dopamine-Modulated Plasticity (F4)

- Status: Accepted
- Date: 2026-11-21
- Related: ADR 0035 (connectome species-pack), ADR 0036 (bit-identical CPU/CUDA LIF), ADR 0037 (F2 browser observation), ADR 0038 (F3 screen & input search), `docs/plans/connectome_plan.md`

## Context

Phase F4 adds *Drosophila* Mushroom Body olfactory associative learning (Pavlovian conditioning) directly to the connectome substrate, while strictly obeying two scientific boundaries from `AGENTS.md`:

1. **No external reward function in the substrate:** learning must arise solely from the connectome's own dopaminergic reinforcement neurons (DANs) modulating **only** Mushroom Body `KC -> MBON` excitatory synapses (62,261 of 15,091,983 edges in FlyWire v783); all other 15,029,722 edges remain strictly fixed.
2. **Honest scientific framing:** the plasticity update is an **explicitly simplified phenomenological three-factor eligibility-trace rule** grounded in *Drosophila* literature (Hige et al. 2015; Cohn et al. 2015; Handler et al. 2019; Jiang & Litwin-Kumar 2021) — not a detailed cAMP/PKA/cGMP biochemical model.
3. **Preserved reference mode & bit-identical determinism:** plasticity is disabled by default (`plasticity = None`), forbidden in `mode="reference"` (Shiu et al. 2024 static wiring remains intact and spec digests unchanged), and produces **bitwise-identical** trajectories on CPU (`numpy`) and GPU (`cupy` with CUDA graphs) as well as across checkpoint save/restore.

## Decision

### 1. Simplified three-factor `KC -> MBON` plasticity (`src/alife_lab/connectome/plasticity.py`)

At every 1 ms tick (`dt_tick_ms = 1.0`), using deterministic `float32`:

1. **Presynaptic KC eligibility trace** `e_i` (decay `tau_elig_ms = 200 ms`): incremented by `1` on each spike of an unsilenced KC `i`.
2. **Compartmental DAN reinforcement trace** `d_j` (decay `tau_dan_ms = 150 ms`): driven by anatomical `DAN -> MBON` synapses (`>= min_dan_synapses = 5`) from unsilenced DANs, normalized by total anatomical DAN synapse count onto MBON `j` so `d_j` is on a consistent physical scale (weighted mean compartmental DAN spike count over `tau_dan_ms`) across both synthetic and full FlyWire circuits.
3. **Phasic burst threshold `dan_threshold` (default `18.0`):** only phasic dopamine bursts above baseline `d_eff_j = max(0, d_j - dan_threshold)` drive saturating three-factor LTD on eligible `KC_i -> MBON_j` multipliers `m_e in [w_min, w_max]`:
   `m_e <- clip(m_e - lr_ltd * e_i * d_eff_j * max(m_e - w_min, 0), w_min, w_max)`.
   This threshold was motivated directly by two measured FlyWire v783 anatomical findings:
   - Unpunished Odor A KCs drive recurrent `KC -> PPL101` spikes (`mbon11_dan <= 10.08`).
   - Depressing GABAergic `MBON11` weakens its 358-synapse direct inhibitory projection onto `PPL101` (`-95, -102, -81, -80`), disinhibiting `PPL101` during post-training Odor A alone up to `mbon11_dan = 17.11` (a real positive-feedback memory-maintenance loop, cf. Felsenberg et al. 2018), whereas external phasic punishment bursts (`220 Hz`) reach `~27–30`.
4. **Unreinforced extinction:** when KC `i` spikes while `d_j <= dan_threshold`, depressed multipliers (`m_e < 1.0`) recover exponentially toward `1.0` at rate `lr_ext = 0.03`.
5. **Deterministic integer projection & zero-rebuild GPU updates:** updated multipliers are projected via `rint(base_weight * scale).astype(int32)` back into the per-world CSR `signed_weight` array and patched in place into `CupyLIFRunner.d_weight` via `update_edge_weights()`. Because CSR structure (`indptr`, `post`) never changes and GPU device pointers stay fixed, captured CUDA graphs remain valid without re-capture while keeping CPU and GPU **100% bit-identical**.

### 2. Sparse KC odor ports and offline synthetic MB circuit

- In the uniform-weight Shiu et al. (2024) LIF model (which lacks APL divisive inhibition), stimulating distinct olfactory receptor types (`ORN_DM1` vs `ORN_DL5` even at 10 Hz) recruits ~3,790 / 5,177 KCs with Jaccard overlap `0.998`. To isolate Mushroom Body associative plasticity cleanly without altering the reference LIF dynamics, `flywire_783` exposes two disjoint 250-cell sparse `KCg-m` odor ensembles (`odor_a_kc`, `odor_b_kc`, 4.8% biological sparsity), alongside `ppl1_punishment_dan` (`PPL101`, 2 bilateral cells), `mbon11_approach` (`MBON11`, 2 cells), and `mbon01_avoidance` (`MBON01`, 2 cells).
- For fast offline CPU-only CI and interactive browser learning, `make_synthetic_mb_pack()` (`dataset_id="synthetic_mb"`) provides an 8-neuron Mushroom Body microcircuit (2 Odor A `KCg-m`, 2 Odor B `KCg-m`, GABAergic `MBON11`, downstream avoidance `MBON01` inhibited by `MBON11`, `PPL101`, and `PAM01`).

### 3. Protocol runner, CLI, API & UI

- `src/alife_lab/connectome/learning.py` (`run_pavlovian_experiment`) and CLI `alife connectome pavlov <dataset_id>` execute the full multi-seed protocol (Pre -> 1-trial -> N-trials -> Extinction + 1500 ms shifted-timing Unpaired control + `mushroom_body_dan`-silenced control + optional `shuffled_control` wiring) with exact permutation tests and Cliff's delta.
- `ConnectomeAdapter` persists `plastic_scale`, `plastic_kc_elig`, `plastic_mbon_dan`, and `plastic_last_eff_weight` in checkpoints (backward-compatible with non-plastic checkpoints), reports `mbon11_rate_hz`, `mbon01_rate_hz`, `plastic_changed_edges`, `plastic_mean_scale`, `plastic_min_scale`, and streams the top changed `KC -> MBON` edges in preview frames for `ConnectomePanel`.
- Browser presets (`connectome_mb_tiny_pavlov`, `connectome_fly_mb_pavlov`, `connectome_fly_mb_no_dan_control`) combine with Timeline replay and paired-fork so users can compare pre- vs post-training brains side by side.

## Consequences

- Both `synthetic_mb` (offline CPU) and real 138,639-neuron `flywire_783` (GPU CuPy, 5 seeds in 35.2 s) demonstrate all four Pavlovian criteria (acquisition `p = 0.0079`, odor specificity `p = 0.0079`, multi-trial dose-response, extinction recovery `p = 0.0079`) and all three negative controls (unpaired timing `p = 0.0476`, DAN-silenced `0` changed edges `p = 0.0079`, shuffled wiring `0.00 Hz`).
- One honest instability is recorded: in 1 of 5 seeds on FlyWire v783 during the unpaired `PPL101` pulse, stochastic whole-brain recurrence transiently recruited ~27k KCs in the absence of APL inhibition.
