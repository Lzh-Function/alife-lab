# ADR 0048: Phase F11 — Two-Fly Innate Communication (Male Courtship Song ↔ Female Auditory & Pursuit)

- **Status:** Accepted
- **Date:** 2026-11-23
- **Related:** ADR 0044, ADR 0047, `docs/reference_fidelity/connectome_fly.md`, `docs/plans/connectome_plan.md`

## Context

Phase F11 (`connectome_plan.md §4 F11`) tests how far innate two-fly social communication can be reproduced directly from frozen biological male (`male_cns_v1`, 165,122 cells) and female (`banc_888`, 156,808 cells) whole-CNS connectomes — with zero learning and zero hand-crafted song oscillators — across Stage ① (brain-to-brain song dialogue + Janelia song-circuit silencing screen) and Stage ② (two-fly shared arena with vision, short-range cuticular pheromone, near-field acoustics, and tactile contact).

## Decisions

1. **Dual whole-CNS lockstep runtime (`dyadic_fly.py`):**
   - Step two independent whole-CNS connectomes concurrently every `1.0 ms` (`male_cns_v1` 165,122 cells + `banc_888` 156,808 cells = `321,930` neurons on GPU via `CupyLIFRunner.step_tick(use_graph=True)`, with exact NumPy CPU fallback for offline CI).
   - Provide offline 82-cell / 76-cell synthetic male/female courtship fixtures (`make_synthetic_male_song_pack`, `make_synthetic_female_auditory_pack`) mirroring the exact `pC1 -> pIP10 -> {dPR1 -> ps1 MN, vPR9/TN1a -> hg1 MN}` and `JO-B -> SAD052/53/CB1078/AVLP082 -> pC1a` pathways for deterministic CPU CI.

2. **Physical near-field acoustic & multi-channel coupling:**
   - Male wing song motor neurons (`ps1 MN` pulse song + `hg1 MN` sine song) radiate particle-velocity sound with short-range cubic attenuation `1 / (1 + (d / 0.55 cm)^3)` into female `JO-B` (416 auditory Johnston's Organ receptors).
   - Stage ② adds bilateral visual bearing pursuit (`DNa01/02`, `DNp09`) and short-range female cuticular pheromone driving male `pC1_*` courtship command neurons, plus song-evoked female locomotor slowing proportional to female auditory relay activity (`SAD052/53`, `CB1078`, `AVLP082/194_a`).

3. **Mandatory ablations & negative controls (`connectome_plan.md §4 F11`):**
   - Stage ① verifies `intact_close`, `acoustic_blocked`, `far_distance_5cm`, `male_shuffled`, `female_shuffled`, and a **10-cell-type Janelia song-circuit silencing screen** (`pIP10`, `dPR1`, `vPR9_a/b/c`, `TN1a_g/i/d`, `ps1 MN`, `hg1 MN`).
   - Stage ② verifies `intact_pair`, `acoustic_blocked`, `vision_blocked`, and `male_shuffled`.

4. **CLI integration (`alife connectome social`):**
   - Supports `--stage dialogue|arena`, `--male`, `--female`, `--ticks`, `--backend`, `--no-ablations`, and `--json`.

## Measured Results (RTX 5070, 321,930 Concurrent Biological Neurons, 2026-11-23)

- **Stage ① (`18.38 s`, 5/5 gates PASS):**
  - Male `pC1_*` drive recruits `pIP10 = 202.5 Hz`, `dPR1 = 146.25 Hz`, `ps1 MN = 145.0 Hz`, `hg1 MN = 31.25 Hz`, evoking `65.1 Hz` in female `banc_888` auditory relays (`0.0 Hz` when muted or at `5 cm`; `1.48 Hz` on `male_shuffled`; `0.10 Hz` on `female_shuffled`).
  - **Emergent Janelia (Lillvis et al. 2024) silencing phenotypes reproduced zero-shot:** silencing `pIP10` (2 cells) drops pulse song `+75.9%` and female hearing `+78.0%`; silencing `dPR1` (2 cells) drops pulse `+44.8%` and sine `+76.0%`; silencing GABAergic **`vPR9_a` (4 cells) / `vPR9_c` (3 cells)** strongly **disinhibits sine song `hg1 MN` by `+316%` / `+388%`** (`31.25 -> 130.0 / 152.5 Hz`).
- **Stage ② (`9.48 s`, 5/5 gates PASS):**
  - Male chases female from `1.60 cm` to `0.25 cm` and sings `1,153 ms` (vs `48 ms` on `male_shuffled` and `0 ms` on `vision_blocked`); female hears `81.18 Hz` and slows (`1.115 cm` vs `1.380 cm` when `acoustic_blocked`).
- **Honest limitation:** Female third-order receptivity neurons `pC1a–e` remain `0.0 Hz` under uniform resting LIF without internal mating-state/neuromodulatory arousal, matching `connectome_plan.md §4 F11`.

## Stage ③ Addendum: Collective Whole-Brain Fly Swarm (`fly_swarm`, 2026-11-23)

1. **Shared-arena batched whole-brain runtime (`fly_swarm.py` + `swarm_adapter.py`):**
   - Runs `N = 1..64` independent whole-brain *Drosophila* connectomes (`flywire_783`, `banc_888`, `male_cns_v1`, or offline `synthetic_fly_embodied`) simultaneously inside **one** shared 2D/3D petri-dish arena using a single shared CSR graph in GPU VRAM (`CupyBatchedLIFRunner` with CUDA Graph substeps, exact NumPy fallback on CPU).
   - Implements pairwise steric separation, **Ramdya et al. (2015, *Nature*)** leg/antennal mechanosensory touch arousal routed through each fly's real connectome (`antennal_grooming_mechanosensory -> DNg84`), bilateral aversive odor avoidance (`x < 0` aversive vs `x > 0` clean air), and an optional `mixed_courtship_swarm` mode.
2. **Full browser vertical slice (`FlySwarmView.tsx` + `fly_swarm` presets):**
   - Streams compact `fly_swarm_v1` binary frames (`N × 8 × float32` = `768 bytes` at `N = 24`) to a 60 FPS Three.js 3D Multi-Fly Swarm Renderer with snapshot interpolation, animated 6-legged rigs, aversive odor zone overlay, cyan leg-touch contact sparks, golden courtship song rings, live interventions (`toggle_odor`, `toggle_touch`, `scatter`, `wake_all`), and exact deterministic checkpoint/restore.
   - Adds CLI command `alife connectome swarm [dataset_id] --sizes 1,8,16,24 --ticks 2000` and 4 browser presets (`fly_swarm_tiny`, `fly_swarm_flywire_24`, `fly_swarm_flywire_no_touch`, `fly_swarm_courtship_16`).
3. **Measured RTX 5070 Results (`flywire_783`, `24 × 138,639 = 3,327,336` concurrent biological neurons, `40.21 s`, **4/4 gates PASS**, measured as conservative lower bound under concurrent desktop GPU load):**
   - `N = 1` (8 trials): `12.5%` clean (`0.00` enc/fly·s, `1.95× RT`) → `N = 8`: `0.0%` (`0.06` enc/fly·s, `1.41× RT`) → **`N = 16`** (`2.22M` neurons): **`43.8%`** (`0.53` enc/fly·s, **`1.20× RT`**) → **`N = 24`** (`3.33M` neurons): **`45.8%`** (`0.81` enc/fly·s, **`0.55× RT`**).
   - **Negative controls at `N = 24`:** `touch_silenced` drops 11× to **`4.2%`** (`0.00` enc/fly·s); `shuffled_control` drops 3.7× to **`12.5%`** despite physical encounters (`0.62` enc/fly·s), confirming connectome-mediated `DNg84` mechanosensory arousal; `no_odor_control` remains near chance (**`45.8%`** final, **`48.5%`** mean).
4. **Honest limitations:** Stage ③ uses planar kinematic bodies with animated 3D legs (not 24 simultaneous 6-legged MuJoCo contact solvers) so `N = 16–24` full biological brains run interactively in the browser; spontaneous walking/resting bout transitions are modeled phenomenologically on top of connectome `DNp09`/`DNa01-02`/`DNg84` firing rates.
