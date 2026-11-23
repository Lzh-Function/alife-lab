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
