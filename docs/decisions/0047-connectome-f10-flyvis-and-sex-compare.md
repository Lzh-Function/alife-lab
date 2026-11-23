# ADR 0047: Phase F10 — Connectome-Constrained Fly Vision (`flyvis`) and Cross-Sex Connectome Comparison

- **Status:** Accepted
- **Date:** 2026-11-23
- **Related:** ADR 0043, ADR 0044, `docs/reference_fidelity/connectome_fly.md`, `docs/plans/connectome_plan.md`

## Context

Phase F10 (`connectome_plan.md §4 F10`) covers two orthogonal capabilities:
1. **Optic-lobe visual system (`flyvis`):** connecting TuragaLab's connectome-constrained 45,669-neuron, 65-cell-type *Drosophila* visual network (Lappalainen et al. 2024, *Nature*, MIT) to verify canonical motion and looming responses and provide a visual-to-descending bridge for embodied flies.
2. **Cross-sex comparative connectomics:** comparing anatomical cell-type homology, sexually dimorphic circuits, and identical-stimulus spiking physiology across female (`flywire_783`, `banc_888`) and male (`male_cns_v1`) whole-brain / whole-CNS connectomes.

## Decisions

1. **Optional `vision` extra + isolated weight storage (`pyproject.toml`, `vision_flyvis.py`):**
   - Add `flyvis>=1.2` under `[project.optional-dependencies] vision` so CPU-only CI remains zero-overhead and offline-runnable via a pure-NumPy Hassenstein-Reichardt fallback (`run_synthetic_visual_assay`).
   - Store downloaded pretrained `flyvis` weights strictly inside `.alife/datasets/flyvis/` (gitignored, 10 MiB, checksum-verified) via `FLYVIS_ROOT_DIR`.
   - Expose `visual_to_descending_hz(optomotor_opponent, looming_contrast)` to map optic-lobe outputs to bounded `DNa01/02` steering and `MDN` escape rates.

2. **Cross-sex comparative assay (`sex_compare.py`):**
   - Map exact cell-type overlap across `flywire_783`, `banc_888`, and `male_cns_v1`, explicitly tracking dimorphic courtship/song/receptivity anchors (`pIP10`, `pMP2`, `dPR1`, `vPR6`, `vMS11` vs `pC1a–e` and `JO-B` auditory neurons).
   - Run identical-stimulus LIF trials (`sugar_150hz`, `sugar_plus_bitter`, `antennal_poke_180hz`, `forward_dn_150hz`) on both `reference` and degree-preserving `shuffled_control` across all datasets, reporting honest cross-dataset differences (e.g., MaleCNS taste-GRN annotation coverage vs FlyWire/BANC).

3. **CLI integration (`alife connectome vision`, `alife connectome compare-sex`):**
   - Provide human-readable and `--json` outputs for both assays.

## Measured Results (RTX 5070, 2026-11-23)

- **`flyvis 1.2.0` (`model_id="000"`, `45,669` cells, `65` cell types, `cuda:0`, `7.87 s`):**
  - Leftward vs rightward drifting gratings reverse horizontal T4/T5 opponency cleanly (`+0.2061` vs `-0.1836`, baseline-corrected `+0.1472` vs `-0.2425`).
  - Expanding dark looming shadow elicits `3.67×` stronger T4+T5 temporal modulation (`0.3221`) than a static dark disc (`0.0877`), while static uniform gray stays quiescent (`0.0045`).
- **Cross-sex comparison (`flywire_783` + `banc_888` + `male_cns_v1`, `460,569` cells total, `9.92 s`):**
  - Shared cell types: `4,419` across all 3; `7,560` FlyWire×BANC; `7,445` BANC×MaleCNS.
  - Dimorphism verified verbatim: male-only `pIP10 (2)`, `pMP2 (2)`, `dPR1 (2)`, `vPR6 (8)`, `vMS11 (14)` vs female-only `pC1a–e (10)`.
  - Identical-stimulus physiology: `antennal_poke_180hz -> DNg84` fires strongly in all 3 (`66.7 / 75.6 / 273.3 Hz` intact vs `0.0 Hz` shuffled); `sugar -> MN9` with bitter suppression reproduces in FlyWire (`71.7 -> 0.0 Hz`, `100%`) and BANC (`10.0 -> 1.7 Hz`, `83.3%`), whereas MaleCNS `sugar_grn` remains `0.0 Hz` (recorded honestly as an annotation/wiring coverage difference).
