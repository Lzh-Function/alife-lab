# Stage 4 Reference Fidelity & Empirical Report: S2–S4 Species Packs, Cross-Species Comparison, and F13a–c Chimeras

- **Date:** 2026-11-27
- **ADR:** [ADR 0053](../decisions/0053-species-packs-s2-s4-and-f13-chimeras.md)
- **Implementation:** [`species_packs.py`](../../src/alife_lab/connectome/species_packs.py), [`cross_species.py`](../../src/alife_lab/connectome/cross_species.py), [`chimera.py`](../../src/alife_lab/connectome/chimera.py), [`graded.py`](../../src/alife_lab/connectome/graded.py)
- **Evidence Levels:** S2–S4 structural packs **E2** (exact checksum + reference count/motif verification against published tables); Cross-species physiological probes & F13a–c chimeras **E2–E3 exploratory** (multi-arm paired controls and causal ablations; chimeras are explicitly non-biological artificial-life hybrids).

---

## 1. Empirical Dataset Provenance & Reference Checks (S2–S4)

All raw datasets remain outside Git under `.alife/datasets/<dataset_id>/raw/` and are verified via SHA-256 on load (zero pickles):

| Dataset | Archive / Files | Total Size | Primary SHA-256 (prefix) | Reference Checks |
|---|---|---:|---|---|
| `fly_larva_l1` | `fly_larva.csv.zip` (Winding et al. 2023 *Science*) | 508,166 B | `49f87128185feb1f…` | **PASS** — 2,956 neurons, 110,677 collapsed edges, 352,611 exact synapses, ORN/PN/KC/MBON/MBIN/DN-VNC ports |
| `platynereis_3d` | 7 CSV/TXT files pinned to JekelyLab commit `ab750290…` (Verasztó et al. 2024/2025 *eLife*) | 13,142,784 B | `full_connectome_adjacency_matrix.csv` verified | **PASS** — 1,720 connected cells (matrix subset of 9,162 EM somatic cells), 8,451 edges, 19,991 synapses, PRC/IN1/ciliomotor/muscle ports |
| `ciona_larva` | `cintestinalis.csv.zip` (Ryan, Lu & Meinertzhagen 2016 *eLife*) | 14,664 B | `d036aeba31e94dee…` | **PASS** — 205 cells, 2,903 edges, 13,295 units (0.1 µm synaptic depth), photoreceptor/relay/ACIN/tail-MN ports |

Offline CPU CI counterparts (`synthetic_fly_larva`, `synthetic_platynereis`, `synthetic_ciona`, `synthetic_chimera_worm_fly_mb`) also pass 100% of reference and adapter checkpoint round-trip checks without network access.

---

## 2. Five-Species Cross-Species Structural & Sensorimotor Comparison

Measured on CPU (`seed=1`, `ticks=120` for S1–S4, `ticks=50` for `flywire_783`, total wall time `3.90 s`):

### 2.1 Structural Topology Across ~700× Scale

| Dataset | Cells | Directed Edges | Total Synapses / Units | Density | Mean Out-Deg (±SD) | Max In / Out | Top-1% Hub Share | Reciprocity | Excit. Chem. Frac. | Gap Edge Frac. | Model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ciona_larva` | 205 | 2,903 | 13,295 | 0.06942 | 14.16 (±9.99) | 50 / 41 | 6.01% | 0.2091 | 90.04% | 0.00% | `spiking_lif` |
| `celegans_302` | 302 | 5,764 | 29,114 | 0.06341 | 19.09 (±12.50) | 107 / 91 | 7.81% | 0.6728 | 97.00% | 37.47% | `graded` |
| `platynereis_3d` | 1,720 | 8,451 | 19,991 | 0.00286 | 4.91 (±7.27) | 48 / 90 | 12.86% | 0.1580 | 84.40% | 0.00% | `graded` |
| `fly_larva_l1` | 2,956 | 110,677 | 352,611 | 0.01267 | 37.44 (±27.88) | 210 / 160 | 5.86% | 0.2606 | 88.79% | 0.00% | `spiking_lif` |
| `flywire_783` | 138,639 | 15,091,983 | 54,492,922 | 0.00079 | 108.86 (±163.20) | 10,356 / 9,783 | 14.95% | 0.2660 | 60.03% | 0.00% | `spiking_lif` |

### 2.2 Paired Sensorimotor Selectivity (`reference` vs degree-preserving `shuffled_control`)

| Dataset | Mode (`w_syn_mv`) | Sensory → Motor Ports | Reference Motor | Shuffled Motor | Wiring Delta | Selectivity Ratio | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| `ciona_larva` | `spiking_lif` (`1.45`) | `ciona_photoreceptors` (30) → `ciona_motor_mn` (10) | **104.17 Hz** | 69.17 Hz | +35.00 Hz | **1.506×** | Biological visual-motor pathway drives tail MNs 50.6% above degree-matched random wiring despite lower population background (85.0 vs 94.7 Hz). |
| `celegans_302` | `graded` (`0.275`) | `role_sensory` (84) → `role_motor` (130) | 28.87 ΔmV | 30.62 ΔmV | −1.75 ΔmV | 0.943× | Broad stimulation of all 84 sensory neurons saturates the dense, high-reciprocity (0.673) worm graph globally. Targeted touch/chemotaxis selectivity is validated separately in S1 (`connectome_s1_celegans.md`). |
| `platynereis_3d` | `graded` (`0.275`) | `platy_photoreceptor_prc` (24) → `platy_ciliomotor_mn` (106) | **22.41 ΔmV** | 15.18 ΔmV | +7.22 ΔmV | **1.476×** | Dedicated visual-ciliomotor circuit focuses drive onto effectors (+47.6%) while keeping whole-body background lower (9.65 vs 12.04 ΔmV). |
| `fly_larva_l1` | `spiking_lif` (`0.85`) | `larva_olfactory_sn` (42) → `larva_dn_vnc` (182) | 59.94 Hz | 80.68 Hz | −20.74 Hz | 0.743× | **Honest finding:** Larval local inhibitory circuits (`LN`, `MB-FBN`) dampen global excitation (`pop=58.71 Hz`). Degree-preserving shuffling destroys targeted inhibition, causing runaway excitation (`pop=78.02 Hz`) that inflates unselective motor firing. |
| `flywire_783` | `spiking_lif` (`0.275`) | `sugar_grn` (32) → `mn9_proboscis_extension` (2) | **60.00 Hz** | 0.00 Hz | +60.00 Hz | **>10⁵×** | Reproduces Shiu et al. (2024) sparse, highly specific feeding reflex (`pop=0.09 Hz`). |

---

## 3. Exploratory Chimera Assays (F13a–c)

> **Framing Notice:** All F13 chimeras are strictly `mode="exploratory"` artificial-life hybrid circuits. They test modular neural composition and interface principles in silico and do **not** claim to represent real biological organisms.

### 3.1 F13a — Intra-Species Cell-Type-Aligned Crossover

| Pair | Shared / Crossed Types | Provenance (Cells) | Edges Transferred | Mean Weight L1 Δ | Hybrid Pop (Ref vs Shuf) | Parent A Pop (Ref vs Shuf) | Wall Time |
|---|---:|---|---:|---:|---:|---:|---:|
| `male_cns_v1 × banc_888` (`blend=0.5`, 165,122 cells, 25.56M edges) | 7,445 / 3,664 | 67.97% `male_cns_v1` / 32.03% `banc_888` | 5,087,391 | 3.593 | 0.434 vs 0.263 Hz | 0.437 vs 0.284 Hz | 22.64 s |
| `celegans_302 × synthetic_celegans_302` (`blend=0.5`, 302 cells) | 302 / 154 | 49.01% `celegans_302` / 50.99% `synthetic` | 276 | 4.826 | 30.91 vs 31.68 ΔmV | 31.00 vs 31.74 ΔmV | 0.058 s |

### 3.2 F13b — Cross-Species Brain Transplant into 12-Segment RFT Body (`bridge_trials=8`, `eval_steps=80`)

| Donor Brain | Provenance (Brain / Bridge) | `donor_intact` Dist (Approach) | `donor_shuffled` Dist (Approach) | `donor_silenced` Dist | Intact − Shuffled | Wall Time | Honest Assessment |
|---|---|---:|---:|---:|---:|---:|---|
| `ciona_larva` | 98.64% / 1.36% | **0.2003 mm** (+0.1648 mm) | 0.0807 mm (+0.0505 mm) | 0.0000 mm | **+0.1196 mm (2.48×)** | 2.84 s | Biological visual-motor wiring clearly outperforms shuffled wiring under identical bridge optimization budget. |
| `platynereis_3d` | 98.49% / 1.51% | 0.4010 mm (+0.3395 mm) | 0.4103 mm (+0.3877 mm) | 0.0000 mm | −0.0093 mm | 2.31 s | Both intact and shuffled graded brains drive strong RFT propulsion; 4-parameter bridge absorbs wiring differences (silencing yields 0.0 mm). |
| `fly_larva_l1` | 99.80% / 0.20% | 0.0968 mm (+0.0873 mm) | 0.1312 mm (+0.1191 mm) | 0.0000 mm | −0.0344 mm | 6.48 s | Shuffled larva brain fires more aggressively (disrupted inhibitory gating), giving slightly larger scalar drive into the 4-parameter bridge; silencing yields 0.0 mm. |

### 3.3 F13c — Circuit Graft (`celegans_302` Host + `fly_larva_l1` Mushroom Body)

- **Unified hybrid pack:** `chimera_worm_fly_mb` — **370 cells** (`81.62%` *C. elegans* graded host / `18.38%` *Drosophila* larva spiking MB graft), **6,630 edges** (`86.94%` host / `10.59%` graft / `2.47%` explicit sensory/DAN/MBON bridge), wall time `1.29 s`.
- **Pavlovian odor-aversion conditioning (`train_trials=4`, `seed=1`):**

| Experimental Arm | Odor A Δ Avoid Bias (`AVA − AVB`) | Odor B Δ Avoid Bias | Odor Specificity (`ΔA − ΔB`) | Post Odor A MBON-Approach | Mean Plastic Scale | Result |
|---|---:|---:|---:|---:|---:|---|
| `paired_intact` (Odor A + ASH shock) | **+0.732 mV** | 0.000 mV | **+0.732 mV** | 41.67 → **0.00 Hz** | 0.5634 (min 0.08) | Selective learned avoidance |
| `unpaired_control` (Odor A & shock separated) | +0.166 mV | 0.000 mV | +0.166 mV | 41.67 → 41.67 Hz | 0.9589 | 4.4× weaker than paired |
| `graft_silenced` (graft MB silenced) | **0.000 mV** | 0.000 mV | **0.000 mV** | 0.00 → 0.00 Hz | 1.0000 | Learning completely abolished |
| `shuffled_graft` (KC→MBON shuffled) | +0.873 mV | +0.479 mV | +0.394 mV | 50.00 → 2.78 Hz | 0.3291 | Leaks to unpunished Odor B (−46% specificity) |

- **All 5/5 checks PASS** on both empirical (`celegans_302 + fly_larva_l1`) and offline synthetic (`synthetic_celegans_302 + synthetic_fly_larva`, `paired_intact ΔA=+12.21 mV`, `unpaired=0.00 mV`, `graft_silenced=0.00 mV`, `shuffled_graft specificity=+10.57 mV`) builds.

---

## 4. Measured Resource Usage

- Full 7-dataset local benchmark script (loading `ciona_larva`, `celegans_302`, `platynereis_3d`, `fly_larva_l1`, `flywire_783`, `male_cns_v1`, `banc_888`, running 5-species comparison + F13a adult-fly & worm hybrids + F13b × 3 species + F13c):
  - Dataset load + SHA-256 verification: `2.90 s`
  - Total benchmark wall time: `46.04 s` (of which `22.64 s` is the 165k-cell × 25.56M-edge `male_cns_v1 × banc_888` F13a crossover)
  - Peak RSS (all 7 datasets resident concurrently on CPU): `5,197.4 MiB`
- Small-species / F13c interactive workloads (`ciona_larva`, `platynereis_3d`, `fly_larva_l1`, `chimera_worm_fly_mb`) run comfortably in `< 2 s` and `< 250 MiB` RAM on CPU-only environments.
