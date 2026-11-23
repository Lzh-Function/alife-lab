# ADR 0042: Connectome Fitting to Real Public *Drosophila* Activity Recordings (F7)

- Status: Accepted
- Date: 2026-11-23
- Related: ADR 0035 (species-pack), ADR 0040 (F5 hidden-truth sysid), ADR 0041 (F6 fixed-topology task learning), `docs/plans/connectome_plan.md`

## Context

Phase F7 (`docs/plans/connectome_plan.md` §4 F7) extends the F5 system-identification pipeline from
synthetic hidden-truth circuits to **real, publicly available *in vivo* *Drosophila* neural
activity recordings** mapped directly onto the FlyWire v783 adult brain connectome.

Mandatory engineering and scientific constraints:

1. **Zero pickles, SHA-256 verified raw data outside Git:** External datasets live under
   `.alife/datasets/f7_real_activity/raw/` (gitignored), verified by SHA-256 (`verify_f7_file`),
   and loaded exclusively via standard CSV and `np.load(..., allow_pickle=False)`.
2. **Strict held-out generalization (no teacher forcing):** Evaluate exclusively on held-out
   future time segments, completely unseen held-out flies, held-out unobserved brain regions, and
   held-out unseen odorants — never reporting training-set fits as generalization.
3. **Mandatory negative controls & honest reporting:** Compare real FlyWire v783 anatomical
   connectivity (`reference`) against row- and sign-preserving `shuffled_control` connectivity and
   an `unconnected` behavioral-only baseline, documenting both positive matches and null/negative
   results honestly.
4. **CPU-only CI & API-process separation:** PyTorch is imported lazily inside `real_fit.py`; pure
   NumPy anatomical mapping, neuropil aggregation, shuffling, and hash verification run in CPU-only
   CI without `torch` or downloaded real datasets.

## Decision

### 1. Public real-animal datasets selected (user-approved)

1. **Aimon et al. (2023, *eLife* 85202; Zenodo `7849014`):** `Walk_anatomical_regions.zip`
   (176.09 MiB, pure `.npy` arrays). Whole-brain light-field calcium imaging across 75 JRC2018
   anatomical brain regions ($\Delta F/F$) with synchronized `Walk`, `Left`, and `Right`
   behavioral regressors in head-fixed walking adult *Drosophila* (29 pan-neuronal `Nsyb` + 5
   `GMR57C10` flies).
   - **FlyWire v783 anatomical bridge:** 74 of 75 JRC2018 regions map directly onto FlyWire v783
     neuropils (`MBPED/MBVL/MBML/MBCA -> MB_*`, `IVLP_L/R -> WED_L/R`; `EPA_R` has no assigned
     cell in FlyWire v783), aggregating **22,086,546** inter-neuropil excitatory and inhibitory
     synapses into a row-normalized `(75, 75)` E/I connectome matrix (`NeuropilConnectome`).
   - Note: Two small `.npy` files from the companion GitHub repository (`FuncRPANTrials.npy`,
     `CompTrialsAllFreq.npy`) were inspected and discarded before fitting because they contain
     integer trial counts (`0..42`), not fluorescence time series.
2. **DoOR 2.0 (Münch & Galizia 2016, *Scientific Reports* 6:21841) + FlyWire mapping:**
   `door_or_merged.csv` + `door_to_flywire_mapping.csv`. 692 real odorants across 52 receptors,
   of which 41 map 1:1 onto **37 unique FlyWire v783 `ORN_<glomerulus>` cell types**. Coupled via
   FlyWire's 2-hop antennal-lobe inter-glomerular network (`ORN -> ALLN/ALPN/ALIN -> ORN`).

### 2. Implementation (`src/alife_lab/connectome/real_fit.py` & `alife connectome fit-real`)

- `build_neuropil_connectome()` / `shuffle_neuropil_connectome()` aggregate and control-shuffle
  FlyWire v783 inter-neuropil E/I matrices (log-compressed row normalization so large optic-lobe
  neuropils do not drown central brain regions).
- `run_aimon2023_fit()` splits both **flies** (70% train flies, 30% completely unseen test flies)
  and **time** (first 70% train, last 30% test, uniform 5 Hz), evaluating:
  1. Within-fly temporal holdout open-loop rollout (zero teacher forcing).
  2. Cross-fly zero-shot open-loop rollout on unseen flies.
  3. Spatial unobserved-region prediction on unseen flies (observing 50% of brain regions to
     predict the remaining 50% unrecorded regions through fixed FlyWire E/I wiring).
- `run_door2_fit()` holds out 25% of odorants (173 unseen real odorants) and 50% of glomeruli per
  seed, testing whether FlyWire's 2-hop antennal-lobe coupling predicts masked peripheral ORN
  responses better than shuffled coupling.
- Exposed via `uv run alife connectome fit-real {aimon2023,door2,synthetic}` and tested offline in
  `tests/reference/test_connectome_f7.py`.

## Consequences

- **Aimon et al. (2023) spatial unobserved-region prediction (10 pan-neuronal flies, 5 seeds):**
  FlyWire `reference` wiring predicts unrecorded brain regions in completely unseen flies at
  $R^2 = +0.5415 \pm 0.0345$ ($r = +0.6969 \pm 0.0235$), beating `unconnected` behavioral-only
  prediction ($R^2 = -1.2823$, $p = 0.0079$, $\delta = +1.00$) and beating matched
  `shuffled_control` in **5/5 seeds** ($\Delta R^2 = +0.0422$, 95% CI $[+0.0040, +0.0819]$,
  $\delta = +0.60$, unpaired exact $p = 0.1111$). Dense global walk-state covariance explains why
  even shuffled regional connectivity achieves $R^2 \approx 0.50$.
- **Aimon et al. (2023) pure open-loop rollout from 3 behavioral regressors:** Raw $R^2$ is
  negative across all arms (`-0.298` temporal, `-1.606` cross-fly for `reference`) due to
  unmodeled photobleaching/baseline drift and spontaneous internal brain states; on scale- and
  drift-invariant cross-fly Pearson correlation, `reference` ($r = +0.4629$) beats
  `shuffled_control` ($r = +0.3449 \pm 0.0550$) in 5/5 seeds, while `unconnected` avoids recurrent
  drift amplification ($r = +0.5214$, $R^2 = -0.7328$).
- **DoOR 2.0 peripheral odor prediction (692 odors × 37 glomeruli, 5 seeds):** Clean null result
  between `reference` ($R^2 = +0.1746$, $r = +0.4297$) and `shuffled_control` ($R^2 = +0.1753$,
  $r = +0.4298$, $p = 0.9444$), consistent with biology: DoOR 2.0 measures *peripheral* receptor
  chemical tuning prior to synaptic antennal-lobe processing, whereas FlyWire's 2-hop `ALLN`
  network performs broad all-to-all gain normalization rather than encoding odorant chemistry.
