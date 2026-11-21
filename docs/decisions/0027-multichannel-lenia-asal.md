# ADR 0027 — Multichannel Lenia substrate and ASAL on it

Status: accepted (2026-11-21)

## Context
ASAL (ADR 0026) on single-channel Lenia shown as grey images gave CLIP scores barely above an
empty world and no prompt specificity. The ASAL paper's Lenia is multichannel (3 channels, 15
kernels, creature 5N7KKM) and is shown to CLIP as RGB.

## Decision
- New substrate `lenia_mc` with the update of SakanaAI/asal `lenia_impl.py` (Leniabreeder,
  MIT), NumPy rFFT float64. Upstream was run once with the owner's permission; agreement and
  a CPU-only fixture are documented in `docs/reference_fidelity/lenia_mc.md`. Chan's own
  normalised variant (`LeniaNDKC.py`) differs and is documented, not implemented.
- Omitted `dynamics` are filled from the pattern before validation, so the canonical spec and
  its hash contain every kernel value.
- Search targets may be `lenia` or `lenia_mc` (discriminated union). Kernel parameters are
  searchable as `dynamics.kernels.<i>.<m|s|h|r>`; a `logit` scale reproduces ASAL's
  `sigmoid(logit(base) + delta)` parameterisation; up to 64 parameters.
- `asal.crop` (omitted when absent, digest-stable) shows CLIP a square centred on the
  pattern's circular centroid (ASAL: 64 × 64 "phenotype"); multichannel frames are RGB.
- Inside searches with torch backends, multichannel candidates are rolled out in a batched
  PyTorch float64 implementation on the search device (tested against NumPy to 1e-9).
- The dispatcher now enforces the profile's `max_active_gpu_jobs` (declared since Phase 3 but
  not enforced; four GPU sweep members had run at once).

## Consequences
- ASAL on `lenia_mc` gives prompt-specific CLIP scores and CMA-ES beats random search
  (sweep `search_asal_lenia_mc_cma_vs_random`, p = 0.024); single-channel results stay
  documented as a negative finding.
- Remaining deviations from ASAL: full CMA-ES instead of Sep-CMA-ES, initial cells not
  searched, 40 generations instead of 1000, centring by circular centroid.
