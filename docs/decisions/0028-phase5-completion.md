# ADR 0028 — Completing Phase 5: prompt-specificity assay, novelty search, IMGEP

Status: accepted (2026-11-21)

## Context
Phase 5 asks for novelty search and curiosity-driven exploration next to ES/QD, ASAL's VLM
evaluation, and discovery evidence with assays. CLIP candidates could not yet be checked in
the ledger, and a CLIP score alone is not evidence (ADR 0026, 0027).

## Decision
- Claim `clip_prompt_specific` (assay plan `clip-1`, `alife_lab.evolution.assays_clip`):
  CPU re-run with CPU CLIP of the search's view; checks `reproduce` (≤ 2 % from the record),
  `above_blank` (≥ +0.02 over an empty world), `specificity` (≥ +0.03 over each of six fixed
  unrelated prompts), `longer` (2 × horizon still ≥ +0.02 over blank) and `baseline`
  (≥ +0.02 over the search's base creature). Margins and control prompts are constants of the
  plan version. It states the model's judgement, not what the pattern is; E2 at most.
- Algorithms `novelty` (kNN novelty with an archive, binary tournaments, Gaussian mutation)
  and `imgep` (random goals in the descriptor box, nearest known candidate mutated) in the
  search plugin, configured by an optional `explore` section (digest-stable when absent).
  Behaviour space: the `qd` descriptors scaled to [0, 1], or CLIP embeddings (novelty only),
  which makes novelty in CLIP space an illumination-style search. Own NumPy implementations
  (QDax/evosax not adopted, ADR 0023/0024).
- Metric `illumination` = ASAL illumination of 32 viable candidates chosen by greedy
  farthest-point selection. The illumination of all candidates was rejected after a pilot: it
  falls with the number of candidates, so it favoured the search that found fewer viable ones.
- Fair comparisons through sweeps with pre-declared endpoints (coverage, illumination).

## Consequences
- Phase 5 DoD met twice: the fast Orbium variant (motion claim) and a CLIP cell-like
  multichannel candidate (prompt-specific claim) are validated by pre-declared assays, and
  negative examples are rejected by them.
- Still not implemented: Sep-CMA-ES / OpenES / PGPE, CVT grids, the ASAL illumination
  genetic algorithm itself, E3/E4 assays, a discovery gallery and lineage viewer page.
