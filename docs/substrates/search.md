# Plugin contract — `search`

- Optimisation run (exploratory only). Clock unit: generation. Backend: `numpy`.
- Spec: `target` (full base Lenia experiment; its seed is replaced by evaluation seeds),
  `space` (1–8 parameters from the allow-list `SEARCH_PATHS`, `low`/`high`, `linear`|`log`;
  integer paths are rounded), `dynamics` (`algorithm` random|cma_es, `population`, `sigma0`,
  `objective`, `max_alive_fraction`, `eval_steps`, `eval_seeds`, `holdout_seeds`),
  `runner.max_steps` = generations. Budget = population × eval_seeds × generations evaluations.
- Optional `qd`: two descriptors (`speed`, `mass`, `alive_fraction`) with fixed bounds/bins,
  `mutation_sigma`, `archive_version`; required by `algorithm: map_elites`. Omitted from the
  canonical dump when absent (digest-stable).
- Optional `asal` (`model`, `prompts`, `frames`) for the CLIP objectives
  `lenia_clip_target` / `lenia_clip_open_endedness`; numeric policy `float32_clip_torch`,
  backends `torch` / `torch_cuda`. Frames are taken at the ends of `frames` equal chunks.
- State: archive of every candidate (`u` normalised, `x` values, `score`, `viable`,
  `alive_fraction`, `speed`, `mass`, `parent`, `generation`); the QD grid is derived, holdout scores of former bests, CMA-ES state
  (JSON arrays), NumPy RNG state, best candidate's final field (display only).
- Interventions: none. Inspection `{x, y}`: nearest candidate on the map (within 4 px) and the
  best candidate, each with params, scores and a materialisable target spec.
- Metrics: `best_score`, `generation_best`, `generation_mean`, `viable_fraction`,
  `evaluations`, `holdout_best` (NaN without holdout), `sigma` (NaN for random); with `qd`:
  `coverage`, `qd_score`, `archive_size`; with CLIP objectives: `blank_score`. CLIP runs store
  the last-frame embedding of every candidate (`embedding`, serialize_version 3).
- Display codec `field_u8`, 3 channels, height H, width H + W (+ H with `qd`): parameter map
  of the first two axes (colour = score, grey = not viable, cyan = ancestors of the best), the
  QD grid (descriptor 0 right, 1 up) and the best candidate's final state. Inspection of the
  grid returns the cell bounds and its elite (with lineage).
- Comparison: `archive_score_max_abs_difference` with `exact_equal` (archive, holdout, CMA-ES).
