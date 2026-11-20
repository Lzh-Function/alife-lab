# Plugin contract — `search`

- Optimisation run (exploratory only). Clock unit: generation. Backend: `numpy`.
- Spec: `target` (full base Lenia experiment; its seed is replaced by evaluation seeds),
  `space` (1–8 parameters from the allow-list `SEARCH_PATHS`, `low`/`high`, `linear`|`log`;
  integer paths are rounded), `dynamics` (`algorithm` random|cma_es, `population`, `sigma0`,
  `objective`, `max_alive_fraction`, `eval_steps`, `eval_seeds`, `holdout_seeds`),
  `runner.max_steps` = generations. Budget = population × eval_seeds × generations evaluations.
- State: archive of every candidate (`u` normalised, `x` values, `score`, `viable`,
  `alive_fraction`, `speed`, `generation`), holdout scores of former bests, CMA-ES state
  (JSON arrays), NumPy RNG state, best candidate's final field (display only).
- Interventions: none. Inspection `{x, y}`: nearest candidate on the map (within 4 px) and the
  best candidate, each with params, scores and a materialisable target spec.
- Metrics: `best_score`, `generation_best`, `generation_mean`, `viable_fraction`,
  `evaluations`, `holdout_best` (NaN without holdout), `sigma` (NaN for random).
- Display codec `field_u8`, 3 channels, width H + W, height H: parameter map of the first two
  axes (colour = score, grey = not viable) and the best candidate's final state.
- Comparison: `archive_score_max_abs_difference` with `exact_equal` (archive, holdout, CMA-ES).
