# Substrate contract — `pd_nca`

- One tick = one epoch: `steps_before_update` CA steps without gradients, `steps_per_update`
  steps with per-entity gradient isolation, one online update (unless `learning: frozen`).
- State: grid float32[H, W, M + S + Hd] (aliveness of sun + N NCAs, state, hidden), per-NCA
  weights, optimizer state, fixed sun vector and seed vectors. No RNG after initialisation.
- Controls: `learning: frozen`, `competition: uniform`. Option `revert_on_extinction`.
- Interventions: `seed {x, y, radius, agent}`, `erase {x, y, radius}` (cells return to the sun).
- Inspection: aliveness of a cell, dominant entity, territories, interaction matrix (mean cosine
  attack→defence over cells where both entities are alive).
- Metrics: `territory_loss`, `grad_norm`, `alive_ncas`, `territory_entropy`, `sun_fraction`,
  `reverted`, plus `territory.sun` / `territory.nca_<k>`.
- Backends `jax` (CPU) and `jax_cuda` (`numeric_policy: float32_jax`). Display: RGB mixture of
  entity colours weighted by aliveness.
