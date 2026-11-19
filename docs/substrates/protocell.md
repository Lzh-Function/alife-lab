# Substrate contract — `protocell`

- Exploratory only. State: field food/waste (int64 [H, W]); cells as arrays (ids, x, y,
  contents int64 [n, 4] = food/catalyst/membrane/waste, genome float64 [n, 3]); exact ledger;
  lineage `id → (parent, birth tick, end tick)`; last 200 division records; PCG64 state.
- Controls: `membrane`, `metabolism`, `division`, `partition`, `heritable`.
- Interventions: `add_food {x, y, count}` and `add_cell {x, y}` (booked as inflow),
  `kill {x, y, radius}` (contents released to the field).
- Inspection: nearest cell (contents, traits, radius, pressure, ancestors), field at the point,
  full ledger and residual, recent divisions.
- Metrics: population, divisions, ruptures, deaths, mean_radius, total_matter, field_food,
  ledger_residual (always 0), mean_k_replicate, mean_permeability, partition_asymmetry.
- Display: food (green) / waste (blue) field with cells coloured by catalyst concentration,
  rendered at 3× the grid resolution.
