# Substrate contract — `particle_life`

- State: `pos`, `vel` float64[N,2], `types` int64[N], stable `ids`, interaction `matrix`
  float64[K,K], `next_id`, `tick`, PCG64 RNG state.
- Mode: exploratory only. Interventions `spawn` (optional `type`), `remove`. Inspectable.
- Metrics: `population`, `mean_speed`, `mean_neighbor_count`, `segregation_index`.
- Display codec `particles_typed_f32`: header `{count, types, world_width, world_height}`,
  payload float32 LE `[x, y, vx, vy, type]` × N.
