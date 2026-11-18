# Substrate contract — `boids`

- State: `pos float64[N,2]`, `vel float64[N,2]`, `ids int64[N]` (stable), `next_id`, `tick`,
  PCG64 RNG state. World size comes from the spec (not stored in checkpoints).
- Clock: `tick`. Safe boundary for commands: between ticks.
- Mode: exploratory only. Capabilities: pause, step, interventions `spawn`/`remove`,
  checkpoint, deterministic CPU (same build).
- Metrics: `population`, `polarization`, `mean_speed`, `mean_neighbor_count`.
- Display codec `particles_f32`: header `{count, world_width, world_height}`, payload
  `float32 LE [x, y, vx, vy] × N`.
- Checkpoint: NPZ `alife_npz_v1`, arrays `pos`, `vel`, `ids`; `serialize_version` 1.
