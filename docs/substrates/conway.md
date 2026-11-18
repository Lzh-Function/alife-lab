# Substrate contract — `conway`

- State: `grid: uint8[H, W] ∈ {0,1}`, `tick: int`, passive `CentroidTracker`, recorded RNG state.
- Clock: `tick` (monotonic integer). Safe boundary for commands: between ticks.
- Capabilities: pause, step, brush intervention, checkpoint, deterministic CPU. No
  inspection/lineage/differentiation.
- Interventions: `brush` with `{"shape": "cells", "cells": [[x,y],...], "value": 0|1}` or
  `{"shape": "circle", "x", "y", "radius" (0..64), "value"}`; at most 65 536 cells.
- Metrics: `alive_cell_count` (every tick), `centroid_displacement` / `glider_displacement`
  (every tick; null when undefined).
- Checkpoint: NPZ `alife_npz_v1`, `serialize_version` 1; restore requires an identical
  spec digest and grid shape.
