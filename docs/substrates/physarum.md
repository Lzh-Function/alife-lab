# Substrate contract — `physarum`

- State: agent arrays `x`, `y`, `h` (float64), `trail` float64[H,W] ≥ 0, `tick`, PCG64 RNG state,
  `last_blocked`; the occupancy grid is derived from positions on load.
- Modes: reference (Jones rules with exclusion) and exploratory. Clock unit: tick. Backend:
  `numpy` (+ Numba-compiled motor loop when installed; bit-identical).
- Parameters: SA, RA (degrees), SO, step size, deposit, decay, p_random_turn, collisions,
  sensing (false = control), food sources `{x, y, radius, amount}`.
- Interventions: `attract {x, y, radius, amount}` adds to the trail; `erase {x, y, radius}`.
- Inspection `{x, y, radius}`: trail value, agents in radius, nearest agent with its three sensor
  readings and the decision the sensory rule will take.
- Metrics: `trail_mean`, `trail_coverage`, `trail_contrast`, `heading_order`, `blocked_fraction`.
- Display codec `field_u8` (sqrt(trail / 50)). Comparison: trail max |difference| + exact flag.
