# Substrate contract — `gray_scott`

- State: `u`, `v` float64[N,N] (NumPy or CuPy), `tick`. RNG used only at initialisation.
- Modes: reference (Pearson 1993 setup) and exploratory. Clock unit: tick.
  Backends: `numpy`, `cupy` (bit-identical state).
- Parameters: Du, Dv, F, k, dt (stability-checked), physical domain side `length`,
  `reaction` (false = diffusion-only control). Square periodic domain only.
- Initialisation: `pearson` (centred square + noise) or `uniform` (trivial steady state).
- Interventions: `seed {x, y, radius}`, `erase {x, y, radius}`.
- Inspection `{x, y}`: u, v, both Laplacians, u v², du/dt, dv/dt.
- Metrics: `mean_u`, `mean_v`, `v_area_fraction` (v > 0.1), `v_std`, `total_u_plus_v`.
- Display codec `field_u8` (1 channel, v / 0.4). Comparison: `max_abs_difference`.
