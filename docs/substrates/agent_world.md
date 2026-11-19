# Substrate contract — `agent_world`

- Exploratory, autonomous (no external fitness). State: agent arrays (ids, parent, birth,
  position, heading, energy, age, genetic weights W1/b1/W2/b2, learned ΔW2, reward baseline,
  per-agent history of inputs/hidden/probabilities/action/reward), food and signal grids,
  counters, PCG64 state.
- Controllers `baseline` / `random` / `mlp`; modes `fixed` / `evolution` / `lifetime`;
  controls `communication`, `shuffle_policies`.
- Interventions: `add_food {x, y, radius}` (fill to capacity), `remove {x, y, radius}`.
- Inspection: nearest agent with lineage, energy, the last `history` ticks of sensors, hidden
  activations, action probabilities, actions and rewards, and the norm of learned weights.
- Metrics: population, mean_energy, births, deaths, food_total, mean_age, signal_rate,
  forward_rate, mean_reward, genome_spread.
- Display: food (green) and signal (blue) grids with agents coloured by heading.
