# Substrate contract — `reaction_network`

- Spec: `species` (name, initial count, `constant`), `reactions` (id, reactant/product
  stoichiometry maps, stochastic rate constant, informational label), `time_per_tick`,
  `max_events_per_tick`, `disabled_reactions`. Non-spatial: there is no `world`.
- State: species counts (int64), cumulative firing counts, enabled mask, simulated time, tick,
  PCG64 RNG state. A display-only history (species × last 256 ticks) is rebuilt after loads.
- Modes: reference (e.g. Gillespie's Lotka reactions) and exploratory. Clock unit: tick.
- Interventions: `add` / `remove` `{species | y, count}`, `disable_reaction` /
  `enable_reaction` `{reaction}` (knock-out control at a tick boundary).
- Inspection: all species, the selected species (by name or row), every reaction with its
  current propensity and firing count, and cycle candidates.
- Metrics: `total_molecules`, `species_present`, `events`, `diversity`, `time`, plus
  `count.<species>` per row when `record_species_counts`.
- Display codec `field_u8`: rows = species, columns = last 256 ticks, log1p(count) / max.
- Comparison: L1 difference of counts (`exact_equal` also requires equal time and firings).
