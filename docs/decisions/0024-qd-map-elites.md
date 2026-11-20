# ADR 0024 — Quality-diversity (MAP-Elites) on the search plugin; additive spec sections

Status: accepted (2026-11-20)

## Context
Spec §7.2 asks for MAP-Elites with pre-declared, fixed descriptor bounds and separate
quality / diversity / coverage / QD-score metrics; §7.4–7.5 ask for lineage records. The search
plugin (ADR 0023) already holds an evaluation archive and supports sweeps.

## Decision
- Optional `qd` section on `SearchExperiment`: two descriptors (`speed`, `mass`,
  `alive_fraction`) with fixed `low`/`high`/`bins`, `mutation_sigma`, `archive_version`.
  The grid is a pure function of the evaluation history (recomputed on load, not stored).
  Entry: viable on every search seed; quality = objective score; strictly better replaces.
- With `qd`, every algorithm reports `coverage`, `qd_score`, `archive_size`, so MAP-Elites,
  CMA-ES and random search are compared on the same grid at equal evaluation budgets.
- `map_elites`: uniform parent selection over filled cells, Gaussian mutation in the
  normalised box (clipped); uniform sampling while the grid is empty. Own NumPy
  implementation (≈ 20 lines); QDax (JAX) not adopted for the same reasons as evosax.
- Lineage: `parent` per candidate (−1 for uniform samples and CMA-ES draws, which have no
  single parent). Ancestry is a tree here; a DAG is not needed without recombination.
- Schema evolution: new optional sections are omitted from the canonical dump when absent
  (`model_serializer`), so existing specs keep their SHA-256 and their checkpoints restore.
  Checkpoint serialize_version 2 adds `mass` and `parent`; version 1 loads with NaN / −1.

## Consequences
- Descriptor bounds were set from a pilot (seed 99) and recorded in the scenario; changing
  them requires a new `archive_version`.
- Planned: novelty search (kNN archive), CVT grids, more descriptors and target substrates.
