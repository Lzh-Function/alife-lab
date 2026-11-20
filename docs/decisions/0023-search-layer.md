# ADR 0023 — Search layer: runs that optimise target-world parameters

Status: accepted (2026-11-20)

## Context
Phase 5 adds external search (spec §7). Searches must stay separate from substrate dynamics
(no reward inside a reference substrate), be checkpointable, reproducible and comparable at an
equal budget, and reuse the run/replay/sweep machinery instead of a parallel system.

## Decision
- A search is an ordinary run of the `search` plugin (clock `generation`, one tick = one
  generation). Its spec embeds the full base target spec (`target`, currently Lenia only), the
  searchable box (`space`, a per-substrate allow-list of paths, linear or log scale), the
  algorithm and a **pre-declared, versioned objective** (`alife_lab.evolution`).
- Candidates are complete target specs (validated, hashable) run headless with the target's own
  adapter; the objective only observes them. Any candidate can be opened as a normal run.
- Algorithms: `random` (uniform baseline) and `cma_es`. CMA-ES is a NumPy implementation of the
  update equations of Hansen's `purecma` (public domain) with array state (checkpointable
  without pickle); pycma 4.5 is a dev dependency used only as the test oracle. evosax 0.3.2
  (JAX) was not adopted for this slice: it would make the CPU-only base install depend on JAX
  and its state is a JAX pytree; it can be added later as an optional backend.
- Common random numbers: all candidates use the same search seeds; the best candidate is
  re-scored on disjoint holdout seeds that never influence the search (tested).
- Fair comparison of algorithms uses the existing sweeps (factor `dynamics.algorithm`, equal
  population × generations × seeds, pre-declared endpoint).

## Consequences
- Pause/step/checkpoint/replay verification, provenance and sweeps work unchanged.
- One generation is the pause granularity; long evaluations make pausing coarse.
- Objective changes require a version bump (`OBJECTIVE_VERSIONS`).
- Planned: QD (MAP-Elites) archives and novelty search on the same plugin; more target
  substrates; GPU batch evaluation.
