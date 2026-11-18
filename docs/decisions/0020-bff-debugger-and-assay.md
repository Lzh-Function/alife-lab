# ADR 0020 — BFF VM debugger and replicator assay

Status: accepted (2026-11-18)

## Context
BFF results so far were entropy transitions (E1). Claims about replicators need a way to look at
individual interactions and a falsifiable test with controls, without changing a run.

## Decision
- Debugger: a columnar step trace produced by a copy of the scalar oracle loop
  (`trace_scalar`), tested equal to the oracle. Exposed as an inspection mode
  (`trace_next_epoch` reconstructs the exact mutated pair tape of the next epoch,
  `trace_self`) and as a stateless tool endpoint for manual programs. Programs are data only.
- Assay: pair assay + soup assay with a scrambled-program control and a no-injection control
  on a shared background, several replicates, and a verdict rule fixed in code and stored with
  every result. Compute is bounded (≤ 2·10⁶ pair evaluations) and runs in the controller's
  worker pool, never in the API process.
- Run-linked assays are published as immutable artifacts (`assays/*.json`, kind `assay`) with an
  `assay_completed` event, so negative results are kept as well as positive ones.

## Consequences
- An assay outcome is evidence "in this assay"; it is reported with its parameters and never as
  a proof of open-ended self-replication. Evidence level for a run-linked positive result: E2
  (measured with controls), E3 only with reproductions across independent runs.
