# ADR 0001 — Controller / worker process model

- Status: accepted (Phase 1)
- Date: 2026-11-18

## Context
Spec §2.2–2.7 requires a controller that is the single authoritative writer of run
metadata, workers isolated from the API process (GPU runtimes must never load in the
API), safe-boundary interventions, and an `Executor` seam for future remote/cluster
targets.

## Decision
- `alife_lab.service.controller.Controller` is the only writer of the SQLite DB. A
  non-blocking `flock` on `.alife/controller.lock` prevents a second controller
  (e.g. `alife run` while `alife serve` is running).
- `LocalExecutor` starts one worker per job with `multiprocessing` **spawn** (no fork
  of the API process). `FakeExecutor` runs the identical worker loop in a thread for
  tests.
- Controller ↔ worker messages are dicts of primitives/bytes over a private
  `multiprocessing` pipe (trusted, local, same program). Every worker message carries
  `job_id` + `lease_token`; messages with a stale lease are dropped.
- Workers write only run-scoped data-plane files (metrics JSONL, checkpoints via the
  ArtifactStore staging protocol, worker logs) and report receipts; the controller
  re-verifies sha256 before indexing a checkpoint.
- Commands are applied only at tick boundaries and acknowledged with the
  `applied_clock`.
- Remote/multi-GPU/cluster roles are rejected by validation with an explicit
  "planned but not implemented" error.

## Consequences
- A pipe-based local executor cannot survive controller restarts: on restart, active
  runs become `interrupted` and must be restored explicitly (spec §2.3).
- Pickle is used only by `multiprocessing` on the private local pipe; no pickle is
  read from disk or from the browser.
