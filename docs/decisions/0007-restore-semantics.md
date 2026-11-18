# ADR 0007 — Restore semantics, epochs and superseded data

- Status: accepted for Phase 1; fork/replay planned for Phase 2
- Date: 2026-11-18

- Each worker job of a run has an incrementing `epoch` and a `start_tick`.
- Restore is explicit (`POST /experiments/{id}/restore`) and allowed only from
  `interrupted`/`failed`; it resumes **in place** from the **latest** checkpoint.
  Restoring an older checkpoint would rewrite history and is refused with
  `fork_required` — that is Phase 2's fork.
- Metric rows of epoch *e* with `tick > start_tick(e')` for any later epoch *e'* are
  excluded from the effective history (`superseded_rows` is reported).
- Graceful shutdown (SIGINT/SIGTERM of `alife serve`) sends `shutdown` to workers,
  which write a `shutdown` checkpoint at the next tick boundary before exiting.
- Determinism: for the CPU integer CA, 1000 continuous ticks equal 437 ticks +
  controller restart + restore + 563 ticks bit-for-bit (integration test).
