# ADR 0012 — Unclean-stop recovery and independent backup/restore

- Status: accepted (Phase 2, slice A)
- Date: 2026-11-18

## Unclean controller stop (SIGKILL / power loss)
Observed: when `alife serve` is killed with SIGKILL, each worker sees EOF on its pipe and
exits within ≈0.4 s (no orphan simulation keeps running). On the next start the controller:

1. marks runs that were queued/active as `interrupted` (`run_interrupted`, reason
   `controller_restart`);
2. moves leftover `*.staging` files into `<run>/quarantine/` (never deletes them; event
   `staging_quarantined`);
3. reports checkpoint files present on disk but not indexed (`unindexed_checkpoint_found`)
   **without adopting them**: such a state may include an intervention whose acknowledgement
   was never recorded, which would break the intervention log that replay relies on;
4. records `PRAGMA quick_check` in `/system/capabilities → startup_integrity`.

Restore then uses the latest *indexed* checkpoint; metric rows of the lost epoch beyond it
are superseded (ADR 0007). The test kills the server with SIGKILL, plants a `.staging` file,
restarts, restores and requires exact replay verification and a gap-free metric history.

## Backup / restore (`alife backup create|verify|restore`)
- Full backups only (DB + all run directories). DB via the SQLite online backup API, files
  copied with SHA-256, every DB artifact/checkpoint reference must match; assembled in
  `<id>.partial` and atomically renamed. The destination must be outside `ALIFE_HOME` and
  the runs directory.
- Excluded: session token, controller lock, WAL/SHM, test outputs, `*.staging`.
- Runs that were active are listed in `live_runs_at_backup` (their append-only logs are
  captured up to the copy time; checkpoints are immutable).
- `verify` re-hashes all files, runs `PRAGMA integrity_check` and re-checks DB references.
- `restore` verifies first and writes only into empty/non-existent directories; restored
  active runs come up as `interrupted` and can be restored from their latest checkpoint.
- Tested: tamper detection (one flipped byte), incomplete backup (no manifest), refusal to
  restore over existing data, and a restored installation passing artifact + replay
  verification and resuming a run.

## Not covered
- Remote/off-site storage targets (only a local directory; copying it to another medium is
  the operator's step). Retention/rotation of backups. Incremental backups.
