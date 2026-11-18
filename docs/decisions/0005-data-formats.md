# ADR 0005 — Phase 1 data formats (metrics JSONL, NPZ checkpoints)

- Status: accepted for Phase 1; Parquet/Zarr planned
- Date: 2026-11-18

## Decision
- Checkpoints: compressed NPZ written with `np.savez_compressed`, loaded with
  `allow_pickle=False`; object arrays are refused. Metadata is a JSON string member
  (`__meta__`, format `alife_npz_v1`). Published via stage → fsync → sha256 →
  atomic rename; immutable once published.
- Metrics: append-only `metrics/history.jsonl` per run, one row per tick per epoch,
  written by the worker and flushed at least every 0.25 s and before every
  checkpoint. Spec §2.1 recommends Parquet; JSONL is used until volumes require it
  (avoids a pyarrow dependency in the base environment).
- Events: SQLite `events` (authoritative, per-run monotonic `seq`) plus an
  append-only `logs/events.jsonl` export in the run directory.

## Consequences
- Metric reads scan the JSONL (O(rows)); acceptable for Phase 1 run lengths, must
  change for long runs (Phase 2: Parquet chunks).
- Metric rows not yet flushed when a worker is killed are lost, but the state is
  rewound to the last checkpoint on restore, so those ticks are recomputed; rows from
  an epoch beyond the next epoch's start tick are excluded as `superseded`.
