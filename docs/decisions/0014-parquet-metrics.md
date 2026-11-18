# ADR 0014 — Parquet metric chunks with a JSONL journal (supersedes the storage part of ADR 0005)

- Status: accepted (Phase 2, slice C)
- Date: 2026-11-18

## Decision
- Worker: every metric row gets `(epoch, seq)`; rows are appended+flushed to
  `metrics/tail_eEEE.jsonl` (journal, fsync'd at every checkpoint). Every 10 000 rows and on a
  clean stop, buffered rows become an immutable Parquet chunk
  `metrics/chunks/eEEE_sSSSSSSSSSSSS.parquet` (zstd; columns `tick, epoch, seq,
  post_intervention, m.<metric>`), published through the ArtifactStore and indexed by the
  controller as `metrics_chunk` artifacts (checksummed, included in verify/backup). Then the
  journal is truncated. Chunks are not written per checkpoint to avoid tiny files.
- Reader: legacy `history.jsonl` + chunks + journals; drop duplicate `(epoch, seq)` (crash
  between publish and truncation); exclude superseded epochs; last row per tick in
  `(epoch, seq)` order. A Hypothesis property test checks equivalence with the Phase 1
  line-by-line reader on random multi-epoch histories with post-intervention rows.
- API: `GET /experiments/{id}/metrics` gains `until_tick` (with `since_tick`, `max_points`).
- Dependency: `pyarrow` 26.0.1 (≈166 MB installed).

## Measurements (2026-11-18, synthetic 1 M rows, 2 float metrics)
| | JSONL (Phase 1) | Parquet chunks |
|---|---|---|
| Size | 113.1 MB (113 B/row) | 16.9 MB (16.9 B/row, 100 chunks) |
| Read, 2000-point downsample | 2.10 s | 0.14 s |
| Read, 1000-tick window | — | 0.13 s (all chunks are scanned; no pruning yet) |

End-to-end Conway throughput unchanged within noise (128² glider 5 719 vs 5 920 ticks/s,
512² 499 vs 505). **Cost:** importing pyarrow raised resident memory of each worker from
≈53 MB to ≈113 MB and of the API process from ≈31 MB to ≈103 MB.

## Limitations / follow-ups
- Range reads scan all chunks (could prune by per-chunk tick statistics).
- Old journals of killed epochs are read but never compacted.
- Done (ADR 0015): pyarrow is imported lazily; short interactive runs stay at ≈54 MB.
