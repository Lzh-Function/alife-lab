# Architecture (implemented state, Phase 0–1)

```text
Browser (React + TS, Canvas2D)            CLI (alife …)
   │ REST /api/v1 + WS (token, Origin, Host checks)
   ▼                                          │ same service layer
FastAPI app (alife_lab.api.app)  ─────────────┘
   │
Controller (alife_lab.service.controller) — single DB writer, flock-guarded
   ├─ SQLite WAL (.alife/alife.sqlite3): experiments, runs, jobs, checkpoints,
   │    events, commands, interventions, artifacts, idempotency_keys
   ├─ LocalArtifactStore (runs/<run_id>/…): stage → fsync → sha256 → atomic rename
   └─ Executor
        ├─ LocalExecutor: multiprocessing *spawn* worker per job   [implemented]
        ├─ FakeExecutor: same worker loop in a thread (tests)      [implemented]
        └─ remote_compute / multi_gpu / cluster_batch              [planned, rejected by validation]
Worker (alife_lab.execution.worker) — owns substrate state; applies commands at tick
boundaries; writes metrics JSONL / checkpoints / logs; sends previews (droppable),
metrics, acks, receipts, heartbeats.
```

## Run directory
```text
runs/<run_id>/
  manifest.json            immutable provenance (source digest, git, spec/profile hashes, backend)
  resolved_spec.json       ExperimentSpec (canonical content of the spec hash)
  execution_profile.json   ExecutionProfile
  environment.json         package versions, probe of the target
  checkpoints/eEEE_tTTTTTTTTTTTT.npz
  metrics/tail_eEEE.jsonl  journal of rows not yet compacted ({tick, epoch, seq, values})
  metrics/chunks/*.parquet immutable metric chunks (10 000 rows; ADR 0014)
  metrics/history.jsonl    legacy Phase 1 format (still readable)
  logs/events.jsonl        mirror of DB events
  logs/worker_eEEE.jsonl   structured worker log
  artifacts/index.json     artifact + checkpoint index (atomically replaced)
```

## State machine
`pending → queued → starting → running ⇄ (pausing →) paused (→ resuming → running)`;
terminal `completed | cancelled | failed`; `interrupted` (controller restart, worker
loss, walltime, shutdown) and `failed` can be restored → `queued`.
See `src/alife_lab/core/states.py`.

## Live data channels
- Durable: metrics JSONL, events (DB + JSONL), checkpoints, commands/interventions.
- Droppable: preview frames. Each WS subscriber keeps only the latest frame; the
  count of dropped frames is reported. JSON messages use a bounded queue (512); on
  overflow the client receives `resync_required` and refetches over REST.
