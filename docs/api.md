# API (v1, implemented subset)

All JSON responses: `{"api_version": "v1", "server_version": "...", "data": ...}`;
errors: `{"api_version": "v1", "error": {"code", "message"}}`. Interactive OpenAPI:
`/api/v1/docs`. Non-GET requests require `X-ALife-Token`.

| Method | Path | Notes |
|---|---|---|
| GET | /health, /system/capabilities, /system/resources | resources = live probe |
| GET | /compute-targets, /execution-profiles | planned roles flagged `implemented: false` |
| GET | /substrates, /substrates/{slug}/schema, /presets | |
| POST | /experiments/validate | resolves preset+overrides, returns report + hashes |
| POST | /experiments | `Idempotency-Key` header supported; `auto_start`, `start_paused` |
| GET | /experiments, /experiments/{id} | `{id}` is the run id |
| POST | /experiments/{id}/start, /pause, /resume, /step, /speed, /checkpoints, /interventions | body may carry `command_id` (UUID, idempotent), `expected_revision`, `requested_clock`; response contains the worker ack with `applied_clock` |
| POST | /experiments/{id}/restore | latest checkpoint only (`fork_required` otherwise) |
| POST | /experiments/{id}/cancel, /verify | verify = sha256 of every indexed artifact |
| GET | /experiments/{id}/checkpoints, /checkpoints/{cid}/frame, /metrics, /events, /interventions, /provenance | frame = binary preview of a checkpoint |
| POST | /experiments/{id}/fork | `{checkpoint_id="latest", label, spec_overrides, intervention, profile_id, ticks_per_second, auto_start, start_paused}`; `Idempotency-Key` supported |
| POST | /experiments/{id}/paired-fork | `{checkpoint_id="now", treatment_intervention, name, start, start_paused}` → comparison with `control`/`treatment` arms |
| GET | /experiments/{id}/inspect?x=&y=&radius=[&tick=] | read-only inspection (live worker, or replayed state at `tick`) |
| GET | /experiments/{id}/timeline | checkpoints, interventions, marker events, replay mode |
| GET | /experiments/{id}/replay/frames?start=&count=&stride= | binary: `uint32 n` + n × (`uint32 len` + frame); header has `source`, `base_tick`, `steps_from_base`, `interventions_replayed` |
| POST | /experiments/{id}/replay/verify | re-simulate each checkpoint segment; exact comparison |
| POST | /experiments/{id}/fork with `{"tick": T}` | fork from an arbitrary tick via deterministic replay |
| POST/GET | /comparisons, /comparisons/{cid} | 2–4 runs |
| GET | /comparisons/{cid}/report | `divergence.points[].hamming_to_first` at common checkpoint ticks + aligned metrics |
| WS | /stream/experiments/{id}?token=… | JSON: hello, run_state, metrics, event, command_ack, heartbeat, resync_required; binary preview frames |

Binary frame: `uint32 BE header length` + JSON header (`tick,width,height,codec,
dropped_previews`) + payload. Codec `bitpack1` = `numpy.packbits` (row-major, MSB first).

## Substrate-specific payloads
- Interventions (`POST /experiments/{id}/interventions`, `{kind, payload}`): `conway` → `brush`
  (`{shape: "cells"|"circle", ...}`); `boids` → `spawn {x, y, radius, count}`, `particle_life` → `spawn {x, y, radius, count, type?}` / `remove`,
  `remove {x, y, radius}`; `bff` → `inject_program {indices, program, encoding}`, `zero_programs {indices}`. Kinds not declared by the substrate are rejected (400).
- Frame codecs: `bff_soup` (`programs`, `columns`; raw bytes of up to 4096 programs) for BFF; `bitpack1` (`width`, `height`) for Conway; `particles_typed_f32` (`count`, `types`, world size; float32 `[x, y, vx, vy, type]`) for Particle Life; `particles_f32` (`count`,
  `world_width`, `world_height`; float32 LE `[x, y, vx, vy]` per particle) for Boids.
- Comparison reports: `divergence.metric` (`hamming` | `rms_position_distance`) and
  `points[].distance_to_first`.

## Sweeps
| Method | Path | Notes |
|---|---|---|
| POST | /sweeps/validate | body = SweepSpec; returns the plan (conditions, member spec hashes) |
| POST | /sweeps | creates and queues all member runs; `Idempotency-Key` supported |
| GET | /sweeps, /sweeps/{id} | progress by run state |
| GET | /sweeps/{id}/report | per-condition summaries, comparisons (Δ mean, bootstrap CI, Cliff's δ, permutation p, Holm), per-run endpoint values, exclusions, time course, note codes |
| POST | /sweeps/{id}/cancel | cancels non-terminal members |


## Gallery and lineage (ADR 0029)

| Method | Path | Notes |
|---|---|---|
| GET | `/api/v1/gallery` | visible entries, newest first |
| POST | `/api/v1/gallery` | `{run_id, tick, title, note, png_base64}`; PNG validated (≤ 2 MiB, ≤ 4096²) |
| GET | `/api/v1/gallery/{id}/image` | `image/png`, SHA-256 checked on read |
| DELETE | `/api/v1/gallery/{id}` | hides the entry (artifact kept) |
| GET | `/api/v1/experiments/{id}/lineage?tick=` | search runs: per-candidate generation, parent, score, viability, position |
| GET | `/api/v1/experiments/{id}/inspect?x=0&y=0&candidate=i` | search runs: one candidate by archive index |
