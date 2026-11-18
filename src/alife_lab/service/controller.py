"""Run controller: the single authoritative writer of run metadata.

The controller owns the SQLite database, dispatches worker jobs through an
Executor, applies the run state machine, persists commands / interventions /
events, and fans out live messages to subscribers. Both the HTTP API and the
CLI drive runs exclusively through this class.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from alife_lab import API_CONTRACT_VERSION, __version__
from alife_lab.artifacts.store import CHECKPOINT_FORMAT, LocalArtifactStore
from alife_lab.core.hashing import canonical_json
from alife_lab.core.spec import ExecutionProfile, ExperimentSpec
from alife_lab.core.states import (
    ACTIVE_STATES,
    RESTORABLE_STATES,
    TERMINAL_STATES,
    InvalidTransition,
    RunState,
    check_transition,
)
from alife_lab.execution.executors import Executor, JobHandle, ValidationReport
from alife_lab.execution.worker import WorkerJob
from alife_lab.provenance import environment_record
from alife_lab.service.catalog import Preset, load_presets, load_profiles
from alife_lab.service.settings import Settings
from alife_lab.storage.db import Database
from alife_lab.substrates import get_manifest

log = logging.getLogger("alife_lab.controller")

ACK_TIMEOUT_SECONDS = 15.0
SHUTDOWN_GRACE_SECONDS = 10.0
SUBSCRIBER_QUEUE_SIZE = 512
CONTROL_KINDS = ("play", "pause", "step", "intervene", "checkpoint", "set_speed", "cancel")


class ControllerError(Exception):
    status_code = 400

    def __init__(self, message: str, code: str = "bad_request") -> None:
        super().__init__(message)
        self.code = code


class NotFound(ControllerError):
    status_code = 404

    def __init__(self, message: str) -> None:
        super().__init__(message, "not_found")


class Conflict(ControllerError):
    status_code = 409

    def __init__(self, message: str, code: str = "conflict") -> None:
        super().__init__(message, code)


class ControllerLocked(RuntimeError):
    pass


def new_run_id() -> str:
    return "run_" + secrets.token_hex(8)


# --------------------------------------------------------------------------- #
# Subscribers (WebSocket fan-out with bounded queues and droppable previews)
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class Subscriber:
    run_id: str
    queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
    )
    latest_preview: dict[str, Any] | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    overflowed: bool = False
    dropped_previews: int = 0

    def push(self, message: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            # Durable data remains available through REST; the client must resync.
            self.overflowed = True
        self.wake.set()

    def push_preview(self, preview: dict[str, Any]) -> None:
        if self.latest_preview is not None:
            self.dropped_previews += 1
        self.latest_preview = preview
        self.wake.set()


@dataclass
class ActiveJob:
    job_id: str
    run_id: str
    epoch: int
    lease_token: str
    handle: JobHandle
    spec: ExperimentSpec
    pending: dict[str, asyncio.Future[dict[str, Any]]] = field(default_factory=dict)
    finished: bool = False
    last_heartbeat: float = field(default_factory=time.time)
    steps_per_second: float = 0.0
    peak_rss_mb: float = 0.0
    last_rss_mb: float | None = None
    cpu_percent: float | None = None
    started_at: float = field(default_factory=time.time)
    process: psutil.Process | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    latest_preview: dict[str, Any] | None = None


class Controller:
    def __init__(self, settings: Settings, executor: Executor) -> None:
        self.settings = settings
        self.executor = executor
        settings.home.mkdir(parents=True, exist_ok=True)
        self._lock_handle = _acquire_lock(settings.lock_path)
        self.db = Database(settings.db_path)
        self.store = LocalArtifactStore(settings.runs_root)
        self.presets: dict[str, Preset] = load_presets(settings.configs_dir)
        self.profiles: dict[str, ExecutionProfile] = load_profiles(settings.configs_dir)
        self.jobs: dict[str, ActiveJob] = {}  # run_id -> active job
        self.subscribers: dict[str, set[Subscriber]] = {}
        self.queue: deque[tuple[str, bool]] = deque()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.started_at = time.time()
        self.environment = environment_record()
        self.target = executor.probe()
        self._closed = False
        self._db_closed = False

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._recover_after_restart()

    def _recover_after_restart(self) -> None:
        """Runs that were active when the previous controller stopped become interrupted."""
        stale = [s.value for s in (*ACTIVE_STATES, RunState.QUEUED)]
        placeholders = ",".join("?" * len(stale))
        rows = self.db.all(f"SELECT id, state FROM runs WHERE state IN ({placeholders})", stale)  # noqa: S608
        for row in rows:
            self._set_state(
                row["id"], RunState.INTERRUPTED, reason="controller_restart", force=True
            )
            self.db.execute(
                "UPDATE jobs SET status='lost', finished_at=? WHERE run_id=? AND status IN "
                "('starting','running')",
                (time.time(), row["id"]),
            )
            self._emit_event(
                row["id"],
                "run_interrupted",
                {"reason": "controller_restart", "previous_state": row["state"]},
                severity="warning",
            )

    async def shutdown(self) -> None:
        """Ask every worker to checkpoint and exit at its next safe boundary."""
        if self._closed:
            return
        self._closed = True
        waits = []
        for job in list(self.jobs.values()):
            if job.finished:
                continue
            with contextlib.suppress(OSError, BrokenPipeError):
                job.handle.conn.send(
                    {"command_id": str(uuid.uuid4()), "kind": "shutdown", "payload": {}}
                )
            waits.append(job.done.wait())
        if waits:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.gather(*waits), SHUTDOWN_GRACE_SECONDS)
        for job in list(self.jobs.values()):
            if not job.finished:
                job.handle.terminate()
                self._finish_job(job, RunState.INTERRUPTED, "shutdown_timeout", None)
        self._db_closed = True
        self.db.close()
        _release_lock(self._lock_handle)

    # ------------------------------------------------------------------ queries

    def system_info(self) -> dict[str, Any]:
        return {
            "api_version": "v1",
            "server_version": __version__,
            "api_contract_version": API_CONTRACT_VERSION,
            "started_at": self.started_at,
            "db_schema_version": self.db.schema_version,
            "target": self.target.to_json(),
            "active_jobs": len([j for j in self.jobs.values() if not j.finished]),
            "max_active_jobs": self.settings.max_active_jobs,
            "queued": len(self.queue),
            "environment": self.environment,
        }

    def _run_row(self, run_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if row is None:
            raise NotFound(f"run {run_id} not found")
        return dict(row)

    def run_summary(self, run_id: str) -> dict[str, Any]:
        row = self._run_row(run_id)
        exp = self.db.one("SELECT * FROM experiments WHERE id=?", (row["experiment_id"],))
        assert exp is not None
        latest = self.latest_checkpoint(run_id)
        job = self.jobs.get(run_id)
        return {
            "id": row["id"],
            "experiment_id": row["experiment_id"],
            "name": row["label"] or exp["name"],
            "label": row["label"],
            "parent_run_id": row["parent_run_id"],
            "parent_checkpoint_id": row["parent_checkpoint_id"],
            "children": [
                r["id"]
                for r in self.db.all(
                    "SELECT id FROM runs WHERE parent_run_id=? ORDER BY created_at", (run_id,)
                )
            ],
            "comparisons": [
                r["comparison_id"]
                for r in self.db.all(
                    "SELECT comparison_id FROM comparison_members WHERE run_id=? UNION "
                    "SELECT id AS comparison_id FROM comparisons WHERE parent_run_id=?",
                    (run_id, run_id),
                )
            ],
            "substrate": exp["substrate"],
            "declared_mode": exp["mode"],
            "effective_mode": row["effective_mode"],
            "intervened": bool(row["intervened"]),
            "state": row["state"],
            "revision": row["revision"],
            "seed": row["seed"],
            "backend": row["backend"],
            "target_id": row["target_id"],
            "profile_id": row["profile_id"],
            "spec_hash": exp["spec_hash"],
            "profile_hash": row["profile_hash"],
            "spec": json.loads(exp["spec_json"]),
            "clock": {"unit": "tick", "value": row["tick"]},
            "epoch": row["epoch"],
            "ticks_per_second": row["ticks_per_second"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "finish_reason": row["finish_reason"],
            "last_error": row["last_error"],
            "latest_checkpoint": latest,
            "restorable": row["state"] in {s.value for s in RESTORABLE_STATES},
            "live": None
            if job is None or job.finished
            else {
                "job_id": job.job_id,
                "steps_per_second": job.steps_per_second,
                "worker_rss_mb": job.last_rss_mb,
                "worker_peak_rss_mb": round(job.peak_rss_mb, 1),
                "worker_cpu_percent": job.cpu_percent,
                "last_heartbeat": job.last_heartbeat,
            },
        }

    def list_runs(self, state: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = (
            "SELECT r.id, r.state, r.tick, r.created_at, r.effective_mode, r.intervened, "
            "r.profile_id, COALESCE(r.label, e.name) AS name, e.substrate, e.spec_hash, "
            "r.parent_run_id FROM runs r "
            "JOIN experiments e ON e.id = r.experiment_id"
        )
        params: list[Any] = []
        if state:
            sql += " WHERE r.state = ?"
            params.append(state)
        sql += " ORDER BY r.created_at DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))
        return [dict(r) for r in self.db.all(sql, params)]

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        self._run_row(run_id)
        rows = self.db.all(
            "SELECT * FROM checkpoints WHERE run_id=? ORDER BY created_at, simulation_clock",
            (run_id,),
        )
        return [dict(r) for r in rows]

    def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.one(
            "SELECT * FROM checkpoints WHERE run_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (run_id,),
        )
        return dict(row) if row else None

    def list_events(
        self, run_id: str, since_seq: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        self._run_row(run_id)
        rows = self.db.all(
            "SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
            (run_id, since_seq, max(1, min(limit, 5000))),
        )
        out = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item.pop("payload_json"))
            out.append(item)
        return out

    def list_interventions(self, run_id: str) -> list[dict[str, Any]]:
        self._run_row(run_id)
        rows = self.db.all(
            "SELECT * FROM interventions WHERE run_id=? ORDER BY requested_at", (run_id,)
        )
        out = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item.pop("payload_json"))
            item["result"] = json.loads(item.pop("result_json") or "null")
            out.append(item)
        return out

    def metrics(self, run_id: str, since_tick: int = -1, max_points: int = 2000) -> dict[str, Any]:
        """Effective metric history: rows superseded by a later restore are excluded."""
        self._run_row(run_id)
        jobs = self.db.all(
            "SELECT epoch, start_tick FROM jobs WHERE run_id=? ORDER BY epoch", (run_id,)
        )
        start_ticks = [(int(j["epoch"]), int(j["start_tick"])) for j in jobs]
        path = self.store.resolve(run_id, "metrics/history.jsonl")
        by_tick: dict[int, dict[str, Any]] = {}
        superseded = 0
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn final line after a hard kill
                    tick, epoch = int(row["tick"]), int(row["epoch"])
                    later = [s for e, s in start_ticks if e > epoch]
                    if later and tick > min(later):
                        superseded += 1
                        continue
                    if tick > since_tick:
                        by_tick[tick] = row
        ticks = sorted(by_tick)
        stride = max(1, -(-len(ticks) // max(1, max_points)))
        sampled = ticks[::stride]
        if ticks and sampled[-1] != ticks[-1]:
            sampled.append(ticks[-1])
        return {
            "run_id": run_id,
            "clock_unit": "tick",
            "total_points": len(ticks),
            "stride": stride,
            "superseded_rows": superseded,
            "rows": [by_tick[t] for t in sampled],
        }

    def provenance(self, run_id: str) -> dict[str, Any]:
        self._run_row(run_id)
        manifest = self.store.resolve(run_id, "manifest.json")
        result: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
        return result

    def checkpoint_frame(self, run_id: str, checkpoint_id: str) -> tuple[dict[str, Any], bytes]:
        """Decode a checkpoint into a preview frame for display (not the research state)."""
        import numpy as np

        from alife_lab.artifacts.store import decode_checkpoint
        from alife_lab.core.hashing import sha256_bytes

        row = self.db.one(
            "SELECT * FROM checkpoints WHERE run_id=? AND id=?", (run_id, checkpoint_id)
        )
        if row is None:
            raise NotFound("checkpoint not found")
        data = self.store.resolve(run_id, row["logical_path"]).read_bytes()
        if sha256_bytes(data) != row["sha256"]:
            raise Conflict("checkpoint checksum mismatch", "checksum_mismatch")
        payload = decode_checkpoint(data)
        grid = payload["arrays"]["grid"]
        header = {
            "tick": int(payload["meta"]["tick"]),
            "width": int(grid.shape[1]),
            "height": int(grid.shape[0]),
            "codec": "bitpack1",
        }
        return header, np.packbits(grid.astype(np.uint8), axis=None).tobytes()

    def verify_run(self, run_id: str) -> dict[str, Any]:
        self._run_row(run_id)
        results = []
        for row in self.db.all("SELECT * FROM artifacts WHERE run_id=?", (run_id,)):
            ok = self.store.verify(run_id, row["logical_path"], row["sha256"])
            results.append({"logical_path": row["logical_path"], "ok": ok})
        for row in self.db.all("SELECT * FROM checkpoints WHERE run_id=?", (run_id,)):
            ok = self.store.verify(run_id, row["logical_path"], row["sha256"])
            results.append({"logical_path": row["logical_path"], "ok": ok})
        return {"run_id": run_id, "ok": all(r["ok"] for r in results), "artifacts": results}

    # ------------------------------------------------------------------ create

    def validate(
        self, spec: ExperimentSpec, profile_id: str
    ) -> tuple[ExecutionProfile, ValidationReport]:
        profile = self.profiles.get(profile_id)
        if profile is None:
            raise NotFound(f"execution profile {profile_id!r} not found")
        return profile, self.executor.validate(spec, profile)

    def create_run(
        self,
        spec: ExperimentSpec,
        profile_id: str,
        ticks_per_second: float | None = None,
        author: str = "local_user",
        lineage: dict[str, Any] | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        profile, report = self.validate(spec, profile_id)
        if not report.ok:
            raise ControllerError("; ".join(report.errors), "validation_failed")
        tps = (
            self.settings.default_ticks_per_second if ticks_per_second is None else ticks_per_second
        )
        if not (tps == 0 or 0.1 <= tps <= 10_000):
            raise ControllerError("ticks_per_second must be 0 (unlimited) or within 0.1..10000")
        run_id = new_run_id()
        experiment_id = "exp_" + secrets.token_hex(8)
        now = time.time()
        spec_json = spec.model_dump(mode="json")
        profile_json = profile.model_dump(mode="json")
        reuse_experiment = False
        effective_mode = spec.experiment.mode
        if lineage is not None:
            parent_exp = self.db.one(
                "SELECT id, spec_hash FROM experiments WHERE id=?", (lineage["experiment_id"],)
            )
            assert parent_exp is not None
            if parent_exp["spec_hash"] == spec.digest():
                experiment_id = parent_exp["id"]  # paired fork: same scientific spec
                reuse_experiment = True
            if lineage["parent_effective_mode"] == "exploratory":
                effective_mode = "exploratory"  # inherits state produced by interventions
        manifest = self._manifest(run_id, experiment_id, spec, profile, report, now)
        if lineage is not None:
            manifest["lineage"] = {
                k: lineage[k]
                for k in (
                    "parent_run_id",
                    "parent_checkpoint_id",
                    "parent_checkpoint_sha256",
                    "parent_spec_sha256",
                    "fork_tick",
                )
            }
        receipts = []
        for logical, value in (
            ("manifest.json", manifest),
            ("resolved_spec.json", spec_json),
            ("execution_profile.json", profile_json),
            (
                "environment.json",
                {"environment": self.environment, "target": self.target.to_json()},
            ),
        ):
            data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
            receipt = self.store.publish_bytes(
                run_id, logical, data, "provenance", "application/json"
            )
            receipts.append(receipt.to_json())
        with self.db.transaction() as conn:
            if not reuse_experiment:
                conn.execute(
                    "INSERT INTO experiments(id, name, substrate, mode, created_at, spec_hash, "
                    "spec_json) VALUES (?,?,?,?,?,?,?)",
                    (
                        experiment_id,
                        spec.experiment.name,
                        spec.experiment.substrate,
                        spec.experiment.mode,
                        now,
                        spec.digest(),
                        canonical_json(spec_json),
                    ),
                )
            conn.execute(
                "INSERT INTO runs(id, experiment_id, state, seed, backend, target_id, profile_id, "
                "profile_hash, profile_json, effective_mode, ticks_per_second, created_at, "
                "manifest_path, parent_run_id, parent_checkpoint_id, label) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    experiment_id,
                    RunState.PENDING.value,
                    spec.experiment.seed,
                    report.resolved_backend,
                    report.target_id,
                    profile.execution.id,
                    profile.digest(),
                    canonical_json(profile_json),
                    effective_mode,
                    tps,
                    now,
                    f"artifact://{run_id}/manifest.json",
                    lineage["parent_run_id"] if lineage else None,
                    lineage["parent_checkpoint_id"] if lineage else None,
                    label,
                ),
            )
        for receipt_json in receipts:
            self._record_artifact(receipt_json)
        if lineage is not None:
            self._install_fork_origin(run_id, lineage)
        self._emit_event(
            run_id,
            "run_created",
            {"author": author, "warnings": report.warnings, "spec_hash": spec.digest()},
        )
        return self.run_summary(run_id)

    def _manifest(
        self,
        run_id: str,
        experiment_id: str,
        spec: ExperimentSpec,
        profile: ExecutionProfile,
        report: ValidationReport,
        now: float,
    ) -> dict[str, Any]:
        manifest = get_manifest(spec.experiment.substrate)
        env = self.environment
        return {
            "schema_version": 1,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "created_at_unix": now,
            "source": {
                "git_commit": env["git_commit"],
                "git_dirty": env["git_dirty"],
                "source_sha256": env["source_digest"],
                "uv_lock_sha256": env["uv_lock_sha256"],
                "frontend_lock_sha256": env["frontend_lock_sha256"],
                "alife_lab_version": env["alife_lab_version"],
                "reproducibility_class": (
                    "release"
                    if env["git_commit"] and env["git_dirty"] is False
                    else "non_release_reproducibility"
                ),
            },
            "science": {
                "experiment_spec_sha256": spec.digest(),
                "substrate": spec.experiment.substrate,
                "substrate_plugin_id": manifest.plugin_id,
                "substrate_plugin_version": manifest.plugin_version,
                "declared_mode": spec.experiment.mode,
                "seed": spec.experiment.seed,
                "rng_policy": "numpy PCG64(seed); consumed only during initialization",
                "numeric_policy": spec.experiment.dynamics.numeric_policy,
                "clock_unit": spec.experiment.dynamics.clock_unit,
            },
            "execution": {
                "target_id": report.target_id,
                "role": profile.execution.target_selector.role,
                "profile_id": profile.execution.id,
                "profile_sha256": profile.digest(),
                "backend": report.resolved_backend,
                "backend_versions": {"numpy": env["packages"].get("numpy")},
                "target_capabilities": self.target.to_json()["capabilities"],
                "validation_warnings": report.warnings,
            },
            "checkpoint_compatibility": {
                "format": CHECKPOINT_FORMAT,
                "serialize_version": manifest.serialize_version,
                "restore_requires": ["identical experiment_spec_sha256"],
            },
            "artifact_index": f"artifact://{run_id}/artifacts/index.json",
        }

    # ------------------------------------------------------------------ start / restore

    async def start_run(self, run_id: str, start_paused: bool = False) -> dict[str, Any]:
        row = self._run_row(run_id)
        state = RunState(row["state"])
        if state != RunState.PENDING:
            raise Conflict(f"run is {state.value}; only pending runs can be started")
        self._set_state(run_id, RunState.QUEUED)
        self.queue.append((run_id, start_paused))
        self._dispatch_queue()
        return self.run_summary(run_id)

    async def restore_run(
        self, run_id: str, checkpoint_id: str = "latest", start_paused: bool = True
    ) -> dict[str, Any]:
        row = self._run_row(run_id)
        state = RunState(row["state"])
        if state not in RESTORABLE_STATES:
            raise Conflict(
                f"run is {state.value}; restore is allowed only from "
                f"{sorted(s.value for s in RESTORABLE_STATES)}"
            )
        latest = self.latest_checkpoint(run_id)
        if checkpoint_id != "latest" and (latest is None or checkpoint_id != latest["id"]):
            raise Conflict(
                "Phase 1 restores only the latest checkpoint in place; restoring an older "
                "checkpoint requires a fork (planned for Phase 2)",
                "fork_required",
            )
        self._set_state(run_id, RunState.QUEUED)
        self.queue.append((run_id, start_paused))
        self._dispatch_queue()
        return self.run_summary(run_id)

    def _mark_rolled_back(self, run_id: str, checkpoint: dict[str, Any]) -> None:
        """Interventions applied after the restored checkpoint are no longer in the state."""
        rows = self.db.all(
            "SELECT id, applied_clock FROM interventions WHERE run_id=? AND status='applied' "
            "AND applied_at > ?",
            (run_id, checkpoint["created_at"]),
        )
        self.db.execute(
            "UPDATE interventions SET status='aborted' WHERE run_id=? AND status='requested'",
            (run_id,),
        )
        if not rows:
            return
        self.db.execute(
            "UPDATE interventions SET status='rolled_back' WHERE run_id=? AND status='applied' "
            "AND applied_at > ?",
            (run_id, checkpoint["created_at"]),
        )
        self._emit_event(
            run_id,
            "interventions_rolled_back",
            {
                "restored_checkpoint_id": checkpoint["id"],
                "intervention_ids": [r["id"] for r in rows],
                "reason": "state restored from a checkpoint taken before these interventions",
            },
            clock=int(checkpoint["simulation_clock"]),
            severity="warning",
        )

    # ------------------------------------------------------------------ fork / compare

    def _resolve_checkpoint(self, run_id: str, checkpoint_id: str) -> dict[str, Any]:
        if checkpoint_id == "latest":
            row = self.latest_checkpoint(run_id)
            if row is None:
                raise NotFound(f"run {run_id} has no checkpoint")
            return row
        found = self.db.one(
            "SELECT * FROM checkpoints WHERE run_id=? AND id=?", (run_id, checkpoint_id)
        )
        if found is None:
            raise NotFound(f"checkpoint {checkpoint_id} not found in run {run_id}")
        return dict(found)

    def _install_fork_origin(self, run_id: str, lineage: dict[str, Any]) -> None:
        """Copy the verified parent checkpoint into the child as its origin state."""
        parent = lineage["parent_checkpoint"]
        data = self.store.resolve(parent["run_id"], parent["logical_path"]).read_bytes()
        from alife_lab.core.hashing import sha256_bytes

        if sha256_bytes(data) != parent["sha256"]:
            raise Conflict("parent checkpoint checksum mismatch", "checksum_mismatch")
        tick = int(parent["simulation_clock"])
        receipt = self.store.publish_bytes(
            run_id,
            f"checkpoints/fork_origin_t{tick:012d}.npz",
            data,
            kind="checkpoint",
            mime_type="application/x-npz",
        )
        self.db.execute(
            "INSERT INTO checkpoints(id, run_id, job_id, epoch, simulation_clock, clock_unit, "
            "logical_path, storage_uri, sha256, size_bytes, reason, created_at, spec_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "ckpt_" + secrets.token_hex(8),
                run_id,
                "fork",
                0,
                tick,
                "tick",
                receipt.logical_path,
                receipt.storage_uri,
                receipt.sha256,
                receipt.size_bytes,
                "fork_origin",
                time.time(),
                parent["spec_hash"] or lineage["parent_spec_sha256"],
            ),
        )
        self.db.execute("UPDATE runs SET tick=? WHERE id=?", (tick, run_id))
        self._write_artifact_index(run_id)

    def fork_run(
        self,
        parent_run_id: str,
        checkpoint_id: str = "latest",
        label: str | None = None,
        spec_overrides: dict[str, Any] | None = None,
        intervention: dict[str, Any] | None = None,
        profile_id: str | None = None,
        ticks_per_second: float | None = None,
        author: str = "local_user",
    ) -> dict[str, Any]:
        """Create a child run whose initial state is a parent checkpoint (pending state)."""
        parent = self._run_row(parent_run_id)
        checkpoint = self._resolve_checkpoint(parent_run_id, checkpoint_id)
        exp = self.db.one(
            "SELECT spec_json, spec_hash FROM experiments WHERE id=?", (parent["experiment_id"],)
        )
        assert exp is not None
        spec_json = json.loads(exp["spec_json"])
        if spec_overrides:
            allowed = {"name", "mode", "runner", "dynamics", "observations"}
            unknown = set(spec_overrides) - allowed
            if unknown:
                raise ControllerError(
                    f"fork cannot change {sorted(unknown)}; world, substrate, seed and "
                    "initialization are fixed by the parent state"
                )
            body = spec_json["experiment"]
            for key, value in spec_overrides.items():
                if isinstance(value, dict) and isinstance(body.get(key), dict):
                    body[key] = {**body[key], **value}
                else:
                    body[key] = value
        try:
            spec = ExperimentSpec.model_validate(spec_json)
        except ValueError as exc:
            raise ControllerError(f"invalid fork spec: {exc}", "invalid_spec") from exc
        fork_tick = int(checkpoint["simulation_clock"])
        if spec.experiment.runner.max_steps <= fork_tick:
            raise ControllerError(
                f"max_steps ({spec.experiment.runner.max_steps}) must exceed the fork tick "
                f"({fork_tick})"
            )
        if label is not None and not (0 < len(label) <= 120):
            raise ControllerError("label must be 1..120 characters")
        if intervention is not None:
            manifest = get_manifest(spec.experiment.substrate)
            if intervention.get("kind") not in manifest.capabilities.intervention_types:
                raise ControllerError(f"unsupported intervention {intervention.get('kind')!r}")
            if not isinstance(intervention.get("payload"), dict):
                raise ControllerError("intervention payload must be an object")
        lineage = {
            "experiment_id": parent["experiment_id"],
            "parent_run_id": parent_run_id,
            "parent_checkpoint_id": checkpoint["id"],
            "parent_checkpoint_sha256": checkpoint["sha256"],
            "parent_spec_sha256": exp["spec_hash"],
            "parent_effective_mode": parent["effective_mode"],
            "fork_tick": fork_tick,
            "parent_checkpoint": checkpoint,
        }
        tps = parent["ticks_per_second"] if ticks_per_second is None else ticks_per_second
        child = self.create_run(
            spec,
            profile_id or parent["profile_id"],
            tps,
            author=author,
            lineage=lineage,
            label=label,
        )
        self._emit_event(
            parent_run_id,
            "run_forked",
            {"child_run_id": child["id"], "checkpoint_id": checkpoint["id"], "label": label},
            clock=fork_tick,
        )
        self._emit_event(
            child["id"],
            "forked_from",
            {
                "parent_run_id": parent_run_id,
                "checkpoint_id": checkpoint["id"],
                "spec_changed": spec.digest() != exp["spec_hash"],
            },
            clock=fork_tick,
        )
        if intervention is not None:
            self._queue_initial_intervention(child["id"], intervention, fork_tick, author)
        return self.run_summary(child["id"])

    def _queue_initial_intervention(
        self, run_id: str, intervention: dict[str, Any], clock: int, author: str
    ) -> None:
        command_id = str(uuid.uuid4())
        payload = {"kind": intervention["kind"], "payload": intervention["payload"]}
        row = self._run_row(run_id)
        exp = self.db.one("SELECT spec_hash FROM experiments WHERE id=?", (row["experiment_id"],))
        assert exp is not None
        origin = self.latest_checkpoint(run_id)
        now = time.time()
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO commands(command_id, run_id, kind, payload_json, status, "
                "requested_clock, created_at) VALUES (?,?,?,?,?,?,?)",
                (command_id, run_id, "intervene", canonical_json(payload), "initial", clock, now),
            )
            conn.execute(
                "INSERT INTO interventions(id, run_id, command_id, kind, payload_json, "
                "author, spec_hash, parent_checkpoint_id, requested_clock, epoch, "
                "requested_at, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "intv_" + secrets.token_hex(8),
                    run_id,
                    command_id,
                    str(intervention["kind"]),
                    canonical_json(intervention["payload"]),
                    author,
                    exp["spec_hash"],
                    origin["id"] if origin else None,
                    clock,
                    1,
                    now,
                    "requested",
                ),
            )

    async def paired_fork(
        self,
        parent_run_id: str,
        checkpoint_id: str = "now",
        treatment_intervention: dict[str, Any] | None = None,
        name: str | None = None,
        ticks_per_second: float | None = None,
        start: bool = True,
        start_paused: bool = True,
        author: str = "local_user",
    ) -> dict[str, Any]:
        """Pause-free one-shot: checkpoint (if live) -> two forks -> comparison record."""
        if checkpoint_id == "now":
            job = self.jobs.get(parent_run_id)
            if job is not None and not job.finished:
                result = await self.control(parent_run_id, "checkpoint", author=author)
                if result.get("status") != "applied":
                    raise Conflict("could not checkpoint the parent run", "checkpoint_failed")
                receipt = (result.get("ack") or {}).get("result") or {}
                row = self.db.one(
                    "SELECT id FROM checkpoints WHERE run_id=? AND logical_path=?",
                    (parent_run_id, receipt.get("logical_path")),
                )
                if row is None:
                    raise Conflict("checkpoint was not indexed", "checkpoint_failed")
                checkpoint_id = row["id"]
            else:
                checkpoint_id = "latest"
        checkpoint = self._resolve_checkpoint(parent_run_id, checkpoint_id)
        tick = int(checkpoint["simulation_clock"])
        control = self.fork_run(
            parent_run_id,
            checkpoint["id"],
            label=f"control@t{tick}",
            ticks_per_second=ticks_per_second,
            author=author,
        )
        treatment = self.fork_run(
            parent_run_id,
            checkpoint["id"],
            label=f"treatment@t{tick}",
            intervention=treatment_intervention,
            ticks_per_second=ticks_per_second,
            author=author,
        )
        comparison = self.create_comparison(
            [control["id"], treatment["id"]],
            name=name or f"paired fork of {parent_run_id} at t={tick}",
            design="paired_fork",
            arms=["control", "treatment"],
            parent_run_id=parent_run_id,
            parent_checkpoint_id=checkpoint["id"],
        )
        if start:
            for rid in (control["id"], treatment["id"]):
                await self.start_run(rid, start_paused=start_paused)
        return self.comparison_summary(comparison["id"])

    def create_comparison(
        self,
        run_ids: list[str],
        name: str,
        design: str = "manual",
        arms: list[str] | None = None,
        parent_run_id: str | None = None,
        parent_checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        if not 2 <= len(run_ids) <= 4 or len(set(run_ids)) != len(run_ids):
            raise ControllerError("a comparison needs 2 to 4 distinct runs")
        for rid in run_ids:
            self._run_row(rid)
        arms = arms or [f"arm{i}" for i in range(len(run_ids))]
        comparison_id = "cmp_" + secrets.token_hex(8)
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO comparisons(id, name, design, parent_run_id, parent_checkpoint_id, "
                "created_at) VALUES (?,?,?,?,?,?)",
                (
                    comparison_id,
                    name[:200],
                    design,
                    parent_run_id,
                    parent_checkpoint_id,
                    time.time(),
                ),
            )
            for i, (rid, arm) in enumerate(zip(run_ids, arms, strict=True)):
                conn.execute(
                    "INSERT INTO comparison_members(comparison_id, run_id, arm, position) "
                    "VALUES (?,?,?,?)",
                    (comparison_id, rid, arm, i),
                )
        for rid in run_ids:
            self._emit_event(rid, "comparison_created", {"comparison_id": comparison_id})
        return self.comparison_summary(comparison_id)

    def comparison_summary(self, comparison_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM comparisons WHERE id=?", (comparison_id,))
        if row is None:
            raise NotFound(f"comparison {comparison_id} not found")
        members = self.db.all(
            "SELECT run_id, arm FROM comparison_members WHERE comparison_id=? ORDER BY position",
            (comparison_id,),
        )
        return {
            **dict(row),
            "members": [{"arm": m["arm"], "run": self.run_summary(m["run_id"])} for m in members],
        }

    def list_comparisons(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.all(
            "SELECT * FROM comparisons ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)
        )
        return [dict(r) for r in rows]

    def comparison_report(self, comparison_id: str, max_points: int = 1000) -> dict[str, Any]:
        from alife_lab.service.compare import state_divergence

        summary = self.comparison_summary(comparison_id)
        run_ids = [m["run"]["id"] for m in summary["members"]]

        def query(sql: str, params: Any) -> list[dict[str, Any]]:
            return [dict(r) for r in self.db.all(sql, params)]

        return {
            "comparison": summary,
            "divergence": state_divergence(query, self.store, run_ids),
            "metrics": {rid: self.metrics(rid, max_points=max_points)["rows"] for rid in run_ids},
        }

    def _dispatch_queue(self) -> None:
        while self.queue:
            active = len([j for j in self.jobs.values() if not j.finished])
            if active >= self.settings.max_active_jobs:
                return
            run_id, start_paused = self.queue.popleft()
            row = self._run_row(run_id)
            if row["state"] != RunState.QUEUED.value:
                continue
            try:
                self._launch(row, start_paused)
            except Exception as exc:
                log.exception("launch failed for %s", run_id)
                self._set_state(run_id, RunState.STARTING, force=True)
                self._set_state(run_id, RunState.FAILED, reason="launch_failed", force=True)
                self.db.execute("UPDATE runs SET last_error=? WHERE id=?", (str(exc), run_id))
                self._emit_event(run_id, "run_failed", {"error": str(exc)}, severity="error")

    def _launch(self, row: dict[str, Any], start_paused: bool) -> None:
        run_id = row["id"]
        exp = self.db.one("SELECT spec_json FROM experiments WHERE id=?", (row["experiment_id"],))
        assert exp is not None
        spec = ExperimentSpec.model_validate_json(exp["spec_json"])
        profile = ExecutionProfile.model_validate_json(row["profile_json"])
        latest = self.latest_checkpoint(run_id)
        restore_from = None
        start_tick = 0
        if latest is not None:
            restore_from = {
                "logical_path": latest["logical_path"],
                "sha256": latest["sha256"],
                "expected_spec_digest": latest["spec_hash"],
                "fork_origin": latest["reason"] == "fork_origin",
            }
            start_tick = int(latest["simulation_clock"])
        if latest is not None:
            self._mark_rolled_back(run_id, latest)
        initial_commands = []
        for cmd in self.db.all(
            "SELECT command_id, kind, payload_json FROM commands "
            "WHERE run_id=? AND status='initial' ORDER BY created_at",
            (run_id,),
        ):
            initial_commands.append(
                {
                    "command_id": cmd["command_id"],
                    "kind": cmd["kind"],
                    "payload": json.loads(cmd["payload_json"]),
                }
            )
            self.db.execute(
                "UPDATE commands SET status='sent' WHERE command_id=?", (cmd["command_id"],)
            )
        epoch = int(row["epoch"]) + 1
        job_id = "job_" + secrets.token_hex(8)
        lease = secrets.token_hex(16)
        now = time.time()
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs(id, run_id, epoch, lease_token, status, executor, start_tick, "
                "restored_from_checkpoint_id, created_at, started_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    run_id,
                    epoch,
                    lease,
                    "starting",
                    self.executor.target_id,
                    start_tick,
                    latest["id"] if latest else None,
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE runs SET epoch=?, tick=?, started_at=COALESCE(started_at, ?) WHERE id=?",
                (epoch, start_tick, now, run_id),
            )
        self._set_state(run_id, RunState.STARTING)
        job_spec = WorkerJob(
            run_id=run_id,
            job_id=job_id,
            epoch=epoch,
            lease_token=lease,
            spec=spec.model_dump(mode="json"),
            profile=profile.model_dump(mode="json"),
            runs_root=str(self.settings.runs_root),
            start_paused=start_paused,
            ticks_per_second=float(row["ticks_per_second"]),
            preview_fps=profile.execution.preview.max_fps,
            restore_from=restore_from,
            initial_commands=initial_commands,
        )
        handle = self.executor.submit(job_spec)
        job = ActiveJob(
            job_id=job_id, run_id=run_id, epoch=epoch, lease_token=lease, handle=handle, spec=spec
        )
        pid = getattr(getattr(handle, "process", None), "pid", None)
        if pid is not None:
            self.db.execute("UPDATE jobs SET worker_pid=? WHERE id=?", (pid, job_id))
            with contextlib.suppress(psutil.Error):
                job.process = psutil.Process(pid)
        self.jobs[run_id] = job
        self._emit_event(
            run_id,
            "job_started",
            {
                "job_id": job_id,
                "epoch": epoch,
                "executor": self.executor.target_id,
                "worker_pid": pid,
                "restored_from_checkpoint": latest["id"] if latest else None,
                "start_tick": start_tick,
                "source_sha256": self.environment["source_digest"],
            },
        )
        thread = threading.Thread(
            target=self._reader, args=(job,), name=f"reader-{job_id}", daemon=True
        )
        thread.start()

    def _reader(self, job: ActiveJob) -> None:
        """Blocking reader thread: forwards worker messages to the event loop."""
        assert self.loop is not None
        while True:
            try:
                message = job.handle.conn.recv()
            except (EOFError, OSError):
                break
            try:
                self.loop.call_soon_threadsafe(self._on_message, job, message)
            except RuntimeError:  # event loop already closed during shutdown
                return
            if message.get("type") in ("finished", "error"):
                break
        job.handle.join(timeout=5)
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self._on_worker_exit, job)

    # ------------------------------------------------------------------ worker messages

    def _on_message(self, job: ActiveJob, message: dict[str, Any]) -> None:
        if self._db_closed:
            return
        if message.get("lease_token") != job.lease_token or job.finished:
            log.warning("dropping message with stale lease for %s", job.run_id)
            return
        kind = message.get("type")
        run_id = job.run_id
        now = time.time()
        job.last_heartbeat = now
        if "tick" in message:
            self.db.execute("UPDATE runs SET tick=? WHERE id=?", (int(message["tick"]), run_id))
        if kind == "ready":
            self.db.execute(
                "UPDATE jobs SET status='running', heartbeat_at=? WHERE id=?", (now, job.job_id)
            )
            self._set_state(run_id, RunState(message["state"]))
        elif kind == "state":
            self._set_state(run_id, RunState(message["state"]))
        elif kind == "metrics":
            job.steps_per_second = float(message.get("steps_per_second", 0.0))
            self._broadcast(
                run_id,
                {
                    "type": "metrics",
                    "rows": message["rows"],
                    "steps_per_second": job.steps_per_second,
                },
            )
        elif kind == "preview":
            preview = {
                "type": "preview",
                "run_id": run_id,
                "tick": message["tick"],
                "width": message["width"],
                "height": message["height"],
                "codec": message["codec"],
                "data": message["data"],
            }
            job.latest_preview = preview
            for sub in self.subscribers.get(run_id, set()):
                sub.push_preview(preview)
        elif kind == "heartbeat":
            job.steps_per_second = float(message.get("steps_per_second", 0.0))
            self._sample_resources(job)
            self.db.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?", (now, job.job_id))
            self._broadcast(
                run_id,
                {
                    "type": "heartbeat",
                    "clock": {"unit": "tick", "value": message["tick"]},
                    "steps_per_second": job.steps_per_second,
                    "worker_rss_mb": job.last_rss_mb,
                    "worker_cpu_percent": job.cpu_percent,
                },
            )
        elif kind == "checkpoint":
            self._on_checkpoint(job, message)
        elif kind == "ack":
            self._on_ack(job, message)
        elif kind == "finished":
            reason = message["reason"]
            target = {
                "completed": RunState.COMPLETED,
                "cancelled": RunState.CANCELLED,
                "interrupted": RunState.INTERRUPTED,
                "disk_limit": RunState.FAILED,
            }.get(reason, RunState.FAILED)
            self._finish_job(job, target, reason, message.get("detail"))
        elif kind == "error":
            self._finish_job(job, RunState.FAILED, "worker_error", message.get("error"))
            self._emit_event(
                run_id,
                "worker_error",
                {"error": message.get("error"), "traceback": message.get("traceback")},
                severity="error",
            )

    def _sample_resources(self, job: ActiveJob) -> None:
        if job.process is None:
            return
        try:
            rss = job.process.memory_info().rss / 2**20
            job.cpu_percent = job.process.cpu_percent(interval=None)
        except psutil.Error:
            return
        job.last_rss_mb = round(rss, 1)
        job.peak_rss_mb = max(job.peak_rss_mb, rss)

    def _on_checkpoint(self, job: ActiveJob, message: dict[str, Any]) -> None:
        receipt = message["receipt"]
        if not self.store.verify(job.run_id, receipt["logical_path"], receipt["sha256"]):
            self._emit_event(
                job.run_id,
                "checkpoint_rejected",
                {"logical_path": receipt["logical_path"], "reason": "checksum verification failed"},
                severity="error",
            )
            return
        checkpoint_id = "ckpt_" + secrets.token_hex(8)
        self.db.execute(
            "INSERT INTO checkpoints(id, run_id, job_id, epoch, simulation_clock, clock_unit, "
            "logical_path, storage_uri, sha256, size_bytes, reason, created_at, spec_hash) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                checkpoint_id,
                job.run_id,
                job.job_id,
                job.epoch,
                int(message["tick"]),
                "tick",
                receipt["logical_path"],
                receipt["storage_uri"],
                receipt["sha256"],
                receipt["size_bytes"],
                message["reason"],
                time.time(),
                job.spec.digest(),
            ),
        )
        self._write_artifact_index(job.run_id)
        payload = {
            "checkpoint_id": checkpoint_id,
            "reason": message["reason"],
            "sha256": receipt["sha256"],
            "size_bytes": receipt["size_bytes"],
            "logical_path": receipt["logical_path"],
        }
        self._emit_event(job.run_id, "checkpoint_ready", payload, clock=int(message["tick"]))

    def _on_ack(self, job: ActiveJob, message: dict[str, Any]) -> None:
        command_id = message.get("command_id")
        if command_id is None:
            return
        ack = {
            "command_id": command_id,
            "kind": message.get("kind"),
            "status": message["status"],
            "applied_clock": message.get("applied_clock"),
            "clock_unit": "tick",
            "result": message.get("result"),
            "error": message.get("error"),
        }
        self.db.execute(
            "UPDATE commands SET status=?, ack_json=? WHERE command_id=?",
            (message["status"], canonical_json(ack), command_id),
        )
        if message.get("kind") == "intervene":
            self._on_intervention_ack(job, ack)
        elif message.get("kind") == "set_speed" and message["status"] == "applied":
            result_obj: dict[str, Any] = message.get("result") or {}
            tps = result_obj.get("ticks_per_second")
            if tps is not None:
                self.db.execute("UPDATE runs SET ticks_per_second=? WHERE id=?", (tps, job.run_id))
        if message["status"] == "applied" and message.get("kind") in ("pause", "play"):
            self._emit_event(
                job.run_id,
                f"{message['kind']}_applied",
                {"command_id": command_id},
                clock=message.get("applied_clock"),
            )
        self._broadcast(job.run_id, {"type": "command_ack", **ack})
        future = job.pending.pop(command_id, None)
        if future is not None and not future.done():
            future.set_result(ack)

    def _on_intervention_ack(self, job: ActiveJob, ack: dict[str, Any]) -> None:
        applied = ack["status"] == "applied"
        self.db.execute(
            "UPDATE interventions SET status=?, applied_clock=?, applied_at=?, result_json=? "
            "WHERE command_id=?",
            (
                ack["status"],
                ack["applied_clock"] if applied else None,
                time.time() if applied else None,
                canonical_json(ack["result"] if applied else {"error": ack["error"]}),
                ack["command_id"],
            ),
        )
        if not applied:
            return
        row = self._run_row(job.run_id)
        self.db.execute(
            "UPDATE runs SET intervened=1, revision=revision+1 WHERE id=?", (job.run_id,)
        )
        intervention = self.db.one(
            "SELECT id, kind, payload_json FROM interventions WHERE command_id=?",
            (ack["command_id"],),
        )
        assert intervention is not None
        self._emit_event(
            job.run_id,
            "intervention_applied",
            {
                "intervention_id": intervention["id"],
                "kind": intervention["kind"],
                "command_id": ack["command_id"],
                "result": ack["result"],
            },
            clock=ack["applied_clock"],
        )
        if row["effective_mode"] == "reference":
            self.db.execute(
                "UPDATE runs SET effective_mode='exploratory' WHERE id=?", (job.run_id,)
            )
            self._emit_event(
                job.run_id,
                "mode_changed",
                {
                    "from": "reference",
                    "to": "exploratory",
                    "reason": "manual intervention applied to a reference-mode run",
                },
                clock=ack["applied_clock"],
                severity="warning",
            )

    def _on_worker_exit(self, job: ActiveJob) -> None:
        if self._db_closed:
            return
        code = job.handle.exitcode()
        if code is not None:
            self.db.execute("UPDATE jobs SET exit_code=? WHERE id=?", (code, job.job_id))
        if not job.finished:
            code = job.handle.exitcode()
            self._finish_job(
                job,
                RunState.INTERRUPTED,
                "worker_lost",
                f"worker exited without a final message (exit code {code})",
            )

    def _finish_job(
        self, job: ActiveJob, target: RunState, reason: str, detail: str | None
    ) -> None:
        if job.finished:
            return
        job.finished = True
        now = time.time()
        self._sample_resources(job)
        code = job.handle.exitcode()
        status = {"completed": "completed", "cancelled": "cancelled"}.get(reason, "ended")
        self.db.execute(
            "UPDATE jobs SET status=?, finished_at=?, exit_code=?, error_json=? WHERE id=?",
            (
                status if target != RunState.FAILED else "failed",
                now,
                code,
                canonical_json({"reason": reason, "detail": detail}),
                job.job_id,
            ),
        )
        self.db.execute(
            "UPDATE commands SET status='aborted' WHERE run_id=? AND status='sent'", (job.run_id,)
        )
        for future in job.pending.values():
            if not future.done():
                future.set_result({"status": "aborted", "error": f"job ended: {reason}"})
        job.pending.clear()
        self.db.execute(
            "UPDATE runs SET finished_at=?, finish_reason=?, last_error=? WHERE id=?",
            (now, reason, detail if target == RunState.FAILED else None, job.run_id),
        )
        self._set_state(job.run_id, target, reason=reason, force=True)
        self._emit_event(
            job.run_id,
            "job_finished",
            {
                "job_id": job.job_id,
                "reason": reason,
                "detail": detail,
                "exit_code": code,
                "wall_seconds": round(now - job.started_at, 3),
                "worker_peak_rss_mb": round(job.peak_rss_mb, 1) if job.process else None,
            },
            severity="info" if target != RunState.FAILED else "error",
        )
        job.done.set()
        self._dispatch_queue()

    # ------------------------------------------------------------------ control

    async def control(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any] | None = None,
        command_id: str | None = None,
        expected_revision: int | None = None,
        requested_clock: int | None = None,
        author: str = "local_user",
        wait: bool = True,
    ) -> dict[str, Any]:
        if kind not in CONTROL_KINDS:
            raise ControllerError(f"unknown control kind {kind!r}")
        payload = payload or {}
        command_id = command_id or str(uuid.uuid4())
        try:
            uuid.UUID(command_id)
        except ValueError as exc:
            raise ControllerError("command_id must be a UUID") from exc
        existing = self.db.one("SELECT * FROM commands WHERE command_id=?", (command_id,))
        if existing is not None:
            if existing["run_id"] != run_id or existing["kind"] != kind:
                raise Conflict(
                    "command_id already used for a different command", "duplicate_command"
                )
            prior_ack = json.loads(existing["ack_json"]) if existing["ack_json"] else None
            return {
                "command_id": command_id,
                "duplicate": True,
                "status": existing["status"],
                "ack": prior_ack,
            }
        row = self._run_row(run_id)
        if expected_revision is not None and expected_revision != row["revision"]:
            raise Conflict(
                f"run revision is {row['revision']}, expected {expected_revision}", "stale_revision"
            )
        state = RunState(row["state"])
        job = self.jobs.get(run_id)
        if job is None or job.finished or state not in ACTIVE_STATES:
            raise Conflict(f"run is {state.value}; no live worker accepts commands", "not_live")
        if kind == "intervene":
            self._validate_intervention(job, payload)
        if kind == "pause":
            self._set_state(run_id, RunState.PAUSING)
        elif kind == "play":
            if state not in (RunState.PAUSED, RunState.PAUSING):
                raise Conflict(f"run is {state.value}; play requires paused")
            self._set_state(run_id, RunState.RESUMING)
        elif kind == "step" and state != RunState.PAUSED:
            raise Conflict(f"run is {state.value}; step requires paused")
        now = time.time()
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO commands(command_id, run_id, kind, payload_json, status, "
                "requested_clock, created_at) VALUES (?,?,?,?,?,?,?)",
                (command_id, run_id, kind, canonical_json(payload), "sent", requested_clock, now),
            )
            if kind == "intervene":
                exp = conn.execute(
                    "SELECT spec_hash FROM experiments WHERE id=?", (row["experiment_id"],)
                ).fetchone()
                latest = self.latest_checkpoint(run_id)
                conn.execute(
                    "INSERT INTO interventions(id, run_id, command_id, kind, payload_json, "
                    "author, spec_hash, parent_checkpoint_id, requested_clock, epoch, "
                    "requested_at, status) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "intv_" + secrets.token_hex(8),
                        run_id,
                        command_id,
                        str(payload.get("kind")),
                        canonical_json(payload.get("payload") or {}),
                        author,
                        exp["spec_hash"],
                        latest["id"] if latest else None,
                        requested_clock,
                        job.epoch,
                        now,
                        "requested",
                    ),
                )
        assert self.loop is not None
        future: asyncio.Future[dict[str, Any]] = self.loop.create_future()
        job.pending[command_id] = future
        try:
            job.handle.conn.send({"command_id": command_id, "kind": kind, "payload": payload})
        except (OSError, BrokenPipeError) as exc:
            job.pending.pop(command_id, None)
            self.db.execute("UPDATE commands SET status='failed' WHERE command_id=?", (command_id,))
            raise Conflict(f"worker channel closed: {exc}", "not_live") from exc
        if not wait:
            return {"command_id": command_id, "status": "sent", "ack": None}
        try:
            ack: dict[str, Any] = await asyncio.wait_for(
                asyncio.shield(future), ACK_TIMEOUT_SECONDS
            )
        except TimeoutError:
            return {"command_id": command_id, "status": "pending", "ack": None}
        return {"command_id": command_id, "status": ack.get("status"), "ack": ack}

    def _validate_intervention(self, job: ActiveJob, payload: dict[str, Any]) -> None:
        manifest = get_manifest(job.spec.experiment.substrate)
        kind = payload.get("kind")
        if not manifest.capabilities.supports_interventions:
            raise ControllerError("substrate does not support interventions")
        if kind not in manifest.capabilities.intervention_types:
            raise ControllerError(
                f"intervention kind {kind!r} not supported; allowed: "
                f"{list(manifest.capabilities.intervention_types)}"
            )
        if not isinstance(payload.get("payload"), dict):
            raise ControllerError("intervention payload must be an object")
        if len(canonical_json(payload)) > 1_000_000:
            raise ControllerError("intervention payload too large")

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        row = self._run_row(run_id)
        state = RunState(row["state"])
        if state in TERMINAL_STATES:
            raise Conflict(f"run is already {state.value}")
        job = self.jobs.get(run_id)
        if job is not None and not job.finished:
            await self.control(run_id, "cancel")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(job.done.wait(), 10)
        else:
            self.queue = deque(item for item in self.queue if item[0] != run_id)
            self._set_state(run_id, RunState.CANCELLED, reason="user_cancel", force=True)
            self.db.execute(
                "UPDATE runs SET finished_at=?, finish_reason='cancelled' WHERE id=?",
                (time.time(), run_id),
            )
        return self.run_summary(run_id)

    # ------------------------------------------------------------------ subscriptions

    def subscribe(self, run_id: str) -> Subscriber:
        self._run_row(run_id)
        sub = Subscriber(run_id=run_id)
        self.subscribers.setdefault(run_id, set()).add(sub)
        job = self.jobs.get(run_id)
        if job is not None and job.latest_preview is not None:
            sub.push_preview(job.latest_preview)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        self.subscribers.get(sub.run_id, set()).discard(sub)

    def _broadcast(self, run_id: str, message: dict[str, Any]) -> None:
        message.setdefault("run_id", run_id)
        message.setdefault("timestamp", time.time())
        for sub in self.subscribers.get(run_id, set()):
            sub.push(message)

    # ------------------------------------------------------------------ idempotency

    def idempotency_lookup(self, key: str, scope: str, request_hash: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM idempotency_keys WHERE key=?", (key,))
        if row is None:
            return None
        if row["scope"] != scope or row["request_hash"] != request_hash:
            raise Conflict(
                "Idempotency-Key reused with a different request", "idempotency_mismatch"
            )
        result: dict[str, Any] = json.loads(row["response_json"])
        return result

    def idempotency_store(
        self, key: str, scope: str, request_hash: str, response: dict[str, Any]
    ) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO idempotency_keys(key, scope, request_hash, response_json, "
            "created_at) VALUES (?,?,?,?,?)",
            (key, scope, request_hash, canonical_json(response), time.time()),
        )

    # ------------------------------------------------------------------ helpers

    def _set_state(
        self, run_id: str, target: RunState, reason: str | None = None, force: bool = False
    ) -> None:
        row = self._run_row(run_id)
        current = RunState(row["state"])
        if current == target:
            return
        if not force:
            try:
                check_transition(current, target)
            except InvalidTransition as exc:
                raise Conflict(str(exc), "invalid_transition") from exc
        self.db.execute(
            "UPDATE runs SET state=?, revision=revision+1 WHERE id=?", (target.value, run_id)
        )
        self._broadcast(
            run_id,
            {
                "type": "run_state",
                "state": target.value,
                "previous": current.value,
                "reason": reason,
                "revision": row["revision"] + 1,
                "clock": {"unit": "tick", "value": row["tick"]},
            },
        )

    def _emit_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        clock: int | None = None,
        severity: str = "info",
        source: str = "controller",
    ) -> dict[str, Any]:
        row = self._run_row(run_id)
        clock_value = row["tick"] if clock is None else clock
        now = time.time()
        with self.db.transaction() as conn:
            seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id=?", (run_id,)
            ).fetchone()
            seq = int(seq_row[0])
            conn.execute(
                "INSERT INTO events(run_id, seq, clock_unit, clock_value, observed_at_utc, source, "
                "type, severity, schema_version, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    seq,
                    "tick",
                    clock_value,
                    now,
                    source,
                    event_type,
                    severity,
                    1,
                    canonical_json(payload),
                ),
            )
        envelope = {
            "run_id": run_id,
            "seq": seq,
            "clock_unit": "tick",
            "clock_value": clock_value,
            "observed_at_utc": now,
            "source": source,
            "type": event_type,
            "severity": severity,
            "schema_version": 1,
            "payload": payload,
        }
        with contextlib.suppress(OSError):
            path = self.store.resolve(run_id, "logs/events.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(envelope, sort_keys=True) + "\n")
        self._broadcast(run_id, {"type": "event", "event": envelope})
        return envelope

    def _record_artifact(self, receipt: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO artifacts(id, run_id, kind, logical_path, storage_uri, mime_type, "
            "size_bytes, sha256, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "art_" + secrets.token_hex(8),
                receipt["run_id"],
                receipt["kind"],
                receipt["logical_path"],
                receipt["storage_uri"],
                receipt["mime_type"],
                receipt["size_bytes"],
                receipt["sha256"],
                time.time(),
            ),
        )

    def _write_artifact_index(self, run_id: str) -> None:
        artifacts = [
            dict(r) for r in self.db.all("SELECT * FROM artifacts WHERE run_id=?", (run_id,))
        ]
        checkpoints = self.list_checkpoints(run_id)
        self.store.write_json_atomic(
            run_id,
            "artifacts/index.json",
            {"run_id": run_id, "artifacts": artifacts, "checkpoints": checkpoints},
        )


def _acquire_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ControllerLocked(
            f"another ALife controller is running (lock {path}); stop it or use its API"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _release_lock(handle: Any) -> None:
    with contextlib.suppress(OSError):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
