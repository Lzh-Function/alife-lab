"""Simulation worker loop.

Runs inside a separate process (LocalExecutor) or a thread (FakeExecutor).
It owns the substrate state and is the only code that mutates it. Commands
from the controller are applied only at tick boundaries (the documented safe
point). The worker never touches the SQLite database: it writes run-scoped
data-plane files (metrics history, checkpoints, worker log) and reports
receipts to the controller over the connection.

Message envelopes are plain dicts of primitive values and bytes; the channel is
a private, local inter-process pipe between trusted processes of this program.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from alife_lab.artifacts.store import (
    ArtifactError,
    LocalArtifactStore,
    decode_checkpoint,
    encode_checkpoint,
)
from alife_lab.core.hashing import sha256_bytes
from alife_lab.core.spec import ExecutionProfile, ExperimentSpec
from alife_lab.storage.metrics_store import MetricsWriter
from alife_lab.substrates import get_adapter

MAX_STEP_PER_COMMAND = 10_000
METRIC_FLUSH_SECONDS = 0.25
HEARTBEAT_SECONDS = 1.0
UNLIMITED_TICK_BUDGET_SECONDS = 0.02


class Connection(Protocol):
    def send(self, obj: Any) -> None: ...
    def recv(self) -> Any: ...
    def poll(self, timeout: float | None = ...) -> bool: ...


@dataclass
class WorkerJob:
    run_id: str
    job_id: str
    epoch: int
    lease_token: str
    spec: dict[str, Any]
    profile: dict[str, Any]
    runs_root: str
    start_paused: bool
    ticks_per_second: float
    preview_fps: float
    # {"logical_path", "sha256", optional "expected_spec_digest", optional "fork_origin"}
    restore_from: dict[str, Any] | None = None
    # Commands applied before the first tick (e.g. the intervention of a fork arm).
    initial_commands: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _json_safe(value: float) -> float | None:
    return None if math.isnan(value) or math.isinf(value) else value


class _Worker:
    def __init__(self, job: WorkerJob, conn: Connection) -> None:
        self.job = job
        self.conn = conn
        self.spec = ExperimentSpec.model_validate(job.spec)
        self.profile = ExecutionProfile.model_validate(job.profile)
        self.spec_digest = self.spec.digest()
        self.adapter = get_adapter(self.spec.experiment.substrate)
        self.store = LocalArtifactStore(Path(job.runs_root))
        self.run_dir = self.store.run_dir(job.run_id)
        self.running = not job.start_paused
        self.ticks_per_second = job.ticks_per_second
        self.preview_interval = 1.0 / max(job.preview_fps, 0.1)
        self.metric_buffer: list[dict[str, Any]] = []
        self.last_checkpoint_tick: int | None = None
        self.started_monotonic = time.monotonic()
        self.limits = self.profile.execution.limits
        self.dirty_preview = True
        self.ticks_since_rate = 0
        self.rate_window_start = time.monotonic()
        self.steps_per_second = 0.0
        (self.run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.metrics = MetricsWriter(self.store, job.run_id, job.epoch)
        self.log_file = (self.run_dir / "logs" / f"worker_e{job.epoch:03d}.jsonl").open(
            "a", encoding="utf-8"
        )

    # ------------------------------------------------------------------ io

    def send(self, message: dict[str, Any]) -> None:
        message["job_id"] = self.job.job_id
        message["lease_token"] = self.job.lease_token
        message["run_id"] = self.job.run_id
        self.conn.send(message)

    def log(self, level: str, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "level": level, "event": event, "pid": os.getpid(), **fields}
        self.log_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.log_file.flush()

    # --------------------------------------------------------------- state

    def init_state(self) -> None:
        if self.job.restore_from is None:
            self.state = self.adapter.initialize(self.spec)
            self.record_metrics()
            self.checkpoint("initial")
            return
        logical = self.job.restore_from["logical_path"]
        expected = self.job.restore_from["sha256"]
        path = self.store.resolve(self.job.run_id, logical)
        data = path.read_bytes()
        if sha256_bytes(data) != expected:
            raise ArtifactError(f"checkpoint {logical} failed checksum verification")
        payload = decode_checkpoint(data)
        meta = payload["meta"]
        expected_spec = self.job.restore_from.get("expected_spec_digest") or self.spec_digest
        if meta.get("spec_digest") != expected_spec:
            raise ArtifactError("checkpoint belongs to a different scientific spec")
        self.state = self.adapter.deserialize_state(payload, self.spec)
        self.last_checkpoint_tick = int(meta["tick"])
        self.log("info", "restored", checkpoint=logical, tick=self.state.tick)
        if self.job.restore_from.get("fork_origin"):
            # A fork is a new run: its metric history starts at the fork tick.
            self.record_metrics()

    @property
    def tick(self) -> int:
        return int(self.state.tick)

    def record_metrics(self) -> None:
        values = self.adapter.metrics(self.state, self.spec)
        row = {
            "tick": self.tick,
            "epoch": self.job.epoch,
            "values": {k: _json_safe(v) for k, v in values.items()},
        }
        self.metric_buffer.append(row)

    def flush_metrics(self) -> None:
        if not self.metric_buffer:
            return
        rows, self.metric_buffer = self.metric_buffer, []
        self.metrics.append(rows)
        self.send(
            {"type": "metrics", "rows": rows, "steps_per_second": round(self.steps_per_second, 2)}
        )
        self.compact_metrics(force=False)

    def compact_metrics(self, force: bool) -> None:
        receipt = self.metrics.compact(force=force)
        if receipt is not None:
            self.send({"type": "metrics_chunk", "receipt": receipt.to_json()})

    def send_preview(self) -> None:
        header, data = self.adapter.render(self.state)
        self.send({"type": "preview", "tick": self.tick, **header, "data": data})
        self.dirty_preview = False

    def checkpoint(self, reason: str) -> dict[str, Any] | None:
        if self.last_checkpoint_tick == self.tick and reason not in ("manual",):
            return None
        usage_mb = self.store.disk_usage_bytes(self.job.run_id) / 1e6
        if usage_mb >= self.limits.max_disk_mb:
            raise _Stop("disk_limit", f"run disk usage {usage_mb:.1f} MB >= limit")
        self.flush_metrics()
        self.metrics.sync()
        serialized = self.adapter.serialize_state(self.state)
        meta = {
            **serialized["meta"],
            "run_id": self.job.run_id,
            "job_id": self.job.job_id,
            "epoch": self.job.epoch,
            "clock_unit": self.adapter.manifest.capabilities.clock_unit,
            "spec_digest": self.spec_digest,
            "substrate": self.spec.experiment.substrate,
            "backend": "numpy",
            "backend_version": np.__version__,
            "python": platform.python_version(),
            "created_at_unix": time.time(),
            "reason": reason,
        }
        data = encode_checkpoint(serialized["arrays"], meta)
        base = f"checkpoints/e{self.job.epoch:03d}_t{self.tick:012d}"
        logical = f"{base}.npz"
        suffix = 1
        while self.store.resolve(self.job.run_id, logical).exists():
            # Same tick, different state (e.g. after an intervention) or a repeated manual save.
            logical = f"{base}_n{suffix}.npz"
            suffix += 1
        receipt = self.store.publish_bytes(
            self.job.run_id, logical, data, kind="checkpoint", mime_type="application/x-npz"
        )
        self.last_checkpoint_tick = self.tick
        message = {
            "type": "checkpoint",
            "tick": self.tick,
            "epoch": self.job.epoch,
            "reason": reason,
            "receipt": receipt.to_json(),
        }
        self.send(message)
        self.log("info", "checkpoint", tick=self.tick, reason=reason, sha256=receipt.sha256)
        return message

    def advance(self) -> None:
        self.state = self.adapter.step(self.state, self.spec)
        self.record_metrics()
        self.dirty_preview = True
        self.ticks_since_rate += 1
        interval = self.spec.experiment.runner.checkpoint_interval
        if self.tick % interval == 0:
            self.checkpoint("periodic")
        if self.tick >= self.spec.experiment.runner.max_steps:
            self.checkpoint("final")
            raise _Stop("completed", "max_steps reached")

    # ------------------------------------------------------------ commands

    def ack(self, command: dict[str, Any], status: str, **extra: Any) -> None:
        self.send(
            {
                "type": "ack",
                "command_id": command.get("command_id"),
                "kind": command.get("kind"),
                "status": status,
                "applied_clock": self.tick,
                "clock_unit": self.adapter.manifest.capabilities.clock_unit,
                **extra,
            }
        )

    def handle(self, command: dict[str, Any]) -> None:
        kind = command.get("kind")
        payload = command.get("payload") or {}
        try:
            if kind == "play":
                self.running = True
                self.ack(command, "applied")
                self.send({"type": "state", "state": "running", "tick": self.tick})
            elif kind == "pause":
                self.running = False
                self.flush_metrics()
                self.ack(command, "applied")
                self.send({"type": "state", "state": "paused", "tick": self.tick})
                self.send_preview()
            elif kind == "step":
                if self.running:
                    self.ack(command, "rejected", error="step requires the run to be paused")
                    return
                count = int(payload.get("count", 1))
                if not 1 <= count <= MAX_STEP_PER_COMMAND:
                    raise ValueError(f"count must be within 1..{MAX_STEP_PER_COMMAND}")
                start = self.tick
                try:
                    for _ in range(count):
                        self.advance()
                finally:
                    self.flush_metrics()
                    self.send_preview()
                self.ack(command, "applied", result={"from_tick": start, "to_tick": self.tick})
            elif kind == "intervene":
                intervention_kind = str(payload.get("kind"))
                self.state, result = self.adapter.apply_intervention(
                    self.state, self.spec, intervention_kind, dict(payload.get("payload") or {})
                )
                # The state now differs from any checkpoint at this tick.
                self.last_checkpoint_tick = None
                # Re-record the metrics row for the current tick: post-intervention state.
                self.record_metrics()
                self.metric_buffer[-1]["post_intervention"] = True
                self.flush_metrics()
                self.send_preview()
                self.ack(command, "applied", result=result)
            elif kind == "checkpoint":
                message = self.checkpoint("manual")
                self.ack(command, "applied", result=message["receipt"] if message else None)
            elif kind == "inspect":
                # Read-only query; never changes state, metrics or checkpoints.
                self.ack(
                    command, "applied", result=self.adapter.inspect(self.state, self.spec, payload)
                )
            elif kind == "set_speed":
                tps = float(payload.get("ticks_per_second", self.ticks_per_second))
                if not (tps == 0 or 0.1 <= tps <= 10_000):
                    raise ValueError("ticks_per_second must be 0 (unlimited) or within 0.1..10000")
                self.ticks_per_second = tps
                self.ack(command, "applied", result={"ticks_per_second": tps})
            elif kind == "cancel":
                self.ack(command, "applied")
                raise _Stop("cancelled", "cancel requested")
            elif kind == "shutdown":
                self.checkpoint("shutdown")
                self.ack(command, "applied")
                raise _Stop("interrupted", "controller shutdown")
            else:
                self.ack(command, "rejected", error=f"unknown command kind {kind!r}")
        except _Stop:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            self.ack(command, "rejected", error=str(exc))

    # ---------------------------------------------------------------- loop

    def run(self) -> None:
        self.init_state()
        for command in self.job.initial_commands:
            self.handle(command)
        self.send(
            {"type": "ready", "tick": self.tick, "state": "running" if self.running else "paused"}
        )
        self.send_preview()
        last_flush = last_preview = last_heartbeat = time.monotonic()
        next_tick_due = time.monotonic()
        while True:
            now = time.monotonic()
            if now - self.started_monotonic > self.limits.max_walltime_seconds:
                self.checkpoint("walltime")
                raise _Stop("interrupted", "max_walltime_seconds reached")
            if self.running:
                if self.ticks_per_second == 0:
                    budget_end = now + UNLIMITED_TICK_BUDGET_SECONDS
                    while time.monotonic() < budget_end and self.running:
                        self.advance()
                        if self.conn.poll(0):
                            self.handle(self.conn.recv())
                    timeout = 0.0
                else:
                    period = 1.0 / self.ticks_per_second
                    ticks_done = 0
                    while self.running and time.monotonic() >= next_tick_due and ticks_done < 1000:
                        self.advance()
                        ticks_done += 1
                        next_tick_due += period
                        if self.conn.poll(0):
                            self.handle(self.conn.recv())
                    if time.monotonic() - next_tick_due > 1.0:
                        next_tick_due = time.monotonic()  # do not accumulate unbounded debt
                    timeout = max(0.0, min(next_tick_due - time.monotonic(), 0.05))
            else:
                timeout = 0.05
                next_tick_due = time.monotonic()
            if self.conn.poll(timeout):
                self.handle(self.conn.recv())
                while self.conn.poll(0):
                    self.handle(self.conn.recv())
            now = time.monotonic()
            if now - self.rate_window_start >= 1.0:
                self.steps_per_second = self.ticks_since_rate / (now - self.rate_window_start)
                self.ticks_since_rate = 0
                self.rate_window_start = now
            if now - last_flush >= METRIC_FLUSH_SECONDS:
                self.flush_metrics()
                last_flush = now
            if self.dirty_preview and now - last_preview >= self.preview_interval:
                self.send_preview()
                last_preview = now
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                self.send(
                    {
                        "type": "heartbeat",
                        "tick": self.tick,
                        "running": self.running,
                        "steps_per_second": round(self.steps_per_second, 2),
                    }
                )
                last_heartbeat = now

    def close(self) -> None:
        try:
            self.compact_metrics(force=True)
        except (OSError, ValueError, BrokenPipeError, EOFError) as exc:
            self.log("warning", "final_metrics_compaction_failed", error=str(exc))
        with contextlib.suppress(OSError, ValueError):
            self.metrics.close()
        for handle in (self.log_file,):
            try:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            except (OSError, ValueError):
                pass


class _Stop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def worker_main(job: WorkerJob, conn: Connection) -> None:
    """Entry point for both process and thread workers."""
    worker: _Worker | None = None
    try:
        worker = _Worker(job, conn)
        worker.run()
    except _Stop as stop:
        assert worker is not None
        try:
            worker.flush_metrics()
            # Final chunk must be reported before "finished" (later messages are ignored).
            worker.compact_metrics(force=True)
            worker.send_preview()
        except Exception as exc:  # best effort before the final message
            worker.log("warning", "final_flush_failed", error=str(exc))
        worker.log("info", "finished", reason=stop.reason, detail=stop.detail, tick=worker.tick)
        worker.send(
            {"type": "finished", "reason": stop.reason, "detail": stop.detail, "tick": worker.tick}
        )
    except Exception as exc:
        message = {
            "type": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
            "job_id": job.job_id,
            "lease_token": job.lease_token,
            "run_id": job.run_id,
        }
        if worker is not None:
            worker.log("error", "worker_error", error=message["error"])
        with contextlib.suppress(OSError, EOFError, BrokenPipeError):
            conn.send(message)
    finally:
        if worker is not None:
            worker.close()
