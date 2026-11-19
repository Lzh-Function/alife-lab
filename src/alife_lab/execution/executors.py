"""Executor boundary (spec section 2.7).

``LocalExecutor`` (spawned worker processes) and ``FakeExecutor`` (in-process
threads, for tests) are implemented. Remote / multi-GPU / cluster executors are
*not* implemented; their roles are rejected by validation with an explicit
reason instead of silently falling back to local execution.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import threading
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from typing import Any, Literal, Protocol

from alife_lab.core.spec import ExecutionProfile, ExperimentSpec
from alife_lab.execution.probe import probe_gpus
from alife_lab.execution.worker import WorkerJob, worker_main
from alife_lab.substrates import get_manifest, has_substrate

IMPLEMENTED_ROLES = ("cpu_dev", "local_interactive")


GPU_BACKENDS = frozenset({"cupy", "jax_cuda"})


def available_backends(gpu_count: int = 0) -> tuple[str, ...]:
    """Backends usable by workers (checked without importing them: no CUDA in the API)."""
    out = ["numpy"]
    if importlib.util.find_spec("numba") is not None:
        out.append("numba")
    if importlib.util.find_spec("jax") is not None:
        out.append("jax")  # CPU platform (forced in the NCA model module)
    if gpu_count > 0 and importlib.util.find_spec("cupy") is not None:
        out.append("cupy")
    if gpu_count > 0 and importlib.util.find_spec("jax_cuda13_plugin") is not None:
        out.append("jax_cuda")
    return tuple(out)


PLANNED_ROLES = ("remote_compute", "multi_gpu", "cluster_batch")


@dataclass(frozen=True)
class TargetCapabilities:
    pause_live: bool
    checkpoint_on_signal: bool
    interactive_preview: bool
    remote_cancel: bool
    multi_gpu: bool
    backends: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionTarget:
    target_id: str
    kind: Literal["local", "fake"]
    roles: tuple[str, ...]
    gpu_count: int
    capabilities: TargetCapabilities
    probe: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    resolved_backend: str | None
    target_id: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class JobHandle(Protocol):
    conn: Connection

    def is_alive(self) -> bool: ...
    def exitcode(self) -> int | None: ...
    def terminate(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...


class Executor(Protocol):
    target_id: str

    def probe(self) -> ExecutionTarget: ...
    def validate(self, spec: ExperimentSpec, profile: ExecutionProfile) -> ValidationReport: ...
    def submit(self, job: WorkerJob) -> JobHandle: ...


def _validate(
    target: ExecutionTarget, spec: ExperimentSpec, profile: ExecutionProfile
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    selector = profile.execution.target_selector
    role = selector.role
    if role in PLANNED_ROLES:
        errors.append(f"execution role {role!r} is planned but not implemented")
    elif role not in target.roles:
        errors.append(f"target {target.target_id!r} does not provide role {role!r}")
    if selector.gpu_count > target.gpu_count:
        errors.append(
            f"profile requires gpu_count={selector.gpu_count} but target has {target.gpu_count}"
        )
    resolved_backend: str | None = None
    substrate = spec.experiment.substrate
    if not has_substrate(substrate):
        errors.append(f"unknown substrate {substrate!r}")
    else:
        manifest = get_manifest(substrate)
        if selector.backend in ("auto", "gpu"):
            # "auto" never picks a GPU backend; "gpu" picks the substrate's first GPU backend.
            candidates = [
                b
                for b in manifest.backends
                if b in target.capabilities.backends
                and (b in GPU_BACKENDS) == (selector.backend == "gpu")
            ]
            resolved_backend = candidates[0] if candidates else None
        elif selector.backend in manifest.backends:
            resolved_backend = selector.backend
        if resolved_backend is None:
            errors.append(
                f"no backend for substrate {substrate!r} matches selector {selector.backend!r} "
                f"on target {target.target_id!r}"
            )
        if manifest.gpu_required and target.gpu_count == 0:
            errors.append(f"substrate {substrate!r} requires a GPU")
        allowed = manifest.backend_numeric_policies.get(resolved_backend or "")
        policy = getattr(spec.experiment.dynamics, "numeric_policy", None)
        if allowed is not None and policy not in allowed:
            errors.append(
                f"backend {resolved_backend!r} requires numeric_policy in {list(allowed)}; the "
                f"spec says {policy!r} (results would differ from the declared policy)"
            )
        if resolved_backend in GPU_BACKENDS:
            if target.gpu_count == 0:
                errors.append(f"backend {resolved_backend!r} needs a GPU; target has none")
            if selector.gpu_count == 0:
                errors.append(
                    f"backend {resolved_backend!r} needs a GPU reservation (gpu_count >= 1)"
                )
        elif selector.gpu_count > 0 and not manifest.gpu_required:
            warnings.append(
                f"profile reserves a GPU but the resolved backend {resolved_backend!r} runs on "
                "the CPU; no GPU memory is used"
            )
    runner = spec.experiment.runner
    if runner.checkpoint_interval > runner.max_steps:
        warnings.append("checkpoint_interval exceeds max_steps; only final checkpoint will exist")
    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        resolved_backend=resolved_backend,
        target_id=target.target_id,
    )


class _ProcessHandle:
    def __init__(self, process: multiprocessing.process.BaseProcess, conn: Connection) -> None:
        self.process = process
        self.conn = conn

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def exitcode(self) -> int | None:
        return self.process.exitcode

    def terminate(self) -> None:
        self.process.terminate()

    def join(self, timeout: float | None = None) -> None:
        self.process.join(timeout)


class LocalExecutor:
    """Runs each job in a freshly spawned worker process (no fork of the API)."""

    target_id = "local"

    def __init__(self) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._target: ExecutionTarget | None = None

    def probe(self) -> ExecutionTarget:
        gpus = probe_gpus()
        gpu_count = len(gpus["devices"])
        self._target = ExecutionTarget(
            target_id=self.target_id,
            kind="local",
            roles=IMPLEMENTED_ROLES,
            gpu_count=gpu_count,
            capabilities=TargetCapabilities(
                pause_live=True,
                checkpoint_on_signal=True,
                interactive_preview=True,
                remote_cancel=False,
                multi_gpu=False,
                backends=available_backends(gpu_count),
            ),
            probe={"gpu": gpus},
        )
        return self._target

    def target(self) -> ExecutionTarget:
        return self._target or self.probe()

    def validate(self, spec: ExperimentSpec, profile: ExecutionProfile) -> ValidationReport:
        return _validate(self.target(), spec, profile)

    def submit(self, job: WorkerJob) -> JobHandle:
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=worker_main, args=(job, child), name=f"alife-worker-{job.run_id}", daemon=False
        )
        process.start()
        child.close()
        return _ProcessHandle(process, parent)


class _ThreadHandle:
    def __init__(self, thread: threading.Thread, conn: Connection) -> None:
        self.thread = thread
        self.conn = conn

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def exitcode(self) -> int | None:
        return None if self.thread.is_alive() else 0

    def terminate(self) -> None:
        # Threads cannot be killed; the cooperative cancel command is the only stop path.
        pass

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout)


class FakeExecutor:
    """In-process executor for tests. Same message protocol, no process isolation."""

    target_id = "fake"

    def __init__(self, gpu_count: int = 0) -> None:
        self._target = ExecutionTarget(
            target_id=self.target_id,
            kind="fake",
            roles=IMPLEMENTED_ROLES,
            gpu_count=gpu_count,
            capabilities=TargetCapabilities(
                pause_live=True,
                checkpoint_on_signal=True,
                interactive_preview=True,
                remote_cancel=False,
                multi_gpu=False,
                backends=available_backends(gpu_count),
            ),
        )

    def probe(self) -> ExecutionTarget:
        return self._target

    def validate(self, spec: ExperimentSpec, profile: ExecutionProfile) -> ValidationReport:
        return _validate(self._target, spec, profile)

    def submit(self, job: WorkerJob) -> JobHandle:
        parent, child = multiprocessing.Pipe(duplex=True)
        thread = threading.Thread(
            target=worker_main, args=(job, child), name=f"fake-worker-{job.run_id}", daemon=True
        )
        thread.start()
        return _ThreadHandle(thread, parent)
