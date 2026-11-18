"""FastAPI application: REST + WebSocket + built Web GUI delivery."""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import re
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from alife_lab import __version__
from alife_lab.api.security import AccessPolicy, BodyTooLarge, SecurityMiddleware
from alife_lab.core.hashing import sha256_json
from alife_lab.core.spec import ExperimentSpec
from alife_lab.core.states import RunState
from alife_lab.core.sweep import SweepSpec
from alife_lab.execution.executors import Executor
from alife_lab.execution.probe import probe_system
from alife_lab.service.controller import Controller, ControllerError, Subscriber
from alife_lab.service.settings import Settings
from alife_lab.substrates import get_manifest, has_substrate, list_substrates

API = "/api/v1"
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9_\-]{8,128}$")


def envelope(data: Any) -> dict[str, Any]:
    return {"api_version": "v1", "server_version": __version__, "data": data}


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateExperimentRequest(_Req):
    preset_id: str | None = None
    spec: dict[str, Any] | None = None
    overrides: dict[str, Any] | None = None
    profile_id: str = "local_interactive"
    ticks_per_second: float | None = None
    auto_start: bool = False
    start_paused: bool = False


class ValidateRequest(_Req):
    preset_id: str | None = None
    spec: dict[str, Any] | None = None
    overrides: dict[str, Any] | None = None
    profile_id: str = "local_interactive"


class ControlRequest(_Req):
    command_id: str | None = None
    expected_revision: int | None = None
    requested_clock: int | None = None


class StepRequest(ControlRequest):
    count: Annotated[int, Field(ge=1, le=10_000)] = 1


class SpeedRequest(ControlRequest):
    ticks_per_second: Annotated[float, Field(ge=0, le=10_000)]


class InterventionRequest(ControlRequest):
    # Validated against the substrate's declared intervention_types by the controller.
    kind: Annotated[str, Field(pattern=r"^[a-z_]{1,32}$")]
    payload: dict[str, Any]


class InterventionSpec(_Req):
    kind: Annotated[str, Field(pattern=r"^[a-z_]{1,32}$")]
    payload: dict[str, Any]


class ForkRequest(_Req):
    checkpoint_id: str = "latest"
    # Alternative to checkpoint_id: fork at an arbitrary tick via deterministic replay.
    tick: Annotated[int, Field(ge=0)] | None = None
    label: Annotated[str, Field(min_length=1, max_length=120)] | None = None
    spec_overrides: dict[str, Any] | None = None
    intervention: InterventionSpec | None = None
    profile_id: str | None = None
    ticks_per_second: Annotated[float, Field(ge=0, le=10_000)] | None = None
    auto_start: bool = False
    start_paused: bool = True


class PairedForkRequest(_Req):
    checkpoint_id: str = "now"
    treatment_intervention: InterventionSpec | None = None
    name: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    ticks_per_second: Annotated[float, Field(ge=0, le=10_000)] | None = None
    start: bool = True
    start_paused: bool = True


class ComparisonRequest(_Req):
    run_ids: Annotated[list[str], Field(min_length=2, max_length=4)]
    name: Annotated[str, Field(min_length=1, max_length=200)] = "manual comparison"


class RestoreRequest(_Req):
    checkpoint_id: str = "latest"
    start_paused: bool = True


_ALLOWED_OVERRIDE_KEYS = {
    "name",
    "seed",
    "mode",
    "world",
    "initialization",
    "runner",
    "dynamics",
    "observations",
}


def _resolve_spec(
    controller: Controller,
    preset_id: str | None,
    spec: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> ExperimentSpec:
    if (preset_id is None) == (spec is None):
        raise ControllerError("provide exactly one of preset_id or spec")
    if spec is not None:
        base = spec
    else:
        assert preset_id is not None
        preset = controller.presets.get(preset_id)
        if preset is None:
            raise ControllerError(f"unknown preset {preset_id!r}", "not_found")
        base = preset.spec.model_dump(mode="json")
    if overrides:
        unknown = set(overrides) - _ALLOWED_OVERRIDE_KEYS
        if unknown:
            raise ControllerError(f"unsupported override keys: {sorted(unknown)}")
        merged = json.loads(json.dumps(base))
        exp = merged["experiment"]
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(exp.get(key), dict):
                exp[key] = {**exp[key], **value}
            else:
                exp[key] = value
        base = merged
    try:
        return ExperimentSpec.model_validate(base)
    except ValidationError as exc:
        raise ControllerError(_format_validation(exc), "invalid_spec") from exc


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:10]:
        loc = ".".join(str(p) for p in err["loc"])
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #


def create_app(
    settings: Settings,
    executor: Executor,
    token: str,
    port: int,
    extra_hosts: tuple[str, ...] = (),
) -> FastAPI:
    holder: dict[str, Controller] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        controller = Controller(settings, executor)
        await controller.start()
        holder["c"] = controller
        try:
            yield
        finally:
            await controller.shutdown()

    app = FastAPI(
        title="ALife Lab",
        version=__version__,
        lifespan=lifespan,
        docs_url=f"{API}/docs",
        openapi_url=f"{API}/openapi.json",
        redoc_url=None,
    )
    policy = AccessPolicy(token, port, extra_hosts)

    def ctl() -> Controller:
        return holder["c"]

    @app.exception_handler(ControllerError)
    async def _controller_error(_: Request, exc: ControllerError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"api_version": "v1", "error": {"code": exc.code, "message": str(exc)}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"]} for e in exc.errors()[:20]]
        return JSONResponse(
            status_code=422,
            content={
                "api_version": "v1",
                "error": {
                    "code": "invalid_request",
                    "message": "request validation failed",
                    "details": errors,
                },
            },
        )

    @app.exception_handler(BodyTooLarge)
    async def _too_large(_: Request, __: BodyTooLarge) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={
                "api_version": "v1",
                "error": {"code": "body_too_large", "message": "Request body too large"},
            },
        )

    # ------------------------------------------------------------- system

    @app.get(f"{API}/health")
    async def health() -> dict[str, Any]:
        return envelope({"status": "ok"})

    @app.get(f"{API}/system/capabilities")
    async def capabilities() -> dict[str, Any]:
        return envelope(ctl().system_info())

    @app.get(f"{API}/system/resources")
    async def resources() -> dict[str, Any]:
        info = await run_in_threadpool(probe_system, settings.runs_root, settings.frontend_dist)
        return envelope(info)

    @app.get(f"{API}/compute-targets")
    async def compute_targets() -> dict[str, Any]:
        target = ctl().target.to_json()
        planned = [
            {"target_id": role, "status": "not_implemented", "roles": [role]}
            for role in ("remote_compute", "multi_gpu", "cluster_batch")
        ]
        return envelope({"targets": [{**target, "status": "ready"}], "planned": planned})

    @app.get(f"{API}/execution-profiles")
    async def execution_profiles() -> dict[str, Any]:
        out = []
        for pid, profile in ctl().profiles.items():
            selector = profile.execution.target_selector
            implemented = selector.role in ("cpu_dev", "local_interactive")
            out.append(
                {
                    "id": pid,
                    "role": selector.role,
                    "implemented": implemented,
                    "profile_hash": profile.digest(),
                    "profile": profile.model_dump(mode="json"),
                }
            )
        return envelope(out)

    # ------------------------------------------------------------- catalog

    @app.get(f"{API}/substrates")
    async def substrates() -> dict[str, Any]:
        return envelope([_manifest_json(m) for m in list_substrates()])

    @app.get(f"{API}/substrates/{{slug}}/schema")
    async def substrate_schema(slug: str) -> dict[str, Any]:
        if not has_substrate(slug):
            raise ControllerError(f"unknown substrate {slug!r}", "not_found")
        return envelope(
            {
                "substrate": slug,
                "experiment_spec_schema": ExperimentSpec.model_json_schema(),
                "manifest": _manifest_json(get_manifest(slug)),
            }
        )

    @app.get(f"{API}/presets")
    async def presets() -> dict[str, Any]:
        return envelope([p.to_json() for p in ctl().presets.values()])

    # ------------------------------------------------------------- experiments

    @app.post(f"{API}/experiments/validate")
    async def validate(body: ValidateRequest) -> dict[str, Any]:
        spec = _resolve_spec(ctl(), body.preset_id, body.spec, body.overrides)
        profile, report = ctl().validate(spec, body.profile_id)
        return envelope(
            {
                "report": report.to_json(),
                "spec": spec.model_dump(mode="json"),
                "spec_hash": spec.digest(),
                "profile_hash": profile.digest(),
            }
        )

    @app.post(f"{API}/experiments", status_code=201)
    async def create_experiment(
        body: CreateExperimentRequest,
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        c = ctl()
        request_hash = sha256_json(body.model_dump(mode="json"))
        if idempotency_key is not None:
            if not _IDEMPOTENCY_RE.match(idempotency_key):
                raise ControllerError("invalid Idempotency-Key")
            previous = c.idempotency_lookup(idempotency_key, "create_experiment", request_hash)
            if previous is not None:
                return envelope({**c.run_summary(previous["run_id"]), "idempotent_replay": True})
        spec = _resolve_spec(c, body.preset_id, body.spec, body.overrides)
        summary = c.create_run(
            spec, body.profile_id, body.ticks_per_second, author="browser_or_api"
        )
        if idempotency_key is not None:
            c.idempotency_store(
                idempotency_key, "create_experiment", request_hash, {"run_id": summary["id"]}
            )
        if body.auto_start:
            summary = await c.start_run(summary["id"], start_paused=body.start_paused)
        return envelope(summary)

    @app.get(f"{API}/experiments")
    async def list_experiments(
        status: Annotated[str | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> dict[str, Any]:
        if status is not None and status not in {s.value for s in RunState}:
            raise ControllerError(f"unknown status {status!r}")
        return envelope(ctl().list_runs(status, limit))

    @app.get(f"{API}/experiments/{{run_id}}")
    async def get_experiment(run_id: str) -> dict[str, Any]:
        return envelope(ctl().run_summary(run_id))

    @app.post(f"{API}/experiments/{{run_id}}/start")
    async def start(run_id: str, start_paused: bool = False) -> dict[str, Any]:
        return envelope(await ctl().start_run(run_id, start_paused=start_paused))

    async def _control(
        run_id: str, kind: str, body: ControlRequest, payload: dict[str, Any]
    ) -> dict[str, Any]:
        result = await ctl().control(
            run_id,
            kind,
            payload,
            command_id=body.command_id,
            expected_revision=body.expected_revision,
            requested_clock=body.requested_clock,
            author="browser_or_api",
        )
        return envelope({**result, "run": ctl().run_summary(run_id)})

    @app.post(f"{API}/experiments/{{run_id}}/pause")
    async def pause(run_id: str, body: ControlRequest | None = None) -> dict[str, Any]:
        return await _control(run_id, "pause", body or ControlRequest(), {})

    @app.post(f"{API}/experiments/{{run_id}}/resume")
    async def resume(run_id: str, body: ControlRequest | None = None) -> dict[str, Any]:
        return await _control(run_id, "play", body or ControlRequest(), {})

    @app.post(f"{API}/experiments/{{run_id}}/step")
    async def step(run_id: str, body: StepRequest | None = None) -> dict[str, Any]:
        body = body or StepRequest()
        return await _control(run_id, "step", body, {"count": body.count})

    @app.post(f"{API}/experiments/{{run_id}}/speed")
    async def speed(run_id: str, body: SpeedRequest) -> dict[str, Any]:
        return await _control(
            run_id, "set_speed", body, {"ticks_per_second": body.ticks_per_second}
        )

    @app.post(f"{API}/experiments/{{run_id}}/interventions")
    async def intervene(run_id: str, body: InterventionRequest) -> dict[str, Any]:
        return await _control(
            run_id, "intervene", body, {"kind": body.kind, "payload": body.payload}
        )

    @app.get(f"{API}/experiments/{{run_id}}/interventions")
    async def interventions(run_id: str) -> dict[str, Any]:
        return envelope(ctl().list_interventions(run_id))

    @app.post(f"{API}/experiments/{{run_id}}/checkpoints")
    async def checkpoint(run_id: str, body: ControlRequest | None = None) -> dict[str, Any]:
        return await _control(run_id, "checkpoint", body or ControlRequest(), {})

    @app.get(f"{API}/experiments/{{run_id}}/checkpoints")
    async def checkpoints(run_id: str) -> dict[str, Any]:
        return envelope(ctl().list_checkpoints(run_id))

    @app.get(f"{API}/experiments/{{run_id}}/checkpoints/{{checkpoint_id}}/frame")
    async def checkpoint_frame(run_id: str, checkpoint_id: str) -> Response:
        header, data = await run_in_threadpool(ctl().checkpoint_frame, run_id, checkpoint_id)
        return Response(content=_pack_frame(header, data), media_type="application/octet-stream")

    @app.post(f"{API}/experiments/{{run_id}}/restore")
    async def restore(run_id: str, body: RestoreRequest | None = None) -> dict[str, Any]:
        body = body or RestoreRequest()
        return envelope(await ctl().restore_run(run_id, body.checkpoint_id, body.start_paused))

    @app.post(f"{API}/experiments/{{run_id}}/fork", status_code=201)
    async def fork(
        run_id: str,
        body: ForkRequest,
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        c = ctl()
        request_hash = sha256_json({"run": run_id, **body.model_dump(mode="json")})
        if idempotency_key is not None:
            if not _IDEMPOTENCY_RE.match(idempotency_key):
                raise ControllerError("invalid Idempotency-Key")
            previous = c.idempotency_lookup(idempotency_key, "fork", request_hash)
            if previous is not None:
                return envelope({**c.run_summary(previous["run_id"]), "idempotent_replay": True})
        intervention = body.intervention.model_dump() if body.intervention else None
        if body.tick is not None:
            if body.checkpoint_id != "latest" or body.spec_overrides or body.profile_id:
                raise ControllerError(
                    "tick forks cannot be combined with checkpoint_id, spec_overrides or profile_id"
                )
            child = await c.fork_at_tick(
                run_id,
                body.tick,
                label=body.label,
                intervention=intervention,
                ticks_per_second=body.ticks_per_second,
                author="browser_or_api",
            )
        else:
            child = c.fork_run(
                run_id,
                body.checkpoint_id,
                label=body.label,
                spec_overrides=body.spec_overrides,
                intervention=intervention,
                profile_id=body.profile_id,
                ticks_per_second=body.ticks_per_second,
                author="browser_or_api",
            )
        if idempotency_key is not None:
            c.idempotency_store(idempotency_key, "fork", request_hash, {"run_id": child["id"]})
        if body.auto_start:
            child = await c.start_run(child["id"], start_paused=body.start_paused)
        return envelope(child)

    @app.post(f"{API}/experiments/{{run_id}}/paired-fork", status_code=201)
    async def paired_fork(
        run_id: str,
        body: PairedForkRequest,
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        c = ctl()
        request_hash = sha256_json({"run": run_id, **body.model_dump(mode="json")})
        if idempotency_key is not None:
            if not _IDEMPOTENCY_RE.match(idempotency_key):
                raise ControllerError("invalid Idempotency-Key")
            previous = c.idempotency_lookup(idempotency_key, "paired_fork", request_hash)
            if previous is not None:
                return envelope(
                    {**c.comparison_summary(previous["comparison_id"]), "idempotent_replay": True}
                )
        result = await c.paired_fork(
            run_id,
            body.checkpoint_id,
            treatment_intervention=(
                body.treatment_intervention.model_dump() if body.treatment_intervention else None
            ),
            name=body.name,
            ticks_per_second=body.ticks_per_second,
            start=body.start,
            start_paused=body.start_paused,
            author="browser_or_api",
        )
        if idempotency_key is not None:
            c.idempotency_store(
                idempotency_key, "paired_fork", request_hash, {"comparison_id": result["id"]}
            )
        return envelope(result)

    @app.get(f"{API}/experiments/{{run_id}}/inspect")
    async def inspect(
        run_id: str,
        x: Annotated[float, Query()],
        y: Annotated[float, Query()],
        radius: Annotated[float, Query(ge=0, le=1000)] = 2,
        tick: Annotated[int | None, Query(ge=0)] = None,
    ) -> dict[str, Any]:
        query = {"x": x, "y": y, "radius": radius}
        return envelope(await ctl().inspect(run_id, query, tick))

    @app.get(f"{API}/experiments/{{run_id}}/timeline")
    async def timeline(run_id: str) -> dict[str, Any]:
        return envelope(ctl().timeline(run_id))

    @app.get(f"{API}/experiments/{{run_id}}/replay/frames")
    async def replay_frames(
        run_id: str,
        start: Annotated[int, Query(ge=0)],
        count: Annotated[int, Query(ge=1, le=200)] = 1,
        stride: Annotated[int, Query(ge=1, le=100_000)] = 1,
    ) -> Response:
        frames = await ctl().replay_frames(run_id, start, count, stride)
        parts = [struct.pack(">I", len(frames))]
        for frame in frames:
            header = {k: v for k, v in frame.items() if k != "data"}
            packed = _pack_frame(header, frame["data"])
            parts.append(struct.pack(">I", len(packed)))
            parts.append(packed)
        return Response(content=b"".join(parts), media_type="application/octet-stream")

    @app.post(f"{API}/experiments/{{run_id}}/replay/verify")
    async def replay_verify(run_id: str) -> dict[str, Any]:
        return envelope(await ctl().verify_replay(run_id))

    # ------------------------------------------------------------- sweeps

    def _sweep_spec(raw: dict[str, Any]) -> SweepSpec:
        try:
            return SweepSpec.model_validate(raw)
        except ValidationError as exc:
            raise ControllerError(_format_validation(exc), "invalid_sweep") from exc

    def _plan_json(plan: dict[str, Any]) -> dict[str, Any]:
        return {
            **{k: v for k, v in plan.items() if k != "members"},
            "members": [
                {
                    "condition_index": m["condition_index"],
                    "condition_label": m["condition_label"],
                    "seed": m["seed"],
                    "spec_hash": m["spec"].digest(),
                }
                for m in plan["members"]
            ],
        }

    @app.post(f"{API}/sweeps/validate")
    async def sweep_validate(body: dict[str, Any]) -> dict[str, Any]:
        return envelope(_plan_json(ctl().sweeps.plan(_sweep_spec(body))))

    @app.post(f"{API}/sweeps", status_code=201)
    async def sweep_create(
        body: dict[str, Any],
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        c = ctl()
        spec = _sweep_spec(body)
        request_hash = spec.digest()
        if idempotency_key is not None:
            if not _IDEMPOTENCY_RE.match(idempotency_key):
                raise ControllerError("invalid Idempotency-Key")
            previous = c.idempotency_lookup(idempotency_key, "sweep", request_hash)
            if previous is not None:
                return envelope(
                    {**c.sweeps.summary(previous["sweep_id"]), "idempotent_replay": True}
                )
        result = await c.sweeps.create(spec, author="browser_or_api")
        if idempotency_key is not None:
            c.idempotency_store(idempotency_key, "sweep", request_hash, {"sweep_id": result["id"]})
        return envelope(result)

    @app.get(f"{API}/sweeps")
    async def sweep_list() -> dict[str, Any]:
        return envelope(ctl().sweeps.list_sweeps())

    @app.get(f"{API}/sweeps/{{sweep_id}}")
    async def sweep_get(sweep_id: str) -> dict[str, Any]:
        return envelope(ctl().sweeps.summary(sweep_id))

    @app.get(f"{API}/sweeps/{{sweep_id}}/report")
    async def sweep_report(sweep_id: str) -> dict[str, Any]:
        return envelope(await run_in_threadpool(ctl().sweeps.report, sweep_id))

    @app.post(f"{API}/sweeps/{{sweep_id}}/cancel")
    async def sweep_cancel(sweep_id: str) -> dict[str, Any]:
        return envelope(await ctl().sweeps.cancel(sweep_id))

    @app.post(f"{API}/comparisons", status_code=201)
    async def create_comparison(body: ComparisonRequest) -> dict[str, Any]:
        return envelope(ctl().create_comparison(body.run_ids, body.name))

    @app.get(f"{API}/comparisons")
    async def comparisons() -> dict[str, Any]:
        return envelope(ctl().list_comparisons())

    @app.get(f"{API}/comparisons/{{comparison_id}}")
    async def comparison(comparison_id: str) -> dict[str, Any]:
        return envelope(ctl().comparison_summary(comparison_id))

    @app.get(f"{API}/comparisons/{{comparison_id}}/report")
    async def comparison_report(
        comparison_id: str,
        max_points: Annotated[int, Query(ge=1, le=20_000)] = 1000,
    ) -> dict[str, Any]:
        report = await run_in_threadpool(ctl().comparison_report, comparison_id, max_points)
        return envelope(report)

    @app.post(f"{API}/experiments/{{run_id}}/cancel")
    async def cancel(run_id: str) -> dict[str, Any]:
        return envelope(await ctl().cancel_run(run_id))

    @app.get(f"{API}/experiments/{{run_id}}/metrics")
    async def metrics(
        run_id: str,
        since_tick: Annotated[int, Query(ge=-1)] = -1,
        max_points: Annotated[int, Query(ge=1, le=20_000)] = 2000,
        until_tick: Annotated[int | None, Query(ge=0)] = None,
    ) -> dict[str, Any]:
        return envelope(
            await run_in_threadpool(ctl().metrics, run_id, since_tick, max_points, until_tick)
        )

    @app.get(f"{API}/experiments/{{run_id}}/events")
    async def events(
        run_id: str,
        since_seq: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    ) -> dict[str, Any]:
        return envelope(ctl().list_events(run_id, since_seq, limit))

    @app.get(f"{API}/experiments/{{run_id}}/provenance")
    async def provenance(run_id: str) -> dict[str, Any]:
        return envelope(ctl().provenance(run_id))

    @app.post(f"{API}/experiments/{{run_id}}/verify")
    async def verify(run_id: str) -> dict[str, Any]:
        return envelope(await run_in_threadpool(ctl().verify_run, run_id))

    # ------------------------------------------------------------- streaming

    @app.websocket(f"{API}/stream/experiments/{{run_id}}")
    async def stream(websocket: WebSocket, run_id: str) -> None:
        c = ctl()
        try:
            sub = c.subscribe(run_id)
        except ControllerError:
            await websocket.close(code=1008, reason="run_not_found")
            return
        await websocket.accept()
        try:
            await websocket.send_json({"type": "hello", "run": c.run_summary(run_id)})
            await _pump(websocket, sub)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            c.unsubscribe(sub)

    # ------------------------------------------------------------- web GUI

    dist = settings.frontend_dist
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> Response:
        if path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={
                    "api_version": "v1",
                    "error": {"code": "not_found", "message": "unknown API route"},
                },
            )
        index = dist / "index.html"
        if path and "/" not in path and "." in path:
            candidate = (dist / path).resolve()
            if candidate.parent == dist.resolve() and candidate.is_file():
                return FileResponse(candidate)
        if not index.is_file():
            return HTMLResponse(_MISSING_UI, status_code=200)
        page = index.read_text(encoding="utf-8").replace(
            "</head>",
            f'<meta name="alife-token" content="{html.escape(token)}"></head>',
            1,
        )
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    app.add_middleware(SecurityMiddleware, policy=policy)
    return app


async def _pump(websocket: WebSocket, sub: Subscriber) -> None:
    async def receiver() -> None:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_json({"type": "pong"})

    recv_task = asyncio.create_task(receiver())
    try:
        while not recv_task.done():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(sub.wake.wait(), 1.0)
            sub.wake.clear()
            if sub.overflowed:
                while not sub.queue.empty():
                    sub.queue.get_nowait()
                sub.overflowed = False
                await websocket.send_json({"type": "resync_required"})
            while not sub.queue.empty():
                await websocket.send_json(sub.queue.get_nowait())
            preview = sub.latest_preview
            if preview is not None:
                sub.latest_preview = None
                header = {k: v for k, v in preview.items() if k != "data"}
                header["dropped_previews"] = sub.dropped_previews
                await websocket.send_bytes(_pack_frame(header, preview["data"]))
    finally:
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            await recv_task


def _pack_frame(header: dict[str, Any], data: bytes) -> bytes:
    raw = json.dumps(header, sort_keys=True).encode("utf-8")
    return struct.pack(">I", len(raw)) + raw + data


def _manifest_json(manifest: Any) -> dict[str, Any]:
    from dataclasses import asdict

    out: dict[str, Any] = asdict(manifest)
    return out


_MISSING_UI = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>ALife Lab</title></head>
<body style="font-family:sans-serif;max-width:40rem;margin:3rem auto">
<h1>ALife Lab</h1>
<p>Web GUI がビルドされていません / The web GUI has not been built.</p>
<pre>npm --prefix frontend ci
npm --prefix frontend run build</pre>
<p>API: <a href="/api/v1/docs">/api/v1/docs</a></p></body></html>"""
