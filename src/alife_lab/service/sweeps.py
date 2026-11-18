"""Sweeps: create conditions × seeds runs through the normal queue and analyse a pre-declared
endpoint with seed-level statistics. Runs inside the controller process (single DB writer)."""

from __future__ import annotations

import json
import math
import secrets
import time
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from alife_lab.analysis.stats import compare, describe, holm
from alife_lab.core.hashing import canonical_json
from alife_lab.core.spec import ExperimentSpec
from alife_lab.core.states import TERMINAL_STATES, RunState
from alife_lab.core.sweep import SweepSpec
from alife_lab.substrates import get_manifest

if TYPE_CHECKING:
    from alife_lab.service.controller import Controller

ALWAYS_COMPUTED = {"conway": {"alive_cell_count"}, "boids": {"population"}}
UNIT_NOTE = (
    "Unit of analysis: one independently seeded run per value. Time points within a run are not "
    "treated as independent samples."
)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_path(body: dict[str, Any], path: str, value: Any) -> None:
    section, key = path.split(".", 1)
    sub = dict(body.get(section) or {})
    sub[key] = value
    body[section] = sub


class SweepService:
    def __init__(self, controller: Controller) -> None:
        self.c = controller

    # ------------------------------------------------------------------ planning

    def plan(self, spec: SweepSpec) -> dict[str, Any]:
        """Resolve and validate every member spec without writing anything."""
        from alife_lab.service.controller import ControllerError, NotFound

        body = spec.sweep
        if body.base_preset is not None:
            preset = self.c.presets.get(body.base_preset)
            if preset is None:
                raise NotFound(f"preset {body.base_preset!r} not found")
            base = preset.spec.model_dump(mode="json")
        else:
            assert body.base_spec is not None
            base = body.base_spec
        experiment = dict(base["experiment"])
        if body.overrides:
            experiment = _deep_merge(experiment, body.overrides)
        experiment["name"] = body.name
        members: list[dict[str, Any]] = []
        conditions = body.conditions()
        for ci, condition in enumerate(conditions):
            for seed in body.seeds:
                exp = json.loads(json.dumps(experiment))
                for path, value in condition["assignments"].items():
                    _set_path(exp, path, value)
                exp["seed"] = seed
                try:
                    member_spec = ExperimentSpec.model_validate(
                        {"schema_version": 1, "experiment": exp}
                    )
                except ValidationError as err:
                    raise ControllerError(
                        f"condition {condition['label']!r} seed {seed}: "
                        + "; ".join(e["msg"] for e in err.errors()[:3]),
                        "invalid_spec",
                    ) from err
                members.append(
                    {
                        "condition_index": ci,
                        "condition_label": condition["label"],
                        "seed": seed,
                        "spec": member_spec,
                    }
                )
        first = members[0]["spec"]
        substrate = first.experiment.substrate
        _, report = self.c.validate(first, body.profile_id)
        if not report.ok:
            raise ControllerError("; ".join(report.errors), "validation_failed")
        metric = body.endpoint.metric
        known = {m.id for m in get_manifest(substrate).metrics}
        if metric not in known:
            raise ControllerError(f"unknown metric {metric!r} for substrate {substrate!r}")
        for m in members:
            exp = m["spec"].experiment
            if metric not in exp.observations.metrics and metric not in ALWAYS_COMPUTED.get(
                substrate, set()
            ):
                raise ControllerError(f"endpoint metric {metric!r} is not observed by the runs")
            if isinstance(body.endpoint.at, int) and body.endpoint.at > exp.runner.max_steps:
                raise ControllerError("endpoint tick exceeds max_steps")
        total_ticks = sum(m["spec"].experiment.runner.max_steps for m in members)
        return {
            "sweep_hash": spec.digest(),
            "substrate": substrate,
            "conditions": [
                {"index": i, "label": c["label"], "assignments": c["assignments"]}
                for i, c in enumerate(conditions)
            ],
            "reference_condition": body.reference_condition,
            "runs": len(members),
            "total_ticks": total_ticks,
            "members": members,
            "warnings": report.warnings,
        }

    # ------------------------------------------------------------------ lifecycle

    async def create(self, spec: SweepSpec, author: str = "local_user") -> dict[str, Any]:
        plan = self.plan(spec)
        sweep_id = "swp_" + secrets.token_hex(8)
        now = time.time()
        self.c.db.execute(
            "INSERT INTO sweeps(id, name, design, spec_hash, spec_json, state, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                sweep_id,
                spec.sweep.name,
                spec.sweep.design,
                spec.digest(),
                canonical_json(spec.model_dump(mode="json")),
                "running",
                now,
            ),
        )
        for m in plan["members"]:
            run = self.c.create_run(
                m["spec"],
                spec.sweep.profile_id,
                ticks_per_second=0,
                author=author,
                label=f"{spec.sweep.name} | {m['condition_label']} | seed {m['seed']}",
            )
            self.c.db.execute(
                "INSERT INTO sweep_members(sweep_id, run_id, condition_index, condition_label, "
                "seed) VALUES (?,?,?,?,?)",
                (sweep_id, run["id"], m["condition_index"], m["condition_label"], m["seed"]),
            )
            self.c._emit_event(run["id"], "sweep_member", {"sweep_id": sweep_id})
        for row in self._members(sweep_id):
            await self.c.start_run(row["run_id"])
        return self.summary(sweep_id)

    async def cancel(self, sweep_id: str) -> dict[str, Any]:
        self._row(sweep_id)
        for m in self._members(sweep_id):
            state = RunState(self.c._run_row(m["run_id"])["state"])
            if state not in TERMINAL_STATES:
                await self.c.cancel_run(m["run_id"])
        self.c.db.execute(
            "UPDATE sweeps SET state='cancelled', finished_at=? WHERE id=?", (time.time(), sweep_id)
        )
        return self.summary(sweep_id)

    # ------------------------------------------------------------------ queries

    def _row(self, sweep_id: str) -> dict[str, Any]:
        from alife_lab.service.controller import NotFound

        row = self.c.db.one("SELECT * FROM sweeps WHERE id=?", (sweep_id,))
        if row is None:
            raise NotFound(f"sweep {sweep_id} not found")
        return dict(row)

    def _members(self, sweep_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.c.db.all(
                "SELECT m.*, r.state, r.tick FROM sweep_members m JOIN runs r ON r.id = m.run_id "
                "WHERE m.sweep_id=? ORDER BY m.condition_index, m.seed",
                (sweep_id,),
            )
        ]

    def list_sweeps(self) -> list[dict[str, Any]]:
        rows = self.c.db.all(
            "SELECT id, name, design, state, created_at, finished_at FROM sweeps "
            "ORDER BY created_at DESC LIMIT 200"
        )
        return [dict(r) for r in rows]

    def summary(self, sweep_id: str) -> dict[str, Any]:
        row = self._row(sweep_id)
        members = self._members(sweep_id)
        counts: dict[str, int] = {}
        for m in members:
            counts[m["state"]] = counts.get(m["state"], 0) + 1
        terminal = {s.value for s in TERMINAL_STATES}
        if row["state"] == "running" and all(m["state"] in terminal for m in members):
            state = (
                "completed"
                if all(m["state"] == "completed" for m in members)
                else "finished_with_failures"
            )
            self.c.db.execute(
                "UPDATE sweeps SET state=?, finished_at=? WHERE id=?",
                (state, time.time(), sweep_id),
            )
            row = self._row(sweep_id)
        spec = json.loads(row["spec_json"])
        return {
            "id": row["id"],
            "name": row["name"],
            "design": row["design"],
            "state": row["state"],
            "spec_hash": row["spec_hash"],
            "spec": spec,
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
            "progress": {"total": len(members), "by_state": counts},
            "members": [
                {
                    k: m[k]
                    for k in (
                        "run_id",
                        "condition_index",
                        "condition_label",
                        "seed",
                        "state",
                        "tick",
                    )
                }
                for m in members
            ],
        }

    def _endpoint_value(
        self, run_id: str, metric: str, at: str | int
    ) -> tuple[float | None, int | None, str | None]:
        if at == "final":
            rows = self.c.metrics(run_id, max_points=1)["rows"]
            row = rows[-1] if rows else None
        else:
            rows = self.c.metrics(run_id, since_tick=int(at) - 1, max_points=1)["rows"]
            row = rows[0] if rows and rows[0]["tick"] == int(at) else None
        if row is None:
            return None, None, "no metric row at the endpoint tick"
        value = row["values"].get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None, int(row["tick"]), "metric is null at the endpoint (null policy)"
        return float(value), int(row["tick"]), None

    def report(self, sweep_id: str, timecourse_points: int = 150) -> dict[str, Any]:
        summary = self.summary(sweep_id)
        spec = SweepSpec.model_validate(summary["spec"])
        body = spec.sweep
        conditions = body.conditions()
        metric, at = body.endpoint.metric, body.endpoint.at
        values: dict[int, list[float]] = {i: [] for i in range(len(conditions))}
        per_run = []
        excluded = []
        for m in summary["members"]:
            if m["state"] != RunState.COMPLETED.value:
                excluded.append({"run_id": m["run_id"], "reason": f"run state {m['state']}"})
                value, tick, reason = None, None, None
            else:
                value, tick, reason = self._endpoint_value(m["run_id"], metric, at)
                if value is None:
                    excluded.append({"run_id": m["run_id"], "reason": reason})
                else:
                    values[m["condition_index"]].append(value)
            per_run.append({**m, "endpoint_value": value, "endpoint_tick": tick})
        ref = body.reference_condition
        comparisons = []
        for i in range(len(conditions)):
            if i == ref:
                continue
            result = compare(values[i], values[ref], seed=0)
            comparisons.append(
                {
                    "condition_index": i,
                    "condition": conditions[i]["label"],
                    "reference": conditions[ref]["label"],
                    **result,
                }
            )
        adjusted = holm([c["permutation"]["p_value"] for c in comparisons])
        for comp, p_adj in zip(comparisons, adjusted, strict=True):
            comp["p_value_holm"] = p_adj
        timecourse = self._timecourse(
            summary["members"], metric, len(conditions), timecourse_points
        )
        notes: list[dict[str, Any]] = [{"code": "unit_of_analysis", "text": UNIT_NOTE}]
        if body.design == "exploratory":
            notes.append(
                {
                    "code": "exploratory",
                    "text": "Exploratory sweep: p-values and intervals are descriptive and must "
                    "not be reported as confirmatory evidence; select hypotheses before a "
                    "confirmatory sweep.",
                }
            )
        if len(comparisons) > 1:
            notes.append(
                {
                    "code": "holm",
                    "text": "Multiple comparisons: Holm-adjusted p-values are reported.",
                }
            )
        if excluded:
            notes.append(
                {
                    "code": "excluded",
                    "count": len(excluded),
                    "text": f"{len(excluded)} run(s) excluded from the endpoint analysis.",
                }
            )
        notes.append(
            {
                "code": "evidence",
                "text": "Evidence levels (E0-E4) are assigned by review, not by this report. A "
                "factor that ablates a mechanism (e.g. alignment_weight = 0) is an intervention "
                "comparison.",
            }
        )
        return {
            "sweep": {
                k: summary[k] for k in ("id", "name", "design", "state", "spec_hash", "progress")
            },
            "hypothesis": body.hypothesis,
            "endpoint": {"metric": metric, "at": at},
            "conditions": [
                {
                    "index": i,
                    "label": c["label"],
                    "assignments": c["assignments"],
                    "is_reference": i == ref,
                    "summary": describe(values[i]),
                    "values": values[i],
                }
                for i, c in enumerate(conditions)
            ],
            "comparisons": comparisons,
            "per_run": per_run,
            "excluded": excluded,
            "timecourse": timecourse,
            "notes": notes,
        }

    def _timecourse(
        self, members: list[dict[str, Any]], metric: str, n_conditions: int, points: int
    ) -> list[dict[str, Any]]:
        out = []
        for ci in range(n_conditions):
            series: dict[int, list[float]] = {}
            for m in members:
                if m["condition_index"] != ci or m["state"] != RunState.COMPLETED.value:
                    continue
                for row in self.c.metrics(m["run_id"], max_points=points)["rows"]:
                    v = row["values"].get(metric)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        series.setdefault(int(row["tick"]), []).append(float(v))
            ticks = sorted(series)
            out.append(
                {
                    "condition_index": ci,
                    "ticks": ticks,
                    "mean": [describe(series[t])["mean"] for t in ticks],
                    "sd": [describe(series[t])["sd"] for t in ticks],
                    "n": [len(series[t]) for t in ticks],
                }
            )
        return out
