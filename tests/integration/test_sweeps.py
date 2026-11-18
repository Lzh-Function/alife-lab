"""Sweeps through the controller: planning, validation, execution, report, cancellation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from alife_lab.core.spec import ExperimentSpec
from alife_lab.core.sweep import SweepSpec
from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.controller import Controller, ControllerError
from alife_lab.service.settings import Settings
from alife_lab.substrates.boids.boids import BoidsAdapter
from tests.integration.test_controller import arun, wait_for


def sweep(**kw: Any) -> SweepSpec:
    body = {
        "name": "t_sweep",
        "base_preset": "boids_flocking",
        "overrides": {
            "initialization": {"count": 60},
            "world": {"width": 80, "height": 80},
            "runner": {"max_steps": 120, "checkpoint_interval": 60},
        },
        "seeds": [1, 2, 3],
        "factors": [
            {"path": "dynamics.alignment_weight", "values": [0.0, 0.05], "labels": ["off", "on"]}
        ],
        "endpoint": {"metric": "polarization"},
        **kw,
    }
    return SweepSpec.model_validate({"sweep": body})


def direct_final_polarization(spec: ExperimentSpec) -> float:
    adapter = BoidsAdapter()
    state = adapter.initialize(spec)
    for _ in range(spec.experiment.runner.max_steps):
        state = adapter.step(state, spec)
    return adapter.metrics(state, spec)["polarization"]


def test_sweep_runs_and_reports(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            spec = sweep()
            plan = c.sweeps.plan(spec)
            assert plan["runs"] == 6 and [x["label"] for x in plan["conditions"]] == ["off", "on"]
            summary = await c.sweeps.create(spec)
            sid = summary["id"]
            await wait_for(lambda: c.sweeps.summary(sid)["state"] != "running", 60)
            assert c.sweeps.summary(sid)["state"] == "completed"
            report = c.sweeps.report(sid)
            off, on = report["conditions"]
            assert off["is_reference"] and off["summary"]["n"] == 3 and on["summary"]["n"] == 3
            # Platform results equal a direct, in-process computation (no hidden coupling).
            for m in plan["members"]:
                expected = direct_final_polarization(m["spec"])
                cond = report["conditions"][m["condition_index"]]
                assert any(abs(v - expected) == 0.0 for v in cond["values"])
            comp = report["comparisons"][0]
            assert comp["permutation"]["method"] == "exact"
            assert comp["permutation"]["permutations"] == 20  # C(6, 3)
            assert comp["p_value_holm"] == comp["permutation"]["p_value"]
            assert len(report["per_run"]) == 6 and not report["excluded"]
            assert all(r["endpoint_tick"] == 120 for r in report["per_run"])
            tc = report["timecourse"]
            assert len(tc) == 2 and tc[0]["ticks"][-1] == 120 and set(tc[0]["n"]) == {3}
            assert [n["code"] for n in report["notes"]] == [
                "unit_of_analysis",
                "exploratory",
                "evidence",
            ]
            run = c.run_summary(report["per_run"][0]["run_id"])
            assert run["label"].startswith("t_sweep | off | seed")
            assert run["spec"]["experiment"]["seed"] == report["per_run"][0]["seed"]
        finally:
            await c.shutdown()

    arun(main)


def test_sweep_at_tick_and_cancel(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            spec = sweep(endpoint={"metric": "mean_speed", "at": 50}, seeds=[4, 5])
            with pytest.raises(ControllerError):  # a conway metric is not valid for boids
                c.sweeps.plan(sweep(endpoint={"metric": "alive_cell_count"}))
            sid = (await c.sweeps.create(spec))["id"]
            await wait_for(lambda: c.sweeps.summary(sid)["state"] != "running", 60)
            report = c.sweeps.report(sid)
            assert all(r["endpoint_tick"] == 50 for r in report["per_run"])
            # Cancellation of a long sweep leaves no live runs and is reported.
            long = sweep(overrides={"runner": {"max_steps": 10_000_000}}, seeds=[7, 8, 9])
            lid = (await c.sweeps.create(long))["id"]
            cancelled = await c.sweeps.cancel(lid)
            assert cancelled["state"] == "cancelled"
            assert all(m["state"] == "cancelled" for m in cancelled["members"])
            assert len(c.sweeps.report(lid)["excluded"]) == 6
        finally:
            await c.shutdown()

    arun(main)


@pytest.mark.parametrize(
    "kw",
    [
        {"seeds": [1, 1]},
        {"factors": [{"path": "seed", "values": [1, 2]}]},
        {"factors": [{"path": "dynamics.alignment_weight", "values": [0.1, 0.1]}]},
        {"design": "confirmatory"},  # needs a hypothesis
        {
            "seeds": list(range(100)),
            "factors": [{"path": "dynamics.noise", "values": [0.0, 0.1, 0.2]}],
        },  # > 200 runs
        {"reference_condition": 5},
    ],
)
def test_invalid_sweeps(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        sweep(**kw)


def test_plan_rejects_invalid_member_specs(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, FakeExecutor())
        await c.start()
        try:
            bad = sweep(factors=[{"path": "dynamics.alignment_weight", "values": [-1.0]}])
            with pytest.raises(ControllerError):
                c.sweeps.plan(bad)
            with pytest.raises(ControllerError):
                c.sweeps.plan(sweep(endpoint={"metric": "nope"}))
            assert c.sweeps.list_sweeps() == []
        finally:
            await c.shutdown()

    arun(main)
