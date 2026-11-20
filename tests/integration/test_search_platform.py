"""Search runs through the platform: worker process, restart, deterministic replay, past
inspection, and a CMA-ES vs random sweep with a pre-declared endpoint."""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.core.sweep import SweepSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for

SMALL = {"population": 6, "eval_steps": 60}


@pytest.mark.integration
def test_search_restart_replay(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["search_lenia_orbium_cma"].spec.model_dump(mode="json")
        raw["experiment"]["dynamics"].update(SMALL)
        raw["experiment"]["runner"] = {"max_steps": 10, "checkpoint_interval": 3}
        rid = c.create_run(ExperimentSpec.model_validate(raw), "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 4})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 3})  # crosses the checkpoint at 6
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            past = await c2.inspect(rid, {"x": 100, "y": 10}, tick=2)
            assert past["source"]["method"] == "deterministic_replay"
            assert past["evaluated"] == 2 * 6
            assert past["best"]["spec"]["experiment"]["substrate"] == "lenia"
        finally:
            await c2.shutdown()

    arun(main)


@pytest.mark.integration
def test_search_algorithm_sweep(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            body = {
                "name": "t_cma_vs_random",
                "base_preset": "search_lenia_orbium_cma",
                "overrides": {
                    "dynamics": SMALL,
                    "runner": {"max_steps": 3, "checkpoint_interval": 3},
                },
                "seeds": [1, 2],
                "factors": [{"path": "dynamics.algorithm", "values": ["cma_es", "random"]}],
                "reference_condition": 1,
                "endpoint": {"metric": "best_score"},
            }
            sid = (await c.sweeps.create(SweepSpec.model_validate({"sweep": body})))["id"]
            await wait_for(lambda: c.sweeps.summary(sid)["state"] != "running", 120)
            report = c.sweeps.report(sid)
            assert c.sweeps.summary(sid)["state"] == "completed", report
            values = [v for cond in report["conditions"] for v in cond["values"]]
            assert len(values) == 4 and all(v is not None and v >= 0 for v in values)
        finally:
            await c.shutdown()

    arun(main)
