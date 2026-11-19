"""Protocells through the platform: worker, food and kill interventions, restart, replay
verification (exact ledger and lineage reproduced) and past inspection."""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for


@pytest.mark.integration
def test_protocell_restart_replay(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["protocell_default"].spec.model_dump(mode="json")
        raw["experiment"]["world"].update(width=32, height=32)
        raw["experiment"]["initialization"]["count"] = 8
        raw["experiment"]["runner"]["checkpoint_interval"] = 100
        rid = c.create_run(ExperimentSpec.model_validate(raw), "cpu_dev", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 30)
        await c.control(rid, "step", {"count": 60})
        food = {"kind": "add_food", "payload": {"x": 4, "y": 4, "count": 1000}}
        assert (await c.control(rid, "intervene", food))["ack"]["status"] == "applied"
        await c.control(rid, "step", {"count": 20})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 30)
            await c2.control(rid, "step", {"count": 60})  # crosses the checkpoint at 100
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            past = await c2.inspect(rid, {"x": 16, "y": 16}, tick=61)
            assert past["source"]["method"] == "deterministic_replay"
            assert past["ledger_residual"] == 0 and past["ledger"]["inflow"] >= 1000
        finally:
            await c2.shutdown()

    arun(main)
