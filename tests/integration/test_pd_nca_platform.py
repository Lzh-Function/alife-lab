"""PD-NCA through the platform: online-learning epochs in a worker, an erase intervention,
restart, replay verification across a checkpoint (weights and optimizer state included)."""

from __future__ import annotations

import importlib.util

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX not installed"),
]


def test_pd_nca_restart_replay(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["pd_nca_dish"].spec.model_dump(mode="json")
        raw["experiment"]["world"].update(width=24, height=24)
        raw["experiment"]["runner"]["checkpoint_interval"] = 20
        rid = c.create_run(ExperimentSpec.model_validate(raw), "local_interactive", 0)["id"]
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 120)
        await c.control(rid, "step", {"count": 12})
        er = {"kind": "erase", "payload": {"x": 12, "y": 12, "radius": 4}}
        assert (await c.control(rid, "intervene", er))["ack"]["status"] == "applied"
        await c.control(rid, "step", {"count": 4})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 120)
            await c2.control(rid, "step", {"count": 10})  # crosses the checkpoint at 20
            assert c2.run_summary(rid)["clock"]["unit"] == "epoch"
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            past = await c2.inspect(rid, {"x": 12, "y": 12}, tick=13)
            assert past["source"]["method"] == "deterministic_replay" and past["dominant"] == 0
        finally:
            await c2.shutdown()

    arun(main)
