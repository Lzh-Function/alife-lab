"""Platform integration test for connectome substrate (controller, worker, interventions,
checkpoints, restart restore, replay verification, inspection).

Uses built-in ``synthetic_tiny`` so it runs offline on CPU-only CI.
"""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import FakeExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for


def _tiny_spec() -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "connectome_platform_tiny",
                "mode": "reference",
                "substrate": "connectome",
                "seed": 5,
                "dataset_id": "synthetic_tiny",
                "wiring": "reference",
                "stimuli": [{"port": "sugar_grn", "rate_hz": 150.0}],
                "runner": {"max_steps": 500, "checkpoint_interval": 25},
            }
        }
    )


@pytest.mark.integration
def test_connectome_worker_interventions_restore_and_replay_verify(settings: Settings) -> None:
    async def main() -> None:
        ctrl = Controller(settings, FakeExecutor())
        await ctrl.start()
        try:
            rid = ctrl.create_run(_tiny_spec(), "cpu_dev", 0)["id"]
            await ctrl.start_run(rid, start_paused=True)
            await wait_for(lambda: state_of(ctrl, rid) == "paused", 20)

            # Advance 40 ticks with sugar stimulation -> MN9 must fire via interneuron relay.
            await ctrl.control(rid, "step", {"count": 40})
            await wait_for(lambda: ctrl.run_summary(rid)["clock"]["value"] >= 40, 20)
            ins_before = await ctrl.inspect(rid, {"port": "mn9_proboscis_extension"})
            assert ins_before["total_spikes"] > 0

            # Intervene: add bitter stimulation and silence sugar_grn; advance 30 more ticks.
            ack1 = await ctrl.control(
                rid,
                "intervene",
                {"kind": "stimulate", "payload": {"port": "bitter_grn", "rate_hz": 150.0}},
            )
            assert ack1["ack"]["status"] == "applied"
            ack2 = await ctrl.control(
                rid,
                "intervene",
                {"kind": "silence", "payload": {"port": "sugar_grn", "silenced": True}},
            )
            assert ack2["ack"]["status"] == "applied"
            await ctrl.control(rid, "step", {"count": 30})
            await wait_for(lambda: ctrl.run_summary(rid)["clock"]["value"] >= 70, 20)
            assert ctrl.run_summary(rid)["effective_mode"] == "exploratory"

            # Manual checkpoint, then verify deterministic replay across checkpoints.
            await ctrl.control(rid, "checkpoint", {})
            await wait_for(
                lambda: (
                    (cp := ctrl.latest_checkpoint(rid)) is not None and cp["simulation_clock"] == 70
                ),
                20,
            )
            ver = await ctrl.verify_replay(rid)
            assert ver["status"] == "verified", ver["segments"]
            assert len(ver["segments"]) >= 2

            # Past inspection by deterministic replay.
            past = await ctrl.inspect(rid, {"cell_index": 0}, tick=30)
            assert past["source"]["method"] == "deterministic_replay"
            assert past["silenced"] is False
        finally:
            await ctrl.shutdown()

        # Restart controller and restore from latest checkpoint.
        ctrl2 = Controller(settings, FakeExecutor())
        await ctrl2.start()
        try:
            assert state_of(ctrl2, rid) == "interrupted"
            await ctrl2.restore_run(rid)
            await wait_for(lambda: state_of(ctrl2, rid) == "paused", 20)
            assert ctrl2.run_summary(rid)["clock"]["value"] == 70
            ins_after = await ctrl2.inspect(rid, {"cell_index": 0})
            assert ins_after["silenced"] is True
        finally:
            await ctrl2.shutdown()

    arun(main)
