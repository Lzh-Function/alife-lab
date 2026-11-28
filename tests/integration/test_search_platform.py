"""Search runs through the platform: worker process, restart, deterministic replay, past
inspection, and a CMA-ES vs random sweep with a pre-declared endpoint."""

from __future__ import annotations

import importlib.util

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.core.sweep import SweepSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller, ControllerError
from alife_lab.service.settings import Settings
from tests.integration.test_controller import arun, state_of, wait_for

SMALL = {"population": 6, "eval_steps": 60}


@pytest.mark.integration
@pytest.mark.parametrize(
    "preset",
    [
        "search_lenia_orbium_cma",
        "search_lenia_qd_map_elites",
        pytest.param(
            "search_creature_coevolve_cma",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("mujoco") is None, reason="MuJoCo not installed"
            ),
        ),
    ],
)
def test_search_restart_replay(settings: Settings, preset: str) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets[preset].spec.model_dump(mode="json")
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
            target = raw["experiment"]["target"]["substrate"]
            assert past["best"]["spec"]["experiment"]["substrate"] == target
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
            await wait_for(lambda: c.sweeps.summary(sid)["state"] != "running", 240)
            report = c.sweeps.report(sid)
            assert c.sweeps.summary(sid)["state"] == "completed", report
            values = [v for cond in report["conditions"] for v in cond["values"]]
            assert len(values) == 4 and all(v is not None and v >= 0 for v in values)
        finally:
            await c.shutdown()

    arun(main)


@pytest.mark.integration
def test_discovery_ledger(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["search_lenia_orbium_cma"].spec.model_dump(mode="json")
        raw["experiment"]["dynamics"].update(SMALL)
        raw["experiment"]["runner"] = {"max_steps": 3, "checkpoint_interval": 3}
        rid = c.create_run(ExperimentSpec.model_validate(raw), "cpu_dev", 0)["id"]
        await c.start_run(rid)
        await wait_for(lambda: state_of(c, rid) == "completed", 60)
        best = (await c.inspect(rid, {"x": 0, "y": 0}))["best"]["index"]
        d = await c.discoveries.create(rid, best, "moving_localized", "t best")
        did = d["id"]
        assert d["status"] == "new" and d["evidence_level"] == "E1"
        assert d["source"]["candidate_index"] == best and d["assays"][0]["state"] == "planned"
        with pytest.raises(ControllerError):  # verdict statuses only come from assay rules
            c.discoveries.review(did, "validated", None)
        c.discoveries.review(did, "triaged", "looks interesting")
        await c.discoveries.start_assay(did)
        await wait_for(lambda: not c.discoveries.get(did)["running"], 120)
        done = c.discoveries.get(did)
        assert done["status"] in ("validated", "rejected", "inconclusive")
        verdict = done["assays"][-1]["result"]["judgement"]
        assert verdict["status"] == done["status"]
        assert done["evidence_level"] == ("E2" if done["status"] == "validated" else "E1")
        kinds = [e["kind"] for e in done["events"]]
        assert kinds[:4] == ["registered", "status_changed", "note", "assay_started"]
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            again = c2.discoveries.get(did)
            assert again["status"] == done["status"] and len(again["assays"]) == 1
            record = c2.discoveries.record(did)
            assert record["candidate"]["spec_sha256"] == done["candidate_spec_hash"]
            assert record["provenance"]["source"]["run_id"] == rid and record["limitations"]
            assert [x["id"] for x in c2.discoveries.list()] == [did]
        finally:
            await c2.shutdown()

    arun(main)


def _clip_gpu_ready() -> bool:
    try:
        import torch
    except ImportError:
        return False
    from tests.reference.test_asal import weights_present

    return weights_present() and torch.cuda.is_available()


@pytest.mark.integration
@pytest.mark.skipif(not _clip_gpu_ready(), reason="needs PyTorch, CUDA and the CLIP weights")
def test_asal_search_on_gpu_restart_replay(settings: Settings) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        raw = c.presets["search_asal_lenia_cell"].spec.model_dump(mode="json")
        raw["experiment"]["dynamics"].update(population=4, eval_steps=40)
        raw["experiment"]["target"]["world"] = {"width": 64, "height": 64}
        raw["experiment"]["runner"] = {"max_steps": 5, "checkpoint_interval": 2}
        rid = c.create_run(ExperimentSpec.model_validate(raw), "local_gpu", 0)["id"]
        assert c.run_summary(rid)["backend"] == "torch_cuda"
        await c.start_run(rid, start_paused=True)
        await wait_for(lambda: state_of(c, rid) == "paused", 60)
        await c.control(rid, "step", {"count": 3})
        await c.shutdown()

        c2 = Controller(settings, LocalExecutor())
        await c2.start()
        try:
            await c2.restore_run(rid)
            await wait_for(lambda: state_of(c2, rid) == "paused", 60)
            await c2.control(rid, "step", {"count": 2})
            report = await c2.verify_replay(rid)
            assert report["status"] == "verified", report["segments"]
            info = await c2.inspect(rid, {"x": 0, "y": 0})
            assert info["asal"]["prompts"] == ["a biological cell"]
        finally:
            await c2.shutdown()

    arun(main)
