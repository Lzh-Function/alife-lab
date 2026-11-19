"""GPU backend through the platform for float substrates: worker on CuPy, replay verification
with the recorded backend, provenance, and comparison with a CPU run of the same spec."""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.execution.executors import LocalExecutor
from alife_lab.service.controller import Controller
from alife_lab.service.settings import Settings
from alife_lab.substrates.bff.vm_cupy import available
from tests.integration.test_controller import arun, state_of, wait_for

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not available(), reason="cupy / CUDA device unavailable"),
]


def _spec(c: Controller, preset: str) -> ExperimentSpec:
    raw = c.presets[preset].spec.model_dump(mode="json")
    raw["experiment"]["runner"]["checkpoint_interval"] = 40
    if preset.startswith("flow_lenia"):
        raw["experiment"]["world"].update(width=64, height=64)
        raw["experiment"]["initialization"]["size"] = 21
    return ExperimentSpec.model_validate(raw)


@pytest.mark.parametrize("preset", ["lenia_orbium", "flow_lenia_example"])
def test_gpu_run_replays_and_stays_close_to_cpu(settings: Settings, preset: str) -> None:
    async def main() -> None:
        c = Controller(settings, LocalExecutor())
        await c.start()
        try:
            sp = _spec(c, preset)
            runs = {}
            for profile in ("local_gpu", "cpu_dev"):
                rid = c.create_run(sp, profile, 0)["id"]
                await c.start_run(rid, start_paused=True)
                await wait_for(lambda rid=rid: state_of(c, rid) == "paused", 60)
                await c.control(rid, "step", {"count": 60})  # crosses the checkpoint at 40
                runs[profile] = rid
            gpu = runs["local_gpu"]
            assert c.run_summary(gpu)["backend"] == "cupy"
            assert c.provenance(gpu)["execution"]["backend_versions"]["cupy"]
            report = await c.verify_replay(gpu)
            assert report["status"] == "verified", report["segments"]
            cmp = c.create_comparison([runs["cpu_dev"], gpu], name=f"{preset}-gpu-vs-cpu")
            points = c.comparison_report(cmp["id"])["divergence"]["points"]
            assert points and max(p["distance_to_first"][1] for p in points) < 1e-9
        finally:
            await c.shutdown()

    arun(main)
