"""Flow-Lenia GPU backend (CuPy): fused reintegration kernel bit-identical to the NumPy
implementation for identical inputs; full GPU steps within documented tolerances of the CPU
oracle; mass conserved; reproducible; checkpoints. Skipped without CuPy / CUDA."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.bff.vm_cupy import available
from alife_lab.substrates.flow_lenia import core
from alife_lab.substrates.flow_lenia.flow_lenia import FlowLeniaAdapter

pytestmark = pytest.mark.skipif(not available(), reason="cupy / CUDA device unavailable")
EXAMPLE_M = [[3, 1, 0], [0, 3, 1], [1, 0, 3]]


def spec(boundary: str = "wall") -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "f",
                "substrate": "flow_lenia",
                "seed": 0,
                "world": {"width": 64, "height": 64, "boundary": boundary},
                "initialization": {"size": 21},
                "dynamics": {"channels": 3, "rule": {"connectivity": EXAMPLE_M}},
            }
        }
    )


@settings(max_examples=15, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    sigma=st.sampled_from([0.3, 0.65, 1.2]),
    boundary=st.sampled_from(["wall", "torus"]),
    dd=st.integers(1, 6),
)
def test_reintegration_kernel_is_bitwise_equal_to_numpy(
    seed: int, sigma: float, boundary: str, dd: int
) -> None:
    import cupy

    if sigma >= dd:
        return
    rng = np.random.default_rng(seed)
    cells = rng.random((23, 29, 2)) * (rng.random((23, 29, 1)) < 0.7)
    mu = core.target_positions(cells.shape, rng.normal(0, 10, (23, 29, 2, 2)), 0.2, dd, sigma,
                               boundary)  # fmt: skip
    cpu = core.reintegrate(cells, mu, dd, sigma, boundary)
    gpu = core.reintegrate(cupy.asarray(cells), cupy.asarray(mu), dd, sigma, boundary).get()
    assert np.array_equal(cpu, gpu)


@pytest.mark.parametrize("boundary", ["wall", "torus"])
def test_gpu_steps_track_cpu_and_conserve_mass(boundary: str) -> None:
    sp = spec(boundary)
    cpu, gpu = FlowLeniaAdapter(), FlowLeniaAdapter()
    gpu.set_backend("cupy")
    a, b = cpu.initialize(sp), gpu.initialize(sp)
    for _ in range(200):
        a, b = cpu.step(a, sp), gpu.step(b, sp)
    assert gpu.state_distance(a, b)["value"] <= 1e-12
    assert abs(gpu.metrics(b, sp)["mass_drift"]) < 1e-12


def test_gpu_reproducible_checkpoint_and_tools() -> None:
    sp = spec()
    gpu = FlowLeniaAdapter()
    gpu.set_backend("cupy")
    s = gpu.initialize(sp)
    for _ in range(11):
        s = gpu.step(s, sp)
    s, info = gpu.apply_intervention(s, sp, "paint", {"x": 3, "y": 3, "radius": 1, "channel": 0})
    assert info["cells"] == 5 and type(s.cells).__module__.startswith("cupy")
    ser = gpu.serialize_state(s)
    assert isinstance(ser["arrays"]["cells"], np.ndarray)
    r = gpu.deserialize_state(decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp)
    for _ in range(20):
        s, r = gpu.step(s, sp), gpu.step(r, sp)
    assert gpu.state_distance(s, r)["exact_equal"]
    assert len(gpu.inspect(s, sp, {"x": 30, "y": 30})["channels"]) == 3
    assert len(gpu.render(s)[1]) == 64 * 64 * 3
