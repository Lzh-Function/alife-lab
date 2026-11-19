"""Gray-Scott on CuPy: element-wise operations only, measured bit-identical to the NumPy
backend (state); metric reductions differ in the last ulp. Skipped without CuPy / CUDA."""

from __future__ import annotations

import pytest

from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.bff.vm_cupy import available
from alife_lab.substrates.gray_scott.gray_scott import GrayScottAdapter

pytestmark = pytest.mark.skipif(not available(), reason="cupy / CUDA device unavailable")


def test_gpu_is_bitwise_equal_to_numpy() -> None:
    sp = ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "g",
                "substrate": "gray_scott",
                "seed": 1,
                "world": {"width": 128, "height": 128, "length": 1.25},
                "dynamics": {"F": 0.0545, "k": 0.062},
            }
        }
    )
    cpu, gpu = GrayScottAdapter(), GrayScottAdapter()
    gpu.set_backend("cupy")
    a, b = cpu.initialize(sp), gpu.initialize(sp)
    for _ in range(2000):
        a, b = cpu.step(a, sp), gpu.step(b, sp)
    assert gpu.state_distance(a, b)["exact_equal"]
    # Reductions (sum/mean) use a different summation order on the GPU: last-ulp differences.
    ma, mb = cpu.metrics(a, sp), gpu.metrics(b, sp)
    assert all(mb[k] == pytest.approx(ma[k], rel=1e-12, abs=1e-15) for k in ma)
