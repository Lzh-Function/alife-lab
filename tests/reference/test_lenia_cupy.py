"""Lenia GPU (CuPy, float64 cuFFT) backend against the NumPy oracle.

Not bitwise equal (different FFT library); documented tolerances: <= 1e-13 after 10 ticks,
<= 1e-10 after 100 ticks (chaotic amplification grows afterwards). GPU runs are bitwise
reproducible on the same device. Skipped without CuPy / CUDA."""

from __future__ import annotations

import numpy as np
import pytest

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.bff.vm_cupy import available
from alife_lab.substrates.lenia.lenia import LeniaAdapter

pytestmark = pytest.mark.skipif(not available(), reason="cupy / CUDA device unavailable")


def spec(**exp: object) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {"experiment": {"name": "l", "substrate": "lenia", "seed": 0, **exp}}
    )


def adapters() -> tuple[LeniaAdapter, LeniaAdapter]:
    cpu, gpu = LeniaAdapter(), LeniaAdapter()
    gpu.set_backend("cupy")
    return cpu, gpu


def test_gpu_tracks_cpu_oracle_within_tolerance() -> None:
    sp = spec()
    cpu, gpu = adapters()
    a, b = cpu.initialize(sp), gpu.initialize(sp)
    assert type(b.cells).__module__.startswith("cupy")
    for t in range(1, 101):
        a, b = cpu.step(a, sp), gpu.step(b, sp)
        d = gpu.state_distance(a, b)["value"]
        if t == 10:
            assert d <= 1e-13
    assert d <= 1e-10
    ma, mb = cpu.metrics(a, sp), gpu.metrics(b, sp)
    assert all(abs(ma[k] - mb[k]) <= 1e-9 for k in ma)


def test_gpu_is_reproducible_and_orbium_survives() -> None:
    sp = spec()
    _, gpu = adapters()
    runs = []
    for _ in range(2):
        s = gpu.initialize(sp)
        for _ in range(600):
            s = gpu.step(s, sp)
        runs.append(s)
    assert gpu.state_distance(runs[0], runs[1])["exact_equal"]
    assert 60 < gpu.metrics(runs[0], sp)["mass"] < 90


def test_gpu_checkpoint_roundtrip_and_interventions() -> None:
    sp = spec()
    _, gpu = adapters()
    s = gpu.initialize(sp)
    for _ in range(17):
        s = gpu.step(s, sp)
    s, info = gpu.apply_intervention(s, sp, "paint", {"x": 5, "y": 5, "radius": 2})
    assert info["cells"] == 13 and type(s.cells).__module__.startswith("cupy")
    ser = gpu.serialize_state(s)
    assert isinstance(ser["arrays"]["cells"], np.ndarray)
    r = gpu.deserialize_state(decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp)
    for _ in range(30):
        s, r = gpu.step(s, sp), gpu.step(r, sp)
    assert gpu.state_distance(s, r)["exact_equal"]
    _header, data = gpu.render(s)
    assert len(data) == 128 * 128
    assert gpu.inspect(s, sp, {"x": 64, "y": 64})["substrate"] == "lenia"
