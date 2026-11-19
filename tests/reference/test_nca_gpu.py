"""Growing NCA on the CUDA build of JAX (backend jax_cuda): deterministic training with XLA's
deterministic ops, closeness to the CPU backend, and the bundled model regrowing on the GPU.
Skipped without the nca-gpu extra or a CUDA device."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from alife_lab.core.spec import ExperimentSpec


def _cuda() -> bool:
    if importlib.util.find_spec("jax_cuda13_plugin") is None:
        return False
    import jax

    try:
        return len(jax.devices("cuda")) > 0
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _cuda(), reason="JAX CUDA build or device unavailable")

SMALL = {
    "name": "g",
    "substrate": "nca_train",
    "seed": 0,
    "target": {"preset": "disc", "size": 12},
    "dynamics": {"numeric_policy": "float32_jax", "pad": 2, "batch_size": 4, "pool_size": 8,
                 "steps_min": 16, "steps_max": 24, "architecture": {"hidden": 32}},
}  # fmt: skip


def test_gpu_training_is_deterministic_and_close_to_cpu() -> None:
    from alife_lab.substrates.nca.train_adapter import NCATrainAdapter

    sp = ExperimentSpec.model_validate({"experiment": SMALL})
    states = {}
    for name, backend in (("gpu1", "jax_cuda"), ("gpu2", "jax_cuda"), ("cpu", "jax")):
        a = NCATrainAdapter()
        a.set_backend(backend)
        s = a.initialize(sp)
        for _ in range(5):
            s = a.step(s, sp)
        states[name] = s
    a = NCATrainAdapter()
    assert a.state_distance(states["gpu1"], states["gpu2"])["exact_equal"]
    diff = a.state_distance(states["gpu1"], states["cpu"])["value"]
    assert diff < 5e-3  # float32 rounding differs between XLA CPU and GPU kernels


def test_bundled_model_regrows_on_gpu() -> None:
    from pathlib import Path

    from alife_lab.service.catalog import load_presets
    from alife_lab.substrates.nca.nca_adapter import NCAAdapter

    raw = load_presets(Path(__file__).resolve().parents[2] / "configs")[
        "nca_heart24_pretrained"
    ].spec.model_dump(mode="json")
    raw["experiment"]["dynamics"]["numeric_policy"] = "float32_jax"
    sp = ExperimentSpec.model_validate(raw)
    a = NCAAdapter()
    a.set_backend("jax_cuda")
    s = a.initialize(sp)
    for _ in range(200):
        s = a.step(s, sp)
    s, _ = a.apply_intervention(s, sp, "damage", {"x": 18, "y": 14, "radius": 6})
    for _ in range(50):
        s = a.step(s, sp)
    assert a.metrics(s, sp)["target_mse"] < 0.01
    assert isinstance(s.x, np.ndarray)
