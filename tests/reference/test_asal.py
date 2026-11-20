"""ASAL scores (ports of SakanaAI/asal asal_metrics.py), the asal spec section, backend
selection by numeric policy, and (when PyTorch and the pinned CLIP weights are present) a CLIP
search with determinism, checkpoint continuation and the blank-world reference."""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import numpy as np
import pytest

from alife_lab.core.spec import ExecutionProfile, ExperimentSpec
from alife_lab.evolution import asal


def unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def test_supervised_target_pairs_frame_t_with_prompt_t_mod_p() -> None:
    e = np.eye(3)
    z = np.stack([e[0], e[1], e[0], e[1]])  # 4 frames
    assert asal.supervised_target(z, e[:2]) == 1.0  # prompts tiled as p0, p1, p0, p1
    assert asal.supervised_target(z[[1, 0, 1, 0]], e[:2]) == 0.0
    assert asal.supervised_target(np.stack([unit([1, 1, 0])]), e[:1]) == pytest.approx(2**-0.5)
    with pytest.raises(ValueError):
        asal.supervised_target(z[:3], e[:2])


def test_open_endedness_definition() -> None:
    e = np.eye(4)
    assert asal.open_endedness(e) == 0.0  # every frame new (all similarities 0)
    same = np.stack([e[0]] * 4)
    assert asal.open_endedness(same) == pytest.approx(-0.75)  # rows 1..3 max 1, row 0 -> 0
    # Negative similarities are clipped at 0 by the zero-masked tril, as in the original.
    opposite = np.stack([e[0], -e[0]])
    assert asal.open_endedness(opposite) == 0.0


def test_illumination() -> None:
    e = np.eye(3)
    assert asal.illumination(e) == 0.0
    assert asal.illumination(np.stack([e[0], e[0], e[1]])) == pytest.approx(-(1 + 1 + 0) / 3)


def clip_spec(**over: Any) -> ExperimentSpec:
    raw: dict[str, Any] = {
        "name": "t_asal",
        "substrate": "search",
        "seed": 3,
        "target": {"name": "t", "substrate": "lenia", "seed": 0,
                   "world": {"width": 32, "height": 32},
                   "dynamics": {"kernel": {"R": 6}}, "initialization": {"preset": "pattern"}},
        "space": [{"path": "dynamics.growth.m", "low": 0.1, "high": 0.3}],
        "dynamics": {"objective": "lenia_clip_target", "numeric_policy": "float32_clip_torch",
                     "population": 4, "eval_steps": 20, "max_alive_fraction": 1.0},
        "asal": {"prompts": ["a biological cell"], "frames": 2},
        "observations": {"metrics": ["best_score", "blank_score"]},
        "runner": {"max_steps": 4, "checkpoint_interval": 2},
    }  # fmt: skip
    for k, v in over.items():
        raw[k] = v
    return ExperimentSpec.model_validate({"experiment": raw})


def test_asal_section_rules() -> None:
    spec = clip_spec()
    assert spec.model_dump(mode="json")["experiment"]["asal"]["frames"] == 2
    base = spec.model_dump(mode="json")["experiment"]
    bad: list[dict[str, Any]] = [
        {"asal": None},  # CLIP objective without asal
        {"dynamics": {**base["dynamics"], "numeric_policy": "float64_cpu"}},
        {"asal": {"prompts": ["a", "b"], "frames": 3}},  # frames not a multiple of prompts
        {"asal": {"prompts": ["bad\nprompt"], "frames": 1}},
        {"dynamics": {**base["dynamics"], "objective": "lenia_clip_open_endedness"}},  # prompts
        {"observations": {"metrics": ["blank_score"]},
         "dynamics": {**base["dynamics"], "objective": "lenia_moving_localized",
                      "numeric_policy": "float64_cpu"}, "asal": None},
    ]  # fmt: skip
    for patch in bad:
        raw = json.loads(json.dumps(base))
        raw.update(patch)
        if raw.get("asal") is None:
            raw.pop("asal")
        with pytest.raises(ValueError):
            ExperimentSpec.model_validate({"experiment": raw})


def test_auto_backend_respects_numeric_policy() -> None:
    from alife_lab.execution.executors import ExecutionTarget, TargetCapabilities, _validate

    caps = TargetCapabilities(True, True, True, False, False, ("numpy", "torch", "torch_cuda"))
    target = ExecutionTarget("t", "fake", ("local_interactive",), 1, caps)

    def profile(backend: str, gpus: int) -> ExecutionProfile:
        return ExecutionProfile.model_validate(
            {"execution": {"id": f"p_{backend}", "target_selector": {
                "role": "local_interactive", "gpu_count": gpus, "backend": backend}}}
        )  # fmt: skip

    raw = clip_spec().model_dump(mode="json")["experiment"]
    raw.pop("asal")
    raw["dynamics"] = {"population": 4, "eval_steps": 20}
    raw["observations"] = {"metrics": ["best_score"]}
    plain = ExperimentSpec.model_validate({"experiment": raw})
    assert _validate(target, clip_spec(), profile("auto", 0)).resolved_backend == "torch"
    assert _validate(target, plain, profile("auto", 0)).resolved_backend == "numpy"
    gpu = _validate(target, clip_spec(), profile("gpu", 1))
    assert gpu.ok and gpu.resolved_backend == "torch_cuda"
    # The GPU backend refuses the float64 policy of a non-CLIP search.
    assert not _validate(target, plain, profile("gpu", 1)).ok


def weights_present() -> bool:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("open_clip") is None:
        return False
    from alife_lab.evolution.clip_model import MODELS, model_dir

    card = MODELS["openai_vit_b32"]
    return (model_dir() / card.key / card.filename).is_file()


@pytest.mark.skipif(not weights_present(), reason="PyTorch/open_clip or CLIP weights missing")
def test_clip_search_runs_deterministically() -> None:
    import torch

    from alife_lab.substrates.search.adapter import SearchAdapter

    backend = "torch_cuda" if torch.cuda.is_available() else "torch"
    spec = clip_spec()

    def run(gens: int) -> Any:
        a = SearchAdapter()
        a.set_backend(backend)
        st = a.initialize(spec)
        for _ in range(gens):
            st = a.step(st, spec)
        return a, st

    a, full = run(4)
    _, again = run(4)
    assert a.state_distance(full, again)["exact_equal"]
    assert full.embedding.shape == (16, 512)
    assert np.allclose(np.linalg.norm(full.embedding, axis=1), 1.0)
    m = a.metrics(full, spec)
    assert np.isfinite(m["best_score"]) and -1 <= m["blank_score"] <= 1
    _, half = run(2)
    resumed = a.deserialize_state(a.serialize_state(half), spec)
    for _ in range(2):
        resumed = a.step(resumed, spec)
    assert a.state_distance(full, resumed)["exact_equal"]
