"""Assay plan for CLIP target-search candidates (claim clip_prompt_specific): plan contents
and the pre-declared judgement rules (synthetic results; CLIP itself is not needed here)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from alife_lab.core.spec import ExperimentSpec, SearchExperiment
from alife_lab.evolution import assays, assays_clip

ROOT = Path(__file__).parents[2]


def search() -> SearchExperiment:
    raw = yaml.safe_load((ROOT / "configs/scenarios/search_asal_lenia_mc.yaml").read_text())
    exp = ExperimentSpec.model_validate(raw).experiment
    assert isinstance(exp, SearchExperiment)
    return exp


def candidate(exp: SearchExperiment) -> dict[str, Any]:
    raw = exp.target.model_dump(mode="json")
    raw["dynamics"]["kernels"][0]["m"] = 0.25
    raw["name"] = "cand"
    return ExperimentSpec.model_validate({"experiment": raw}).model_dump(mode="json")


def test_plan_contents() -> None:
    exp = search()
    plan = assays.make_plan(exp, candidate(exp), {"score": 0.29}, "clip_prompt_specific")
    assert plan["plan_version"] == "clip-1" and plan["seed_free_initial_state"]
    ids = [c["id"] for c in plan["checks"]]
    assert ids == ["reproduce", "above_blank", "specificity", "longer", "baseline"]
    rep = plan["checks"][0]
    assert rep["prompts"] == ["a biological cell", "two biological cells"] and rep["crop"] == 64
    assert set(rep["controls"]).isdisjoint(rep["prompts"])
    assert plan["checks"][3]["steps"] == 2 * exp.dynamics.eval_steps
    base = plan["checks"][4]["spec"]["experiment"]["dynamics"]["kernels"][0]["m"]
    assert base == exp.target.dynamics.kernels[0].m  # the creature's own value, not 0.25
    with pytest.raises(ValueError):  # motion claims are not defined for CLIP searches
        assays.make_plan(exp, candidate(exp), {"score": 0.29}, "moving_localized")


def res(score: float, blank: float = 0.253, ctl: float = 0.22) -> dict[str, Any]:
    return {"score": score, "blank": blank, "max_control": ctl, "max_control_prompt": "a car"}


def test_judge_rules() -> None:
    exp = search()
    plan = assays.make_plan(exp, candidate(exp), {"score": 0.29}, "clip_prompt_specific")
    good = {"reproduce": res(0.29), "longer": res(0.28), "baseline": res(0.245)}
    out = assays.judge(plan, good)
    assert out["status"] == "validated" and out["evidence_level"] == "E2"
    for broken, check in (
        ({"reproduce": res(0.31)}, "reproduce"),  # 7 % off the record
        ({"reproduce": res(0.29, ctl=0.27)}, "specificity"),  # an unrelated prompt is close
        ({"reproduce": res(0.29, blank=0.28)}, "above_blank"),
        ({"longer": res(0.255)}, "longer"),  # the look fades over a longer rollout
        ({"baseline": res(0.28)}, "baseline"),  # the base creature is about as cell-like
    ):
        verdict = assays.judge(plan, {**good, **broken})
        assert verdict["status"] == "rejected" and verdict["verdicts"][check]["passed"] is False
    missing = assays.judge(plan, {**good, "longer": {"error": "boom"}})
    assert missing["status"] == "inconclusive"


def test_plan_is_json_and_hash_stable() -> None:
    exp = search()
    a = assays.make_plan(exp, candidate(exp), {"score": 0.29}, "clip_prompt_specific")
    b = json.loads(json.dumps(a))
    assert (
        assays_clip.make_plan(exp, candidate(exp), {"score": 0.29})["plan_hash"] == b["plan_hash"]
    )
