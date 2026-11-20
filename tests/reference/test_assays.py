"""Assay plans and pre-declared judgement rules for discovery candidates (with negative
controls: a dissolving candidate and a "faster than itself" claim must be rejected)."""

from __future__ import annotations

from typing import Any

import pytest

from alife_lab.core.spec import ExperimentSpec, SearchExperiment
from alife_lab.evolution import assays

SEARCH = ExperimentSpec.model_validate(
    {
        "experiment": {
            "name": "t",
            "substrate": "search",
            "seed": 4,
            "target": {
                "name": "o",
                "substrate": "lenia",
                "seed": 0,
                "world": {"width": 64, "height": 64},
                "initialization": {"preset": "pattern"},
            },
            "space": [
                {"path": "dynamics.growth.m", "low": 0.1, "high": 0.3},
                {"path": "dynamics.growth.s", "low": 0.005, "high": 0.05, "scale": "log"},
            ],
            "dynamics": {"eval_steps": 300},
        }
    }
).experiment
assert isinstance(SEARCH, SearchExperiment)


def candidate(m: float, s: float) -> dict[str, Any]:
    raw = SEARCH.target.model_dump(mode="json")
    raw["dynamics"]["growth"].update(m=m, s=s)
    return ExperimentSpec.model_validate({"experiment": raw}).model_dump(mode="json")


def plan_for(m: float, s: float, claim: str, speed: float) -> dict[str, Any]:
    return assays.make_plan(SEARCH, candidate(m, s), {"speed": speed}, claim)


def run_plan(plan: dict[str, Any]) -> dict[str, Any]:
    results = {
        c["id"]: assays.run_check(c, plan["seeds"], plan["max_alive_fraction"])
        for c in plan["checks"]
    }
    return assays.judge(plan, results)


def test_plan_is_deterministic_and_seed_free_for_patterns() -> None:
    a = plan_for(0.1358, 0.018, "faster_than_base", 0.77)
    b = plan_for(0.1358, 0.018, "faster_than_base", 0.77)
    assert a["plan_hash"] == b["plan_hash"] and a["seed_free_initial_state"]
    assert len(a["seeds"]) == 1 and [c["id"] for c in a["checks"]] == [
        "reproduce", "longer", "larger_world", "baseline",
    ]  # fmt: skip
    base = a["checks"][3]["spec"]["experiment"]["dynamics"]["growth"]
    assert (base["m"], base["s"]) == (0.15, 0.015)  # the search target's own parameters
    assert a["checks"][2]["spec"]["experiment"]["world"]["width"] == 128
    with pytest.raises(ValueError):
        plan_for(0.15, 0.015, "nonsense", 0.6)


def test_judge_rules() -> None:
    plan = plan_for(0.15, 0.015, "moving_localized", 0.6)
    ok = {"viable_fraction": 1.0, "mean_speed": 0.6}
    res = {c["id"]: dict(ok) for c in plan["checks"]}
    assert assays.judge(plan, res)["status"] == "validated"
    assert assays.judge(plan, res)["evidence_level"] == "E2"
    res["longer"] = {"viable_fraction": 0.0, "mean_speed": 0.0}
    assert assays.judge(plan, res)["status"] == "rejected"
    res["longer"] = {"error": "boom"}
    assert assays.judge(plan, res)["status"] == "inconclusive"
    res = {c["id"]: dict(ok) for c in plan["checks"]}
    res["reproduce"] = {"viable_fraction": 1.0, "mean_speed": 0.7}  # 17 % off the record
    assert assays.judge(plan, res)["verdicts"]["reproduce"]["passed"] is False


def test_found_variant_validated_and_negative_controls_rejected() -> None:
    found = run_plan(plan_for(0.1358, 0.018, "faster_than_base", 0.7711))
    assert found["status"] == "validated", found["verdicts"]
    # Negative control 1: a dissolving candidate (m = 0.17) is rejected.
    dissolving = run_plan(plan_for(0.17, 0.015, "moving_localized", 0.6))
    assert dissolving["status"] == "rejected"
    # Every check fails on viability (none is left "not evaluated" for want of a speed).
    assert all(v["passed"] is False for v in dissolving["verdicts"].values())
    # Negative control 2: the base parameters are not faster than themselves.
    itself = run_plan(plan_for(0.15, 0.015, "faster_than_base", 0.6235))
    assert itself["status"] == "rejected"
    assert itself["verdicts"]["baseline"]["passed"] is False
    assert itself["verdicts"]["larger_world"]["passed"] is True
