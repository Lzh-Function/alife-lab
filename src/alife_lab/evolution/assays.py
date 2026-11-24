"""Assay plans for discovery candidates found by a search (Lenia targets).

A plan is written and hashed *before* any check runs; its pass rules never look at results.
Checks run the unchanged Lenia substrate headless with the search's objective measurements
(``evaluate_lenia``). Seeds come from a stream disjoint from the search and holdout seeds;
seed-free (pattern) initial states use one seed.

Claim kinds
-----------
* ``moving_localized``: the candidate stays alive, localised and concentrated, and moves.
  Required checks: ``reproduce``, ``longer``, ``larger_world``.
* ``faster_than_base``: the above, and it moves faster than the base parameters of the search
  target under the ``larger_world`` conditions (``baseline`` check).

Evidence (spec §9.7): a registered candidate is E1 (measured by the search objective). All
required checks passing gives E2 (reproduced under changed duration / world size / seeds).
E3 (intervention-supported mechanism) and E4 (independent implementation / external check)
are out of reach of this plan and are never assigned automatically.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from alife_lab.core.hashing import sha256_json
from alife_lab.core.spec import CulturalWorldExperiment, ExperimentSpec, SearchExperiment

PLAN_VERSION = 1
PASS_VIABLE = 0.8  # fraction of seeds that must be viable
REPRODUCE_TOL = 0.05  # relative speed tolerance vs. the recorded search value
LARGER_TOL = 0.10  # relative speed tolerance vs. the reproduce check
FASTER_MARGIN = 0.05  # candidate must be >= 5 % faster than the base parameters
CLAIM_KINDS = (
    "moving_localized",
    "faster_than_base",
    "clip_prompt_specific",
    "cumulative_culture",
)


def _seeds(search: SearchExperiment, count: int) -> list[int]:
    raw = np.random.SeedSequence([search.seed, 3]).generate_state(count, dtype=np.uint64)
    return [int(v >> np.uint64(1)) for v in raw]


def _with(spec_json: dict[str, Any], **changes: Any) -> dict[str, Any]:
    out = copy.deepcopy(spec_json)
    exp = out["experiment"]
    for path, value in changes.items():
        node = exp
        keys = path.split("__")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return out


def make_plan(
    search: SearchExperiment | CulturalWorldExperiment,
    candidate_spec: dict[str, Any],
    recorded: dict[str, float],
    claim_kind: str,
) -> dict[str, Any]:
    if claim_kind not in CLAIM_KINDS:
        raise ValueError(f"unknown claim kind {claim_kind!r}")
    if claim_kind == "cumulative_culture":
        if not isinstance(search, CulturalWorldExperiment):
            raise ValueError("claim cumulative_culture requires a CulturalWorldExperiment")
        from alife_lab.evolution import cultural_assays

        return cultural_assays.make_plan(search, candidate_spec, recorded)
    if not isinstance(search, SearchExperiment):
        raise ValueError(f"claim {claim_kind!r} requires a SearchExperiment")
    if claim_kind == "clip_prompt_specific":
        from alife_lab.evolution import assays_clip

        return assays_clip.make_plan(search, candidate_spec, recorded)
    if search.asal is not None or search.target.substrate != "lenia":
        raise ValueError(f"{claim_kind} is defined for single-channel, non-CLIP Lenia searches")
    ExperimentSpec.model_validate(candidate_spec)  # must be a valid target spec
    seed_free = candidate_spec["experiment"]["initialization"].get("preset") == "pattern"
    seeds = _seeds(search, 1 if seed_free else 5)
    steps = search.dynamics.eval_steps
    long_steps = max(3000, 10 * steps)
    w = candidate_spec["experiment"]["world"]
    big = {"width": min(1024, 2 * w["width"]), "height": min(1024, 2 * w["height"])}
    big_world = {**w, **big}
    checks: list[dict[str, Any]] = [
        {
            "id": "reproduce",
            "purpose": "Same conditions as the search: viable and same speed (quantified).",
            "spec": candidate_spec,
            "steps": steps,
            "rule": {"viable_at_least": PASS_VIABLE,
                     "speed_within": {"of": "recorded", "relative": REPRODUCE_TOL}},
        },
        {
            "id": "longer",
            "purpose": f"Persistence: {long_steps} ticks instead of {steps}.",
            "spec": candidate_spec,
            "steps": long_steps,
            "rule": {"viable_at_least": PASS_VIABLE},
        },
        {
            "id": "larger_world",
            "purpose": f"Other world size ({big['width']} x {big['height']}), {long_steps} ticks.",
            "spec": _with(candidate_spec, world=big_world),
            "steps": long_steps,
            "rule": {"viable_at_least": PASS_VIABLE,
                     "speed_within": {"of": "reproduce", "relative": LARGER_TOL}},
        },
    ]  # fmt: skip
    if claim_kind == "faster_than_base":
        base = copy.deepcopy(candidate_spec)
        target = search.target.model_dump(mode="json")
        for p in search.space:
            node, src = base["experiment"], target
            keys = p.path.split(".")
            for k in keys[:-1]:
                node, src = node[k], src[k]
            node[keys[-1]] = src[keys[-1]]
        base["experiment"]["name"] = f"{candidate_spec['experiment']['name']}-base"
        checks.append(
            {
                "id": "baseline",
                "purpose": "Comparison: the search target's own parameters under the "
                "larger_world conditions; the candidate must be faster.",
                "spec": _with(base, world=big_world),
                "steps": long_steps,
                "rule": {
                    "candidate_faster_than_this_by": FASTER_MARGIN,
                    "candidate": "larger_world",
                },
            }
        )
    plan = {
        "plan_version": PLAN_VERSION,
        "claim_kind": claim_kind,
        "objective": search.dynamics.objective,
        "max_alive_fraction": search.dynamics.max_alive_fraction,
        "seeds": seeds,
        "seed_free_initial_state": seed_free,
        "recorded": recorded,
        "checks": checks,
        "status_rule": "validated (E2) if every check passes; rejected if any check fails; "
        "inconclusive if a check could not be evaluated",
        "max_evidence_level": "E2",
    }
    plan["plan_hash"] = sha256_json(plan)
    return plan


def run_check(check: dict[str, Any], seeds: list[int], max_alive: float) -> dict[str, Any]:
    """Run one check on every seed (in a worker process)."""
    from alife_lab.evolution.objectives import evaluate_lenia

    if check.get("kind") == "clip":
        from alife_lab.evolution import assays_clip

        return assays_clip.run_check(check, seeds)
    if check.get("kind") == "cultural":
        from alife_lab.evolution import cultural_assays

        return cultural_assays.run_check(check, seeds)

    t0 = time.perf_counter()
    rows = []
    for s in seeds:
        spec = ExperimentSpec.model_validate(_with(check["spec"], seed=s))
        ev = evaluate_lenia(spec, int(check["steps"]), max_alive)
        rows.append({"seed": s, "viable": ev.viable, "speed": ev.speed, "mass": ev.mass,
                     "alive_fraction": ev.alive_fraction, "survived": ev.survived,
                     "spec_hash": spec.digest()})  # fmt: skip
    viable = float(np.mean([r["viable"] for r in rows]))
    return {
        "id": check["id"],
        "per_seed": rows,
        "viable_fraction": viable,
        "mean_speed": float(np.mean([r["speed"] for r in rows])),
        "mean_mass": float(np.mean([r["mass"] for r in rows])),
        "seconds": time.perf_counter() - t0,
    }


def judge(plan: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-declared rules; returns per-check verdicts and the overall status."""
    if plan["claim_kind"] == "clip_prompt_specific":
        from alife_lab.evolution import assays_clip

        return assays_clip.judge(plan, results)
    if plan["claim_kind"] == "cumulative_culture":
        from alife_lab.evolution import cultural_assays

        return cultural_assays.judge(plan, results)
    verdicts: dict[str, dict[str, Any]] = {}
    for check in plan["checks"]:
        cid = check["id"]
        res = results.get(cid)
        rule = check["rule"]
        if res is None or "error" in res:
            verdicts[cid] = {"passed": None, "reason": (res or {}).get("error", "not run")}
            continue
        reasons, ok = [], True
        if "viable_at_least" in rule:
            good = res["viable_fraction"] >= rule["viable_at_least"]
            ok &= good
            need = rule["viable_at_least"]
            reasons.append(f"viable {res['viable_fraction']:.2f} >= {need}: {good}")
            if not good:  # a pattern that does not persist fails; speed is not compared
                verdicts[cid] = {"passed": False, "reason": reasons[0]}
                continue
        if "speed_within" in rule:
            ref_name = rule["speed_within"]["of"]
            ref = (plan["recorded"].get("speed") if ref_name == "recorded"
                   else (results.get(ref_name) or {}).get("mean_speed"))  # fmt: skip
            if ref is None or not np.isfinite(ref) or ref <= 0:
                verdicts[cid] = {"passed": None, "reason": f"no reference speed ({ref_name})"}
                continue
            rel = abs(res["mean_speed"] - ref) / ref
            good = rel <= rule["speed_within"]["relative"]
            ok &= good
            tol = rule["speed_within"]["relative"]
            reasons.append(
                f"speed {res['mean_speed']:.4f} vs {ref_name} {ref:.4f} "
                f"(rel. {rel:.3f} <= {tol}): {good}"
            )
        if "candidate_faster_than_this_by" in rule:
            cand = (results.get(rule["candidate"]) or {}).get("mean_speed")
            if cand is None:
                verdicts[cid] = {"passed": None, "reason": "candidate check missing"}
                continue
            margin = rule["candidate_faster_than_this_by"]
            good = cand >= res["mean_speed"] * (1 + margin)
            ok &= good
            base = res["mean_speed"]
            reasons.append(f"candidate {cand:.4f} >= base {base:.4f} x {1 + margin}: {good}")
        verdicts[cid] = {"passed": bool(ok), "reason": "; ".join(reasons)}
    passed = [v["passed"] for v in verdicts.values()]
    if any(p is False for p in passed):
        status, level = "rejected", "E1"
    elif all(p is True for p in passed):
        status, level = "validated", plan["max_evidence_level"]
    else:
        status, level = "inconclusive", "E1"
    return {"verdicts": verdicts, "status": status, "evidence_level": level}
