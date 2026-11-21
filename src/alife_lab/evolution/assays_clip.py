"""Assay plan for candidates of CLIP target searches (claim ``clip_prompt_specific``).

Claim: "CLIP rates the candidate's rollout as closer to the search prompts than an empty world,
than a fixed set of unrelated prompts and than the search's base creature, reproducibly and
over a longer rollout." This is a statement about the foundation model's judgement (evidence
E1 -> E2), never about the pattern being what the prompt names.

The plan (control prompts, margins, horizons) is fixed before any check runs and hashed. Checks
re-run the unchanged target substrate on the CPU (NumPy) and embed frames with the pinned CLIP
model on the CPU, independently of the GPU batch used during the search.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from alife_lab.core.hashing import sha256_json
from alife_lab.core.spec import ExperimentSpec, SearchExperiment

PLAN_VERSION = "clip-1"
CONTROL_PROMPTS = (
    "a car",
    "a city at night",
    "a cat",
    "a printed page of text",
    "a chair",
    "a landscape photograph",
)
REPRODUCE_TOL = 0.02  # relative difference to the score recorded by the search
BLANK_MARGIN = 0.02  # target score must exceed the empty world's score by this much
CONTROL_MARGIN = 0.03  # ... and the best unrelated prompt's mean similarity by this much
BASE_MARGIN = 0.02  # ... and the base creature's target score by this much


def make_plan(
    search: SearchExperiment, candidate_spec: dict[str, Any], recorded: dict[str, float]
) -> dict[str, Any]:
    if search.asal is None or search.dynamics.objective != "lenia_clip_target":
        raise ValueError("clip_prompt_specific needs a CLIP target search")
    ExperimentSpec.model_validate(candidate_spec)
    a = search.asal
    seed_free = candidate_spec["experiment"]["initialization"].get("preset") == "pattern"
    raw = np.random.SeedSequence([search.seed, 3]).generate_state(1 if seed_free else 5,
                                                                  dtype=np.uint64)  # fmt: skip
    seeds = [int(v >> np.uint64(1)) for v in raw]
    steps = search.dynamics.eval_steps
    view = {"frames": a.frames, "crop": a.crop, "prompts": list(a.prompts),
            "controls": list(CONTROL_PROMPTS), "model": a.model}  # fmt: skip
    base = copy.deepcopy(candidate_spec)
    target = search.target.model_dump(mode="json")
    for p in search.space:  # the search target's own values at every searched path
        node: Any = base["experiment"]
        src: Any = target
        keys = p.path.split(".")
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
            src = src[int(k)] if isinstance(src, list) else src[k]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = src[int(last)]
        else:
            node[last] = src[last]
    base["experiment"]["name"] = f"{candidate_spec['experiment']['name']}-base"
    checks = [
        {"id": "reproduce", "kind": "clip", "spec": candidate_spec, "steps": steps, **view,
         "purpose": "Same rollout and view as the search; CPU re-run, CPU CLIP.",
         "rule": {"score_within_recorded": REPRODUCE_TOL}},
        {"id": "above_blank", "kind": "derived", "from": "reproduce",
         "purpose": "The target score beats an empty world shown the same way.",
         "rule": {"above_blank_by": BLANK_MARGIN}},
        {"id": "specificity", "kind": "derived", "from": "reproduce",
         "purpose": "The target prompts beat every pre-declared unrelated prompt.",
         "rule": {"above_controls_by": CONTROL_MARGIN}},
        {"id": "longer", "kind": "clip", "spec": candidate_spec, "steps": 2 * steps, **view,
         "purpose": f"Frames sampled over {2 * steps} ticks instead of {steps}.",
         "rule": {"above_blank_by": BLANK_MARGIN}},
        {"id": "baseline", "kind": "clip", "spec": base, "steps": steps, **view,
         "purpose": "The search target's own parameters under the same view.",
         "rule": {"candidate_above_this_by": BASE_MARGIN, "candidate": "reproduce"}},
    ]  # fmt: skip
    plan: dict[str, Any] = {
        "plan_version": PLAN_VERSION,
        "claim_kind": "clip_prompt_specific",
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


def run_check(check: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    """Re-run the rollout on every seed and score it with CLIP on the CPU."""
    from alife_lab.evolution import asal as scores
    from alife_lab.evolution.clip_model import get_embedder
    from alife_lab.evolution.objectives import rollout_lenia_frames

    t0 = time.perf_counter()
    emb = get_embedder(check["model"], "cpu")
    txt = emb.embed_texts(check["prompts"])
    ctl = emb.embed_texts(check["controls"])
    rows = []
    for s in seeds:
        raw = copy.deepcopy(check["spec"])
        raw["experiment"]["seed"] = s
        spec = ExperimentSpec.model_validate(raw)
        ro = rollout_lenia_frames(spec, int(check["steps"]), int(check["frames"]), 1.0,
                                  check.get("crop"))  # fmt: skip
        z = emb.embed_fields(ro.frames)
        blank = emb.embed_fields(np.zeros_like(ro.frames))
        controls = {p: float(np.mean(z @ ctl[i])) for i, p in enumerate(check["controls"])}
        row: dict[str, Any] = {
            "seed": s,
            "score": scores.supervised_target(z, txt),
            "blank": scores.supervised_target(blank, txt),
            "controls": controls,
            "spec_hash": spec.digest(),
        }
        rows.append(row)
    ctl_mean = {p: float(np.mean([r["controls"][p] for r in rows])) for p in check["controls"]}
    best_ctl = max(ctl_mean, key=lambda p: ctl_mean[p])
    return {
        "id": check["id"],
        "per_seed": rows,
        "score": float(np.mean([r["score"] for r in rows])),
        "blank": float(np.mean([r["blank"] for r in rows])),
        "controls": ctl_mean,
        "max_control": ctl_mean[best_ctl],
        "max_control_prompt": best_ctl,
        "device": "cpu",
        "seconds": time.perf_counter() - t0,
    }


def judge(plan: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    verdicts: dict[str, dict[str, Any]] = {}
    for check in plan["checks"]:
        cid, rule = check["id"], check["rule"]
        res = results.get(check.get("from", cid))
        if res is None or "error" in res or "score" not in res:
            verdicts[cid] = {"passed": None, "reason": (res or {}).get("error", "not run")}
            continue
        score = res["score"]
        if "score_within_recorded" in rule:
            rec = plan["recorded"].get("score")
            if rec is None or rec == 0:
                verdicts[cid] = {"passed": None, "reason": "no recorded score"}
                continue
            rel = abs(score - rec) / abs(rec)
            ok = rel <= rule["score_within_recorded"]
            reason = f"score {score:.4f} vs recorded {rec:.4f} (rel. {rel:.4f})"
        elif "above_blank_by" in rule:
            gap = score - res["blank"]
            ok = gap >= rule["above_blank_by"]
            reason = f"score {score:.4f} - blank {res['blank']:.4f} = {gap:+.4f}"
        elif "above_controls_by" in rule:
            gap = score - res["max_control"]
            ok = gap >= rule["above_controls_by"]
            worst = f'"{res["max_control_prompt"]}" {res["max_control"]:.4f}'
            reason = f"score {score:.4f} - best unrelated prompt {worst} = {gap:+.4f}"
        else:
            cand = (results.get(rule["candidate"]) or {}).get("score")
            if cand is None:
                verdicts[cid] = {"passed": None, "reason": "candidate check missing"}
                continue
            gap = cand - score
            ok = gap >= rule["candidate_above_this_by"]
            reason = f"candidate {cand:.4f} - base {score:.4f} = {gap:+.4f}"
        verdicts[cid] = {"passed": bool(ok), "reason": reason}
    passed = [v["passed"] for v in verdicts.values()]
    if any(p is False for p in passed):
        status, level = "rejected", "E1"
    elif all(p is True for p in passed):
        status, level = "validated", plan["max_evidence_level"]
    else:
        status, level = "inconclusive", "E1"
    return {"verdicts": verdicts, "status": status, "evidence_level": level}
