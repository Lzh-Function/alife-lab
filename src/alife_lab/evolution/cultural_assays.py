"""Pre-registered 5-criterion cumulative culture assay plans and E2/E3 judgement rules (P5).

A plan is hashed *before* any check runs; its pass rules are fixed and never inspect results first:
1. ``post_inventor_retention``: survivors across >= 80% of seeds, deaths > 0, retention >= 0.15.
2. ``compositional_refinement``: depth >= 2.0 and advantage over paired ``no_composition``.
3. ``transmission_ablation``: retention & utility/energy advantage over paired ``no_transmission``.
4. ``erase_culture_causal``: midpoint ``erase_culture`` + ``destroy_objects`` & ``mute_tx`` forks.
5. ``heldout_transfer``: generalization under unseen faster shifting environment + phase shock.

Evidence: E1 at registration (rejected on failure), E2 when checks 1/2/3/5 pass cleanly, and E3
when all 5 checks including causal ``erase_culture_causal`` pass cleanly.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from alife_lab.analysis.stats import cliffs_delta
from alife_lab.core.hashing import sha256_json
from alife_lab.core.spec import CulturalWorldExperiment, ExperimentSpec

PLAN_VERSION = 1
CLAIM_KIND = "cumulative_culture"
PASS_VIABLE = 0.8


def _seeds(exp: CulturalWorldExperiment, count: int) -> list[int]:
    raw = np.random.SeedSequence([exp.seed, 5]).generate_state(count, dtype=np.uint64)
    return [int(v >> np.uint64(1)) for v in raw]


def _with_exp(spec_json: dict[str, Any], seed: int | None = None, **updates: Any) -> dict[str, Any]:
    out = copy.deepcopy(spec_json)
    exp = out["experiment"]
    if seed is not None:
        exp["seed"] = int(seed)
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(exp.get(key), dict):
            _deep_update(exp[key], val)
        else:
            exp[key] = val
    return out


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def _step_run(
    spec_json: dict[str, Any], steps: int, *, shock_at: int | None = None
) -> dict[str, Any]:
    from alife_lab.substrates import get_adapter

    adapter = get_adapter("cultural_world")
    spec = ExperimentSpec.model_validate(spec_json)
    state = adapter.initialize(spec)
    for t in range(1, steps + 1):
        state = adapter.step(state, spec)
        if shock_at is not None and t == shock_at:
            state, _ = adapter.apply_intervention(
                state, spec, "shock_environment", {"phase_shift": 0.35}
            )
    m = adapter.metrics(state, spec)
    return {
        "metrics": m,
        "viable": bool(m["population"] > 0.0),
        "deaths": float(state.deaths),
    }


def make_plan(
    exp: CulturalWorldExperiment,
    candidate_spec: dict[str, Any],
    recorded: dict[str, float],
    *,
    n_seeds: int = 5,
    steps: int = 600,
) -> dict[str, Any]:
    """Build and hash the pre-declared 5-criterion cumulative culture assay plan."""
    ExperimentSpec.model_validate(candidate_spec)
    seeds = _seeds(exp, max(2, int(n_seeds)))
    steps = max(80, int(steps))
    t_mid = max(20, steps // 2)
    followup = min(60, max(10, steps - t_mid))

    ctrl_nocomp = _with_exp(
        candidate_spec,
        dynamics={"technology": {"allow_composition": False}},
    )
    ctrl_notx = _with_exp(
        candidate_spec,
        dynamics={
            "transmission_regime": "none",
            "transmission": {"vertical": False, "horizontal": False, "oblique": False},
        },
    )
    heldout_period = max(40, round(exp.world.regime_period * 0.65))
    heldout_cand = _with_exp(
        candidate_spec,
        world={"environment_regime": "shifting_patches", "regime_period": heldout_period},
    )
    heldout_ctrl = _with_exp(
        ctrl_notx,
        world={"environment_regime": "shifting_patches", "regime_period": heldout_period},
    )

    checks: list[dict[str, Any]] = [
        {
            "id": "post_inventor_retention",
            "kind": "cultural",
            "purpose": (
                "Viability >= 80%, nonzero inventor turnover (deaths > 0), and post-inventor "
                "skill retention >= 0.15 across independent seeds."
            ),
            "spec": candidate_spec,
            "steps": steps,
            "rule": {
                "viable_at_least": PASS_VIABLE,
                "min_deaths": 1.0,
                "min_retention": 0.15,
            },
        },
        {
            "id": "compositional_refinement",
            "kind": "cultural",
            "purpose": (
                "Refinement depth >= 2.0 and statistically positive improvement over paired "
                "allow_composition=False control."
            ),
            "spec": candidate_spec,
            "ctrl_spec": ctrl_nocomp,
            "steps": steps,
            "rule": {"min_depth": 2.0, "min_utility_ratio": 1.03},
        },
        {
            "id": "transmission_ablation",
            "kind": "cultural",
            "purpose": (
                "Higher skill retention and net utility / energy than paired "
                "transmission_regime='none' baseline sharing identical RNG env streams."
            ),
            "spec": candidate_spec,
            "ctrl_spec": ctrl_notx,
            "steps": steps,
            "rule": {
                "min_retention_gain": 0.10,
                "min_utility_ratio": 1.03,
                "min_energy_ratio": 1.05,
            },
        },
        {
            "id": "erase_culture_causal",
            "kind": "cultural",
            "purpose": (
                "Midpoint checkpoint paired-fork causal check: erase_culture + destroy_objects "
                "drops depth within 60 ticks and set_transmission mute reduces final utility."
            ),
            "spec": candidate_spec,
            "steps": steps,
            "midpoint": t_mid,
            "followup": followup,
            "rule": {"min_utility_ratio": 1.02, "min_energy_ratio": 1.05},
        },
        {
            "id": "heldout_transfer",
            "kind": "cultural",
            "purpose": (
                "Transfer to unseen faster shifting_patches regime + midpoint phase shock "
                "vs paired no-transmission control."
            ),
            "spec": heldout_cand,
            "ctrl_spec": heldout_ctrl,
            "steps": steps,
            "shock_at": t_mid,
            "rule": {"viable_at_least": PASS_VIABLE, "min_retention": 0.10},
        },
    ]
    plan: dict[str, Any] = {
        "plan_version": PLAN_VERSION,
        "claim_kind": CLAIM_KIND,
        "seeds": seeds,
        "seed_free_initial_state": False,
        "max_alive_fraction": 0.0,
        "recorded": recorded,
        "checks": checks,
        "status_rule": (
            "validated (E3) if all 5 checks pass; validated (E2) if checks 1, 2, 3, 5 pass "
            "and only erase_culture_causal fails; rejected (E1) otherwise; inconclusive on error"
        ),
        "max_evidence_level": "E3",
    }
    plan["plan_hash"] = sha256_json(plan)
    return plan


def run_check(check: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    """Execute one pre-registered cumulative-culture check across independent seeds."""
    from alife_lab.substrates import get_adapter

    cid = str(check["id"])
    steps = int(check["steps"])
    t0 = time.perf_counter()

    if cid == "post_inventor_retention":
        rows = [_step_run(_with_exp(check["spec"], seed=s), steps) for s in seeds]
        viable = float(np.mean([1.0 if r["viable"] else 0.0 for r in rows]))
        mean_deaths = float(np.mean([r["deaths"] for r in rows]))
        mean_ret = float(np.mean([r["metrics"]["skill_retention_after_death"] for r in rows]))
        return {
            "id": cid,
            "viable_fraction": viable,
            "mean_deaths": mean_deaths,
            "mean_retention": mean_ret,
            "summary_text": (
                f"viable={viable * 100:.0f}% · deaths={mean_deaths:.1f} · ret={mean_ret:.3f}"
            ),
            "seconds": round(time.perf_counter() - t0, 3),
        }

    if cid == "compositional_refinement":
        cand_rows = [_step_run(_with_exp(check["spec"], seed=s), steps) for s in seeds]
        ctrl_rows = [_step_run(_with_exp(check["ctrl_spec"], seed=s), steps) for s in seeds]
        cand_d = [float(r["metrics"]["improvement_depth"]) for r in cand_rows]
        ctrl_d = [float(r["metrics"]["improvement_depth"]) for r in ctrl_rows]
        cand_u = [float(r["metrics"]["tech_utility"]) for r in cand_rows]
        ctrl_u = [float(r["metrics"]["tech_utility"]) for r in ctrl_rows]
        cand_cf = float(np.mean([r["metrics"]["compositional_skill_fraction"] for r in cand_rows]))
        return {
            "id": cid,
            "cand_depth": float(np.mean(cand_d)),
            "ctrl_depth": float(np.mean(ctrl_d)),
            "cand_net_utility": float(np.mean(cand_u)),
            "ctrl_net_utility": float(np.mean(ctrl_u)),
            "compositional_fraction": cand_cf,
            "cliffs_delta_depth": cliffs_delta(cand_d, ctrl_d),
            "cliffs_delta_utility": cliffs_delta(cand_u, ctrl_u),
            "summary_text": (
                f"depth={np.mean(cand_d):.2f} vs {np.mean(ctrl_d):.2f} · "
                f"net_util={np.mean(cand_u):.3f} vs {np.mean(ctrl_u):.3f} · comp={cand_cf:.2f}"
            ),
            "seconds": round(time.perf_counter() - t0, 3),
        }

    if cid == "transmission_ablation":
        cand_rows = [_step_run(_with_exp(check["spec"], seed=s), steps) for s in seeds]
        ctrl_rows = [_step_run(_with_exp(check["ctrl_spec"], seed=s), steps) for s in seeds]
        cr = float(np.mean([r["metrics"]["skill_retention_after_death"] for r in cand_rows]))
        br = float(np.mean([r["metrics"]["skill_retention_after_death"] for r in ctrl_rows]))
        cu = float(np.mean([r["metrics"]["tech_utility"] for r in cand_rows]))
        bu = float(np.mean([r["metrics"]["tech_utility"] for r in ctrl_rows]))
        ce = float(np.mean([r["metrics"]["mean_energy"] for r in cand_rows]))
        be = float(np.mean([r["metrics"]["mean_energy"] for r in ctrl_rows]))
        return {
            "id": cid,
            "cand_retention": cr,
            "ctrl_retention": br,
            "cand_utility": cu,
            "ctrl_utility": bu,
            "cand_energy": ce,
            "ctrl_energy": be,
            "summary_text": (
                f"ret={cr:.3f} vs {br:.3f} · net_util={cu:.3f} vs {bu:.3f} · E={ce:.2f} vs {be:.2f}"
            ),
            "seconds": round(time.perf_counter() - t0, 3),
        }

    if cid == "erase_culture_causal":
        adapter = get_adapter("cultural_world")
        t_mid = int(check["midpoint"])
        followup = int(check["followup"])
        rem = max(0, steps - t_mid - followup)
        i_d60, w_d60 = [], []
        i_uf, m_uf = [], []
        i_ef, m_ef = [], []
        h6_ext_lrn = []
        has_ext = False
        for s in seeds:
            spec_obj = ExperimentSpec.model_validate(_with_exp(check["spec"], seed=s))
            exp_obj = spec_obj.experiment
            assert isinstance(exp_obj, CulturalWorldExperiment)
            has_ext = has_ext or exp_obj.dynamics.technology.allow_externalization
            st = adapter.initialize(spec_obj)
            for _ in range(t_mid):
                st = adapter.step(st, spec_obj)
            snap = adapter.serialize_state(st)

            st_intact = adapter.deserialize_state(snap, spec_obj)
            for _ in range(followup):
                st_intact = adapter.step(st_intact, spec_obj)
            i_d60.append(float(adapter.metrics(st_intact, spec_obj)["improvement_depth"]))
            for _ in range(rem):
                st_intact = adapter.step(st_intact, spec_obj)
            m_int_end = adapter.metrics(st_intact, spec_obj)
            i_uf.append(float(m_int_end["tech_utility"]))
            i_ef.append(float(m_int_end["mean_energy"]))

            st_wiped = adapter.deserialize_state(snap, spec_obj)
            st_wiped, _ = adapter.apply_intervention(st_wiped, spec_obj, "erase_culture", {})
            st_wiped, _ = adapter.apply_intervention(st_wiped, spec_obj, "destroy_objects", {})
            for _ in range(followup):
                st_wiped = adapter.step(st_wiped, spec_obj)
            w_d60.append(float(adapter.metrics(st_wiped, spec_obj)["improvement_depth"]))

            st_mute = adapter.deserialize_state(snap, spec_obj)
            st_mute, _ = adapter.apply_intervention(
                st_mute,
                spec_obj,
                "set_transmission",
                {"vertical": False, "horizontal": False, "oblique": False, "externalized": False},
            )
            for _ in range(followup + rem):
                st_mute = adapter.step(st_mute, spec_obj)
            m_mut_end = adapter.metrics(st_mute, spec_obj)
            m_uf.append(float(m_mut_end["tech_utility"]))
            m_ef.append(float(m_mut_end["mean_energy"]))

            if exp_obj.dynamics.technology.allow_externalization:
                st_h6 = adapter.deserialize_state(snap, spec_obj)
                st_h6, _ = adapter.apply_intervention(st_h6, spec_obj, "erase_culture", {})
                for _ in range(followup):
                    st_h6 = adapter.step(st_h6, spec_obj)
                h6_ext_lrn.append(
                    float(adapter.metrics(st_h6, spec_obj)["externalized_learning_fraction"])
                )

        id60, wd60 = float(np.mean(i_d60)), float(np.mean(w_d60))
        iuf, muf = float(np.mean(i_uf)), float(np.mean(m_uf))
        ief, mef = float(np.mean(i_ef)), float(np.mean(m_ef))
        h6_val = float(np.mean(h6_ext_lrn)) if h6_ext_lrn else 0.0
        return {
            "id": cid,
            "intact_depth_60": id60,
            "wiped_depth_60": wd60,
            "intact_final_utility": iuf,
            "mute_final_utility": muf,
            "intact_final_energy": ief,
            "mute_final_energy": mef,
            "h6_erase_only_ext_fraction_60": h6_val if has_ext else None,
            "summary_text": (
                f"d60={id60:.2f} vs wiped={wd60:.2f} · mute net_util={iuf:.3f} vs {muf:.3f} · "
                f"E={ief:.2f} vs {mef:.2f}"
            ),
            "seconds": round(time.perf_counter() - t0, 3),
        }

    # heldout_transfer
    shock_at = int(check.get("shock_at", steps // 2))
    cand_h = [_step_run(_with_exp(check["spec"], seed=s), steps, shock_at=shock_at) for s in seeds]
    ctrl_h = [
        _step_run(_with_exp(check["ctrl_spec"], seed=s), steps, shock_at=shock_at) for s in seeds
    ]
    viable_h = float(np.mean([1.0 if r["viable"] else 0.0 for r in cand_h]))
    cr_h = float(np.mean([r["metrics"]["skill_retention_after_death"] for r in cand_h]))
    cu_h = float(np.mean([r["metrics"]["tech_utility"] for r in cand_h]))
    bu_h = float(np.mean([r["metrics"]["tech_utility"] for r in ctrl_h]))
    return {
        "id": cid,
        "viable_fraction": viable_h,
        "cand_heldout_retention": cr_h,
        "cand_heldout_utility": cu_h,
        "ctrl_heldout_utility": bu_h,
        "summary_text": (
            f"viable={viable_h * 100:.0f}% · ret={cr_h:.3f} · net_util={cu_h:.3f} vs {bu_h:.3f}"
        ),
        "seconds": round(time.perf_counter() - t0, 3),
    }


def judge(plan: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Judge pre-declared 5-criterion rules and assign E1/E2/E3 cleanly."""
    verdicts: dict[str, dict[str, Any]] = {}
    for check in plan["checks"]:
        cid = str(check["id"])
        rule = check["rule"]
        res = results.get(cid)
        if res is None or "error" in res:
            verdicts[cid] = {"passed": None, "reason": (res or {}).get("error", "not run")}
            continue

        if cid == "post_inventor_retention":
            ok_v = res["viable_fraction"] >= rule["viable_at_least"]
            ok_d = res["mean_deaths"] >= rule["min_deaths"]
            ok_r = res["mean_retention"] >= rule["min_retention"]
            ok = ok_v and ok_d and ok_r
            verdicts[cid] = {
                "passed": ok,
                "reason": (
                    f"viable={res['viable_fraction']:.2f}, deaths={res['mean_deaths']:.1f}, "
                    f"ret={res['mean_retention']:.3f}"
                ),
            }
        elif cid == "compositional_refinement":
            ok_depth = res["cand_depth"] >= rule["min_depth"]
            d_depth = res.get("cliffs_delta_depth") or 0.0
            d_util = res.get("cliffs_delta_utility") or 0.0
            better_depth = res["cand_depth"] > res["ctrl_depth"] and d_depth > 0.0
            better_util = (
                res["cand_net_utility"] > res["ctrl_net_utility"] * rule["min_utility_ratio"]
                and d_util > 0.0
            )
            ok = ok_depth and (better_depth or better_util) and (max(d_depth, d_util) > 0.0)
            verdicts[cid] = {
                "passed": ok,
                "reason": (
                    f"depth {res['cand_depth']:.2f} vs {res['ctrl_depth']:.2f} "
                    f"(d={d_depth:+.2f}), net_util {res['cand_net_utility']:.3f} vs "
                    f"{res['ctrl_net_utility']:.3f} (d={d_util:+.2f})"
                ),
            }
        elif cid == "transmission_ablation":
            ok_ret = res["cand_retention"] >= res["ctrl_retention"] + rule["min_retention_gain"]
            ok_util = res["cand_utility"] >= res["ctrl_utility"] * rule["min_utility_ratio"]
            ok_eng = res["cand_energy"] >= res["ctrl_energy"] * rule["min_energy_ratio"]
            ok = ok_ret and (ok_util or ok_eng)
            verdicts[cid] = {
                "passed": ok,
                "reason": (
                    f"ret {res['cand_retention']:.3f} vs {res['ctrl_retention']:.3f}, "
                    f"util {res['cand_utility']:.3f} vs {res['ctrl_utility']:.3f}, "
                    f"E {res['cand_energy']:.2f} vs {res['ctrl_energy']:.2f}"
                ),
            }
        elif cid == "erase_culture_causal":
            ok_d = res["intact_depth_60"] > res["wiped_depth_60"]
            ok_u = (
                res["intact_final_utility"] >= res["mute_final_utility"] * rule["min_utility_ratio"]
            )
            ok_e = res["intact_final_energy"] >= res["mute_final_energy"] * rule["min_energy_ratio"]
            ok = ok_d and (ok_u or ok_e)
            verdicts[cid] = {
                "passed": ok,
                "reason": (
                    f"depth+60 {res['intact_depth_60']:.2f} vs wiped {res['wiped_depth_60']:.2f}; "
                    f"mute util {res['intact_final_utility']:.3f} vs "
                    f"{res['mute_final_utility']:.3f}"
                ),
            }
        elif cid == "heldout_transfer":
            ok_v = res["viable_fraction"] >= rule["viable_at_least"]
            ok_r = res["cand_heldout_retention"] >= rule["min_retention"]
            ok_u = res["cand_heldout_utility"] >= res["ctrl_heldout_utility"]
            ok = ok_v and ok_r and ok_u
            verdicts[cid] = {
                "passed": ok,
                "reason": (
                    f"viable={res['viable_fraction']:.2f}, "
                    f"ret={res['cand_heldout_retention']:.3f}, "
                    f"util {res['cand_heldout_utility']:.3f} vs {res['ctrl_heldout_utility']:.3f}"
                ),
            }
        else:
            verdicts[cid] = {"passed": None, "reason": "unknown check"}

    if any(v["passed"] is None for v in verdicts.values()):
        return {"verdicts": verdicts, "status": "inconclusive", "evidence_level": "E1"}
    core_ids = (
        "post_inventor_retention",
        "compositional_refinement",
        "transmission_ablation",
        "heldout_transfer",
    )
    core_ok = all(bool(verdicts[cid]["passed"]) for cid in core_ids)
    causal_ok = bool(verdicts["erase_culture_causal"]["passed"])
    if core_ok and causal_ok:
        return {"verdicts": verdicts, "status": "validated", "evidence_level": "E3"}
    if core_ok:
        return {"verdicts": verdicts, "status": "validated", "evidence_level": "E2"}
    return {"verdicts": verdicts, "status": "rejected", "evidence_level": "E1"}


def run_cultural_assay(
    spec: ExperimentSpec,
    *,
    n_seeds: int = 5,
    steps: int = 600,
) -> dict[str, Any]:
    """Run the complete 5-criterion cumulative culture assay headless and return report."""
    if not isinstance(spec.experiment, CulturalWorldExperiment):
        raise TypeError("run_cultural_assay requires a CulturalWorldExperiment spec")
    t0 = time.perf_counter()
    raw_spec = spec.model_dump(mode="json")
    plan = make_plan(spec.experiment, raw_spec, {}, n_seeds=n_seeds, steps=steps)
    results = {c["id"]: run_check(c, plan["seeds"]) for c in plan["checks"]}
    verdict = judge(plan, results)
    return {
        "preset_name": spec.experiment.name,
        "spec_hash": spec.digest(),
        "plan_hash": plan["plan_hash"],
        "seeds": plan["seeds"],
        "steps": steps,
        "results": results,
        "judgement": verdict,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


def run_cultural_phase_diagram(
    spec: ExperimentSpec,
    *,
    regime_periods: tuple[int, ...] = (80, 200, 500),
    tx_costs: tuple[float, ...] = (0.0, 0.015, 0.04),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 500,
) -> dict[str, Any]:
    """Run an exploratory 2D phase sweep over (regime_period × transmission_cost)."""
    if not isinstance(spec.experiment, CulturalWorldExperiment):
        raise TypeError("run_cultural_phase_diagram requires a CulturalWorldExperiment spec")
    t0 = time.perf_counter()
    raw_spec = spec.model_dump(mode="json")
    cells: list[dict[str, Any]] = []
    for period in regime_periods:
        for cost in tx_costs:
            cond_spec = _with_exp(
                raw_spec,
                world={"regime_period": int(period)},
                dynamics={"transmission": {"transmission_cost": float(cost)}},
            )
            runs = [_step_run(_with_exp(cond_spec, seed=s), steps) for s in seeds]
            cells.append(
                {
                    "regime_period": int(period),
                    "transmission_cost": float(cost),
                    "viable_fraction": float(np.mean([1.0 if r["viable"] else 0.0 for r in runs])),
                    "mean_energy": round(
                        float(np.mean([r["metrics"]["mean_energy"] for r in runs])), 4
                    ),
                    "mean_depth": round(
                        float(np.mean([r["metrics"]["improvement_depth"] for r in runs])), 3
                    ),
                    "mean_retention": round(
                        float(np.mean([r["metrics"]["skill_retention_after_death"] for r in runs])),
                        4,
                    ),
                    "mean_net_utility": round(
                        float(np.mean([r["metrics"]["tech_utility"] for r in runs])), 4
                    ),
                    "mean_comp_fraction": round(
                        float(
                            np.mean([r["metrics"]["compositional_skill_fraction"] for r in runs])
                        ),
                        4,
                    ),
                }
            )
    return {
        "preset_name": spec.experiment.name,
        "seeds": list(seeds),
        "steps": steps,
        "regime_periods": list(regime_periods),
        "transmission_costs": list(tx_costs),
        "cells": cells,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }
