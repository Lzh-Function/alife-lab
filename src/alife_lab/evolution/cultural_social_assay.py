"""P6 H4 atomic barter & division-of-labor assay plus P7 scaling/GPU benchmarks."""

from __future__ import annotations

import contextlib
import copy
import itertools
import math
import resource
import sys
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from alife_lab.analysis.stats import compare
from alife_lab.core.spec import CulturalWorldExperiment, ExperimentSpec

_P6_REQUIRED_METRICS: tuple[str, ...] = (
    "population",
    "mean_energy",
    "resource_ledger_residual",
    "tech_utility",
    "trade_count",
    "trade_volume",
    "specialization_index",
    "division_of_labor_index",
    "interdependence_index",
    "repeat_partner_rate",
)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def _with_exp(spec_json: dict[str, Any], seed: int | None = None, **updates: Any) -> dict[str, Any]:
    out = copy.deepcopy(spec_json)
    exp = out["experiment"]
    if seed is not None:
        exp["seed"] = int(seed)
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(exp.get(key), dict):
            _deep_update(exp[key], val)
        else:
            exp[key] = copy.deepcopy(val)
    return out


def _ensure_social_metrics(spec_json: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(spec_json)
    obs = out["experiment"].setdefault("observations", {})
    existing = list(obs.get("metrics", ()))
    seen = set(existing)
    for name in _P6_REQUIRED_METRICS:
        if name not in seen:
            existing.append(name)
            seen.add(name)
    obs["metrics"] = existing
    return out


def _finite_float(val: Any, default: float = 0.0) -> float:
    f = float(val)
    return f if math.isfinite(f) else default


def _step_residual(state: Any) -> float:
    from alife_lab.substrates.cultural_world.ecology import max_ledger_residual

    a = state.agents
    n = a.n
    art_a = sum(float(art.mass_a) for art in state.artifacts)
    art_b = sum(float(art.mass_b) for art in state.artifacts)
    return float(
        max_ledger_residual(
            state.eco,
            (float(a.inv_a.sum()) if n else 0.0) + art_a,
            (float(a.inv_b.sum()) if n else 0.0) + art_b,
        )
    )


def _paired_seeds(exp: CulturalWorldExperiment, count: int) -> list[int]:
    raw = np.random.SeedSequence([int(exp.seed), 6]).generate_state(count, dtype=np.uint64)
    return [int(v >> np.uint64(1)) for v in raw]


def _paired_permutation_test(
    x: list[float], y: list[float], *, exact_limit: int = 16_384, seed: int = 0
) -> dict[str, Any]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return {"p_value": None, "method": "undefined", "permutations": 0}
    diffs = a - b
    n = int(diffs.size)
    observed = float(abs(diffs.mean()))
    tol = 1e-12 * max(1.0, observed)
    total = 1 << n
    if total <= exact_limit:
        count = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            val = float(abs(np.dot(signs, diffs) / n))
            if val >= observed - tol:
                count += 1
        return {
            "p_value": float(count / total),
            "method": "exact_paired",
            "permutations": total,
        }
    rng = np.random.default_rng(seed)
    n_mc = 20_000
    signs_mat = rng.choice((-1.0, 1.0), size=(n_mc, n))
    vals = np.abs((signs_mat * diffs[None, :]).mean(axis=1))
    count = int((vals >= observed - tol).sum())
    return {
        "p_value": float((count + 1) / (n_mc + 1)),
        "method": "monte_carlo_paired",
        "permutations": n_mc,
        "seed": seed,
    }


def _paired_comparison(x: list[float], y: list[float], *, seed: int) -> dict[str, Any]:
    comp = compare(x, y, seed=seed)
    comp["paired_permutation"] = _paired_permutation_test(x, y, seed=seed)
    return comp


def _run_simple_arm(
    spec_json: dict[str, Any], seeds: list[int], steps: int
) -> list[dict[str, Any]]:
    from alife_lab.substrates import get_adapter

    adapter = get_adapter("cultural_world")
    rows: list[dict[str, Any]] = []
    for s in seeds:
        spec_obj = ExperimentSpec.model_validate(_with_exp(spec_json, seed=s))
        st = adapter.initialize(spec_obj)
        max_res = _step_residual(st)
        for _ in range(steps):
            st = adapter.step(st, spec_obj)
            res = _step_residual(st)
            if res > max_res:
                max_res = res
        m = adapter.metrics(st, spec_obj)
        rows.append(
            {
                "metrics": m,
                "viable": bool(m["population"] > 0.0),
                "max_ledger_residual": float(max(max_res, m["resource_ledger_residual"])),
            }
        )
    return rows


def _summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    energies = [_finite_float(r["metrics"]["mean_energy"]) for r in rows if r["viable"]]
    return {
        "mean_energy": round(float(np.mean(energies)) if energies else 0.0, 4),
        "mean_tech_utility": round(
            float(np.mean([_finite_float(r["metrics"]["tech_utility"]) for r in rows])), 4
        ),
        "mean_trade_count": round(
            float(np.mean([_finite_float(r["metrics"]["trade_count"]) for r in rows])), 2
        ),
        "mean_trade_volume": round(
            float(np.mean([_finite_float(r["metrics"]["trade_volume"]) for r in rows])), 4
        ),
        "mean_specialization_index": round(
            float(np.mean([_finite_float(r["metrics"]["specialization_index"]) for r in rows])), 4
        ),
        "mean_division_of_labor_index": round(
            float(np.mean([_finite_float(r["metrics"]["division_of_labor_index"]) for r in rows])),
            4,
        ),
        "mean_interdependence_index": round(
            float(np.mean([_finite_float(r["metrics"]["interdependence_index"]) for r in rows])), 4
        ),
        "mean_repeat_partner_rate": round(
            float(np.mean([_finite_float(r["metrics"]["repeat_partner_rate"]) for r in rows])), 4
        ),
        "max_ledger_residual": float(max(r["max_ledger_residual"] for r in rows)),
        "viable_fraction": round(float(np.mean([1.0 if r["viable"] else 0.0 for r in rows])), 4),
    }


def _col(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [_finite_float(r["metrics"].get(key, 0.0)) for r in rows]


def run_h4_exchange_assay(
    spec: ExperimentSpec,
    n_seeds: int = 3,
    steps: int = 400,
) -> dict[str, Any]:
    """Run the 4-arm H4 exchange, role-shuffle, and midpoint forbid_exchange causal assay."""
    if not isinstance(spec.experiment, CulturalWorldExperiment):
        raise TypeError("run_h4_exchange_assay requires a CulturalWorldExperiment spec")

    from alife_lab.substrates import get_adapter

    t0 = time.perf_counter()
    n_seeds = max(1, int(n_seeds))
    steps = max(4, int(steps))
    t_mid = max(2, steps // 2)
    seeds = _paired_seeds(spec.experiment, n_seeds)

    base_json = _ensure_social_metrics(spec.model_dump(mode="json"))
    intact_tmpl = _with_exp(
        base_json,
        dynamics={
            "social": {
                "allow_exchange": True,
                "interdependent_resources": True,
                "shuffle_roles_control": False,
            }
        },
    )
    no_ex_tmpl = _with_exp(
        base_json,
        dynamics={
            "social": {
                "allow_exchange": False,
                "interdependent_resources": True,
                "shuffle_roles_control": False,
            }
        },
    )
    shuf_tmpl = _with_exp(
        base_json,
        dynamics={
            "social": {
                "allow_exchange": True,
                "interdependent_resources": True,
                "shuffle_roles_control": True,
            }
        },
    )

    adapter = get_adapter("cultural_world")
    intact_rows: list[dict[str, Any]] = []
    forbid_rows: list[dict[str, Any]] = []

    for s in seeds:
        spec_obj = ExperimentSpec.model_validate(_with_exp(intact_tmpl, seed=s))
        st_intact = adapter.initialize(spec_obj)
        max_res_mid = _step_residual(st_intact)
        for _ in range(t_mid):
            st_intact = adapter.step(st_intact, spec_obj)
            res = _step_residual(st_intact)
            if res > max_res_mid:
                max_res_mid = res

        mid_metrics = adapter.metrics(st_intact, spec_obj)
        mid_trades = _finite_float(mid_metrics["trade_count"])
        snap = adapter.serialize_state(st_intact)

        st_forbid = adapter.deserialize_state(snap, spec_obj)
        st_forbid, _ = adapter.apply_intervention(
            st_forbid, spec_obj, "forbid_exchange", {"forbid": True}
        )

        max_res_intact = max_res_mid
        max_res_forbid = max_res_mid
        for _ in range(steps - t_mid):
            st_intact = adapter.step(st_intact, spec_obj)
            r_i = _step_residual(st_intact)
            if r_i > max_res_intact:
                max_res_intact = r_i

            st_forbid = adapter.step(st_forbid, spec_obj)
            r_f = _step_residual(st_forbid)
            if r_f > max_res_forbid:
                max_res_forbid = r_f

        m_intact = adapter.metrics(st_intact, spec_obj)
        m_forbid = adapter.metrics(st_forbid, spec_obj)
        m_intact_ext = dict(m_intact)
        m_intact_ext["post_midpoint_trades"] = max(
            0.0, _finite_float(m_intact["trade_count"]) - mid_trades
        )
        m_forbid_ext = dict(m_forbid)
        m_forbid_ext["post_midpoint_trades"] = max(
            0.0, _finite_float(m_forbid["trade_count"]) - mid_trades
        )

        intact_rows.append(
            {
                "metrics": m_intact_ext,
                "viable": bool(m_intact["population"] > 0.0),
                "max_ledger_residual": float(
                    max(max_res_intact, m_intact["resource_ledger_residual"])
                ),
            }
        )
        forbid_rows.append(
            {
                "metrics": m_forbid_ext,
                "viable": bool(m_forbid["population"] > 0.0),
                "max_ledger_residual": float(
                    max(max_res_forbid, m_forbid["resource_ledger_residual"])
                ),
            }
        )

    no_ex_rows = _run_simple_arm(no_ex_tmpl, seeds, steps)
    shuf_rows = _run_simple_arm(shuf_tmpl, seeds, steps)

    arms = {
        "exchange_intact": _summarize_arm(intact_rows),
        "no_exchange_control": _summarize_arm(no_ex_rows),
        "role_shuffled_control": _summarize_arm(shuf_rows),
        "forbid_exchange_midpoint": _summarize_arm(forbid_rows),
    }

    comparisons = {
        "division_of_labor_intact_vs_no_exchange": _paired_comparison(
            _col(intact_rows, "division_of_labor_index"),
            _col(no_ex_rows, "division_of_labor_index"),
            seed=601,
        ),
        "division_of_labor_intact_vs_role_shuffled": _paired_comparison(
            _col(intact_rows, "division_of_labor_index"),
            _col(shuf_rows, "division_of_labor_index"),
            seed=602,
        ),
        "interdependence_intact_vs_no_exchange": _paired_comparison(
            _col(intact_rows, "interdependence_index"),
            _col(no_ex_rows, "interdependence_index"),
            seed=603,
        ),
        "interdependence_intact_vs_role_shuffled": _paired_comparison(
            _col(intact_rows, "interdependence_index"),
            _col(shuf_rows, "interdependence_index"),
            seed=604,
        ),
        "interdependence_intact_vs_forbid_midpoint": _paired_comparison(
            _col(intact_rows, "interdependence_index"),
            _col(forbid_rows, "interdependence_index"),
            seed=605,
        ),
        "post_midpoint_trades_intact_vs_forbid_midpoint": _paired_comparison(
            _col(intact_rows, "post_midpoint_trades"),
            _col(forbid_rows, "post_midpoint_trades"),
            seed=606,
        ),
        "trade_count_intact_vs_forbid_midpoint": _paired_comparison(
            _col(intact_rows, "trade_count"),
            _col(forbid_rows, "trade_count"),
            seed=607,
        ),
        "tech_utility_intact_vs_no_exchange": _paired_comparison(
            _col(intact_rows, "tech_utility"),
            _col(no_ex_rows, "tech_utility"),
            seed=608,
        ),
        "mean_energy_intact_vs_no_exchange": _paired_comparison(
            _col(intact_rows, "mean_energy"),
            _col(no_ex_rows, "mean_energy"),
            seed=609,
        ),
    }

    mean_post_intact = float(np.mean(_col(intact_rows, "post_midpoint_trades")))
    mean_post_forbid = float(np.mean(_col(forbid_rows, "post_midpoint_trades")))

    checks = {
        "mass_conservation": max(a["max_ledger_residual"] for a in arms.values()) <= 1e-9,
        "positive_trades": (
            arms["exchange_intact"]["mean_trade_count"] > 0.0
            and arms["exchange_intact"]["mean_trade_volume"] > 0.0
        ),
        "no_exchange_zero_trades": arms["no_exchange_control"]["mean_trade_count"] == 0.0,
        "division_of_labor_gt_no_exchange": (
            arms["exchange_intact"]["mean_division_of_labor_index"]
            > arms["no_exchange_control"]["mean_division_of_labor_index"]
        ),
        "division_of_labor_gt_role_shuffled": (
            arms["exchange_intact"]["mean_division_of_labor_index"]
            > arms["role_shuffled_control"]["mean_division_of_labor_index"]
        ),
        "forbid_exchange_halts_subsequent_trades": (
            mean_post_forbid == 0.0 and mean_post_intact > 0.0
        ),
        "forbid_exchange_reduces_interdependence": (
            arms["exchange_intact"]["mean_interdependence_index"]
            > arms["forbid_exchange_midpoint"]["mean_interdependence_index"]
        ),
    }

    return {
        "preset_name": spec.experiment.name,
        "seeds": seeds,
        "steps": steps,
        "arms": arms,
        "comparisons": comparisons,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


def _peak_rss_mib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return round(raw / (1024.0 * 1024.0), 2)
    return round(raw / 1024.0, 2)


@contextlib.contextmanager
def _phase_timer(totals_sec: dict[str, float]) -> Iterator[None]:
    """Instrument ``cultural_world.model`` phase boundaries without mutating substrate files."""
    from typing import cast

    from alife_lab.substrates.cultural_world import model as cw_model

    mod = cast(Any, cw_model)
    hook_names = (
        "step_ecology",
        "decay_artifacts",
        "compute_sensors",
        "choose_actions",
        "apply_social_transmission",
        "learn_from_artifacts",
        "apply_local_exchange",
        "step_world",
    )
    orig: dict[str, Any] = {name: getattr(mod, name, None) for name in hook_names}
    marks: dict[str, float] = {}

    def _wrap(
        fn_name: str,
        *,
        mark_before: str | None = None,
        mark_after: str | None = None,
    ) -> Any:
        fn = orig[fn_name]
        if fn is None:
            return None

        def _inner(*args: Any, **kwargs: Any) -> Any:
            if mark_before is not None:
                marks[mark_before] = time.perf_counter()
            res = fn(*args, **kwargs)
            if mark_after is not None:
                marks[mark_after] = time.perf_counter()
            return res

        return _inner

    def wrapped_step_world(*args: Any, **kwargs: Any) -> Any:
        marks.clear()
        res = orig["step_world"](*args, **kwargs)
        t_end = time.perf_counter()
        if "eco_0" in marks and "eco_1" in marks:
            totals_sec["ecology"] += marks["eco_1"] - marks["eco_0"]
        if "sp_0" in marks and "sp_1" in marks:
            totals_sec["sensors_policy"] += marks["sp_1"] - marks["sp_0"]
        if "sp_1" in marks and "tx_0" in marks:
            totals_sec["skills_tech"] += marks["tx_0"] - marks["sp_1"]
        if "tx_0" in marks and "tx_1" in marks:
            totals_sec["transmission"] += marks["tx_1"] - marks["tx_0"]
        if "ex_0" in marks and "ex_1" in marks:
            totals_sec["exchange"] += marks["ex_1"] - marks["ex_0"]
            totals_sec["mortality_repro"] += t_end - marks["ex_1"]
        elif "tx_1" in marks:
            totals_sec["mortality_repro"] += t_end - marks["tx_1"]
        return res

    replacements: dict[str, Any] = {
        "step_ecology": _wrap("step_ecology", mark_before="eco_0"),
        "decay_artifacts": _wrap("decay_artifacts", mark_after="eco_1"),
        "compute_sensors": _wrap("compute_sensors", mark_before="sp_0"),
        "choose_actions": _wrap("choose_actions", mark_after="sp_1"),
        "apply_social_transmission": _wrap("apply_social_transmission", mark_before="tx_0"),
        "learn_from_artifacts": _wrap("learn_from_artifacts", mark_after="tx_1"),
        "apply_local_exchange": _wrap(
            "apply_local_exchange", mark_before="ex_0", mark_after="ex_1"
        ),
        "step_world": wrapped_step_world,
    }
    for name, fn in replacements.items():
        if fn is not None:
            setattr(mod, name, fn)
    try:
        yield
    finally:
        for name, fn in orig.items():
            if fn is not None:
                setattr(mod, name, fn)


def _probe_gpu_mlp(n_agents: int, iters: int, seed: int) -> dict[str, Any]:
    """Safely benchmark batched sensor+MLP pass on CUDA vs NumPy CPU when requested."""
    rng = np.random.default_rng(seed)
    n = max(64, int(n_agents))
    iters = max(20, int(iters))
    r = 32
    w_world, h_world = 64.0, 64.0
    h_dim = 8
    n_in = 12
    n_act = 6
    food_np = rng.random((r, r), dtype=np.float64)
    mata_np = rng.random((r, r), dtype=np.float64) * 0.8
    matb_np = rng.random((r, r), dtype=np.float64) * 0.8
    ax_np = rng.random(n, dtype=np.float64) * w_world
    ay_np = rng.random(n, dtype=np.float64) * h_world
    hd_np = rng.random(n, dtype=np.float64) * (2.0 * math.pi)
    inv_a_np = rng.random(n, dtype=np.float64)
    inv_b_np = rng.random(n, dtype=np.float64)
    eng_np = rng.uniform(2.0, 9.0, size=n).astype(np.float64)
    has_sk_np = (rng.random(n) > 0.3).astype(np.float64)

    w1_np = rng.standard_normal((n, n_in, h_dim), dtype=np.float64) / math.sqrt(n_in)
    b1_np = np.zeros((n, h_dim), dtype=np.float64)
    w2_np = rng.standard_normal((n, h_dim, n_act), dtype=np.float64) / math.sqrt(h_dim)
    b2_np = np.zeros((n, n_act), dtype=np.float64)
    temp = 1.0
    ang = math.radians(40.0)
    sensor_dist = max(1.0, w_world / r)
    s_sin, s_cos = math.sin(0.5), math.cos(0.5)

    def _cpu_pass() -> np.ndarray:
        x_buf = np.zeros((n, n_in), dtype=np.float64)
        for k, off in enumerate((ang, 0.0, -ang)):
            sx = (ax_np + sensor_dist * np.cos(hd_np + off)) % w_world
            sy = (ay_np + sensor_dist * np.sin(hd_np + off)) % h_world
            ci = np.floor(sy / h_world * r).astype(np.int64) % r
            cj = np.floor(sx / w_world * r).astype(np.int64) % r
            x_buf[:, k] = food_np[ci, cj]
        ci0 = np.floor(ay_np / h_world * r).astype(np.int64) % r
        cj0 = np.floor(ax_np / w_world * r).astype(np.int64) % r
        x_buf[:, 3] = mata_np[ci0, cj0] / 0.8
        x_buf[:, 4] = matb_np[ci0, cj0] / 0.8
        x_buf[:, 5] = np.clip(inv_a_np, 0.0, 2.0) * 0.5
        x_buf[:, 6] = np.clip(inv_b_np, 0.0, 2.0) * 0.5
        x_buf[:, 7] = np.clip(eng_np / 9.5, 0.0, 2.0)
        x_buf[:, 8] = has_sk_np
        x_buf[:, 9] = s_sin
        x_buf[:, 10] = s_cos
        x_buf[:, 11] = 1.0

        hid = np.tanh(np.einsum("ni,nih->nh", x_buf, w1_np) + b1_np)
        logits = np.einsum("nh,nha->na", hid, w2_np) + b2_np
        z = logits / temp
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return np.asarray(e / e.sum(axis=1, keepdims=True), dtype=np.float64)

    # Probe PyTorch CUDA first (lazy import so controller never imports GPU libs).
    try:
        import torch

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            ref = _cpu_pass()

            t0 = time.perf_counter()
            for _ in range(iters):
                _ = _cpu_pass()
            cpu_sec = max(1e-9, time.perf_counter() - t0)

            dev = torch.device("cuda:0")
            food_t = torch.from_numpy(food_np).to(dev)
            mata_t = torch.from_numpy(mata_np).to(dev)
            matb_t = torch.from_numpy(matb_np).to(dev)
            ax_t = torch.from_numpy(ax_np).to(dev)
            ay_t = torch.from_numpy(ay_np).to(dev)
            hd_t = torch.from_numpy(hd_np).to(dev)
            inva_t = torch.from_numpy(inv_a_np).to(dev)
            invb_t = torch.from_numpy(inv_b_np).to(dev)
            eng_t = torch.from_numpy(eng_np).to(dev)
            hsk_t = torch.from_numpy(has_sk_np).to(dev)
            w1_t = torch.from_numpy(w1_np).to(dev)
            b1_t = torch.from_numpy(b1_np).to(dev)
            w2_t = torch.from_numpy(w2_np).to(dev)
            b2_t = torch.from_numpy(b2_np).to(dev)

            def _torch_pass(
                f_t: Any,
                ma_t: Any,
                mb_t: Any,
                x_pos: Any,
                y_pos: Any,
                hdg: Any,
                ia: Any,
                ib: Any,
                en: Any,
                hs: Any,
                dt_w1: Any,
                dt_b1: Any,
                dt_w2: Any,
                dt_b2: Any,
            ) -> Any:
                cols = []
                for off in (ang, 0.0, -ang):
                    sx = torch.remainder(x_pos + sensor_dist * torch.cos(hdg + off), w_world)
                    sy = torch.remainder(y_pos + sensor_dist * torch.sin(hdg + off), h_world)
                    ci = torch.remainder(torch.floor(sy / h_world * r).long(), r)
                    cj = torch.remainder(torch.floor(sx / w_world * r).long(), r)
                    cols.append(f_t[ci, cj])
                ci0 = torch.remainder(torch.floor(y_pos / h_world * r).long(), r)
                cj0 = torch.remainder(torch.floor(x_pos / w_world * r).long(), r)
                cols.append(ma_t[ci0, cj0] / 0.8)
                cols.append(mb_t[ci0, cj0] / 0.8)
                cols.append(torch.clamp(ia, 0.0, 2.0) * 0.5)
                cols.append(torch.clamp(ib, 0.0, 2.0) * 0.5)
                cols.append(torch.clamp(en / 9.5, 0.0, 2.0))
                cols.append(hs)
                cols.append(torch.full_like(hs, s_sin))
                cols.append(torch.full_like(hs, s_cos))
                cols.append(torch.ones_like(hs))
                x_in = torch.stack(cols, dim=1)
                hid = torch.tanh(torch.bmm(x_in.unsqueeze(1), dt_w1).squeeze(1) + dt_b1)
                logits = torch.bmm(hid.unsqueeze(1), dt_w2).squeeze(1) + dt_b2
                return torch.softmax(logits / temp, dim=1)

            with torch.no_grad():
                for _ in range(5):
                    out_t = _torch_pass(
                        food_t,
                        mata_t,
                        matb_t,
                        ax_t,
                        ay_t,
                        hd_t,
                        inva_t,
                        invb_t,
                        eng_t,
                        hsk_t,
                        w1_t,
                        b1_t,
                        w2_t,
                        b2_t,
                    )
                torch.cuda.synchronize()
                max_diff = float(np.max(np.abs(out_t.cpu().numpy() - ref)))

                out32 = _torch_pass(
                    food_t.float(),
                    mata_t.float(),
                    matb_t.float(),
                    ax_t.float(),
                    ay_t.float(),
                    hd_t.float(),
                    inva_t.float(),
                    invb_t.float(),
                    eng_t.float(),
                    hsk_t.float(),
                    w1_t.float(),
                    b1_t.float(),
                    w2_t.float(),
                    b2_t.float(),
                )
                diff32 = float(np.max(np.abs(out32.cpu().numpy().astype(np.float64) - ref)))

                torch.cuda.synchronize()
                t1 = time.perf_counter()
                for _ in range(iters):
                    _ = _torch_pass(
                        food_t,
                        mata_t,
                        matb_t,
                        ax_t,
                        ay_t,
                        hd_t,
                        inva_t,
                        invb_t,
                        eng_t,
                        hsk_t,
                        w1_t,
                        b1_t,
                        w2_t,
                        b2_t,
                    )
                torch.cuda.synchronize()
                gpu_sec = max(1e-9, time.perf_counter() - t1)

            speedup = round(cpu_sec / gpu_sec, 3)
            return {
                "available": True,
                "backend": "torch_cuda",
                "speedup_vs_cpu": speedup,
                "max_abs_diff": max_diff,
                "note": (
                    f"{dev_name}: N={n}, iters={iters}, float64 CPU={cpu_sec * 1e3:.2f}ms vs "
                    f"GPU={gpu_sec * 1e3:.2f}ms (float32 max_abs_diff={diff32:.2e})."
                ),
            }
    except Exception as exc:
        torch_err = str(exc)
    else:
        torch_err = "torch.cuda.is_available() returned False"

    # Fallback probe: CuPy
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) > 0:
            ref = _cpu_pass()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = _cpu_pass()
            cpu_sec = max(1e-9, time.perf_counter() - t0)

            food_g = cp.asarray(food_np)
            mata_g = cp.asarray(mata_np)
            matb_g = cp.asarray(matb_np)
            ax_g = cp.asarray(ax_np)
            ay_g = cp.asarray(ay_np)
            hd_g = cp.asarray(hd_np)
            inva_g = cp.asarray(inv_a_np)
            invb_g = cp.asarray(inv_b_np)
            eng_g = cp.asarray(eng_np)
            hsk_g = cp.asarray(has_sk_np)
            w1_g = cp.asarray(w1_np)
            b1_g = cp.asarray(b1_np)
            w2_g = cp.asarray(w2_np)
            b2_g = cp.asarray(b2_np)

            def _cp_pass() -> Any:
                cols = []
                for off in (ang, 0.0, -ang):
                    sx = (ax_g + sensor_dist * cp.cos(hd_g + off)) % w_world
                    sy = (ay_g + sensor_dist * cp.sin(hd_g + off)) % h_world
                    ci = cp.floor(sy / h_world * r).astype(cp.int64) % r
                    cj = cp.floor(sx / w_world * r).astype(cp.int64) % r
                    cols.append(food_g[ci, cj])
                ci0 = cp.floor(ay_g / h_world * r).astype(cp.int64) % r
                cj0 = cp.floor(ax_g / w_world * r).astype(cp.int64) % r
                cols.append(mata_g[ci0, cj0] / 0.8)
                cols.append(matb_g[ci0, cj0] / 0.8)
                cols.append(cp.clip(inva_g, 0.0, 2.0) * 0.5)
                cols.append(cp.clip(invb_g, 0.0, 2.0) * 0.5)
                cols.append(cp.clip(eng_g / 9.5, 0.0, 2.0))
                cols.append(hsk_g)
                cols.append(cp.full_like(hsk_g, s_sin))
                cols.append(cp.full_like(hsk_g, s_cos))
                cols.append(cp.ones_like(hsk_g))
                x_g = cp.stack(cols, axis=1)
                hid = cp.tanh(cp.einsum("ni,nih->nh", x_g, w1_g) + b1_g)
                logits = cp.einsum("nh,nha->na", hid, w2_g) + b2_g
                z = logits / temp
                z -= z.max(axis=1, keepdims=True)
                e = cp.exp(z)
                return e / e.sum(axis=1, keepdims=True)

            out_g = _cp_pass()
            cp.cuda.get_current_stream().synchronize()
            max_diff = float(np.max(np.abs(cp.asnumpy(out_g) - ref)))
            t1 = time.perf_counter()
            for _ in range(iters):
                _ = _cp_pass()
            cp.cuda.get_current_stream().synchronize()
            gpu_sec = max(1e-9, time.perf_counter() - t1)

            return {
                "available": True,
                "backend": "cupy",
                "speedup_vs_cpu": round(cpu_sec / gpu_sec, 3),
                "max_abs_diff": max_diff,
                "note": (
                    f"CuPy CUDA: N={n}, iters={iters}, float64 CPU={cpu_sec * 1e3:.2f}ms vs "
                    f"GPU={gpu_sec * 1e3:.2f}ms."
                ),
            }
    except Exception as exc:
        cupy_err = str(exc)
    else:
        cupy_err = "no CUDA devices detected"

    return {
        "available": False,
        "backend": "none",
        "speedup_vs_cpu": None,
        "max_abs_diff": None,
        "note": f"CUDA unavailable (torch: {torch_err}; cupy: {cupy_err}).",
    }


def _default_benchmark_spec() -> ExperimentSpec:
    from alife_lab.service.catalog import load_presets
    from alife_lab.service.settings import load_settings

    presets = load_presets(load_settings().configs_dir)
    if "cultural_world_exchange" in presets:
        return presets["cultural_world_exchange"].spec
    if "cultural_world_compositional" in presets:
        raw = presets["cultural_world_compositional"].spec.model_dump(mode="json")
        return ExperimentSpec.model_validate(
            _with_exp(
                raw,
                dynamics={"social": {"allow_exchange": True, "interdependent_resources": True}},
            )
        )
    raise RuntimeError("no cultural_world preset found for scaling benchmark")


def run_cultural_scaling_benchmark(
    populations: tuple[int, ...] = (80, 200, 400),
    max_skills_grid: tuple[int, ...] = (2, 4, 8),
    steps: int = 100,
    seed: int = 1,
    include_gpu: bool = False,
    *,
    spec: ExperimentSpec | None = None,
) -> dict[str, Any]:
    """Benchmark CulturalWorld CPU scaling, per-phase hotspots, and optional GPU MLP probe."""
    from alife_lab.substrates import get_adapter
    from alife_lab.substrates.cultural_world import model as cw_model

    t0 = time.perf_counter()
    base_spec = spec if spec is not None else _default_benchmark_spec()
    if not isinstance(base_spec.experiment, CulturalWorldExperiment):
        raise TypeError("run_cultural_scaling_benchmark requires a CulturalWorldExperiment spec")

    steps = max(1, int(steps))
    seed = int(seed)
    base_json = _ensure_social_metrics(base_spec.model_dump(mode="json"))
    adapter = get_adapter("cultural_world")

    grid: list[dict[str, Any]] = []
    for pop in populations:
        pop_i = max(4, int(pop))
        for max_sk in max_skills_grid:
            sk_i = max(1, int(max_sk))
            cell_json = _with_exp(
                base_json,
                seed=seed,
                initialization={"count": pop_i},
                dynamics={
                    "max_population": max(int(base_spec.experiment.dynamics.max_population), pop_i),
                    "technology": {"max_skills_per_agent": sk_i},
                },
            )
            cell_spec = ExperimentSpec.model_validate(cell_json)
            exp_obj = cell_spec.experiment
            assert isinstance(exp_obj, CulturalWorldExperiment)

            phase_sec: dict[str, float] = {
                "ecology": 0.0,
                "sensors_policy": 0.0,
                "skills_tech": 0.0,
                "transmission": 0.0,
                "exchange": 0.0,
                "mortality_repro": 0.0,
            }
            st = adapter.initialize(cell_spec)
            max_res = _step_residual(st)
            total_agent_ticks = 0

            run_t0 = time.perf_counter()
            with _phase_timer(phase_sec):
                for _ in range(steps):
                    total_agent_ticks += int(st.agents.n)
                    st = cw_model.step_world(st, exp_obj)
                    res = _step_residual(st)
                    if res > max_res:
                        max_res = res
            wall_cell = max(1e-9, time.perf_counter() - run_t0)

            grid.append(
                {
                    "population": pop_i,
                    "max_skills": sk_i,
                    "steps": steps,
                    "wall_seconds": round(wall_cell, 4),
                    "ms_per_tick": round((wall_cell / steps) * 1000.0, 3),
                    "agent_ticks_per_sec": round(total_agent_ticks / wall_cell, 1),
                    "peak_rss_mib": _peak_rss_mib(),
                    "max_ledger_residual": float(max_res),
                    "hotspots_ms": {k: round(v * 1000.0, 3) for k, v in phase_sec.items()},
                }
            )

    if include_gpu:
        gpu_probe = _probe_gpu_mlp(max(populations) if populations else 400, steps, seed)
    else:
        gpu_probe = {
            "available": False,
            "backend": "none",
            "speedup_vs_cpu": None,
            "max_abs_diff": None,
            "note": "GPU probe not requested (pass --include-gpu to run).",
        }

    return {
        "grid": grid,
        "gpu_probe": gpu_probe,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }
