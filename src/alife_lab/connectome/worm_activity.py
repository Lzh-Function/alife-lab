"""Whole-brain calcium fluorescence forward model & empirical C. elegans activity fitting (S1).

Includes:
1. Pure-CSV/JSON & synthetic offline benchmark (``run_worm_activity_benchmark``) preserving
   synthetic-to-synthetic controls (``reference`` vs ``no_gap_junctions`` vs ``shuffled_control``).
2. Pure-Parquet loader & 5-arm connectome-constrained empirical fitting
   (``load_celegans_activity_recordings``, ``run_real_worm_activity_fit``) evaluated against real
   whole-brain C. elegans GCaMP datasets stored under ``.alife/datasets/celegans_activity/``
   (ZERO pickles). PyTorch is imported lazily inside fitting functions only.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.analysis.stats import compare, describe
from alife_lab.connectome.celegans import (
    canonical_celegans_302_names,
    make_synthetic_celegans_302_pack,
    normalize_celegans_name,
)
from alife_lab.connectome.graded import (
    GradedCircuit,
    GradedParams,
    make_initial_graded_state,
    step_graded_tick_cpu,
)
from alife_lab.connectome.lif import make_shuffled_csr
from alife_lab.connectome.schema import (
    SYNAPSE_CHEMICAL,
    SYNAPSE_GAP_JUNCTION,
    ConnectomePack,
)
from alife_lab.connectome.sysid import _pearson_r, _r2_score, _resolve_device
from alife_lab.core.hashing import sha256_file

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]

CELEGANS_ACTIVITY_DIR_NAME = "celegans_activity"
CELEGANS_ACTIVITY_FILENAME = "worm_data_short.parquet"
CELEGANS_ACTIVITY_COMMIT = "a28aacad1a997ac7aee98f1c52993ab8c0599ee8"
CELEGANS_ACTIVITY_URL = (
    "https://huggingface.co/datasets/qsimeon/celegans_neural_data/resolve/"
    f"{CELEGANS_ACTIVITY_COMMIT}/worm_data_short.parquet"
)
CELEGANS_ACTIVITY_SIZE_BYTES = 669_511_381
CELEGANS_ACTIVITY_SHA256 = "17701963149851d37d40955096de181d8c5378c256fe4ede93864380f9363ddc"


def calcium_from_voltage(
    v_trace: F64,
    *,
    e_leak_mv: float = -35.0,
    tau_ca_s: float = 1.2,
    dt_s: float = 0.01,
) -> F64:
    """Transform simulated membrane voltages (T, N) into low-pass calcium dF/F traces."""
    v = np.asarray(v_trace, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("v_trace must be 2D (T, N)")
    t_len, n = v.shape
    out = np.zeros((t_len, n), dtype=np.float64)
    decay = math.exp(-dt_s / max(tau_ca_s, 1e-4))
    drive = np.maximum(0.0, (v - e_leak_mv) * 0.08)
    ca = np.zeros(n, dtype=np.float64)
    for t in range(t_len):
        ca = ca * decay + (1.0 - decay) * drive[t]
        out[t] = ca
    return out


def load_worm_activity_table(path: Path) -> tuple[list[str], F64]:
    """Load neural calcium fluorescence table (CSV rows=time, cols=neuron_name; or JSON)."""
    if not path.is_file():
        raise FileNotFoundError(f"worm activity table not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        neurons = [str(k) for k in payload["neurons"]]
        traces = np.asarray(payload["traces"], dtype=np.float64)
        return neurons, traces
    if path.suffix.lower() != ".csv":
        raise ValueError(f"unsupported worm activity format {path.suffix!r} (CSV or JSON only)")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(v) for v in r] for r in reader if r]
    return [str(h).strip() for h in header], np.asarray(rows, dtype=np.float64)


def _simulate_worm_voltages(
    *,
    gap_junctions_enabled: bool,
    shuffled: bool,
    seed: int,
    steps: int,
    stim_schedule: F64,
) -> F64:
    pack = make_synthetic_celegans_302_pack()
    base_csr = pack.edges.to_csr(pack.cells.num_cells)
    csr = make_shuffled_csr(base_csr, seed) if shuffled else base_csr
    circuit = GradedCircuit.from_csr(csr)
    params = GradedParams(dt_ms=1.0, substeps_per_tick=10)
    state = make_initial_graded_state(pack.cells.num_cells, params)
    out = np.zeros((steps, pack.cells.num_cells), dtype=np.float64)
    for t in range(steps):
        ext = stim_schedule[t].astype(np.float32)
        step_graded_tick_cpu(
            state,
            circuit,
            params,
            gap_junctions_enabled=gap_junctions_enabled,
            external_current=ext,
        )
        out[t] = state.v.astype(np.float64)
    return out


def run_worm_activity_benchmark(seed: int = 7, steps: int = 120) -> dict[str, Any]:
    """Compare intact 302-cell C. elegans wiring vs no-gap and shuffled controls on dF/F traces."""
    pack = make_synthetic_celegans_302_pack()
    n = pack.cells.num_cells
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    # Held-out stimulation sequence across posterior touch, anterior touch, & chemosensory pools
    stim = np.zeros((steps, n), dtype=np.float64)
    ant = pack.io_ports["anterior_touch"].cell_indices
    post = pack.io_ports["posterior_touch"].cell_indices
    chemo = pack.io_ports["chemosensory_attract"].cell_indices
    stim[0:steps, post] = 1.6
    stim[steps // 3 : steps // 2, ant] = 3.2
    stim[2 * steps // 3 : steps, chemo] = 2.0 + 0.3 * rng.standard_normal(
        (steps - 2 * steps // 3, chemo.size)
    )
    stim = np.maximum(0.0, stim)

    v_truth = _simulate_worm_voltages(
        gap_junctions_enabled=True, shuffled=False, seed=seed, steps=steps, stim_schedule=stim
    )
    ca_truth = calcium_from_voltage(v_truth)

    # Evaluate downstream motor & interneuron non-sensory cells (held-out from direct sensory stim)
    non_sens = pack.cells.role != 0
    y_true = ca_truth[steps // 4 :, non_sens]

    results: dict[str, dict[str, float]] = {}
    for arm, gap_on, shuf in (
        ("reference", True, False),
        ("no_gap_junctions", False, False),
        ("shuffled_control", True, True),
    ):
        v_pred = _simulate_worm_voltages(
            gap_junctions_enabled=gap_on, shuffled=shuf, seed=seed, steps=steps, stim_schedule=stim
        )
        ca_pred = calcium_from_voltage(v_pred)[steps // 4 :, non_sens]
        results[arm] = {
            "heldout_r2": round(float(_r2_score(y_true, ca_pred)), 5),
            "heldout_pearson_r": round(float(_pearson_r(y_true, ca_pred)), 5),
            "rmse": round(float(np.sqrt(np.mean((y_true - ca_pred) ** 2))), 5),
        }
    return {
        "dataset": "synthetic_celegans_302_offline_calcium",
        "steps": steps,
        "evaluated_neurons": int(non_sens.sum()),
        "arms": results,
    }


# --------------------------------------------------------------------------- #
# Empirical C. elegans whole-brain GCaMP verification, loading & fitting
# --------------------------------------------------------------------------- #


def verify_celegans_activity_file(datasets_dir: Path, *, verify_hash: bool = True) -> Path:
    """Verify the downloaded worm_data_short.parquet file size and SHA-256."""
    path = datasets_dir / CELEGANS_ACTIVITY_DIR_NAME / "raw" / CELEGANS_ACTIVITY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"C. elegans activity dataset missing: {path} "
            f"(expected under .alife/datasets/{CELEGANS_ACTIVITY_DIR_NAME}/raw/; "
            "requires user approval before download)"
        )
    actual_size = path.stat().st_size
    if actual_size != CELEGANS_ACTIVITY_SIZE_BYTES:
        raise ValueError(
            f"size mismatch for {CELEGANS_ACTIVITY_FILENAME}: "
            f"expected {CELEGANS_ACTIVITY_SIZE_BYTES}, got {actual_size}"
        )
    if verify_hash:
        actual_sha = sha256_file(path)
        if actual_sha != CELEGANS_ACTIVITY_SHA256:
            raise ValueError(
                f"SHA-256 mismatch for {CELEGANS_ACTIVITY_FILENAME}: "
                f"expected {CELEGANS_ACTIVITY_SHA256[:16]}…, got {actual_sha[:16]}…"
            )
    return path


@dataclass(frozen=True)
class WormCalciumRecording:
    """Standardized real worm recording aligned to canonical 302 C. elegans cell indices."""

    source_dataset: str
    worm_id: str
    neuron_names: tuple[str, ...]
    cell_indices: I32
    traces: F32  # shape (T, M), chronological uniform calcium traces


def load_celegans_activity_recordings(
    datasets_dir: Path,
    *,
    max_worms: int = 12,
    min_matched_neurons: int = 40,
    max_frames: int = 600,
    source_filter: str | None = None,
    verify_hash: bool = True,
) -> list[WormCalciumRecording]:
    """Load real C. elegans GCaMP recordings from pure Apache Parquet (ZERO pickles)."""
    import pyarrow.parquet as pq

    path = verify_celegans_activity_file(datasets_dir, verify_hash=verify_hash)
    canon_names = [nm for nm, _ in canonical_celegans_302_names()]
    canon_lookup = {nm: i for i, nm in enumerate(canon_names)}

    filters: list[tuple[str, str, Any]] = [("is_labeled_neuron", "==", True)]
    if source_filter:
        filters.append(("source_dataset", "==", source_filter))

    tbl = pq.read_table(
        path,
        columns=[
            "source_dataset",
            "worm",
            "neuron",
            "is_labeled_neuron",
            "calcium_data",
            "max_timesteps",
        ],
        filters=filters,
    )
    df = tbl.to_pandas()
    if df.empty:
        return []

    candidates: list[WormCalciumRecording] = []
    for (src_ds, worm_id), grp in df.groupby(["source_dataset", "worm"], sort=False):
        per_cell_raw: dict[int, list[F32]] = {}
        for _, row in grp.iterrows():
            nm = normalize_celegans_name(str(row["neuron"]))
            cidx = canon_lookup.get(nm)
            if cidx is None:
                continue
            arr = np.asarray(row["calcium_data"], dtype=np.float32)
            if arr.ndim != 1 or arr.size < 100:
                continue
            per_cell_raw.setdefault(cidx, []).append(arr)

        if len(per_cell_raw) < min_matched_neurons:
            continue

        min_len = min(
            max_frames,
            min(arr.size for arrs in per_cell_raw.values() for arr in arrs),
        )
        if min_len < 100:
            continue

        cleaned: dict[int, F32] = {}
        for cidx, arrs in per_cell_raw.items():
            valid = [
                a[:min_len]
                for a in arrs
                if bool(np.all(np.isfinite(a[:min_len]))) and float(np.std(a[:min_len])) >= 1e-5
            ]
            if valid:
                cleaned[cidx] = np.mean(np.stack(valid, axis=0), axis=0).astype(np.float32)

        if len(cleaned) < min_matched_neurons:
            continue

        sorted_idxs = sorted(cleaned.keys())
        traces = np.stack([cleaned[i] for i in sorted_idxs], axis=1).astype(np.float32)
        candidates.append(
            WormCalciumRecording(
                source_dataset=str(src_ds),
                worm_id=str(worm_id),
                neuron_names=tuple(canon_names[i] for i in sorted_idxs),
                cell_indices=np.asarray(sorted_idxs, dtype=np.int32),
                traces=traces,
            )
        )

    candidates.sort(key=lambda r: (-int(r.cell_indices.size), r.source_dataset, r.worm_id))
    return candidates[:max_worms]


def _build_worm_connectome_matrices(pack: ConnectomePack) -> tuple[F32, F32, F32]:
    """Build (302, 302) log1p-compressed row-normalized chemical and gap matrices."""
    n = pack.cells.num_cells
    e = pack.edges
    w_chem_raw_counts = np.zeros((n, n), dtype=np.float64)
    w_chem_raw_signed = np.zeros((n, n), dtype=np.float64)
    w_gap_raw_counts = np.zeros((n, n), dtype=np.float64)

    chem_mask = e.kind == SYNAPSE_CHEMICAL
    gap_mask = e.kind == SYNAPSE_GAP_JUNCTION
    np.add.at(
        w_chem_raw_counts,
        (e.post[chem_mask], e.pre[chem_mask]),
        e.count[chem_mask].astype(np.float64),
    )
    np.add.at(
        w_chem_raw_signed,
        (e.post[chem_mask], e.pre[chem_mask]),
        e.sign[chem_mask].astype(np.float64) * e.count[chem_mask].astype(np.float64),
    )
    np.add.at(
        w_gap_raw_counts, (e.post[gap_mask], e.pre[gap_mask]), e.count[gap_mask].astype(np.float64)
    )

    chem_mag = np.log1p(w_chem_raw_counts)
    chem_row = np.maximum(chem_mag.sum(axis=1, keepdims=True), 1.0)
    w_chem_mag = (chem_mag / chem_row).astype(np.float32)

    chem_signed_log = np.sign(w_chem_raw_signed) * np.log1p(np.abs(w_chem_raw_signed))
    w_chem_signed = (chem_signed_log / chem_row).astype(np.float32)

    gap_mag = np.log1p(w_gap_raw_counts)
    gap_mag = 0.5 * (gap_mag + gap_mag.T)
    gap_row = np.maximum(gap_mag.sum(axis=1, keepdims=True), 1.0)
    w_gap_mag = (gap_mag / gap_row).astype(np.float32)
    return w_chem_mag, w_chem_signed, w_gap_mag


def _zscore_train_test_traces(traces: F32) -> tuple[F32, F32]:
    t_tot = int(traces.shape[0])
    cut = max(20, min(t_tot - 20, round(t_tot * 0.7)))
    tr = traces[:cut]
    te = traces[cut:]
    mu = tr.mean(axis=0, keepdims=True)
    sd = np.maximum(tr.std(axis=0, keepdims=True), np.float32(1e-4))
    return ((tr - mu) / sd).astype(np.float32), ((te - mu) / sd).astype(np.float32)


def _init_connectome_operator(
    arm: str,
    w_chem_base: F32,
    w_chem_raw_signed: F32,
    w_gap_base: F32,
    dev: str,
) -> tuple[list[Any], Any, Any]:
    """Create shared differentiable connectome parameters & regularizer for one evaluation arm."""
    import torch
    import torch.nn.functional as F

    n = int(w_chem_base.shape[0])
    wc_t = torch.as_tensor(w_chem_base, dtype=torch.float32, device=dev)
    wcs_t = torch.as_tensor(w_chem_raw_signed, dtype=torch.float32, device=dev)
    wg_t = torch.as_tensor(w_gap_base, dtype=torch.float32, device=dev)
    m_chem = wc_t > 0.0
    m_gap = wg_t > 0.0
    n_chem = max(1, int(m_chem.sum().item()))
    n_gap = max(1, int(m_gap.sum().item()))

    if arm in ("reference", "no_gap_junctions", "shuffled_control"):
        theta_chem = torch.zeros((n, n), dtype=torch.float32, device=dev, requires_grad=True)
        params: list[Any] = [theta_chem]
        use_gap = arm != "no_gap_junctions"
        theta_gap = (
            torch.full((n, n), -1.0, dtype=torch.float32, device=dev, requires_grad=True)
            if use_gap
            else None
        )
        if theta_gap is not None:
            params.append(theta_gap)

        def matrices() -> tuple[Any, Any]:
            w_c = theta_chem * wc_t
            if theta_gap is None:
                return w_c, torch.zeros((n, n), dtype=torch.float32, device=dev)
            g_pos = F.softplus(theta_gap) * wg_t
            w_g = 0.5 * (g_pos + g_pos.transpose(0, 1))
            return w_c, w_g

        def penalty() -> Any:
            reg = torch.sum((theta_chem * m_chem.float()) ** 2) / float(n_chem)
            if theta_gap is not None:
                reg = reg + torch.sum((F.softplus(theta_gap) * m_gap.float()) ** 2) / float(n_gap)
            return reg

        return params, matrices, penalty

    if arm == "raw_em_gaba_only":
        raw_g_chem = torch.full((n,), -0.5, dtype=torch.float32, device=dev, requires_grad=True)
        raw_g_gap = torch.full((n,), -0.5, dtype=torch.float32, device=dev, requires_grad=True)
        params = [raw_g_chem, raw_g_gap]

        def matrices_raw() -> tuple[Any, Any]:
            w_c = wcs_t * F.softplus(raw_g_chem).unsqueeze(1)
            w_g = wg_t * F.softplus(raw_g_gap).unsqueeze(1)
            w_g = 0.5 * (w_g + w_g.transpose(0, 1))
            return w_c, w_g

        def penalty_raw() -> Any:
            return torch.mean(F.softplus(raw_g_chem) ** 2 + F.softplus(raw_g_gap) ** 2)

        return params, matrices_raw, penalty_raw

    # "unconnected"
    zero_mat = torch.zeros((n, n), dtype=torch.float32, device=dev)

    def matrices_zero() -> tuple[Any, Any]:
        return zero_mat, zero_mat

    def penalty_zero() -> Any:
        return torch.tensor(0.0, device=dev)

    return [], matrices_zero, penalty_zero


def _build_rollout_windows(seq: Any, horizon: int = 5) -> tuple[Any, Any]:
    """Slice ``(T, M)`` tensor into ``(B, M)`` starts and ``(B, horizon, M)`` targets."""
    import torch

    t_len = int(seq.shape[0])
    step = horizon
    starts = list(range(0, t_len - horizon, step))
    if not starts:
        starts = [0]
    x0 = torch.stack([seq[s] for s in starts], dim=0)
    targets = torch.stack([seq[s + 1 : s + 1 + horizon] for s in starts], dim=0)
    return x0, targets


def _fit_arm_for_seed(
    arm: str,
    recordings: list[WormCalciumRecording],
    te_list: list[Any],
    splits: list[tuple[Any, Any, Any, Any]],
    idx_tensors: list[Any],
    te_wins: list[tuple[Any, Any]],
    *,
    x_obs_tr: Any,
    y_unobs_tr: Any,
    w_unobs_tr: Any,
    x0_tr_all: Any,
    y_tr_all: Any,
    m_obs_tr: Any,
    w_loss_tr: Any,
    horizon: int,
    n_cells: int,
    w_chem_base: F32,
    w_chem_raw_signed: F32,
    w_gap_base: F32,
    epochs: int,
    lr: float,
    dev: str,
) -> dict[str, float]:
    """Fit both spatial unobserved prediction and open-loop rollout for one seed & arm."""
    import torch

    # 1. Task A: Vectorized spatial unobserved-neuron prediction on held-out final 30% frames
    sp_params, sp_matrices, sp_penalty = _init_connectome_operator(
        arm, w_chem_base, w_chem_raw_signed, w_gap_base, dev
    )
    sp_bias = torch.zeros((n_cells,), dtype=torch.float32, device=dev, requires_grad=True)
    sp_opt = torch.optim.Adam([*sp_params, sp_bias], lr=lr)

    for _ in range(epochs):
        sp_opt.zero_grad(set_to_none=True)
        w_c, w_g = sp_matrices()
        w_tot = w_c + w_g
        pred = torch.matmul(x_obs_tr, w_tot.transpose(0, 1)) + sp_bias
        loss = torch.sum(((pred - y_unobs_tr) ** 2) * w_unobs_tr) + 1e-2 * sp_penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*sp_params, sp_bias], 5.0)
        sp_opt.step()

    sp_r2s, sp_rs = [], []
    with torch.no_grad():
        w_c, w_g = sp_matrices()
        w_tot = w_c + w_g
        for w_idx, (loc_obs, loc_unobs, g_obs, g_unobs) in enumerate(splits):
            w_uo = w_tot[g_unobs][:, g_obs]
            y_te_obs = te_list[w_idx][:, loc_obs]
            y_te_unobs = te_list[w_idx][:, loc_unobs]
            pred_te = torch.matmul(y_te_obs, w_uo.transpose(0, 1)) + sp_bias[g_unobs]
            yt_np = y_te_unobs.cpu().numpy().astype(np.float64)
            yp_np = pred_te.cpu().numpy().astype(np.float64)
            sp_r2s.append(_r2_score(yt_np, yp_np))
            sp_rs.append(_pearson_r(yt_np.ravel(), yp_np.ravel()))

    # 2. Task B: Vectorized batched open-loop temporal rollout (ZERO teacher forcing)
    tm_params, tm_matrices, tm_penalty = _init_connectome_operator(
        arm, w_chem_base, w_chem_raw_signed, w_gap_base, dev
    )
    alpha_logit = torch.zeros((n_cells,), dtype=torch.float32, device=dev, requires_grad=True)
    tm_bias = torch.zeros((n_cells,), dtype=torch.float32, device=dev, requires_grad=True)
    tm_opt = torch.optim.Adam([*tm_params, alpha_logit, tm_bias], lr=lr)

    for _ in range(epochs):
        tm_opt.zero_grad(set_to_none=True)
        w_c, w_g = tm_matrices()
        alpha = (0.05 + 0.90 * torch.sigmoid(alpha_logit)).unsqueeze(0)
        w_c_t = w_c.transpose(0, 1)
        w_g_t = w_g.transpose(0, 1)
        g_diag_all = torch.matmul(m_obs_tr, w_g_t)
        x = x0_tr_all
        preds_list: list[Any] = []
        for _ in range(horizon):
            laplacian = torch.matmul(x, w_g_t) - g_diag_all * x
            drive = torch.matmul(x, w_c_t) + laplacian + tm_bias
            target = torch.tanh(drive) * 4.0
            x = (x + alpha * (target - x)) * m_obs_tr
            preds_list.append(x)
        preds = torch.stack(preds_list, dim=1)
        tloss = torch.sum(((preds - y_tr_all) ** 2) * w_loss_tr) + 1e-2 * tm_penalty()
        tloss.backward()
        torch.nn.utils.clip_grad_norm_([*tm_params, alpha_logit, tm_bias], 5.0)
        tm_opt.step()

    tm_r2s, tm_rs = [], []
    with torch.no_grad():
        w_c, w_g = tm_matrices()
        for w_idx, (x0_te, y_te_win) in enumerate(te_wins):
            c_idx = idx_tensors[w_idx]
            alpha = (0.05 + 0.90 * torch.sigmoid(alpha_logit[c_idx])).unsqueeze(0)
            b_loc = tm_bias[c_idx].unsqueeze(0)
            w_sub_t = w_c[c_idx][:, c_idx].transpose(0, 1)
            g_sub = w_g[c_idx][:, c_idx]
            g_diag = torch.sum(g_sub, dim=1, keepdim=True).transpose(0, 1)
            g_sub_t = g_sub.transpose(0, 1)
            x = x0_te
            preds_te_list: list[Any] = []
            for _ in range(horizon):
                laplacian = torch.matmul(x, g_sub_t) - g_diag * x
                drive = torch.matmul(x, w_sub_t) + laplacian + b_loc
                target = torch.tanh(drive) * 4.0
                x = x + alpha * (target - x)
                preds_te_list.append(x)
            preds_te = torch.stack(preds_te_list, dim=1)
            yt_np = y_te_win.cpu().numpy().astype(np.float64)
            yp_np = preds_te.cpu().numpy().astype(np.float64)
            tm_r2s.append(_r2_score(yt_np, yp_np))
            tm_rs.append(_pearson_r(yt_np.ravel(), yp_np.ravel()))

    return {
        "spatial_unobs_r2": round(float(np.mean(sp_r2s)), 4),
        "spatial_unobs_r": round(float(np.mean(sp_rs)), 4),
        "temporal_rollout_r2": round(float(np.mean(tm_r2s)), 4),
        "temporal_rollout_r": round(float(np.mean(tm_rs)), 4),
    }


def run_real_worm_activity_fit(
    pack: ConnectomePack,
    datasets_dir: Path,
    *,
    max_worms: int = 12,
    min_matched_neurons: int = 40,
    max_frames: int = 600,
    source_filter: str | None = None,
    horizon: int = 15,
    epochs: int = 100,
    lr: float = 0.05,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    device: str = "auto",
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Fit empirical 302-cell EM connectome models against real C. elegans GCaMP traces."""
    import torch

    t0 = time.perf_counter()
    recordings = load_celegans_activity_recordings(
        datasets_dir,
        max_worms=max_worms,
        min_matched_neurons=min_matched_neurons,
        max_frames=max_frames,
        source_filter=source_filter or None,
        verify_hash=verify_hash,
    )
    if not recordings:
        raise ValueError(
            "no C. elegans recordings matched filter criteria "
            f"(source_filter={source_filter!r}, min_matched_neurons={min_matched_neurons})"
        )

    dev = _resolve_device(device)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    w_chem_mag, w_chem_raw_signed, w_gap_mag = _build_worm_connectome_matrices(pack)
    n_cells = pack.cells.num_cells

    tr_list: list[Any] = []
    te_list: list[Any] = []
    for rec in recordings:
        y_tr, y_te = _zscore_train_test_traces(rec.traces)
        tr_list.append(torch.as_tensor(y_tr, dtype=torch.float32, device=dev))
        te_list.append(torch.as_tensor(y_te, dtype=torch.float32, device=dev))

    idx_tensors = [
        torch.as_tensor(rec.cell_indices, dtype=torch.long, device=dev) for rec in recordings
    ]
    te_wins = [_build_rollout_windows(te, horizon=horizon) for te in te_list]

    # Pre-build batched 302-dim training rollout tensors once across all seeds & arms
    n_worms_f = float(len(recordings))
    x0_chunks: list[Any] = []
    y_chunks: list[Any] = []
    m_chunks: list[Any] = []
    wl_chunks: list[Any] = []
    for w_idx, rec in enumerate(recordings):
        x0_loc, y_loc = _build_rollout_windows(tr_list[w_idx], horizon=horizon)
        b_w = int(x0_loc.shape[0])
        m_w = int(rec.cell_indices.size)
        c_idx = idx_tensors[w_idx]
        x0_emb = torch.zeros((b_w, n_cells), dtype=torch.float32, device=dev)
        x0_emb[:, c_idx] = x0_loc
        y_emb = torch.zeros((b_w, horizon, n_cells), dtype=torch.float32, device=dev)
        y_emb[:, :, c_idx] = y_loc
        m_emb = torch.zeros((b_w, n_cells), dtype=torch.float32, device=dev)
        m_emb[:, c_idx] = 1.0
        wl_emb = torch.zeros((b_w, 1, n_cells), dtype=torch.float32, device=dev)
        wl_emb[:, :, c_idx] = 1.0 / (n_worms_f * float(b_w) * float(horizon) * float(m_w))
        x0_chunks.append(x0_emb)
        y_chunks.append(y_emb)
        m_chunks.append(m_emb)
        wl_chunks.append(wl_emb)
    x0_tr_all = torch.cat(x0_chunks, dim=0)
    y_tr_all = torch.cat(y_chunks, dim=0)
    m_obs_tr = torch.cat(m_chunks, dim=0)
    w_loss_tr = torch.cat(wl_chunks, dim=0)

    rows_tr = [int(tr.shape[0]) for tr in tr_list]
    n_total_tr = sum(rows_tr)

    arm_names = (
        "reference",
        "no_gap_junctions",
        "raw_em_gaba_only",
        "shuffled_control",
        "unconnected",
    )
    per_seed: dict[str, list[dict[str, float]]] = {arm: [] for arm in arm_names}

    for s in seeds:
        torch.manual_seed(int(s) * 97 + 3)
        splits: list[tuple[Any, Any, Any, Any]] = []
        x_obs_tr = torch.zeros((n_total_tr, n_cells), dtype=torch.float32, device=dev)
        y_unobs_tr = torch.zeros((n_total_tr, n_cells), dtype=torch.float32, device=dev)
        w_unobs_tr = torch.zeros((n_total_tr, n_cells), dtype=torch.float32, device=dev)
        cursor = 0
        for w_idx, rec in enumerate(recordings):
            m_w = int(rec.cell_indices.size)
            rng_sp = np.random.default_rng(int(s) * 313 + 7 + w_idx)
            perm_loc = rng_sp.permutation(m_w)
            n_obs = m_w // 2
            loc_obs = np.sort(perm_loc[:n_obs])
            loc_unobs = np.sort(perm_loc[n_obs:])
            g_obs = rec.cell_indices[loc_obs]
            g_unobs = rec.cell_indices[loc_unobs]
            loc_obs_t = torch.as_tensor(loc_obs, dtype=torch.long, device=dev)
            loc_unobs_t = torch.as_tensor(loc_unobs, dtype=torch.long, device=dev)
            g_obs_t = torch.as_tensor(g_obs, dtype=torch.long, device=dev)
            g_unobs_t = torch.as_tensor(g_unobs, dtype=torch.long, device=dev)
            splits.append((loc_obs_t, loc_unobs_t, g_obs_t, g_unobs_t))

            t_tr_w = rows_tr[w_idx]
            sl = slice(cursor, cursor + t_tr_w)
            x_obs_tr[sl, g_obs_t] = tr_list[w_idx][:, loc_obs_t]
            y_unobs_tr[sl, g_unobs_t] = tr_list[w_idx][:, loc_unobs_t]
            w_unobs_tr[sl, g_unobs_t] = 1.0 / (n_worms_f * float(t_tr_w) * float(len(g_unobs)))
            cursor += t_tr_w

        rng_shuf = np.random.default_rng(int(s) * 1009 + 17)
        perm_302 = rng_shuf.permutation(n_cells)
        w_chem_shuf = w_chem_mag[perm_302][:, perm_302]
        w_gap_shuf = w_gap_mag[perm_302][:, perm_302]

        for arm in arm_names:
            wc = w_chem_shuf if arm == "shuffled_control" else w_chem_mag
            wg = w_gap_shuf if arm == "shuffled_control" else w_gap_mag
            metrics = _fit_arm_for_seed(
                arm,
                recordings,
                te_list,
                splits,
                idx_tensors,
                te_wins,
                x_obs_tr=x_obs_tr,
                y_unobs_tr=y_unobs_tr,
                w_unobs_tr=w_unobs_tr,
                x0_tr_all=x0_tr_all,
                y_tr_all=y_tr_all,
                m_obs_tr=m_obs_tr,
                w_loss_tr=w_loss_tr,
                horizon=horizon,
                n_cells=n_cells,
                w_chem_base=wc,
                w_chem_raw_signed=w_chem_raw_signed,
                w_gap_base=wg,
                epochs=epochs,
                lr=lr,
                dev=dev,
            )
            per_seed[arm].append({"seed": float(s), **metrics})

    def _vals(arm: str, key: str) -> list[float]:
        return [float(row[key]) for row in per_seed[arm]]

    summary = {
        arm: {
            "spatial_unobs_r2": describe(_vals(arm, "spatial_unobs_r2")),
            "spatial_unobs_r": describe(_vals(arm, "spatial_unobs_r")),
            "temporal_rollout_r2": describe(_vals(arm, "temporal_rollout_r2")),
            "temporal_rollout_r": describe(_vals(arm, "temporal_rollout_r")),
        }
        for arm in arm_names
    }
    comparisons = {
        "spatial_unobs_r2": {
            f"reference_vs_{other}": compare(
                _vals("reference", "spatial_unobs_r2"),
                _vals(other, "spatial_unobs_r2"),
                seed=800 + idx,
            )
            for idx, other in enumerate(
                ("shuffled_control", "no_gap_junctions", "raw_em_gaba_only", "unconnected")
            )
        },
        "temporal_rollout_r2": {
            f"reference_vs_{other}": compare(
                _vals("reference", "temporal_rollout_r2"),
                _vals(other, "temporal_rollout_r2"),
                seed=820 + idx,
            )
            for idx, other in enumerate(
                ("shuffled_control", "no_gap_junctions", "raw_em_gaba_only", "unconnected")
            )
        },
    }
    peak_vram_mb = (
        round(float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0), 2)
        if dev == "cuda"
        else 0.0
    )
    src_counts = dict(Counter(rec.source_dataset for rec in recordings))
    return {
        "dataset": "celegans_activity_worm_data_short",
        "connectome_dataset": pack.dataset_id,
        "connectome_digest": pack.pack_digest,
        "citation": (
            "Simeon et al. (2024) IEEE SoutheastCon 'Scaling Properties for Artificial Neural "
            "Network Models of a Small Nervous System'; OpenWorm CElegansNeuroML."
        ),
        "n_worms": len(recordings),
        "source_datasets": src_counts,
        "source_filter": source_filter or "",
        "rollout_horizon_steps": horizon,
        "mean_matched_neurons": round(
            float(np.mean([rec.cell_indices.size for rec in recordings])), 1
        ),
        "device": dev,
        "peak_vram_mb": peak_vram_mb,
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "framing_note": (
            "Empirical C. elegans GCaMP traces fitted with the 302-cell EM chemical + ohmic "
            "gap-junction connectome on strictly held-out final 30% frames: 50% spatial "
            "unobserved-neuron prediction and 5-step open-loop rollout (zero teacher forcing)."
        ),
        "summary": summary,
        "comparisons": comparisons,
        "per_seed": per_seed,
    }
