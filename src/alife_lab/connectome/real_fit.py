"""Phase F7: Fitting connectome models to real public Drosophila brain activity recordings.

External fitting layer (PyTorch imported lazily inside worker/CLI functions only).
Strictly tabular / pure NumPy ``.npy`` (``allow_pickle=False``) — never opens ``.pkl`` files.

Supported public real-animal datasets (stored outside Git in ``.alife/datasets/f7_real_activity/``):
1. **Aimon et al. (2023, eLife 85202; Zenodo 7849014):**
   Whole-brain light-field calcium imaging (75 JRC2018 anatomical brain regions, deltaF/F) with
   synchronized Walk / Left-turn / Right-turn behavioral regressors in head-fixed adult flies.
   Mapped directly onto FlyWire v783's 78 neuropils (138,639 neurons, 15,091,983 edges) and
   evaluated on:
   - Held-out future time segments (within-fly open-loop rollout, NO teacher forcing)
   - Held-out unseen flies (cross-fly zero-shot generalization)
   - Held-out unobserved brain regions (spatial inter-region connectome coupling test)
   Compared against degree- and weight-preserving ``shuffled_control`` regional connectomes and
   an ``unconnected`` behavioral-regressor-only control.
2. **DoOR 2.0 (Münch & Galizia 2016, Scientific Reports):**
   692 real odorants x 41 olfactory receptors mapped 1:1 to FlyWire v783 ``ORN_<glomerulus>``
   populations. Evaluates held-out odorant prediction of masked glomeruli through FlyWire's
   antennal-lobe circuit (``reference`` vs ``shuffled_control`` vs ``untrained``).
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.analysis.stats import compare, describe
from alife_lab.connectome.schema import ConnectomePack
from alife_lab.connectome.sysid import _pearson_r, _r2_score, _resolve_device

F7_DIR_NAME = "f7_real_activity"

F7_EXPECTED_SHA256: dict[str, str] = {
    "door_or_merged.csv": ("e80cd3ed3496fd2c0e571a7ef928938558e10812b9893f1d17045a418409bb82"),
    "door_to_flywire_mapping.csv": (
        "903abffd67a88f73a3879d345d6a10e95ea2065fa342334f18ded18d4e06331f"
    ),
    "Walk_anatomical_regions.zip": (
        "14039b2ab1a689e58b08324747daf275b8555837395df1951c7528ab3f8d4a91"
    ),
}

# Explicit anatomical name bridge from Aimon 2023 JRC2018 75-region list to FlyWire v783 neuropils.
_AIMON_TO_FLYWIRE_REGION: dict[str, str] = {
    "MBPED_R": "MB_PED_R",
    "MBPED_L": "MB_PED_L",
    "MBVL_R": "MB_VL_R",
    "MBVL_L": "MB_VL_L",
    "MBML_R": "MB_ML_R",
    "MBML_L": "MB_ML_L",
    "MBCA_R": "MB_CA_R",
    "MBCA_L": "MB_CA_L",
    "IVLP_R": "WED_R",
    "IVLP_L": "WED_L",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify_f7_file(raw_dir: Path, filename: str) -> Path:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"F7 dataset file missing: {path} (expected under .alife/datasets/{F7_DIR_NAME}/raw/)"
        )
    expected = F7_EXPECTED_SHA256.get(filename)
    if expected is not None:
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected {expected[:16]}…, got {actual[:16]}…"
            )
    return path


def map_aimon_region_to_flywire(region_name: str, flywire_regions: set[str]) -> str | None:
    """Map an Aimon 2023 anatomical region name onto its FlyWire v783 neuropil."""
    if region_name in flywire_regions:
        return region_name
    mapped = _AIMON_TO_FLYWIRE_REGION.get(region_name)
    if mapped is not None and mapped in flywire_regions:
        return mapped
    return None


@dataclass(frozen=True)
class NeuropilConnectome:
    """75-region excitatory & inhibitory anatomical connectivity aggregated from FlyWire v783."""

    region_names: tuple[str, ...]
    matched_mask: npt.NDArray[np.bool_]
    w_exc: npt.NDArray[np.float32]  # (R, R) row-normalized post <- pre excitatory strength
    w_inh: npt.NDArray[np.float32]  # (R, R) row-normalized post <- pre inhibitory strength
    synapse_total: int
    matched_count: int


def build_neuropil_connectome(
    pack: ConnectomePack,
    region_names: tuple[str, ...],
) -> NeuropilConnectome:
    """Aggregate all 15M FlyWire v783 synapses into a region-by-region E/I matrix."""
    fw_regions = set(pack.cells.region)
    fw_to_ridx: dict[str, int] = {}
    matched = np.zeros(len(region_names), dtype=np.bool_)
    for ridx, rname in enumerate(region_names):
        fw = map_aimon_region_to_flywire(rname, fw_regions)
        if fw is not None:
            fw_to_ridx[fw] = ridx
            matched[ridx] = True

    cell_to_r = np.asarray(
        [fw_to_ridx.get(r, -1) for r in pack.cells.region],
        dtype=np.int32,
    )
    e = pack.edges
    pre_r = cell_to_r[e.pre]
    post_r = cell_to_r[e.post]
    valid = (pre_r >= 0) & (post_r >= 0) & (pre_r != post_r)
    pre_v = pre_r[valid]
    post_v = post_r[valid]
    cnt_v = e.count[valid].astype(np.float64)
    sgn_v = e.sign[valid]

    n_r = len(region_names)
    w_exc = np.zeros((n_r, n_r), dtype=np.float64)
    w_inh = np.zeros((n_r, n_r), dtype=np.float64)
    exc_m = sgn_v > 0
    inh_m = sgn_v < 0
    np.add.at(w_exc, (post_v[exc_m], pre_v[exc_m]), cnt_v[exc_m])
    np.add.at(w_inh, (post_v[inh_m], pre_v[inh_m]), cnt_v[inh_m])

    total_syn = round(float(w_exc.sum() + w_inh.sum()))
    # Log-compressed row normalization prevents a few giant optic-lobe neuropils from
    # drowning out central/MB neuropils while preserving exact anatomical sparsity & ordering.
    w_exc_log = np.log1p(w_exc)
    w_inh_log = np.log1p(w_inh)
    row_tot = np.maximum(
        w_exc_log.sum(axis=1, keepdims=True) + w_inh_log.sum(axis=1, keepdims=True), 1.0
    )
    w_exc_n = (w_exc_log / row_tot).astype(np.float32)
    w_inh_n = (w_inh_log / row_tot).astype(np.float32)

    return NeuropilConnectome(
        region_names=region_names,
        matched_mask=matched,
        w_exc=w_exc_n,
        w_inh=w_inh_n,
        synapse_total=total_syn,
        matched_count=int(matched.sum()),
    )


def shuffle_neuropil_connectome(nc: NeuropilConnectome, seed: int) -> NeuropilConnectome:
    """Row- and sign-preserving shuffled regional connectome (negative control)."""
    rng = np.random.default_rng(seed * 997 + 13)
    n_r = len(nc.region_names)
    we = np.zeros_like(nc.w_exc)
    wi = np.zeros_like(nc.w_inh)
    for row in range(n_r):
        cols = [c for c in range(n_r) if c != row]
        perm_e = rng.permutation(cols)
        perm_i = rng.permutation(cols)
        we[row, cols] = nc.w_exc[row, perm_e]
        wi[row, cols] = nc.w_inh[row, perm_i]
    return NeuropilConnectome(
        region_names=nc.region_names,
        matched_mask=nc.matched_mask,
        w_exc=we,
        w_inh=wi,
        synapse_total=nc.synapse_total,
        matched_count=nc.matched_count,
    )


@dataclass(frozen=True)
class AimonRecording:
    exp_id: str
    fps_hz: float
    # Downsampled uniform 5 Hz arrays: regions (T, R), beh (T, 3) = [walk, left, right]
    regions: npt.NDArray[np.float32]
    behavior: npt.NDArray[np.float32]


def _downsample_mean(x: npt.NDArray[np.float64], factor: int) -> npt.NDArray[np.float32]:
    if factor <= 1:
        return x.astype(np.float32, copy=False)
    n_bins = x.shape[0] // factor
    if n_bins < 4:
        return x.astype(np.float32, copy=False)
    trimmed = x[: n_bins * factor]
    new_shape = (n_bins, factor, *x.shape[1:])
    return trimmed.reshape(new_shape).mean(axis=1).astype(np.float32)


def load_aimon2023_recordings(
    datasets_dir: Path,
    *,
    max_flies: int = 10,
    target_hz: float = 5.0,
    max_frames: int = 400,
    verify_hash: bool = True,
) -> tuple[tuple[str, ...], list[AimonRecording]]:
    """Load pan-neuronal Aimon et al. (2023) whole-brain 75-region recordings from zip."""
    raw_dir = datasets_dir / F7_DIR_NAME / "raw"
    zip_path = (
        verify_f7_file(raw_dir, "Walk_anatomical_regions.zip")
        if verify_hash
        else (raw_dir / "Walk_anatomical_regions.zip")
    )
    recs: list[AimonRecording] = []
    with zipfile.ZipFile(zip_path) as zf:
        raw_lines = zf.read("Walk_anatomical_regions/_RegionNames.txt").decode().splitlines()
        # Skip the first 2 header lines ("Large: ..." and "Small:")
        region_names = tuple(
            line.strip()
            for line in raw_lines
            if line.strip() and not line.startswith(("Large:", "Small:"))
        )
        exps: dict[str, set[str]] = {}
        for nm in zf.namelist():
            base = nm.split("/")[-1]
            for suf in ("_Regions.npy", "_Walk.npy", "_Left.npy", "_Right.npy"):
                if base.endswith(suf):
                    exps.setdefault(base[: -len(suf)], set()).add(suf)

        # Select pan-neuronal (Nsyb / GMR57C10) experiments with all 3 behavioral regressors.
        candidates = sorted(
            k
            for k, sufs in exps.items()
            if {"_Regions.npy", "_Walk.npy", "_Left.npy", "_Right.npy"} <= sufs
            and k.startswith(("Nsyb_", "nsyb_", "GMR57C10_"))
        )
        for k in candidates:
            parts = k.split("_")
            try:
                fps = float(parts[2])
            except (IndexError, ValueError):
                fps = 20.0
            reg_raw = np.load(
                io.BytesIO(zf.read(f"Walk_anatomical_regions/{k}_Regions.npy")),
                allow_pickle=False,
            )  # (R, T)
            if reg_raw.shape[0] != len(region_names) or np.isnan(reg_raw).any():
                continue
            wk = np.load(
                io.BytesIO(zf.read(f"Walk_anatomical_regions/{k}_Walk.npy")), allow_pickle=False
            ).reshape(-1)
            lf = np.load(
                io.BytesIO(zf.read(f"Walk_anatomical_regions/{k}_Left.npy")), allow_pickle=False
            ).reshape(-1)
            rt = np.load(
                io.BytesIO(zf.read(f"Walk_anatomical_regions/{k}_Right.npy")), allow_pickle=False
            ).reshape(-1)
            t_min = min(reg_raw.shape[1], wk.shape[0], lf.shape[0], rt.shape[0])
            reg_t = reg_raw[:, :t_min].T.astype(np.float64)  # (T, R)
            beh_t = np.stack([wk[:t_min], lf[:t_min], rt[:t_min]], axis=1).astype(np.float64)
            if np.isnan(beh_t).any():
                continue
            factor = max(1, round(fps / target_hz))
            reg_ds = _downsample_mean(reg_t, factor)[:max_frames]
            beh_ds = _downsample_mean(beh_t, factor)[:max_frames]
            if reg_ds.shape[0] < 80:
                continue
            # Require actual walking variance so held-out test isn't all-stationary.
            if float(beh_ds[:, 0].std()) < 1e-4:
                continue
            recs.append(
                AimonRecording(exp_id=k, fps_hz=fps / factor, regions=reg_ds, behavior=beh_ds)
            )
            if len(recs) >= max_flies:
                break
    return region_names, recs


def _zscore_train_test(
    x_tr: npt.NDArray[np.float32], x_te: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(x_tr.std(axis=0, keepdims=True), np.float32(1e-4))
    return ((x_tr - mu) / sd).astype(np.float32), ((x_te - mu) / sd).astype(np.float32)


def _fit_aimon_openloop_and_spatial(
    nc: NeuropilConnectome,
    recs: list[AimonRecording],
    *,
    mode: str,  # "reference", "shuffled", "unconnected"
    epochs: int,
    lr: float,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Fit a FlyWire-constrained 75-region dynamical model on Aimon 2023 real flies.

    Evaluates three strict held-out generalizations (zero teacher forcing):
    1. ``temporal_holdout_r2`` & ``temporal_holdout_r``: open-loop prediction on the unseen final
       30% of each fly's recording from Walk/Left/Right regressors.
    2. ``cross_fly_holdout_r2`` & ``cross_fly_holdout_r``: zero-shot prediction on completely
       unseen held-out flies (last 30% of flies never used in training!).
    3. ``spatial_unobs_r2`` & ``spatial_unobs_r``: on unseen held-out time segments, observe 50%
       of brain regions and predict the OTHER 50% UNOBSERVED brain regions via connectome coupling!
    """
    import torch

    dev = _resolve_device(device)
    n_r = len(nc.region_names)
    rng = np.random.default_rng(seed * 313 + 7)

    # Split flies into train flies (70%) and unseen cross-fly test flies (30%).
    n_flies = len(recs)
    n_tr_flies = max(1, round(n_flies * 0.7))
    n_tr_flies = min(n_tr_flies, max(1, n_flies - 1)) if n_flies >= 2 else 1

    # Prepare normalized temporal splits per fly (train = first 70% frames, test = last 30%).
    tr_y_list, tr_u_list = [], []
    te_y_list, te_u_list = [], []
    for rec in recs:
        t_tot = rec.regions.shape[0]
        cut = max(20, round(t_tot * 0.7))
        y_tr, y_te = _zscore_train_test(rec.regions[:cut], rec.regions[cut:])
        u_tr, u_te = _zscore_train_test(rec.behavior[:cut], rec.behavior[cut:])
        tr_y_list.append(torch.as_tensor(y_tr, device=dev))
        tr_u_list.append(torch.as_tensor(u_tr, device=dev))
        te_y_list.append(torch.as_tensor(y_te, device=dev))
        te_u_list.append(torch.as_tensor(u_te, device=dev))

    if mode == "unconnected":
        w_exc_t = torch.zeros((n_r, n_r), dtype=torch.float32, device=dev)
        w_inh_t = torch.zeros((n_r, n_r), dtype=torch.float32, device=dev)
    else:
        w_exc_t = torch.as_tensor(nc.w_exc, dtype=torch.float32, device=dev)
        w_inh_t = torch.as_tensor(nc.w_inh, dtype=torch.float32, device=dev)

    # Shared connectome-constrained dynamical parameters (75 regions):
    # - decay alpha_r in (0.05, 0.95)
    # - regional excitatory & inhibitory coupling gains g_exc_r, g_inh_r >= 0
    # - behavioral input weights B_beh (R, 3) + bias b_r (R,)
    alpha_logit = torch.zeros(n_r, device=dev, requires_grad=True)
    log_g_exc = torch.full((n_r,), -0.5, device=dev, requires_grad=True)
    log_g_inh = torch.full((n_r,), -0.5, device=dev, requires_grad=True)
    b_beh = torch.zeros((n_r, 3), device=dev, requires_grad=True)
    bias = torch.zeros(n_r, device=dev, requires_grad=True)
    params = [alpha_logit, log_g_exc, log_g_inh, b_beh, bias]

    def rollout_openloop(u_seq: Any) -> Any:
        # u_seq: (T, 3) or (B, T, 3) -> returns pred (T, R) or (B, T, R) purely open-loop!
        batched = u_seq.ndim == 3
        u_b = u_seq if batched else u_seq.unsqueeze(0)
        b_sz, t_len, _ = u_b.shape
        alpha = (0.05 + 0.90 * torch.sigmoid(alpha_logit)).unsqueeze(0)
        g_e = torch.exp(torch.clamp(log_g_exc, -3.0, 2.0))
        g_i = torch.exp(torch.clamp(log_g_inh, -3.0, 2.0))
        w_eff_t = ((w_exc_t * g_e.unsqueeze(0)) - (w_inh_t * g_i.unsqueeze(0))).transpose(0, 1)
        drive_all = torch.matmul(u_b, b_beh.transpose(0, 1)) + bias.view(1, 1, n_r)  # (B, T, R)
        x = torch.zeros((b_sz, n_r), device=dev)
        outs: list[Any] = []
        for t in range(t_len):
            syn = x @ w_eff_t
            target = torch.tanh(syn + drive_all[:, t, :]) * 2.5
            x = x + alpha * (target - x)
            outs.append(x)
        out = torch.stack(outs, dim=1)
        return out if batched else out.squeeze(0)

    # Stack training flies to common length so BPTT launches 1 batched kernel per frame.
    min_tr_t = min(int(tr_y_list[i].shape[0]) for i in range(n_tr_flies))
    tr_y_batch = torch.stack([tr_y_list[i][:min_tr_t] for i in range(n_tr_flies)], dim=0)
    tr_u_batch = torch.stack([tr_u_list[i][:min_tr_t] for i in range(n_tr_flies)], dim=0)

    # Evaluate untrained baseline on fly test segments.
    with torch.no_grad():
        untr_r2s = [
            _r2_score(
                te_y_list[i].cpu().numpy().astype(np.float64),
                rollout_openloop(te_u_list[i]).cpu().numpy().astype(np.float64),
            )
            for i in range(n_flies)
        ]

    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        # Train ONLY on the first n_tr_flies' training 70% segments!
        pred = rollout_openloop(tr_u_batch)
        loss = torch.mean((pred - tr_y_batch) ** 2) + 1e-4 * torch.mean(b_beh**2)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

    # Evaluate open-loop generalization on:
    # (a) Temporal holdout on the training flies (unseen last 30% time segment)
    # (b) Cross-fly holdout on unseen flies (entire fly never seen during training!)
    temp_r2s, temp_rs = [], []
    xfly_r2s, xfly_rs = [], []
    with torch.no_grad():
        for i in range(n_tr_flies):
            yt = te_y_list[i].cpu().numpy().astype(np.float64)
            yp = rollout_openloop(te_u_list[i]).cpu().numpy().astype(np.float64)
            temp_r2s.append(_r2_score(yt, yp))
            temp_rs.append(_pearson_r(yt.ravel(), yp.ravel()))
        for i in range(n_tr_flies, n_flies):
            yt = te_y_list[i].cpu().numpy().astype(np.float64)
            yp = rollout_openloop(te_u_list[i]).cpu().numpy().astype(np.float64)
            xfly_r2s.append(_r2_score(yt, yp))
            xfly_rs.append(_pearson_r(yt.ravel(), yp.ravel()))

    # 3. Spatial unobserved-region prediction test (tests whether FlyWire inter-neuropil wiring
    #    predicts held-out UNOBSERVED brain regions from 50% OBSERVED brain regions!).
    perm_reg = rng.permutation(n_r)
    obs_r = np.sort(perm_reg[: n_r // 2])
    unobs_r = np.sort(perm_reg[n_r // 2 :])
    obs_t = torch.as_tensor(obs_r, dtype=torch.long, device=dev)
    unobs_t = torch.as_tensor(unobs_r, dtype=torch.long, device=dev)

    # Learnable regional gains applied strictly to the fixed FlyWire (or shuffled) inter-region
    # submatrix W[unobs, obs] + behavioral regressors -> predicts unobserved regions!
    s_g_exc = torch.zeros(len(obs_r), device=dev, requires_grad=True)
    s_g_inh = torch.zeros(len(obs_r), device=dev, requires_grad=True)
    s_beh = torch.zeros((len(unobs_r), 3), device=dev, requires_grad=True)
    s_scale = torch.ones(len(unobs_r), device=dev, requires_grad=True)
    s_bias = torch.zeros(len(unobs_r), device=dev, requires_grad=True)
    s_params = [s_g_exc, s_g_inh, s_beh, s_scale, s_bias]
    we_sub = w_exc_t[unobs_t][:, obs_t]  # (U, O)
    wi_sub = w_inh_t[unobs_t][:, obs_t]  # (U, O)

    def predict_unobs(y_obs_seq: Any, u_seq: Any) -> Any:
        ge = torch.exp(torch.clamp(s_g_exc, -3.0, 3.0))
        gi = torch.exp(torch.clamp(s_g_inh, -3.0, 3.0))
        w_uo = (we_sub * ge.unsqueeze(0)) - (wi_sub * gi.unsqueeze(0))  # (U, O)
        syn = y_obs_seq @ w_uo.transpose(0, 1)  # (T, U)
        return s_scale.unsqueeze(0) * syn + u_seq @ s_beh.transpose(0, 1) + s_bias.unsqueeze(0)

    s_opt = torch.optim.Adam(s_params, lr=0.05)
    for _ in range(epochs):
        s_opt.zero_grad(set_to_none=True)
        sloss = torch.tensor(0.0, device=dev)
        for i in range(n_tr_flies):
            yp = predict_unobs(tr_y_list[i][:, obs_t], tr_u_list[i])
            sloss = sloss + torch.mean((yp - tr_y_list[i][:, unobs_t]) ** 2)
        sloss = sloss / n_tr_flies + 1e-4 * (torch.mean(s_g_exc**2) + torch.mean(s_g_inh**2))
        sloss.backward()  # type: ignore[no-untyped-call]
        s_opt.step()

    sp_r2s, sp_rs = [], []
    with torch.no_grad():
        for i in range(n_tr_flies, n_flies) if n_flies > n_tr_flies else range(n_flies):
            yt = te_y_list[i][:, unobs_t].cpu().numpy().astype(np.float64)
            yp = (
                predict_unobs(te_y_list[i][:, obs_t], te_u_list[i]).cpu().numpy().astype(np.float64)
            )
            sp_r2s.append(_r2_score(yt, yp))
            sp_rs.append(_pearson_r(yt.ravel(), yp.ravel()))

    return {
        "untrained_r2": round(float(np.mean(untr_r2s)), 4),
        "temporal_holdout_r2": round(float(np.mean(temp_r2s)), 4),
        "temporal_holdout_r": round(float(np.mean(temp_rs)), 4),
        "cross_fly_holdout_r2": round(float(np.mean(xfly_r2s)), 4) if xfly_r2s else None,
        "cross_fly_holdout_r": round(float(np.mean(xfly_rs)), 4) if xfly_rs else None,
        "spatial_unobs_r2": round(float(np.mean(sp_r2s)), 4),
        "spatial_unobs_r": round(float(np.mean(sp_rs)), 4),
    }


def run_aimon2023_fit(
    pack: ConnectomePack,
    datasets_dir: Path,
    *,
    max_flies: int = 10,
    epochs: int = 100,
    lr: float = 0.04,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    device: str = "auto",
) -> dict[str, Any]:
    """Fit FlyWire v783 75-neuropil connectome to real Aimon et al. (2023) whole-brain imaging."""
    t0 = time.perf_counter()
    region_names, recs = load_aimon2023_recordings(datasets_dir, max_flies=max_flies)
    nc_ref = build_neuropil_connectome(pack, region_names)
    dev = _resolve_device(device)

    ref_runs: list[dict[str, Any]] = []
    shuf_runs: list[dict[str, Any]] = []
    unconn_runs: list[dict[str, Any]] = []
    for s in seeds:
        r_ref = _fit_aimon_openloop_and_spatial(
            nc_ref, recs, mode="reference", epochs=epochs, lr=lr, device=dev, seed=int(s)
        )
        nc_shuf = shuffle_neuropil_connectome(nc_ref, seed=int(s))
        r_shuf = _fit_aimon_openloop_and_spatial(
            nc_shuf, recs, mode="shuffled", epochs=epochs, lr=lr, device=dev, seed=int(s)
        )
        r_unc = _fit_aimon_openloop_and_spatial(
            nc_ref, recs, mode="unconnected", epochs=epochs, lr=lr, device=dev, seed=int(s)
        )
        ref_runs.append({"seed": int(s), **r_ref})
        shuf_runs.append({"seed": int(s), **r_shuf})
        unconn_runs.append({"seed": int(s), **r_unc})

    def _col(rows: list[dict[str, Any]], k: str) -> list[float]:
        return [float(r[k]) for r in rows if r[k] is not None]

    return {
        "dataset": "aimon2023_walk",
        "citation": (
            "Aimon et al. (2023), eLife 85202; Zenodo 7849014 (Walk_anatomical_regions.zip)"
        ),
        "flywire_pack_digest": pack.pack_digest,
        "device": dev,
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "n_flies": len(recs),
        "fly_ids": [r.exp_id for r in recs],
        "n_regions": len(region_names),
        "matched_flywire_regions": nc_ref.matched_count,
        "aggregated_flywire_synapses": nc_ref.synapse_total,
        "framing_note": (
            "Real in vivo whole-brain Drosophila calcium imaging (Aimon et al. 2023) fitted with "
            "a FlyWire v783 75-neuropil dynamical connectome. Evaluated strictly on held-out time "
            "segments, held-out unseen flies, and held-out unobserved brain regions."
        ),
        "summary": {
            "reference_temporal_r2": describe(_col(ref_runs, "temporal_holdout_r2")),
            "shuffled_temporal_r2": describe(_col(shuf_runs, "temporal_holdout_r2")),
            "unconnected_temporal_r2": describe(_col(unconn_runs, "temporal_holdout_r2")),
            "reference_cross_fly_r2": describe(_col(ref_runs, "cross_fly_holdout_r2")),
            "shuffled_cross_fly_r2": describe(_col(shuf_runs, "cross_fly_holdout_r2")),
            "unconnected_cross_fly_r2": describe(_col(unconn_runs, "cross_fly_holdout_r2")),
            "reference_spatial_unobs_r2": describe(_col(ref_runs, "spatial_unobs_r2")),
            "shuffled_spatial_unobs_r2": describe(_col(shuf_runs, "spatial_unobs_r2")),
            "unconnected_spatial_unobs_r2": describe(_col(unconn_runs, "spatial_unobs_r2")),
            "reference_spatial_unobs_r": describe(_col(ref_runs, "spatial_unobs_r")),
            "shuffled_spatial_unobs_r": describe(_col(shuf_runs, "spatial_unobs_r")),
        },
        "comparisons": {
            "spatial_unobs_r2_ref_vs_shuffled": compare(
                _col(ref_runs, "spatial_unobs_r2"), _col(shuf_runs, "spatial_unobs_r2"), seed=701
            ),
            "spatial_unobs_r2_ref_vs_unconnected": compare(
                _col(ref_runs, "spatial_unobs_r2"), _col(unconn_runs, "spatial_unobs_r2"), seed=702
            ),
            "cross_fly_r2_ref_vs_shuffled": compare(
                _col(ref_runs, "cross_fly_holdout_r2"),
                _col(shuf_runs, "cross_fly_holdout_r2"),
                seed=703,
            ),
            "cross_fly_r2_ref_vs_unconnected": compare(
                _col(ref_runs, "cross_fly_holdout_r2"),
                _col(unconn_runs, "cross_fly_holdout_r2"),
                seed=704,
            ),
        },
        "per_seed": {"reference": ref_runs, "shuffled": shuf_runs, "unconnected": unconn_runs},
    }


def load_door2_flywire_matrix(
    pack: ConnectomePack,
    datasets_dir: Path,
    *,
    verify_hash: bool = True,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Load DoOR 2.0 real odor matrix (692 odors x 41 mapped FlyWire ORN glomeruli) plus
    FlyWire's exact 2-hop antennal-lobe inter-glomerular E/I coupling matrices.
    """
    raw_dir = datasets_dir / F7_DIR_NAME / "raw"
    door_path = (
        verify_f7_file(raw_dir, "door_or_merged.csv")
        if verify_hash
        else (raw_dir / "door_or_merged.csv")
    )
    map_path = (
        verify_f7_file(raw_dir, "door_to_flywire_mapping.csv")
        if verify_hash
        else (raw_dir / "door_to_flywire_mapping.csv")
    )

    d2f: dict[str, str] = {}
    with map_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["flywire_glomerulus"].startswith("ORN_"):
                d2f[row["door_name"]] = row["flywire_glomerulus"]

    with door_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        hdr = next(reader)
        receptors = hdr[1:]
        odors: list[str] = []
        rows: list[list[float]] = []
        for vals in reader:
            odors.append(vals[0])
            rows.append([float(x) for x in vals[1:]])
    full_m = np.asarray(rows, dtype=np.float32)

    fw_types = set(pack.cells.cell_type)
    # Deduplicate glomeruli (average receptors mapping to the same ORN_<glom> if any).
    glom_to_cols: dict[str, list[int]] = {}
    for ci, rec in enumerate(receptors):
        g = d2f.get(rec)
        if g is not None and g in fw_types:
            glom_to_cols.setdefault(g, []).append(ci)

    gloms = tuple(sorted(glom_to_cols.keys()))
    mat = np.stack(
        [full_m[:, glom_to_cols[g]].mean(axis=1) for g in gloms],
        axis=1,
    ).astype(np.float32)  # (692, G)

    # Build FlyWire v783 2-hop antennal-lobe inter-glomerular connectivity:
    # ORN_g1 -> (ALPN/ALLN) -> ORN_g2/ALPN_g2 aggregated by glomerulus.
    g_to_idx = {g: i for i, g in enumerate(gloms)}
    n_g = len(gloms)
    n_cells = pack.cells.num_cells
    orn_g = np.full(n_cells, -1, dtype=np.int32)
    al_inter = np.zeros(n_cells, dtype=np.bool_)
    for i in range(n_cells):
        ct = pack.cells.cell_type[i]
        if ct in g_to_idx:
            orn_g[i] = g_to_idx[ct]
        elif pack.cells.cell_class[i] in ("ALLN", "ALPN", "ALIN"):
            al_inter[i] = True

    # Aggregate ORN_g -> AL_inter (excitatory) and AL_inter -> ORN_g (exc/inh)
    e = pack.edges
    inter_ids = np.flatnonzero(al_inter)
    inter_remap = np.full(n_cells, -1, dtype=np.int32)
    inter_remap[inter_ids] = np.arange(inter_ids.size, dtype=np.int32)
    n_int = int(inter_ids.size)

    m_in = (orn_g[e.pre] >= 0) & (inter_remap[e.post] >= 0)
    a_in = np.zeros((n_int, n_g), dtype=np.float64)
    np.add.at(
        a_in, (inter_remap[e.post[m_in]], orn_g[e.pre[m_in]]), e.count[m_in].astype(np.float64)
    )

    m_out = (inter_remap[e.pre] >= 0) & (orn_g[e.post] >= 0)
    a_out_e = np.zeros((n_g, n_int), dtype=np.float64)
    a_out_i = np.zeros((n_g, n_int), dtype=np.float64)
    out_e = m_out & (e.sign > 0)
    out_i = m_out & (e.sign < 0)
    np.add.at(
        a_out_e,
        (orn_g[e.post[out_e]], inter_remap[e.pre[out_e]]),
        e.count[out_e].astype(np.float64),
    )
    np.add.at(
        a_out_i,
        (orn_g[e.post[out_i]], inter_remap[e.pre[out_i]]),
        e.count[out_i].astype(np.float64),
    )

    # 2-hop effective coupling + direct ORN->ORN synapses
    w_exc = np.log1p(a_out_e @ a_in)
    w_inh = np.log1p(a_out_i @ a_in)
    np.fill_diagonal(w_exc, 0.0)
    np.fill_diagonal(w_inh, 0.0)
    row_s = np.maximum(w_exc.sum(axis=1, keepdims=True) + w_inh.sum(axis=1, keepdims=True), 1.0)
    return (
        tuple(odors),
        gloms,
        mat,
        (w_exc / row_s).astype(np.float32),
        (w_inh / row_s).astype(np.float32),
    )


def run_door2_fit(
    pack: ConnectomePack,
    datasets_dir: Path,
    *,
    epochs: int = 120,
    lr: float = 0.05,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate FlyWire v783 antennal-lobe coupling on predicting held-out real DoOR 2.0 odors."""
    import torch

    t0 = time.perf_counter()
    _odors, gloms, mat, w_exc, w_inh = load_door2_flywire_matrix(pack, datasets_dir)
    dev = _resolve_device(device)
    n_odors, n_g = mat.shape

    def eval_one(
        w_e_np: npt.NDArray[np.float32], w_i_np: npt.NDArray[np.float32], seed: int
    ) -> dict[str, float]:
        rng = np.random.default_rng(seed * 419 + 5)
        o_perm = rng.permutation(n_odors)
        cut = round(n_odors * 0.75)
        tr_o, te_o = o_perm[:cut], o_perm[cut:]
        x_tr_all, x_te_all = _zscore_train_test(mat[tr_o], mat[te_o])

        g_perm = rng.permutation(n_g)
        obs_g = np.sort(g_perm[: n_g // 2])
        unobs_g = np.sort(g_perm[n_g // 2 :])

        x_tr_obs = torch.as_tensor(x_tr_all[:, obs_g], device=dev)
        y_tr_unobs = torch.as_tensor(x_tr_all[:, unobs_g], device=dev)
        x_te_obs = torch.as_tensor(x_te_all[:, obs_g], device=dev)
        y_te_unobs = x_te_all[:, unobs_g].astype(np.float64)

        we_sub = torch.as_tensor(w_e_np[np.ix_(unobs_g, obs_g)], device=dev)
        wi_sub = torch.as_tensor(w_i_np[np.ix_(unobs_g, obs_g)], device=dev)

        g_e = torch.zeros(len(obs_g), device=dev, requires_grad=True)
        g_i = torch.zeros(len(obs_g), device=dev, requires_grad=True)
        scale = torch.ones(len(unobs_g), device=dev, requires_grad=True)
        bias = torch.zeros(len(unobs_g), device=dev, requires_grad=True)
        opt = torch.optim.Adam([g_e, g_i, scale, bias], lr=lr)

        def pred(x_obs: Any) -> Any:
            w = (we_sub * torch.exp(torch.clamp(g_e, -3.0, 3.0)).unsqueeze(0)) - (
                wi_sub * torch.exp(torch.clamp(g_i, -3.0, 3.0)).unsqueeze(0)
            )
            return scale.unsqueeze(0) * (x_obs @ w.transpose(0, 1)) + bias.unsqueeze(0)

        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((pred(x_tr_obs) - y_tr_unobs) ** 2) + 1e-4 * (
                torch.mean(g_e**2) + torch.mean(g_i**2)
            )
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()

        with torch.no_grad():
            yp = pred(x_te_obs).cpu().numpy().astype(np.float64)
        return {
            "heldout_odor_r2": round(_r2_score(y_te_unobs, yp), 4),
            "heldout_odor_r": round(_pearson_r(y_te_unobs.ravel(), yp.ravel()), 4),
        }

    ref_runs, shuf_runs = [], []
    for s in seeds:
        ref_runs.append({"seed": int(s), **eval_one(w_exc, w_inh, int(s))})
        rng_s = np.random.default_rng(int(s) * 881 + 3)
        we_s = np.zeros_like(w_exc)
        wi_s = np.zeros_like(w_inh)
        for row in range(n_g):
            cols = [c for c in range(n_g) if c != row]
            we_s[row, cols] = w_exc[row, rng_s.permutation(cols)]
            wi_s[row, cols] = w_inh[row, rng_s.permutation(cols)]
        shuf_runs.append({"seed": int(s), **eval_one(we_s, wi_s, int(s))})

    ref_r2 = [float(r["heldout_odor_r2"]) for r in ref_runs]
    shuf_r2 = [float(r["heldout_odor_r2"]) for r in shuf_runs]
    return {
        "dataset": "door2_olfactory",
        "citation": "DoOR 2.0 (Münch & Galizia 2016, Sci Rep 6:21841); FlyWire mapping",
        "flywire_pack_digest": pack.pack_digest,
        "device": dev,
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "n_odors": n_odors,
        "n_glomeruli": n_g,
        "glomeruli": list(gloms),
        "framing_note": (
            "Real DoOR 2.0 odor responses mapped 1:1 to FlyWire v783 ORN glomeruli. Tests "
            "whether FlyWire antennal-lobe 2-hop inter-glomerular coupling predicts held-out "
            "glomeruli on 173 unseen real odorants better than shuffled coupling."
        ),
        "summary": {
            "reference_heldout_r2": describe(ref_r2),
            "shuffled_heldout_r2": describe(shuf_r2),
            "reference_heldout_r": describe([float(r["heldout_odor_r"]) for r in ref_runs]),
            "shuffled_heldout_r": describe([float(r["heldout_odor_r"]) for r in shuf_runs]),
        },
        "comparison": compare(ref_r2, shuf_r2, seed=705),
        "per_seed": {"reference": ref_runs, "shuffled": shuf_runs},
    }


def run_synthetic_f7_check(
    *, seed: int = 1, epochs: int = 60, device: str = "cpu"
) -> dict[str, Any]:
    """Fast offline synthetic whole-brain regional recording check for CPU-only CI."""
    rng = np.random.default_rng(seed * 73 + 11)
    n_r = 12
    rnames = tuple(f"REG_{i}" for i in range(n_r))
    w_exc = np.zeros((n_r, n_r), dtype=np.float32)
    w_inh = np.zeros((n_r, n_r), dtype=np.float32)
    for i in range(n_r):
        w_exc[i, (i - 1) % n_r] = 0.65
        w_exc[i, (i - 2) % n_r] = 0.25
        w_inh[i, (i + 3) % n_r] = 0.35
    nc = NeuropilConnectome(
        region_names=rnames,
        matched_mask=np.ones(n_r, dtype=np.bool_),
        w_exc=w_exc,
        w_inh=w_inh,
        synapse_total=1000,
        matched_count=n_r,
    )
    recs: list[AimonRecording] = []
    b_true = rng.normal(0.0, 0.5, size=(n_r, 3)).astype(np.float32)
    for f_idx in range(4):
        t_len = 120
        t_ax = np.linspace(0, 12, t_len, dtype=np.float32)
        beh = np.stack(
            [
                np.maximum(0.0, np.sin(t_ax + f_idx)),
                np.maximum(0.0, np.cos(t_ax * 0.7 + f_idx)),
                np.maximum(0.0, -np.cos(t_ax * 0.7 + f_idx)),
            ],
            axis=1,
        ).astype(np.float32)
        y = np.zeros((t_len, n_r), dtype=np.float32)
        x = np.zeros(n_r, dtype=np.float32)
        w_eff = 0.8 * w_exc - 0.6 * w_inh
        for t in range(t_len):
            x = x + 0.35 * (np.tanh(w_eff @ x + beh[t] @ b_true.T) * 2.0 - x)
            y[t] = x + rng.normal(0.0, 0.05, size=n_r).astype(np.float32)
        recs.append(AimonRecording(exp_id=f"syn_fly_{f_idx}", fps_hz=5.0, regions=y, behavior=beh))

    ref = _fit_aimon_openloop_and_spatial(
        nc, recs, mode="reference", epochs=epochs, lr=0.05, device=device, seed=seed
    )
    shuf = _fit_aimon_openloop_and_spatial(
        shuffle_neuropil_connectome(nc, seed=seed),
        recs,
        mode="shuffled",
        epochs=epochs,
        lr=0.05,
        device=device,
        seed=seed,
    )
    unc = _fit_aimon_openloop_and_spatial(
        nc, recs, mode="unconnected", epochs=epochs, lr=0.05, device=device, seed=seed
    )
    return {"reference": ref, "shuffled": shuf, "unconnected": unc}
