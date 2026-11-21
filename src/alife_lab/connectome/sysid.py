"""Phase F5: Hidden-truth connectome system identification (differentiable rate model).

External optimization layer (NEVER modifies the core ``connectome`` substrate).
PyTorch is imported lazily inside functions so the FastAPI API process stays GPU-free.

Scientific design (docs/plans/connectome_plan.md §4 F5):
1. Build a hidden ground-truth ("teacher") brain by augmenting the anatomical wiring diagram
   with missing biological mechanisms:
   - ``heterogeneous_tau``: cell-type-specific membrane time constants tau_c in [6, 55] ms
   - ``sign_flips``: hidden neurotransmitter sign inversions on a fraction of active cell types
   - ``neuromodulation``: hidden per-class excitability gain gamma_k and tonic bias b_k
   - ``gap_junctions``: hidden symmetric electrical coupling g_ik * (r_k - r_i) on sparse pairs
2. Record realistic limited observations from the teacher brain:
   - partial cell sampling (``obs_fraction`` or specific ``obs_classes``)
   - coarse temporal binning (``bin_ms``, e.g. 20 ms imaging-like frames)
   - additive Gaussian measurement noise (``noise_sd``)
3. Fit student models using PyTorch autodiff (Adam + cosine LR) on ``reference`` wiring vs
   ``shuffled_control`` wiring vs ``untrained`` baseline, across multiple independent random
   initializations (``n_restarts``) to measure:
   - parameter recovery (tau Pearson r, sign-flip AUROC, neuromod Pearson r, gap-junction AUROC)
   - unseen-stimulus generalization on BOTH observed and unobserved neurons (heldout R^2)
   - non-uniqueness / identifiability (parameter disagreement across equally well-fitting restarts)
4. Provide observability curves (recording fraction / binning) and cell-class observability maps
   ("which cells are most informative to record?").
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.subcircuit import Subcircuit


@dataclass(frozen=True)
class HiddenMechanismSpec:
    """Ground-truth biological mechanisms hidden inside the teacher brain."""

    tau_enabled: bool = True
    sign_flip_fraction: float = 0.15
    neuromod_enabled: bool = True
    gap_junctions_enabled: bool = True
    gap_candidates: int = 128
    gap_active_fraction: float = 0.25


@dataclass(frozen=True)
class ObservationSpec:
    """Realistic limited-recording operator applied to teacher trajectories."""

    obs_fraction: float = 0.25
    obs_classes: tuple[str, ...] = ()
    bin_ms: float = 10.0
    noise_sd: float = 0.015


@dataclass(frozen=True)
class SysIDTrainConfig:
    """External gradient-based fitting configuration."""

    dt_ms: float = 2.0
    sim_ms: float = 100.0
    n_train_stimuli: int = 8
    n_holdout_stimuli: int = 4
    epochs: int = 120
    lr: float = 0.06
    l1_gap: float = 1e-3
    l2_reg: float = 5e-4
    n_restarts: int = 3
    device: str = "auto"


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _auroc(y_true: npt.NDArray[np.float64], y_score: npt.NDArray[np.float64]) -> float:
    """Exact rank-based AUROC (returns NaN if only one class is present)."""
    pos = y_true > 0.5
    n_pos = int(pos.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # Handle ties with average ranks.
    sorted_s = y_score[order]
    i = 0
    n = sorted_s.size
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_r = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_r
        i = j
    sum_pos = float(ranks[pos].sum())
    return (sum_pos - 0.5 * n_pos * (n_pos + 1)) / (n_pos * n_neg)


def _pearson_r(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    if a.size < 2:
        return float("nan")
    da = a - a.mean()
    db = b - b.mean()
    denom = float(np.sqrt((da * da).sum() * (db * db).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((da * db).sum() / denom)


def _r2_score(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _build_stimuli(
    sub: Subcircuit,
    *,
    n_train: int,
    n_holdout: int,
    steps: int,
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Create disjoint multi-glomerular odor pulse patterns (train vs unseen holdout).

    Shape: ``(steps, N, B)`` float32 external drive.
    """
    rng = np.random.default_rng(seed + 101)
    n_groups = len(sub.input_groups)
    total_b = n_train + n_holdout
    patterns: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    k_active = max(1, min(3, max(1, n_groups // 3)))
    for b in range(total_b):
        for _ in range(64):
            if b < n_groups and n_groups >= 4 and b < 4:
                # Guarantee single/pair anchor patterns cover all glomeruli cleanly.
                chosen = tuple(sorted({b % n_groups, (b + 1) % n_groups}))
            else:
                chosen = tuple(
                    sorted(int(x) for x in rng.choice(n_groups, size=k_active, replace=False))
                )
            if chosen not in seen:
                seen.add(chosen)
                patterns.append(chosen)
                break
        else:
            patterns.append((b % n_groups,))

    stim = np.zeros((steps, sub.num_cells, total_b), dtype=np.float32)
    t_on = max(1, steps // 10)
    t_off = max(t_on + 2, int(steps * 0.72))
    for b, grp_ids in enumerate(patterns):
        # Distinct temporal pulse offset/amplitude per odor to probe tau_m dynamics!
        amp = float(rng.uniform(1.6, 2.6))
        local_on = t_on + (b % 3) * 2
        local_off = min(steps - 2, t_off - (b % 2) * 4)
        for gid in grp_ids:
            cells = sub.input_groups[gid][1]
            stim[local_on:local_off, cells, b] = amp
    return stim[:, :, :n_train].copy(), stim[:, :, n_train:].copy()


def _select_gap_candidates(
    sub: Subcircuit,
    *,
    max_candidates: int,
    seed: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Sample undirected candidate electrical gap-junction pairs (intra-class / non-chemical)."""
    rng = np.random.default_rng(seed + 202)
    chem_set = set(zip(sub.pre.tolist(), sub.post.tolist(), strict=True))
    by_class: dict[int, list[int]] = {}
    for i, cid in enumerate(sub.class_ids.tolist()):
        by_class.setdefault(cid, []).append(i)

    # Prefer classes with >=2 cells and non-zero downstream involvement.
    pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    class_keys = [c for c, arr in by_class.items() if len(arr) >= 2]
    rng.shuffle(class_keys)
    attempts = 0
    while len(pairs) < max_candidates and attempts < max_candidates * 40 and class_keys:
        attempts += 1
        cid = int(class_keys[attempts % len(class_keys)])
        arr = by_class[cid]
        i1, i2 = rng.choice(len(arr), size=2, replace=False)
        u, v = int(arr[int(i1)]), int(arr[int(i2)])
        if u > v:
            u, v = v, u
        if (u, v) in seen_pairs or (u, v) in chem_set or (v, u) in chem_set:
            continue
        seen_pairs.add((u, v))
        pairs.append((u, v))

    if not pairs:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    u_arr = np.asarray([p[0] for p in pairs], dtype=np.int32)
    v_arr = np.asarray([p[1] for p in pairs], dtype=np.int32)
    return u_arr, v_arr


def _select_observed_cells(
    sub: Subcircuit,
    obs: ObservationSpec,
    *,
    seed: int,
) -> npt.NDArray[np.int32]:
    """Select observed neuron indices by cell-class filter or random stratified fraction."""
    rng = np.random.default_rng(seed + 303)
    if obs.obs_classes:
        wanted = set(obs.obs_classes)
        pool = np.asarray(
            [i for i in range(sub.num_cells) if sub.cell_classes[i] in wanted],
            dtype=np.int32,
        )
        if pool.size == 0:
            raise ValueError(f"no cells match obs_classes={obs.obs_classes!r}")
        if obs.obs_fraction >= 0.999:
            return pool
        k = max(1, round(pool.size * obs.obs_fraction))
        return np.sort(rng.choice(pool, size=k, replace=False)).astype(np.int32)

    if obs.obs_fraction >= 0.999:
        return np.arange(sub.num_cells, dtype=np.int32)
    k = max(1, round(sub.num_cells * obs.obs_fraction))
    return np.sort(rng.choice(sub.num_cells, size=k, replace=False)).astype(np.int32)


def _base_weight_scale(sub: Subcircuit) -> float:
    """Normalize chemical edge scale so typical synaptic input is O(1) across circuit sizes."""
    if sub.num_edges == 0:
        return 0.01
    # Compute mean absolute in-strength per active postsynaptic neuron.
    in_abs = np.zeros(sub.num_cells, dtype=np.float64)
    np.add.at(in_abs, sub.post, np.abs(sub.signed_weight).astype(np.float64))
    nz = in_abs[in_abs > 0]
    med = float(np.median(nz)) if nz.size > 0 else 100.0
    # Scale so median total in-strength * 0.5 rate ~= 2.5 input units.
    return float(5.0 / max(med, 10.0))


@dataclass
class GroundTruthParams:
    """Hidden biological parameters of the teacher brain (NumPy arrays)."""

    tau_type_ms: npt.NDArray[np.float32]  # (num_types,)
    sign_mult_type: npt.NDArray[np.float32]  # (num_types,) +1 or -1
    flipped_type_mask: npt.NDArray[np.bool_]  # (num_types,)
    neuromod_gain_class: npt.NDArray[np.float32]  # (num_classes,)
    neuromod_bias_class: npt.NDArray[np.float32]  # (num_classes,)
    gap_u: npt.NDArray[np.int32]  # (num_gap_cand,)
    gap_v: npt.NDArray[np.int32]  # (num_gap_cand,)
    gap_weight: npt.NDArray[np.float32]  # (num_gap_cand,)
    active_type_mask: npt.NDArray[np.bool_]  # (num_types,) types with >=1 outgoing edge


def _sample_ground_truth(
    sub: Subcircuit,
    mech: HiddenMechanismSpec,
    *,
    seed: int,
) -> GroundTruthParams:
    rng = np.random.default_rng(seed + 404)
    n_t = sub.num_types
    n_c = sub.num_classes

    # Active presynaptic cell types (have outgoing chemical edges in the subcircuit).
    out_deg = np.bincount(sub.pre, minlength=sub.num_cells)
    active_type = np.zeros(n_t, dtype=np.bool_)
    for i in range(sub.num_cells):
        if out_deg[i] > 0:
            active_type[int(sub.type_ids[i])] = True

    # 1. Heterogeneous per-type membrane time constants.
    if mech.tau_enabled:
        log_tau = rng.uniform(math.log(6.0), math.log(50.0), size=n_t)
        tau_type = np.exp(log_tau).astype(np.float32)
    else:
        tau_type = np.full(n_t, 20.0, dtype=np.float32)

    # 2. Hidden excitatory/inhibitory sign flips on a subset of active cell types.
    sign_mult = np.ones(n_t, dtype=np.float32)
    flipped = np.zeros(n_t, dtype=np.bool_)
    active_ids = np.flatnonzero(active_type)
    if mech.sign_flip_fraction > 0.0 and active_ids.size > 0:
        n_flip = max(1, round(active_ids.size * mech.sign_flip_fraction))
        n_flip = min(n_flip, int(active_ids.size))
        chosen = rng.choice(active_ids, size=n_flip, replace=False)
        sign_mult[chosen] = np.float32(-1.0)
        flipped[chosen] = True

    # 3. Hidden per-class neuromodulatory gain and tonic bias.
    if mech.neuromod_enabled:
        nm_gain = rng.uniform(0.65, 1.50, size=n_c).astype(np.float32)
        nm_bias = rng.uniform(-0.35, 0.35, size=n_c).astype(np.float32)
    else:
        nm_gain = np.ones(n_c, dtype=np.float32)
        nm_bias = np.zeros(n_c, dtype=np.float32)

    # 4. Sparse electrical gap junctions on candidate pairs.
    if mech.gap_junctions_enabled and mech.gap_candidates > 0:
        gap_u, gap_v = _select_gap_candidates(sub, max_candidates=mech.gap_candidates, seed=seed)
        n_g = gap_u.shape[0]
        gap_w = np.zeros(n_g, dtype=np.float32)
        if n_g > 0 and mech.gap_active_fraction > 0.0:
            n_act = max(1, round(n_g * mech.gap_active_fraction))
            act_idx = rng.choice(n_g, size=min(n_act, n_g), replace=False)
            gap_w[act_idx] = rng.uniform(0.30, 0.70, size=act_idx.size).astype(np.float32)
    else:
        gap_u = np.zeros(0, dtype=np.int32)
        gap_v = np.zeros(0, dtype=np.int32)
        gap_w = np.zeros(0, dtype=np.float32)

    return GroundTruthParams(
        tau_type_ms=tau_type,
        sign_mult_type=sign_mult,
        flipped_type_mask=flipped,
        neuromod_gain_class=nm_gain,
        neuromod_bias_class=nm_bias,
        gap_u=gap_u,
        gap_v=gap_v,
        gap_weight=gap_w,
        active_type_mask=active_type,
    )


def _make_sparse_chem_matrix(sub: Subcircuit, device: str) -> Any:
    import torch

    w_scale = _base_weight_scale(sub)
    N = sub.num_cells
    if sub.num_edges == 0:
        idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        vals = torch.zeros(0, dtype=torch.float32, device=device)
    else:
        idx = torch.stack(
            [
                torch.as_tensor(sub.post, dtype=torch.long, device=device),
                torch.as_tensor(sub.pre, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        vals = torch.as_tensor(
            sub.signed_weight.astype(np.float32) * np.float32(w_scale),
            dtype=torch.float32,
            device=device,
        )
    return torch.sparse_coo_tensor(idx, vals, size=(N, N), check_invariants=False).coalesce()


def _simulate_torch(
    *,
    w_chem: Any,
    type_ids_t: Any,
    class_ids_t: Any,
    gap_u_t: Any,
    gap_v_t: Any,
    tau_type: Any,
    sign_mult_type: Any,
    nm_gain_class: Any,
    nm_bias_class: Any,
    gap_weight: Any,
    stim_t: Any,
    dt_ms: float,
    bias_theta: float = 1.2,
) -> Any:
    """Roll out differentiable rate-based connectome dynamics.

    ``stim_t`` has shape ``(T, N, B)``; returns trajectory ``(T, N, B)`` in ``(0, 1)``.
    """
    import torch

    steps, n_cells, batch = stim_t.shape
    alpha = (dt_ms / tau_type[type_ids_t]).unsqueeze(1)  # (N, 1)
    m_cell = sign_mult_type[type_ids_t].unsqueeze(1)  # (N, 1)
    g_cell = nm_gain_class[class_ids_t].unsqueeze(1)  # (N, 1)
    b_cell = nm_bias_class[class_ids_t].unsqueeze(1)  # (N, 1)
    has_gap = gap_u_t.numel() > 0

    r = torch.full((n_cells, batch), 0.05, dtype=torch.float32, device=stim_t.device)
    traj: list[Any] = []
    for t in range(steps):
        syn = torch.sparse.mm(w_chem, r * m_cell)
        if has_gap:
            diff = r[gap_v_t] - r[gap_u_t]  # (E_g, B)
            cur = gap_weight.unsqueeze(1) * diff
            syn = syn.index_add(0, gap_u_t, cur).index_add(0, gap_v_t, -cur)
        drive = g_cell * (syn + stim_t[t]) + b_cell - bias_theta
        target_r = torch.sigmoid(drive)
        r = r + alpha * (target_r - r)
        traj.append(r)
    return torch.stack(traj, dim=0)


def _bin_trajectory_torch(traj: Any, bin_steps: int) -> Any:
    """Average-pool trajectory ``(T, N, B)`` into temporal bins of width ``bin_steps``."""
    steps, n_cells, batch = traj.shape
    n_bins = max(1, steps // bin_steps)
    usable = n_bins * bin_steps
    return traj[:usable].reshape(n_bins, bin_steps, n_cells, batch).mean(dim=1)


def _tau_from_logit(logit: Any) -> Any:
    import torch

    # Smoothly bounded in [4.0, 65.0] ms; logit=0 -> ~20 ms (logit(16/61) ≈ -1.034).
    return 4.0 + 61.0 * torch.sigmoid(logit)


def _logit_from_tau(tau_ms: float) -> float:
    p = min(max((tau_ms - 4.0) / 61.0, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


class SysIDSession:
    """Stateful epoch-by-epoch system identification session (used by CLI & browser adapter)."""

    def __init__(
        self,
        sub_ref: Subcircuit,
        sub_student: Subcircuit,
        mech: HiddenMechanismSpec,
        obs: ObservationSpec,
        cfg: SysIDTrainConfig,
        *,
        seed: int,
        restart_index: int = 0,
    ) -> None:
        import torch

        self.sub_ref = sub_ref
        self.sub_student = sub_student
        self.mech = mech
        self.obs = obs
        self.cfg = cfg
        self.seed = int(seed)
        self.restart_index = int(restart_index)
        self.device = _resolve_device(cfg.device)
        self.steps = max(4, round(cfg.sim_ms / cfg.dt_ms))
        self.bin_steps = max(1, round(obs.bin_ms / cfg.dt_ms))

        # 1. Sample hidden ground-truth teacher parameters & stimuli on the reference wiring.
        self.truth = _sample_ground_truth(sub_ref, mech, seed=self.seed)
        self.obs_idx = _select_observed_cells(sub_ref, obs, seed=self.seed)
        obs_mask = np.zeros(sub_ref.num_cells, dtype=np.bool_)
        obs_mask[self.obs_idx] = True
        self.unobs_idx = np.flatnonzero(~obs_mask).astype(np.int32)

        stim_tr, stim_ho = _build_stimuli(
            sub_ref,
            n_train=cfg.n_train_stimuli,
            n_holdout=cfg.n_holdout_stimuli,
            steps=self.steps,
            seed=self.seed,
        )
        self.stim_tr_t = torch.as_tensor(stim_tr, dtype=torch.float32, device=self.device)
        self.stim_ho_t = torch.as_tensor(stim_ho, dtype=torch.float32, device=self.device)
        self.obs_idx_t = torch.as_tensor(self.obs_idx, dtype=torch.long, device=self.device)

        self.w_ref = _make_sparse_chem_matrix(sub_ref, self.device)
        self.w_stu = (
            self.w_ref
            if sub_student is sub_ref
            else _make_sparse_chem_matrix(sub_student, self.device)
        )
        self.type_ids_t = torch.as_tensor(sub_ref.type_ids, dtype=torch.long, device=self.device)
        self.class_ids_t = torch.as_tensor(sub_ref.class_ids, dtype=torch.long, device=self.device)
        self.gap_u_t = torch.as_tensor(self.truth.gap_u, dtype=torch.long, device=self.device)
        self.gap_v_t = torch.as_tensor(self.truth.gap_v, dtype=torch.long, device=self.device)

        # 2. Simulate teacher brain (no grad) and create noisy binned partial recordings.
        with torch.no_grad():
            tr_true_full = _simulate_torch(
                w_chem=self.w_ref,
                type_ids_t=self.type_ids_t,
                class_ids_t=self.class_ids_t,
                gap_u_t=self.gap_u_t,
                gap_v_t=self.gap_v_t,
                tau_type=torch.as_tensor(self.truth.tau_type_ms, device=self.device),
                sign_mult_type=torch.as_tensor(self.truth.sign_mult_type, device=self.device),
                nm_gain_class=torch.as_tensor(self.truth.neuromod_gain_class, device=self.device),
                nm_bias_class=torch.as_tensor(self.truth.neuromod_bias_class, device=self.device),
                gap_weight=torch.as_tensor(self.truth.gap_weight, device=self.device),
                stim_t=self.stim_tr_t,
                dt_ms=cfg.dt_ms,
            )
            ho_true_full = _simulate_torch(
                w_chem=self.w_ref,
                type_ids_t=self.type_ids_t,
                class_ids_t=self.class_ids_t,
                gap_u_t=self.gap_u_t,
                gap_v_t=self.gap_v_t,
                tau_type=torch.as_tensor(self.truth.tau_type_ms, device=self.device),
                sign_mult_type=torch.as_tensor(self.truth.sign_mult_type, device=self.device),
                nm_gain_class=torch.as_tensor(self.truth.neuromod_gain_class, device=self.device),
                nm_bias_class=torch.as_tensor(self.truth.neuromod_bias_class, device=self.device),
                gap_weight=torch.as_tensor(self.truth.gap_weight, device=self.device),
                stim_t=self.stim_ho_t,
                dt_ms=cfg.dt_ms,
            )
            tr_binned_clean = _bin_trajectory_torch(tr_true_full, self.bin_steps)
            self.ho_true_np = ho_true_full.detach().cpu().numpy().astype(np.float32)

        rng_noise = np.random.default_rng(self.seed + 505)
        obs_clean = tr_binned_clean[:, self.obs_idx_t, :].detach().cpu().numpy()
        noise = rng_noise.normal(0.0, obs.noise_sd, size=obs_clean.shape).astype(np.float32)
        self.y_obs_np = np.clip(obs_clean + noise, 0.0, 1.0).astype(np.float32)
        self.y_obs_t = torch.as_tensor(self.y_obs_np, dtype=torch.float32, device=self.device)

        # 3. Initialize learnable student parameters (with small perturbation for restarts > 0).
        rng_init = np.random.default_rng(self.seed * 1009 + self.restart_index * 73 + 7)
        jitter = 0.0 if restart_index == 0 else 0.18
        tau_init = np.full(sub_student.num_types, _logit_from_tau(20.0), dtype=np.float32)
        sign_init = np.ones(sub_student.num_types, dtype=np.float32)
        gain_init = np.zeros(sub_student.num_classes, dtype=np.float32)  # exp(0) = 1
        bias_init = np.zeros(sub_student.num_classes, dtype=np.float32)
        gap_init = np.full(self.truth.gap_u.shape[0], -2.5, dtype=np.float32)  # softplus ~0.08
        if jitter > 0.0:
            tau_init += rng_init.normal(0.0, jitter * 2.0, size=tau_init.shape).astype(np.float32)
            sign_init += rng_init.normal(0.0, jitter, size=sign_init.shape).astype(np.float32)
            gain_init += rng_init.normal(0.0, jitter, size=gain_init.shape).astype(np.float32)
            bias_init += rng_init.normal(0.0, jitter * 0.5, size=bias_init.shape).astype(np.float32)
            if gap_init.size > 0:
                gap_init += rng_init.normal(0.0, jitter, size=gap_init.shape).astype(np.float32)

        self.tau_logit = torch.tensor(
            tau_init, dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.sign_mult = torch.tensor(
            sign_init, dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.nm_log_gain = torch.tensor(
            gain_init, dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.nm_bias = torch.tensor(
            bias_init, dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.gap_raw = torch.tensor(
            gap_init, dtype=torch.float32, device=self.device, requires_grad=True
        )

        self.params: list[Any] = []
        if mech.tau_enabled:
            self.params.append(self.tau_logit)
        if mech.sign_flip_fraction > 0.0:
            self.params.append(self.sign_mult)
        if mech.neuromod_enabled:
            self.params.extend([self.nm_log_gain, self.nm_bias])
        if mech.gap_junctions_enabled and self.gap_u_t.numel() > 0:
            self.params.append(self.gap_raw)
        if not self.params:
            self.params.append(self.tau_logit)

        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr)
        self.epoch = 0
        self.last_loss = float("nan")
        self.last_ho_pred_np: npt.NDArray[np.float32] | None = None

    def _effective_tensors(self) -> tuple[Any, Any, Any, Any, Any]:
        import torch
        import torch.nn.functional as F

        tau = (
            _tau_from_logit(self.tau_logit)
            if self.mech.tau_enabled
            else torch.full_like(self.tau_logit, 20.0)
        )
        sign = (
            torch.tanh(self.sign_mult) * 1.5
            if self.mech.sign_flip_fraction > 0.0
            else torch.ones_like(self.sign_mult)
        )
        gain = (
            torch.exp(torch.clamp(self.nm_log_gain, -1.5, 1.5))
            if self.mech.neuromod_enabled
            else torch.ones_like(self.nm_log_gain)
        )
        bias = (
            torch.clamp(self.nm_bias, -1.5, 1.5)
            if self.mech.neuromod_enabled
            else torch.zeros_like(self.nm_bias)
        )
        gap = (
            F.softplus(self.gap_raw)
            if (self.mech.gap_junctions_enabled and self.gap_u_t.numel() > 0)
            else torch.zeros_like(self.gap_raw)
        )
        return tau, sign, gain, bias, gap

    def step_epoch(self) -> float:
        """Run one gradient descent epoch on the noisy partial observation."""
        import torch

        # Cosine LR decay from cfg.lr down to 0.15 * cfg.lr.
        prog = min(1.0, self.epoch / max(1, self.cfg.epochs))
        lr = self.cfg.lr * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        tau, sign, gain, bias, gap = self._effective_tensors()
        pred_full = _simulate_torch(
            w_chem=self.w_stu,
            type_ids_t=self.type_ids_t,
            class_ids_t=self.class_ids_t,
            gap_u_t=self.gap_u_t,
            gap_v_t=self.gap_v_t,
            tau_type=tau,
            sign_mult_type=sign,
            nm_gain_class=gain,
            nm_bias_class=bias,
            gap_weight=gap,
            stim_t=self.stim_tr_t,
            dt_ms=self.cfg.dt_ms,
        )
        pred_binned = _bin_trajectory_torch(pred_full, self.bin_steps)[:, self.obs_idx_t, :]
        mse = torch.mean((pred_binned - self.y_obs_t) ** 2)
        reg = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if self.mech.gap_junctions_enabled and gap.numel() > 0:
            reg = reg + self.cfg.l1_gap * torch.mean(torch.abs(gap))
        if self.cfg.l2_reg > 0.0:
            reg = reg + self.cfg.l2_reg * (
                torch.mean((sign - 1.0) ** 2) + torch.mean(self.nm_log_gain**2)
            )
        loss = mse + reg
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=5.0)
        self.optimizer.step()
        self.epoch += 1
        self.last_loss = float(mse.detach().cpu().item())
        return self.last_loss

    def export_param_vectors(self) -> dict[str, npt.NDArray[np.float32]]:
        with self._no_grad():
            tau, sign, gain, bias, gap = self._effective_tensors()
            return {
                "tau_type_ms": tau.detach().cpu().numpy().astype(np.float32),
                "sign_mult_type": sign.detach().cpu().numpy().astype(np.float32),
                "neuromod_gain_class": gain.detach().cpu().numpy().astype(np.float32),
                "neuromod_bias_class": bias.detach().cpu().numpy().astype(np.float32),
                "gap_weight": gap.detach().cpu().numpy().astype(np.float32),
            }

    def export_raw_state(self) -> dict[str, npt.NDArray[np.float32]]:
        return {
            "tau_logit": self.tau_logit.detach().cpu().numpy().astype(np.float32),
            "sign_mult": self.sign_mult.detach().cpu().numpy().astype(np.float32),
            "nm_log_gain": self.nm_log_gain.detach().cpu().numpy().astype(np.float32),
            "nm_bias": self.nm_bias.detach().cpu().numpy().astype(np.float32),
            "gap_raw": self.gap_raw.detach().cpu().numpy().astype(np.float32),
        }

    def load_raw_state(self, arrays: dict[str, npt.NDArray[np.float32]], epoch: int) -> None:
        import torch

        with torch.no_grad():
            self.tau_logit.copy_(torch.as_tensor(arrays["tau_logit"], device=self.device))
            self.sign_mult.copy_(torch.as_tensor(arrays["sign_mult"], device=self.device))
            self.nm_log_gain.copy_(torch.as_tensor(arrays["nm_log_gain"], device=self.device))
            self.nm_bias.copy_(torch.as_tensor(arrays["nm_bias"], device=self.device))
            self.gap_raw.copy_(torch.as_tensor(arrays["gap_raw"], device=self.device))
        self.epoch = int(epoch)

    @staticmethod
    def _no_grad() -> Any:
        import torch

        return torch.no_grad()

    def evaluate(self) -> dict[str, float]:
        """Evaluate observed fit, unseen-stimulus generalization and hidden-parameter recovery."""
        with self._no_grad():
            tau, sign, gain, bias, gap = self._effective_tensors()
            tr_pred = _simulate_torch(
                w_chem=self.w_stu,
                type_ids_t=self.type_ids_t,
                class_ids_t=self.class_ids_t,
                gap_u_t=self.gap_u_t,
                gap_v_t=self.gap_v_t,
                tau_type=tau,
                sign_mult_type=sign,
                nm_gain_class=gain,
                nm_bias_class=bias,
                gap_weight=gap,
                stim_t=self.stim_tr_t,
                dt_ms=self.cfg.dt_ms,
            )
            ho_pred = _simulate_torch(
                w_chem=self.w_stu,
                type_ids_t=self.type_ids_t,
                class_ids_t=self.class_ids_t,
                gap_u_t=self.gap_u_t,
                gap_v_t=self.gap_v_t,
                tau_type=tau,
                sign_mult_type=sign,
                nm_gain_class=gain,
                nm_bias_class=bias,
                gap_weight=gap,
                stim_t=self.stim_ho_t,
                dt_ms=self.cfg.dt_ms,
            )
            tr_bin = (
                _bin_trajectory_torch(tr_pred, self.bin_steps)[:, self.obs_idx_t, :]
                .detach()
                .cpu()
                .numpy()
            )
            ho_np = ho_pred.detach().cpu().numpy().astype(np.float32)
            self.last_ho_pred_np = ho_np

        tr_mse = float(np.mean((tr_bin - self.y_obs_np) ** 2))
        tr_r2 = _r2_score(self.y_obs_np.astype(np.float64), tr_bin.astype(np.float64))
        ho_all_r2 = _r2_score(self.ho_true_np.astype(np.float64), ho_np.astype(np.float64))
        ho_all_mse = float(np.mean((self.ho_true_np - ho_np) ** 2))
        ho_obs_r2 = _r2_score(
            self.ho_true_np[:, self.obs_idx, :].astype(np.float64),
            ho_np[:, self.obs_idx, :].astype(np.float64),
        )
        if self.unobs_idx.size > 0:
            ho_unobs_r2 = _r2_score(
                self.ho_true_np[:, self.unobs_idx, :].astype(np.float64),
                ho_np[:, self.unobs_idx, :].astype(np.float64),
            )
        else:
            ho_unobs_r2 = ho_obs_r2

        vecs = self.export_param_vectors()
        act = self.truth.active_type_mask
        # Tau recovery on all types (or active types).
        tau_r = _pearson_r(
            self.truth.tau_type_ms.astype(np.float64),
            vecs["tau_type_ms"].astype(np.float64),
        )
        tau_mae = float(np.mean(np.abs(self.truth.tau_type_ms - vecs["tau_type_ms"])))

        # Sign-flip detection AUROC: lower learned sign_mult predicts a hidden sign flip!
        if np.any(act) and np.any(self.truth.flipped_type_mask[act]):
            sign_auroc = _auroc(
                self.truth.flipped_type_mask[act].astype(np.float64),
                -vecs["sign_mult_type"][act].astype(np.float64),
            )
            pred_flip = vecs["sign_mult_type"][act] < 0.0
            sign_acc = float(np.mean(pred_flip == self.truth.flipped_type_mask[act]))
        else:
            sign_auroc = float("nan")
            sign_acc = float("nan")

        # Neuromodulation recovery (average of separate gain and bias correlations to avoid
        # artificial correlation from the ~1.0 gain vs ~0.0 bias mean offset).
        g_r = _pearson_r(
            self.truth.neuromod_gain_class.astype(np.float64),
            vecs["neuromod_gain_class"].astype(np.float64),
        )
        b_r = _pearson_r(
            self.truth.neuromod_bias_class.astype(np.float64),
            vecs["neuromod_bias_class"].astype(np.float64),
        )
        valid_nm = [v for v in (g_r, b_r) if not math.isnan(v)]
        nm_r = float(sum(valid_nm) / len(valid_nm)) if valid_nm else float("nan")
        nm_mae = 0.5 * (
            float(np.mean(np.abs(self.truth.neuromod_gain_class - vecs["neuromod_gain_class"])))
            + float(np.mean(np.abs(self.truth.neuromod_bias_class - vecs["neuromod_bias_class"])))
        )

        # Gap-junction discovery AUROC & Pearson r.
        if self.truth.gap_weight.size > 0 and np.any(self.truth.gap_weight > 0.0):
            gap_auroc = _auroc(
                (self.truth.gap_weight > 0.0).astype(np.float64),
                vecs["gap_weight"].astype(np.float64),
            )
            gap_r = _pearson_r(
                self.truth.gap_weight.astype(np.float64),
                vecs["gap_weight"].astype(np.float64),
            )
        else:
            gap_auroc = float("nan")
            gap_r = float("nan")

        return {
            "loss": tr_mse,
            "train_obs_r2": tr_r2,
            "heldout_all_r2": ho_all_r2,
            "heldout_all_mse": ho_all_mse,
            "heldout_obs_r2": ho_obs_r2,
            "heldout_unobs_r2": ho_unobs_r2,
            "tau_pearson_r": tau_r,
            "tau_mae_ms": tau_mae,
            "sign_flip_auroc": sign_auroc,
            "sign_flip_accuracy": sign_acc,
            "neuromod_pearson_r": nm_r,
            "neuromod_mae": nm_mae,
            "gap_auroc": gap_auroc,
            "gap_pearson_r": gap_r,
        }


def _clean_float(val: float) -> float | None:
    return None if (math.isnan(val) or math.isinf(val)) else round(val, 4)


def _clean_metrics(m: dict[str, float]) -> dict[str, float | None]:
    return {k: _clean_float(v) for k, v in m.items()}


def run_hidden_truth_experiment(
    sub_ref: Subcircuit,
    *,
    sub_shuffled: Subcircuit | None = None,
    mech: HiddenMechanismSpec | None = None,
    obs: ObservationSpec | None = None,
    cfg: SysIDTrainConfig | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    """Run the full F5 hidden-truth system identification benchmark + identifiability check.

    Compares:
    1. ``untrained`` baseline (default uniform model with no hidden mechanisms).
    2. ``reference`` wiring trained from ``n_restarts`` independent initializations.
    3. ``shuffled_control`` wiring trained under the exact same observation and budget.
    """
    import torch

    m_spec = mech or HiddenMechanismSpec()
    o_spec = obs or ObservationSpec()
    t_cfg = cfg or SysIDTrainConfig()
    dev = _resolve_device(t_cfg.device)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    # 1. Untrained baseline (0 epochs, restart 0).
    base_sess = SysIDSession(sub_ref, sub_ref, m_spec, o_spec, t_cfg, seed=seed, restart_index=0)
    untrained_metrics = base_sess.evaluate()

    # 2. Multi-restart fits on biological reference wiring (tests identifiability!).
    restart_metrics: list[dict[str, float]] = []
    restart_vecs: list[dict[str, npt.NDArray[np.float32]]] = []
    for r_idx in range(max(1, t_cfg.n_restarts)):
        sess = (
            base_sess
            if r_idx == 0
            else SysIDSession(
                sub_ref, sub_ref, m_spec, o_spec, t_cfg, seed=seed, restart_index=r_idx
            )
        )
        for _ in range(t_cfg.epochs):
            sess.step_epoch()
        restart_metrics.append(sess.evaluate())
        restart_vecs.append(sess.export_param_vectors())

    # 3. Shuffled-wiring negative control (optional, default run when sub_shuffled provided).
    shuffled_metrics: dict[str, float] | None = None
    if sub_shuffled is not None:
        shuf_sess = SysIDSession(
            sub_ref, sub_shuffled, m_spec, o_spec, t_cfg, seed=seed, restart_index=0
        )
        for _ in range(t_cfg.epochs):
            shuf_sess.step_epoch()
        shuffled_metrics = shuf_sess.evaluate()

    if dev == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    else:
        peak_vram_mb = 0.0
    wall_s = time.perf_counter() - t0

    # Best restart by training observed loss (cannot peek at holdout!).
    best_i = int(np.argmin([m["loss"] for m in restart_metrics]))
    best_ref = restart_metrics[best_i]

    # Non-uniqueness / identifiability analysis across restarts on the SAME recording.
    if len(restart_vecs) >= 2:
        tau_stack = np.stack([v["tau_type_ms"] for v in restart_vecs], axis=0)
        sign_stack = np.stack([v["sign_mult_type"] for v in restart_vecs], axis=0)
        gap_stack = np.stack([v["gap_weight"] for v in restart_vecs], axis=0)
        fit_r2s = [m["train_obs_r2"] for m in restart_metrics]
        unobs_r2s = [m["heldout_unobs_r2"] for m in restart_metrics]
        tau_dis = float(np.mean(np.std(tau_stack, axis=0)))
        sign_dis = float(np.mean(np.std(sign_stack, axis=0)))
        gap_dis = float(np.mean(np.std(gap_stack, axis=0))) if gap_stack.shape[1] > 0 else 0.0
        fit_spread = float(max(fit_r2s) - min(fit_r2s))
        unobs_spread = float(max(unobs_r2s) - min(unobs_r2s))
        # Sloppy / non-unique if restarts achieve nearly identical observed fit yet disagree
        # substantially on latent parameters or unobserved predictions.
        if fit_spread < 0.03 and (tau_dis > 4.0 or sign_dis > 0.25 or unobs_spread > 0.05):
            verdict = "non_unique_sloppy"
            note = (
                "Multiple distinct parameter sets achieved nearly identical fit on the limited "
                "recording while disagreeing on latent parameters/unobserved cells."
            )
        elif tau_dis < 2.0 and sign_dis < 0.12 and unobs_spread < 0.03:
            verdict = "well_identified"
            note = (
                "Independent restarts converged to closely agreeing latent parameters and "
                "unseen-stimulus predictions."
            )
        else:
            verdict = "partially_identifiable"
            note = (
                "Observed fit is consistent across restarts, with moderate parameter spread on "
                "weakly driven or unobserved cell types."
            )
    else:
        tau_dis = sign_dis = gap_dis = fit_spread = unobs_spread = 0.0
        verdict = "single_restart"
        note = "Only one restart was run; run >= 2 restarts to assess solution non-uniqueness."

    num_params = int(sum(p.numel() for p in base_sess.params))
    return {
        "dataset_id": sub_ref.dataset_id,
        "subcircuit_id": sub_ref.subcircuit_id,
        "pack_digest": sub_ref.pack_digest,
        "seed": int(seed),
        "device": dev,
        "wall_seconds": round(wall_s, 3),
        "peak_vram_mb": peak_vram_mb,
        "circuit": {
            "num_cells": sub_ref.num_cells,
            "num_edges": sub_ref.num_edges,
            "num_types": sub_ref.num_types,
            "num_classes": sub_ref.num_classes,
            "observed_cells": int(base_sess.obs_idx.size),
            "unobserved_cells": int(base_sess.unobs_idx.size),
            "observed_fraction_actual": round(
                base_sess.obs_idx.size / max(1, sub_ref.num_cells), 4
            ),
            "num_learnable_params": num_params,
            "gap_candidates": int(base_sess.truth.gap_u.size),
            "gap_true_active": int((base_sess.truth.gap_weight > 0).sum()),
            "sign_true_flipped": int(base_sess.truth.flipped_type_mask.sum()),
        },
        "observation": {
            "obs_fraction": o_spec.obs_fraction,
            "obs_classes": list(o_spec.obs_classes),
            "bin_ms": o_spec.bin_ms,
            "noise_sd": o_spec.noise_sd,
        },
        "framing_note": (
            "Differentiable firing-rate connectome system identification (external optimization "
            "layer). Recovery of synthetic hidden mechanisms does not imply biological truth."
        ),
        "untrained": _clean_metrics(untrained_metrics),
        "reference_best": _clean_metrics(best_ref),
        "reference_restarts": [_clean_metrics(m) for m in restart_metrics],
        "shuffled_control": _clean_metrics(shuffled_metrics) if shuffled_metrics else None,
        "identifiability": {
            "n_restarts": len(restart_vecs),
            "fit_r2_spread": round(fit_spread, 4),
            "heldout_unobs_r2_spread": round(unobs_spread, 4),
            "tau_disagreement_ms": round(tau_dis, 3),
            "sign_disagreement": round(sign_dis, 4),
            "gap_disagreement": round(gap_dis, 4),
            "verdict": verdict,
            "note": note,
        },
    }


def run_observability_study(
    sub_ref: Subcircuit,
    *,
    fractions: tuple[float, ...] = (0.05, 0.20, 0.50, 1.0),
    class_targets: tuple[str, ...] = ("ALPN", "ALLN", "Kenyon_Cell", "MBON", "LHLN"),
    mech: HiddenMechanismSpec | None = None,
    bin_ms: float = 10.0,
    noise_sd: float = 0.015,
    cfg: SysIDTrainConfig | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    """Produce F5's two diagnostic maps:

    1. ``fraction_curve``: how much recording coverage is needed to recover hidden mechanisms.
    2. ``cell_class_map``: which cell classes are most informative to record.
    """
    m_spec = mech or HiddenMechanismSpec()
    base_cfg = cfg or SysIDTrainConfig(n_restarts=2)
    t0 = time.perf_counter()

    fraction_rows: list[dict[str, Any]] = []
    for frac in fractions:
        rep = run_hidden_truth_experiment(
            sub_ref,
            sub_shuffled=None,
            mech=m_spec,
            obs=ObservationSpec(obs_fraction=frac, bin_ms=bin_ms, noise_sd=noise_sd),
            cfg=base_cfg,
            seed=seed,
        )
        fraction_rows.append(
            {
                "obs_fraction": frac,
                "observed_cells": rep["circuit"]["observed_cells"],
                "untrained_heldout_all_r2": rep["untrained"]["heldout_all_r2"],
                "reference_best": rep["reference_best"],
                "identifiability": rep["identifiability"],
            }
        )

    avail_classes = set(sub_ref.class_names)
    class_rows: list[dict[str, Any]] = []
    for cls in class_targets:
        if cls not in avail_classes:
            continue
        rep = run_hidden_truth_experiment(
            sub_ref,
            sub_shuffled=None,
            mech=m_spec,
            obs=ObservationSpec(
                obs_fraction=1.0, obs_classes=(cls,), bin_ms=bin_ms, noise_sd=noise_sd
            ),
            cfg=base_cfg,
            seed=seed,
        )
        class_rows.append(
            {
                "obs_class": cls,
                "observed_cells": rep["circuit"]["observed_cells"],
                "observed_fraction_of_circuit": rep["circuit"]["observed_fraction_actual"],
                "reference_best": rep["reference_best"],
                "identifiability": rep["identifiability"],
            }
        )

    class_rows.sort(
        key=lambda r: (
            -999.0
            if r["reference_best"]["heldout_unobs_r2"] is None
            else float(r["reference_best"]["heldout_unobs_r2"])
        ),
        reverse=True,
    )
    return {
        "dataset_id": sub_ref.dataset_id,
        "subcircuit_id": sub_ref.subcircuit_id,
        "seed": int(seed),
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "fraction_curve": fraction_rows,
        "cell_class_map": class_rows,
    }
