"""Phase F6: Fixed-topology connectome task learning (biological wiring vs shuffled wiring).

External learning layer (NEVER modifies the core ``connectome`` substrate).
PyTorch is imported lazily inside functions so the FastAPI API process stays GPU-free.

Mandatory scientific framing (docs/plans/connectome_plan.md §4 F6):
We explicitly refer to these models as **learning machines borrowing the fly connectome
topology**, NEVER as "a fly learning". Every benchmark binds a pre-registered hypothesis and
decision rule (SHA-256 hashed before execution) and compares biological ``reference`` wiring
against degree- and weight-preserving ``shuffled_control`` wiring across multiple independent
seeds on unseen holdout test trials.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from alife_lab.analysis.stats import compare, describe
from alife_lab.connectome.subcircuit import Subcircuit
from alife_lab.connectome.sysid import _base_weight_scale, _make_sparse_chem_matrix, _resolve_device
from alife_lab.core.hashing import sha256_json

TaskName = Literal["odor_classification", "short_memory", "nonlinear_xor"]
TrainScope = Literal["readout_only", "io_only", "partial_weights"]

F6_FRAMING_NOTE = (
    "Learning machine borrowing the fly connectome topology (fixed graph structure); "
    "NOT biological learning by a fly."
)

_OUTPUT_CLASSES = frozenset({"MBON", "LHLN", "LHCENT", "MBIN"})
_PLASTIC_PRESYN_CLASSES = frozenset({"Kenyon_Cell", "ALPN"})


@dataclass(frozen=True)
class F6TaskConfig:
    """Task, training scope, and optimization hyperparameters."""

    task: TaskName = "odor_classification"
    scope: TrainScope = "readout_only"
    dt_ms: float = 2.0
    n_train: int = 64
    n_test: int = 64
    epochs: int = 80
    lr: float = 0.08
    weight_decay: float = 1e-3
    max_readout: int = 256
    device: str = "auto"


@dataclass(frozen=True)
class F6Preregistration:
    """Pre-registered hypothesis and decision rule hashed before any training runs."""

    hypothesis: str
    task: TaskName
    scope: TrainScope
    dataset_id: str
    subcircuit_id: str
    primary_metric: str = "test_acc"
    control_description: str = (
        "Degree- and weight-preserving shuffled_control wiring across independent seeds."
    )
    alpha: float = 0.05
    min_delta_acc: float = 0.05
    min_cliffs_delta: float = 0.50

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "hypothesis": self.hypothesis,
            "task": self.task,
            "scope": self.scope,
            "dataset_id": self.dataset_id,
            "subcircuit_id": self.subcircuit_id,
            "primary_metric": self.primary_metric,
            "control_description": self.control_description,
            "alpha": self.alpha,
            "min_delta_acc": self.min_delta_acc,
            "min_cliffs_delta": self.min_cliffs_delta,
        }
        return {**payload, "plan_digest": sha256_json(payload)}


def make_preregistration(
    dataset_id: str,
    subcircuit_id: str,
    task: TaskName,
    scope: TrainScope,
) -> F6Preregistration:
    return F6Preregistration(
        hypothesis=(
            f"A learning machine borrowing biological {dataset_id} ({subcircuit_id}) wiring "
            f"achieves higher unseen test accuracy on {task} ({scope}) than degree- and "
            "weight-matched shuffled wiring by at least +0.05 with Cliff's delta >= 0.50 "
            "and exact permutation p < 0.05."
        ),
        task=task,
        scope=scope,
        dataset_id=dataset_id,
        subcircuit_id=subcircuit_id,
    )


def evaluate_preregistration(
    plan: F6Preregistration,
    ref_scores: list[float],
    shuf_scores: list[float],
) -> dict[str, Any]:
    """Apply pre-declared decision rule without post-hoc adjustment."""
    cmp = compare(ref_scores, shuf_scores, seed=601)
    diff = float(cmp["mean_difference"])
    delta = float(cmp["cliffs_delta"])
    pval = float(cmp["permutation"]["p_value"])
    if diff >= plan.min_delta_acc and delta >= plan.min_cliffs_delta and pval < plan.alpha:
        verdict = "validated"
        reason = (
            f"Biological wiring beat shuffled control by {diff:+.3f} "
            f"(Cliff's delta={delta:+.2f}, exact p={pval:.4f})."
        )
    elif diff <= -plan.min_delta_acc and delta <= -plan.min_cliffs_delta and pval < plan.alpha:
        verdict = "rejected"
        reason = (
            f"Shuffled wiring beat biological wiring by {-diff:+.3f} "
            f"(Cliff's delta={delta:+.2f}, exact p={pval:.4f})."
        )
    else:
        verdict = "inconclusive"
        reason = (
            f"Difference {diff:+.3f} (Cliff's delta={delta:+.2f}, exact p={pval:.4f}) "
            "did not meet pre-registered thresholds."
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "comparison": cmp,
        "reference_stats": describe(ref_scores),
        "shuffled_stats": describe(shuf_scores),
    }


def select_readout_indices(sub: Subcircuit, max_readout: int = 256) -> npt.NDArray[np.int32]:
    """Deterministically select anatomical downstream output neurons (identical across wirings)."""
    primary = [i for i in range(sub.num_cells) if sub.cell_classes[i] in _OUTPUT_CLASSES]
    if len(primary) < 4:
        primary = [
            i
            for i in range(sub.num_cells)
            if sub.cell_classes[i] in _OUTPUT_CLASSES or sub.cell_classes[i] == "Kenyon_Cell"
        ]
    if not primary:
        primary = list(range(sub.num_cells))
    arr = np.asarray(primary, dtype=np.int32)
    if arr.size <= max_readout:
        return arr
    # Deterministic evenly spaced subsample across anatomical output neurons.
    pick = np.linspace(0, arr.size - 1, max_readout).astype(np.int32)
    return arr[pick]


def select_partial_weight_type_mask(sub: Subcircuit) -> npt.NDArray[np.bool_]:
    """Select presynaptic cell types scaled under ``partial_weights``."""
    mask = np.zeros(sub.num_types, dtype=np.bool_)
    for i in range(sub.num_cells):
        if sub.cell_classes[i] in _PLASTIC_PRESYN_CLASSES:
            mask[int(sub.type_ids[i])] = True
    if not np.any(mask):
        mask[:] = True
    return mask


@dataclass(frozen=True)
class TaskDataset:
    """Deterministic group-level stimulus waveforms and labels (train & unseen holdout test)."""

    task: TaskName
    num_classes: int
    steps: int
    read_start: int
    read_end: int
    # Group-level drive (T, G, B) where G = len(sub.input_groups)
    x_train_groups: npt.NDArray[np.float32]
    y_train: npt.NDArray[np.int64]
    x_test_groups: npt.NDArray[np.float32]
    y_test: npt.NDArray[np.int64]

    @property
    def chance_acc(self) -> float:
        return 1.0 / max(1, self.num_classes)


def build_task_dataset(
    sub: Subcircuit,
    cfg: F6TaskConfig,
    *,
    seed: int,
) -> TaskDataset:
    """Generate deterministic train and unseen holdout trials for one of the three F6 tasks."""
    rng = np.random.default_rng(seed * 7919 + 17)
    n_groups = max(1, len(sub.input_groups))
    total_b = cfg.n_train + cfg.n_test

    if cfg.task == "odor_classification":
        num_classes = 4
        steps = max(12, round(50.0 / cfg.dt_ms))
        read_start = max(1, int(steps * 0.68))
        read_end = steps
        # 4 distinct multi-glomerular odor prototypes with overlapping components.
        protos = np.zeros((num_classes, n_groups), dtype=np.float32)
        for c in range(num_classes):
            if n_groups >= 4:
                base = (c * max(1, n_groups // 4)) % n_groups
                protos[c, base] = 2.4
                protos[c, (base + 1) % n_groups] = 2.0
                if n_groups >= 8:
                    protos[c, (base + 5) % n_groups] = 1.4
            else:
                protos[c, c % n_groups] = 2.4

        y = np.arange(total_b, dtype=np.int64) % num_classes
        rng.shuffle(y[: cfg.n_train])
        rng.shuffle(y[cfg.n_train :])

        x = np.zeros((steps, n_groups, total_b), dtype=np.float32)
        t_on = max(1, steps // 10)
        t_off = max(t_on + 2, int(steps * 0.85))
        for b in range(total_b):
            proto = protos[int(y[b])].copy()
            # 25% glomerular dropout + trial-specific Gaussian jitter
            drop = rng.random(n_groups) < 0.25
            if not np.all(drop):
                proto[drop] = 0.0
            noise = rng.normal(0.0, 0.55, size=(t_off - t_on, n_groups)).astype(np.float32)
            x[t_on:t_off, :, b] = np.maximum(0.0, proto[None, :] + noise)

    elif cfg.task == "short_memory":
        num_classes = 4
        steps = max(24, round(80.0 / cfg.dt_ms))
        cue_on = max(1, round(4.0 / cfg.dt_ms))
        cue_off = max(cue_on + 2, round(20.0 / cfg.dt_ms))
        read_start = max(cue_off + 4, round(68.0 / cfg.dt_ms))
        read_end = steps

        protos = np.zeros((num_classes, n_groups), dtype=np.float32)
        for c in range(num_classes):
            base = (c * max(1, n_groups // 4)) % n_groups
            protos[c, base] = 2.6
            if n_groups >= 4:
                protos[c, (base + 1) % n_groups] = 2.2

        y = np.arange(total_b, dtype=np.int64) % num_classes
        rng.shuffle(y[: cfg.n_train])
        rng.shuffle(y[cfg.n_train :])

        x = np.zeros((steps, n_groups, total_b), dtype=np.float32)
        for b in range(total_b):
            proto = protos[int(y[b])]
            cue_noise = rng.normal(0.0, 0.35, size=(cue_off - cue_on, n_groups)).astype(np.float32)
            x[cue_on:cue_off, :, b] = np.maximum(0.0, proto[None, :] + cue_noise)
            # Distractor noise during the 48 ms silent delay window so passive noiseless decay
            # alone cannot trivially solve the task.
            delay_noise = rng.normal(0.0, 0.35, size=(steps - cue_off, n_groups)).astype(np.float32)
            x[cue_off:, :, b] = np.maximum(0.0, delay_noise)

    elif cfg.task == "nonlinear_xor":
        num_classes = 2
        steps = max(12, round(50.0 / cfg.dt_ms))
        read_start = max(1, int(steps * 0.68))
        read_end = steps
        half = max(1, n_groups // 2)
        grp_a = np.arange(0, half, dtype=np.int32)
        grp_b = np.arange(half, n_groups, dtype=np.int32)
        if grp_b.size == 0:
            grp_b = grp_a

        # Balanced 4 input combinations (0,0), (0,1), (1,0), (1,1) -> label = a ^ b.
        combos = np.arange(total_b, dtype=np.int64) % 4
        rng.shuffle(combos[: cfg.n_train])
        rng.shuffle(combos[cfg.n_train :])
        a_bit = (combos >> 1) & 1
        b_bit = combos & 1
        y = (a_bit ^ b_bit).astype(np.int64)

        x = np.zeros((steps, n_groups, total_b), dtype=np.float32)
        t_on = max(1, steps // 10)
        t_off = max(t_on + 2, int(steps * 0.85))
        for b in range(total_b):
            vec = np.zeros(n_groups, dtype=np.float32)
            if a_bit[b]:
                vec[grp_a] = 2.2
            if b_bit[b]:
                vec[grp_b] = 2.2
            noise = rng.normal(0.0, 0.35, size=(t_off - t_on, n_groups)).astype(np.float32)
            x[t_on:t_off, :, b] = np.maximum(0.0, vec[None, :] + noise)
    else:
        raise ValueError(f"unsupported F6 task: {cfg.task!r}")

    return TaskDataset(
        task=cfg.task,
        num_classes=num_classes,
        steps=steps,
        read_start=read_start,
        read_end=read_end,
        x_train_groups=x[:, :, : cfg.n_train].copy(),
        y_train=y[: cfg.n_train].copy(),
        x_test_groups=x[:, :, cfg.n_train :].copy(),
        y_test=y[cfg.n_train :].copy(),
    )


def _group_to_cell_matrix(sub: Subcircuit, device: str) -> Any:
    """Fixed sparse anatomical projection from glomerular groups (G) -> sensory cells (N)."""
    import torch

    n_groups = max(1, len(sub.input_groups))
    rows: list[int] = []
    cols: list[int] = []
    for g, (_, cell_arr) in enumerate(sub.input_groups):
        for c in cell_arr.tolist():
            rows.append(int(c))
            cols.append(g)
    if not rows:
        idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        vals = torch.zeros(0, dtype=torch.float32, device=device)
    else:
        idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
        vals = torch.ones(len(rows), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        idx, vals, size=(sub.num_cells, n_groups), check_invariants=False
    ).coalesce()


def _rollout_readout_features(
    *,
    w_chem: Any,
    group_map: Any,
    type_ids_t: Any,
    readout_idx_t: Any,
    x_groups_t: Any,
    w_in: Any | None,
    type_scale: Any | None,
    dt_ms: float,
    read_start: int,
    read_end: int,
    tau_ms: float = 20.0,
    bias_theta: float = 1.2,
) -> tuple[Any, Any]:
    """Roll out fixed-topology rate connectome and return readout features ``(B, R)`` and
    readout trajectory ``(T, R)`` for trial 0 (for diagnostic rendering).
    """
    import torch

    steps, _n_groups, batch = x_groups_t.shape
    n_cells = int(type_ids_t.shape[0])
    alpha = float(dt_ms / tau_ms)
    r = torch.full((n_cells, batch), 0.05, dtype=torch.float32, device=x_groups_t.device)
    m_cell = type_scale[type_ids_t].unsqueeze(1) if type_scale is not None else None

    # Pre-project input groups (T, G, B) -> optionally mixed by w_in (G, G).
    xg = torch.einsum("gh,thb->tgb", w_in, x_groups_t) if w_in is not None else x_groups_t

    read_acc = torch.zeros(
        (int(readout_idx_t.shape[0]), batch), dtype=torch.float32, device=x_groups_t.device
    )
    read_count = 0
    tr0_read: list[Any] = []
    for t in range(steps):
        inp = torch.sparse.mm(group_map, xg[t])  # (N, B)
        r_pre = r * m_cell if m_cell is not None else r
        syn = torch.sparse.mm(w_chem, r_pre)
        r = r + alpha * (torch.sigmoid(syn + inp - bias_theta) - r)
        ro = r[readout_idx_t]
        tr0_read.append(ro[:, 0])
        if read_start <= t < read_end:
            read_acc = read_acc + ro
            read_count += 1

    feat = (read_acc / max(1, read_count)).transpose(0, 1)  # (B, R)
    tr0_mat = torch.stack(tr0_read, dim=0)  # (T, R)
    return feat, tr0_mat


class TaskTrainSession:
    """Stateful epoch-by-epoch F6 task training session (CLI benchmark & browser adapter)."""

    def __init__(
        self,
        sub: Subcircuit,
        cfg: F6TaskConfig,
        *,
        seed: int,
        data_seed: int | None = None,
    ) -> None:
        import torch

        self.sub = sub
        self.cfg = cfg
        self.seed = int(seed)
        self.data_seed = int(seed if data_seed is None else data_seed)
        self.device = _resolve_device(cfg.device)
        self.dataset = build_task_dataset(sub, cfg, seed=self.data_seed)
        self.readout_idx = select_readout_indices(sub, cfg.max_readout)
        self.partial_type_mask = select_partial_weight_type_mask(sub)

        self.w_chem = _make_sparse_chem_matrix(sub, self.device)
        self.group_map = _group_to_cell_matrix(sub, self.device)
        self.type_ids_t = torch.as_tensor(sub.type_ids, dtype=torch.long, device=self.device)
        self.readout_idx_t = torch.as_tensor(self.readout_idx, dtype=torch.long, device=self.device)
        self.partial_mask_t = torch.as_tensor(
            self.partial_type_mask, dtype=torch.bool, device=self.device
        )

        self.x_tr_t = torch.as_tensor(
            self.dataset.x_train_groups, dtype=torch.float32, device=self.device
        )
        self.y_tr_t = torch.as_tensor(self.dataset.y_train, dtype=torch.long, device=self.device)
        self.x_te_t = torch.as_tensor(
            self.dataset.x_test_groups, dtype=torch.float32, device=self.device
        )
        self.y_te_t = torch.as_tensor(self.dataset.y_test, dtype=torch.long, device=self.device)

        n_r = int(self.readout_idx.shape[0])
        n_c = self.dataset.num_classes
        n_g = max(1, len(sub.input_groups))

        rng_init = np.random.default_rng(self.seed * 131 + 29)
        w_out_init = rng_init.normal(0.0, 0.05 / math.sqrt(max(1, n_r)), size=(n_c, n_r)).astype(
            np.float32
        )
        self.w_out = torch.tensor(
            w_out_init, dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.b_out = torch.zeros(n_c, dtype=torch.float32, device=self.device, requires_grad=True)

        # Optional scope parameters:
        # - io_only: learnable input projection initialized as identity (G, G)
        # - partial_weights: learnable per-type positive multiplier on KC/ALPN types (init ~1.0)
        self.w_in = torch.eye(n_g, dtype=torch.float32, device=self.device, requires_grad=True)
        # 0.1 + 2.9 * sigmoid(logit) = 1.0 -> sigmoid = 0.9/2.9 -> logit ~= -0.800
        init_logit = math.log((0.9 / 2.9) / (1.0 - 0.9 / 2.9))
        self.type_scale_logit = torch.full(
            (sub.num_types,),
            init_logit,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        self.params: list[Any] = [self.w_out, self.b_out]
        if cfg.scope == "io_only":
            self.params.append(self.w_in)
        elif cfg.scope == "partial_weights":
            self.params.append(self.type_scale_logit)

        # Fast-path: for readout_only the reservoir is 100% frozen, so precompute features once!
        self._cached_tr_feat: Any = None
        self._cached_te_feat: Any = None
        self.last_tr0_read_np: npt.NDArray[np.float32] | None = None
        if cfg.scope == "readout_only":
            with torch.no_grad():
                self._cached_tr_feat, tr0 = _rollout_readout_features(
                    w_chem=self.w_chem,
                    group_map=self.group_map,
                    type_ids_t=self.type_ids_t,
                    readout_idx_t=self.readout_idx_t,
                    x_groups_t=self.x_tr_t,
                    w_in=None,
                    type_scale=None,
                    dt_ms=cfg.dt_ms,
                    read_start=self.dataset.read_start,
                    read_end=self.dataset.read_end,
                )
                self._cached_te_feat, _ = _rollout_readout_features(
                    w_chem=self.w_chem,
                    group_map=self.group_map,
                    type_ids_t=self.type_ids_t,
                    readout_idx_t=self.readout_idx_t,
                    x_groups_t=self.x_te_t,
                    w_in=None,
                    type_scale=None,
                    dt_ms=cfg.dt_ms,
                    read_start=self.dataset.read_start,
                    read_end=self.dataset.read_end,
                )
                self.last_tr0_read_np = tr0.detach().cpu().numpy().astype(np.float32)

        self.optimizer = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.epoch = 0
        self.history_train_acc: list[float] = []
        self.history_test_acc: list[float] = []
        self.last_confusion = np.zeros((n_c, n_c), dtype=np.int32)

    def _effective_type_scale(self) -> Any | None:
        import torch

        if self.cfg.scope != "partial_weights":
            return None
        raw = 0.1 + 2.9 * torch.sigmoid(self.type_scale_logit)
        ones = torch.ones_like(raw)
        return torch.where(self.partial_mask_t, raw, ones)

    def _forward_features(self, is_train: bool) -> Any:
        if self.cfg.scope == "readout_only":
            return self._cached_tr_feat if is_train else self._cached_te_feat
        feat, tr0 = _rollout_readout_features(
            w_chem=self.w_chem,
            group_map=self.group_map,
            type_ids_t=self.type_ids_t,
            readout_idx_t=self.readout_idx_t,
            x_groups_t=self.x_tr_t if is_train else self.x_te_t,
            w_in=self.w_in if self.cfg.scope == "io_only" else None,
            type_scale=self._effective_type_scale(),
            dt_ms=self.cfg.dt_ms,
            read_start=self.dataset.read_start,
            read_end=self.dataset.read_end,
        )
        if not is_train:
            self.last_tr0_read_np = tr0.detach().cpu().numpy().astype(np.float32)
        return feat

    def _standardize_pair(self, tr_feat: Any, te_feat: Any) -> tuple[Any, Any]:
        """Leak-free readout standardization using training-set mean & SD only."""
        mu = tr_feat.mean(dim=0, keepdim=True)
        sd = tr_feat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        return (tr_feat - mu) / sd, (te_feat - mu) / sd

    def step_epoch(self) -> float:
        import torch
        import torch.nn.functional as F

        prog = min(1.0, self.epoch / max(1, self.cfg.epochs))
        lr = self.cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        feat = self._forward_features(is_train=True)
        mu = feat.mean(dim=0, keepdim=True)
        sd = feat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        feat_z = (feat - mu) / sd
        logits = feat_z @ self.w_out.transpose(0, 1) + self.b_out
        loss = F.cross_entropy(logits, self.y_tr_t)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=5.0)
        self.optimizer.step()
        self.epoch += 1
        return float(loss.detach().cpu().item())

    def evaluate(self) -> dict[str, float]:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            tr_raw = self._forward_features(is_train=True)
            te_raw = self._forward_features(is_train=False)
            tr_feat, te_feat = self._standardize_pair(tr_raw, te_raw)
            tr_logits = tr_feat @ self.w_out.transpose(0, 1) + self.b_out
            te_logits = te_feat @ self.w_out.transpose(0, 1) + self.b_out
            tr_loss = float(F.cross_entropy(tr_logits, self.y_tr_t).cpu().item())
            te_loss = float(F.cross_entropy(te_logits, self.y_te_t).cpu().item())
            tr_pred = tr_logits.argmax(dim=1)
            te_pred = te_logits.argmax(dim=1)
            tr_acc = float((tr_pred == self.y_tr_t).float().mean().cpu().item())
            te_acc = float((te_pred == self.y_te_t).float().mean().cpu().item())

            # Mean softmax confidence margin on unseen test trials.
            probs = torch.softmax(te_logits, dim=1)
            true_p = probs[torch.arange(probs.shape[0], device=self.device), self.y_te_t]
            margin = float((true_p - self.dataset.chance_acc).mean().cpu().item())
            ro_norm = float(torch.linalg.norm(self.w_out).cpu().item())

            # Confusion matrix (true x pred) on unseen test set.
            n_c = self.dataset.num_classes
            cm = np.zeros((n_c, n_c), dtype=np.int32)
            yt_np = self.dataset.y_test
            yp_np = te_pred.detach().cpu().numpy()
            for t_i, p_i in zip(yt_np.tolist(), yp_np.tolist(), strict=True):
                cm[t_i, p_i] += 1
            self.last_confusion = cm

        self.history_train_acc.append(tr_acc)
        self.history_test_acc.append(te_acc)
        return {
            "train_loss": tr_loss,
            "test_loss": te_loss,
            "train_acc": tr_acc,
            "test_acc": te_acc,
            "chance_acc": self.dataset.chance_acc,
            "test_margin": margin,
            "readout_norm": ro_norm,
        }

    def export_raw_state(self) -> dict[str, npt.NDArray[np.float32]]:
        return {
            "w_out": self.w_out.detach().cpu().numpy().astype(np.float32),
            "b_out": self.b_out.detach().cpu().numpy().astype(np.float32),
            "w_in": self.w_in.detach().cpu().numpy().astype(np.float32),
            "type_scale_logit": self.type_scale_logit.detach().cpu().numpy().astype(np.float32),
        }

    def load_raw_state(self, arrays: dict[str, npt.NDArray[np.float32]], epoch: int) -> None:
        import torch

        with torch.no_grad():
            self.w_out.copy_(torch.as_tensor(arrays["w_out"], device=self.device))
            self.b_out.copy_(torch.as_tensor(arrays["b_out"], device=self.device))
            self.w_in.copy_(torch.as_tensor(arrays["w_in"], device=self.device))
            self.type_scale_logit.copy_(
                torch.as_tensor(arrays["type_scale_logit"], device=self.device)
            )
        self.epoch = int(epoch)


def run_topology_benchmark(
    sub_ref: Subcircuit,
    shuffled_factory: Any,
    cfg: F6TaskConfig,
    *,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    """Run pre-registered F6 benchmark: biological wiring vs multi-seed shuffled wiring.

    ``shuffled_factory(seed)`` returns a degree- and weight-preserving shuffled ``Subcircuit``
    for each seed so shuffled variance includes BOTH wiring-shuffle and trial/init randomness.
    """
    import torch

    plan = make_preregistration(sub_ref.dataset_id, sub_ref.subcircuit_id, cfg.task, cfg.scope)
    dev = _resolve_device(cfg.device)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    ref_runs: list[dict[str, Any]] = []
    shuf_runs: list[dict[str, Any]] = []
    n_params = 0
    n_readout = 0

    for s in seeds:
        # 1. Biological reference wiring on seed s (same task dataset seed = s).
        sess_ref = TaskTrainSession(sub_ref, cfg, seed=int(s), data_seed=int(s))
        n_params = int(sum(p.numel() for p in sess_ref.params))
        n_readout = int(sess_ref.readout_idx.shape[0])
        for _ in range(cfg.epochs):
            sess_ref.step_epoch()
        m_ref = sess_ref.evaluate()
        ref_runs.append({"seed": int(s), **{k: round(v, 4) for k, v in m_ref.items()}})

        # 2. Independent degree-preserving shuffled wiring on exact same dataset seed = s!
        sub_shuf: Subcircuit = shuffled_factory(int(s))
        sess_shuf = TaskTrainSession(sub_shuf, cfg, seed=int(s), data_seed=int(s))
        for _ in range(cfg.epochs):
            sess_shuf.step_epoch()
        m_shuf = sess_shuf.evaluate()
        shuf_runs.append({"seed": int(s), **{k: round(v, 4) for k, v in m_shuf.items()}})

    if dev == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    else:
        peak_vram_mb = 0.0
    wall_s = time.perf_counter() - t0

    ref_accs = [float(r["test_acc"]) for r in ref_runs]
    shuf_accs = [float(r["test_acc"]) for r in shuf_runs]
    evaluation = evaluate_preregistration(plan, ref_accs, shuf_accs)

    return {
        "dataset_id": sub_ref.dataset_id,
        "subcircuit_id": sub_ref.subcircuit_id,
        "pack_digest": sub_ref.pack_digest,
        "task": cfg.task,
        "scope": cfg.scope,
        "seeds": [int(s) for s in seeds],
        "device": dev,
        "wall_seconds": round(wall_s, 3),
        "peak_vram_mb": peak_vram_mb,
        "framing_note": F6_FRAMING_NOTE,
        "circuit": {
            "num_cells": sub_ref.num_cells,
            "num_edges": sub_ref.num_edges,
            "num_readout_cells": n_readout,
            "num_input_groups": len(sub_ref.input_groups),
            "num_learnable_params": n_params,
            "base_weight_scale": round(_base_weight_scale(sub_ref), 6),
        },
        "preregistration": plan.to_dict(),
        "evaluation": evaluation,
        "reference_runs": ref_runs,
        "shuffled_runs": shuf_runs,
    }
