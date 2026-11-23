"""``connectome_task``: external fixed-topology task learning adapter (Phase F6).

One tick = one PyTorch optimization epoch training a learning machine that borrows a fixed
connectome wiring topology (biological ``reference`` vs degree-preserving ``shuffled_control``)
on odor classification, short-term memory, or nonlinear XOR under one of three training scopes
(``readout_only``, ``io_only``, ``partial_weights``).

PyTorch is imported lazily inside worker methods only (NEVER in the FastAPI API process).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab import API_CONTRACT_VERSION
from alife_lab.connectome.flywire import load_connectome_pack
from alife_lab.connectome.subcircuit import (
    Subcircuit,
    extract_subcircuit,
    make_synthetic_olfactory_pack,
)
from alife_lab.connectome.tasks import (
    F6_FRAMING_NOTE,
    F6TaskConfig,
    TaskTrainSession,
    make_preregistration,
)
from alife_lab.core.spec import ConnectomeTaskExperiment, ExperimentSpec
from alife_lab.service.settings import load_settings
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

SERIALIZE_VERSION = 1
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

_SUBCIRCUIT_CACHE: dict[tuple[str, str, str, int, str], Subcircuit] = {}


def _ct(spec: ExperimentSpec) -> ConnectomeTaskExperiment:
    exp = spec.experiment
    if not isinstance(exp, ConnectomeTaskExperiment):
        raise TypeError("connectome_task adapter received a different spec")
    return exp


def _get_subcircuit(exp: ConnectomeTaskExperiment) -> Subcircuit:
    ds_dir = str(load_settings().resolved_datasets_dir)
    s_seed = 0 if exp.wiring == "reference" else int(exp.seed)
    key = (exp.dataset_id, exp.subcircuit_id, exp.wiring, s_seed, ds_dir)
    cached = _SUBCIRCUIT_CACHE.get(key)
    if cached is not None:
        return cached
    if exp.dataset_id == "synthetic_olfactory":
        pack = make_synthetic_olfactory_pack()
        sc_id = "full"
    else:
        pack = load_connectome_pack(
            exp.dataset_id, load_settings().resolved_datasets_dir, verify_hashes=True
        )
        sc_id = exp.subcircuit_id
    sub = extract_subcircuit(
        pack,
        subcircuit_id=sc_id,
        wiring=exp.wiring,
        shuffle_seed=s_seed,
    )
    _SUBCIRCUIT_CACHE[key] = sub
    return sub


def _task_cfg(exp: ConnectomeTaskExperiment, backend: str) -> F6TaskConfig:
    d = exp.dynamics
    return F6TaskConfig(
        task=d.task,
        scope=d.scope,
        dt_ms=d.dt_ms,
        n_train=d.n_train,
        n_test=d.n_test,
        epochs=exp.runner.max_steps,
        lr=d.lr,
        weight_decay=d.weight_decay,
        max_readout=d.max_readout,
        device="cuda" if backend == "torch_cuda" else "cpu",
    )


def _colormap_coolwarm(v: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    x = np.clip(v, 0.0, 1.0)
    r = np.clip(np.where(x < 0.5, x * 0.5, 0.25 + (x - 0.5) * 1.5), 0.0, 1.0)
    g = np.clip(np.where(x < 0.5, 0.1 + x * 1.5, 0.85 - (x - 0.5) * 0.3), 0.0, 1.0)
    b = np.clip(np.where(x < 0.5, 0.25 + x * 1.4, 0.95 - (x - 0.5) * 1.4), 0.0, 1.0)
    return np.rint(np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _render_task_rgb(session: TaskTrainSession) -> npt.NDArray[np.uint8]:
    """Deterministic 160x120 RGB diagnostic panel for F6 fixed-topology task learning.

    Quadrants:
      Top-left  (76x56): Unseen test confusion matrix (bright diagonal = high accuracy)
      Top-right (76x56): Train (cyan) vs unseen test (gold) accuracy curves + chance line
      Bot-left  (76x56): Input group waveform & shaded readout window for test trial 0
      Bot-right (76x56): Readout neuron activity heatmap (neurons x time) for test trial 0
    """
    H, W = 120, 160
    img = np.full((H, W, 3), 16, dtype=np.uint8)
    img[59:61, :, :] = 46
    img[:, 79:81, :] = 46

    # 1. Top-left: Confusion matrix (normalized per true class row).
    cm = session.last_confusion.astype(np.float32)
    n_c = max(1, cm.shape[0])
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    cm_norm = cm / row_sums
    cell_h = 56 // n_c
    cell_w = 76 // n_c
    for r in range(n_c):
        for c in range(n_c):
            v = float(cm_norm[r, c])
            y0, y1 = 2 + r * cell_h, 2 + (r + 1) * cell_h - 1
            x0, x1 = 2 + c * cell_w, 2 + (c + 1) * cell_w - 1
            if r == c:
                col = (int(28 + v * 46), int(50 + v * 175), int(70 + v * 175))
            else:
                col = (int(30 + v * 210), int(24 + v * 50), int(34 + v * 40))
            img[y0:y1, x0:x1, :] = col

    # 2. Top-right: Train (sky) and unseen test (amber) accuracy curves over epochs.
    chance_y = 57 - round(session.dataset.chance_acc * 52)
    img[max(2, min(57, chance_y)), 82:158, :] = (90, 96, 110)
    hist_tr = session.history_train_acc
    hist_te = session.history_test_acc
    n_pts = len(hist_te)
    if n_pts >= 1:
        xs = np.linspace(83, 156, n_pts).astype(np.int32)
        for i in range(n_pts):
            x = int(xs[i])
            y_tr = max(2, min(57, 57 - round(hist_tr[i] * 52)))
            y_te = max(2, min(57, 57 - round(hist_te[i] * 52)))
            img[max(2, y_tr - 1) : min(58, y_tr + 1), x, :] = (56, 189, 248)
            img[max(2, y_te - 1) : min(58, y_te + 2), x, :] = (251, 191, 36)

    # 3. Bottom-left: Input group drive (G x T) for test trial 0 + readout window highlight.
    x0_g = session.dataset.x_test_groups[:, :, 0]  # (T, G)
    t_steps, n_g = x0_g.shape
    r_idx = np.linspace(0, n_g - 1, 56).astype(np.int32)
    c_idx = np.linspace(0, t_steps - 1, 76).astype(np.int32)
    in_mat = np.clip(x0_g[np.ix_(c_idx, r_idx)].T / 3.0, 0.0, 1.0)
    img[62:118, 2:78, :] = _colormap_coolwarm(in_mat)
    # Subtle green top bar indicating readout window
    rx0 = 2 + round(session.dataset.read_start / max(1, t_steps) * 76)
    rx1 = 2 + round(session.dataset.read_end / max(1, t_steps) * 76)
    img[62:65, max(2, rx0) : min(78, rx1), :] = (74, 222, 128)

    # 4. Bottom-right: Readout neuron activity heatmap (R x T) for test trial 0.
    tr0 = session.last_tr0_read_np
    if tr0 is not None and tr0.size > 0:
        rt, nr = tr0.shape
        rr_idx = np.linspace(0, nr - 1, 56).astype(np.int32)
        rc_idx = np.linspace(0, rt - 1, 76).astype(np.int32)
        ro_mat = tr0[np.ix_(rc_idx, rr_idx)].T
        img[62:118, 82:158, :] = _colormap_coolwarm(ro_mat)
    return img


@dataclass
class TaskWorld:
    session: TaskTrainSession
    backend: str
    prereg: dict[str, Any]
    last_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def tick(self) -> int:
        return self.session.epoch


_M = MetricDefinition
_METRICS = (
    _M(
        "train_loss",
        "metric.train_loss",
        "nats",
        "epoch",
        "Cross-entropy loss on training trials.",
        "min",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "test_loss",
        "metric.test_loss",
        "nats",
        "epoch",
        "Cross-entropy loss on unseen holdout test trials.",
        "min",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "train_acc",
        "metric.train_acc",
        "fraction",
        "epoch",
        "Classification accuracy on training trials.",
        "max",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "test_acc",
        "metric.test_acc",
        "fraction",
        "epoch",
        "Classification accuracy on unseen holdout test trials (primary F6 metric).",
        "max",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "chance_acc",
        "metric.chance_acc",
        "fraction",
        "epoch",
        "Random-guessing baseline accuracy (1 / num_classes).",
        "none",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "test_margin",
        "metric.test_margin",
        "prob",
        "epoch",
        "Mean true-class probability above chance on unseen test trials.",
        "max",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "readout_norm",
        "metric.readout_norm",
        "L2",
        "epoch",
        "L2 norm of linear readout weights.",
        "none",
        "every epoch",
        1,
        "never null",
    ),
)


CONNECTOME_TASK_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.connectome_task",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="connectome_task",
    display_name_key="substrate.connectome_task",
    scientific_modes=("exploratory",),
    capabilities=Capabilities(
        state_kind="grid",
        clock_unit="epoch",
        supports_pause=True,
        supports_interventions=False,
        supports_inspection=True,
        supports_lineage=True,
        supports_differentiation=True,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=(),
    ),
    backends=("torch", "torch_cuda"),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    backend_numeric_policies={"torch": ("float32_torch",), "torch_cuda": ("float32_torch",)},
    reference={
        "description": (
            f"Phase F6 fixed-topology task learning benchmark. {F6_FRAMING_NOTE} "
            "Compares biological connectome wiring against degree- and weight-matched shuffled "
            "wiring under pre-registered criteria."
        ),
        "paper_urls": [],
        "implementation": "this repository (PyTorch fixed-topology reservoir / task learner)",
    },
)


class ConnectomeTaskAdapter:
    manifest = CONNECTOME_TASK_MANIFEST

    def __init__(self) -> None:
        self.backend = "torch"

    def set_backend(self, backend: str) -> None:
        if backend not in ("torch", "torch_cuda"):
            raise ValueError(f"unsupported backend for connectome_task: {backend!r}")
        self.backend = backend

    def initialize(self, spec: ExperimentSpec) -> TaskWorld:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed for connectome_task")
        exp = _ct(spec)
        sub = _get_subcircuit(exp)
        cfg = _task_cfg(exp, self.backend)
        plan = make_preregistration(sub.dataset_id, sub.subcircuit_id, cfg.task, cfg.scope)
        sess = TaskTrainSession(sub, cfg, seed=int(exp.seed), data_seed=int(exp.seed))
        m0 = sess.evaluate()
        return TaskWorld(session=sess, backend=self.backend, prereg=plan.to_dict(), last_metrics=m0)

    def step(self, state: TaskWorld, spec: ExperimentSpec) -> TaskWorld:
        state.session.step_epoch()
        state.last_metrics = state.session.evaluate()
        return state

    def metrics(self, state: TaskWorld, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_ct(spec).observations.metrics)
        return {k: float(v) for k, v in state.last_metrics.items() if k in wanted}

    def render(self, state: TaskWorld) -> tuple[dict[str, Any], bytes]:
        rgb = _render_task_rgb(state.session)
        h, w = rgb.shape[:2]
        return (
            {
                "codec": "field_u8",
                "width": w,
                "height": h,
                "channels": 3,
                "scale": "F6 Confusion / Accuracy / Stimulus / Readout",
            },
            rgb.tobytes(),
        )

    def apply_intervention(
        self, state: TaskWorld, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[TaskWorld, dict[str, Any]]:
        raise ValueError("connectome_task runs accept no live interventions (external learning)")

    def inspect(
        self, state: TaskWorld, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        sess = state.session
        return {
            "substrate": "connectome_task",
            "framing_note": F6_FRAMING_NOTE,
            "epoch": sess.epoch,
            "dataset_id": sess.sub.dataset_id,
            "subcircuit_id": sess.sub.subcircuit_id,
            "wiring": sess.sub.wiring,
            "task": sess.cfg.task,
            "scope": sess.cfg.scope,
            "num_cells": sess.sub.num_cells,
            "num_edges": sess.sub.num_edges,
            "num_readout_cells": int(sess.readout_idx.shape[0]),
            "num_learnable_params": int(sum(p.numel() for p in sess.params)),
            "device": sess.device,
            "preregistration_digest": state.prereg["plan_digest"],
            "confusion_matrix": sess.last_confusion.tolist(),
            "metrics": {k: round(v, 4) for k, v in state.last_metrics.items()},
        }

    def state_distance(self, a: TaskWorld, b: TaskWorld) -> dict[str, Any]:
        ra = a.session.export_raw_state()
        rb = b.session.export_raw_state()
        max_diff = max((float(np.abs(ra[k] - rb[k]).max()) if ra[k].size else 0.0) for k in ra)
        exact = all(np.array_equal(ra[k], rb[k]) for k in ra)
        return {"metric": "params_max_abs_difference", "value": max_diff, "exact_equal": exact}

    def serialize_state(self, state: TaskWorld) -> dict[str, Any]:
        raw = state.session.export_raw_state()
        return {
            "arrays": raw,
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": int(state.session.epoch),
                "dataset_id": state.session.sub.dataset_id,
                "subcircuit_id": state.session.sub.subcircuit_id,
                "wiring": state.session.sub.wiring,
                "task": state.session.cfg.task,
                "scope": state.session.cfg.scope,
                "history_train_acc": state.session.history_train_acc[-200:],
                "history_test_acc": state.session.history_test_acc[-200:],
            },
        }

    def deserialize_state(self, data: dict[str, Any], spec: ExperimentSpec) -> TaskWorld:
        meta = data["meta"]
        if int(meta["serialize_version"]) != SERIALIZE_VERSION:
            raise ValueError("incompatible connectome_task checkpoint serialize_version")
        arr = data["arrays"]
        world = self.initialize(spec)
        raw = {
            k: np.asarray(arr[k], dtype=np.float32)
            for k in ("w_out", "b_out", "w_in", "type_scale_logit")
        }
        world.session.load_raw_state(raw, int(meta["tick"]))
        world.session.history_train_acc = [float(x) for x in meta.get("history_train_acc", [])]
        world.session.history_test_acc = [float(x) for x in meta.get("history_test_acc", [])]
        world.last_metrics = world.session.evaluate()
        return world
