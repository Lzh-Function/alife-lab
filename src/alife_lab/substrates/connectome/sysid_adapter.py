"""``connectome_sysid``: external hidden-truth system identification adapter (Phase F5).

One tick = one PyTorch gradient-descent epoch fitting a differentiable firing-rate connectome
model to noisy, temporally binned, partial recordings from a hidden ground-truth teacher brain.
PyTorch is imported lazily inside worker methods only (NEVER in the FastAPI API process).
"""

from __future__ import annotations

import importlib.util
import math
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
from alife_lab.connectome.sysid import (
    HiddenMechanismSpec,
    ObservationSpec,
    SysIDSession,
    SysIDTrainConfig,
)
from alife_lab.core.spec import ConnectomeSysIDExperiment, ExperimentSpec
from alife_lab.service.settings import load_settings
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

SERIALIZE_VERSION = 1
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

_SUBCIRCUIT_CACHE: dict[tuple[str, str, str, int, str], Subcircuit] = {}


def _cs(spec: ExperimentSpec) -> ConnectomeSysIDExperiment:
    exp = spec.experiment
    if not isinstance(exp, ConnectomeSysIDExperiment):
        raise TypeError("connectome_sysid adapter received a different spec")
    return exp


def _get_subcircuits(exp: ConnectomeSysIDExperiment) -> tuple[Subcircuit, Subcircuit]:
    ds_dir = str(load_settings().resolved_datasets_dir)
    ref_key = (exp.dataset_id, exp.subcircuit_id, "reference", 0, ds_dir)
    sub_ref = _SUBCIRCUIT_CACHE.get(ref_key)
    if sub_ref is None:
        if exp.dataset_id == "synthetic_olfactory":
            pack = make_synthetic_olfactory_pack()
            sc_id = "full"
        else:
            pack = load_connectome_pack(
                exp.dataset_id, load_settings().resolved_datasets_dir, verify_hashes=True
            )
            sc_id = exp.subcircuit_id
        sub_ref = extract_subcircuit(pack, subcircuit_id=sc_id, wiring="reference")
        _SUBCIRCUIT_CACHE[ref_key] = sub_ref

    if exp.wiring == "reference":
        return sub_ref, sub_ref

    shuf_key = (exp.dataset_id, exp.subcircuit_id, "shuffled_control", int(exp.seed), ds_dir)
    sub_stu = _SUBCIRCUIT_CACHE.get(shuf_key)
    if sub_stu is None:
        if exp.dataset_id == "synthetic_olfactory":
            pack = make_synthetic_olfactory_pack()
            sc_id = "full"
        else:
            pack = load_connectome_pack(
                exp.dataset_id, load_settings().resolved_datasets_dir, verify_hashes=True
            )
            sc_id = exp.subcircuit_id
        sub_stu = extract_subcircuit(
            pack,
            subcircuit_id=sc_id,
            wiring="shuffled_control",
            shuffle_seed=int(exp.seed),
        )
        _SUBCIRCUIT_CACHE[shuf_key] = sub_stu
    return sub_ref, sub_stu


def _session_specs(
    exp: ConnectomeSysIDExperiment, backend: str
) -> tuple[HiddenMechanismSpec, ObservationSpec, SysIDTrainConfig]:
    m = exp.mechanisms
    o = exp.observation
    d = exp.dynamics
    device = "cuda" if backend == "torch_cuda" else "cpu"
    return (
        HiddenMechanismSpec(
            tau_enabled=m.tau_enabled,
            sign_flip_fraction=m.sign_flip_fraction,
            neuromod_enabled=m.neuromod_enabled,
            gap_junctions_enabled=m.gap_junctions_enabled,
            gap_candidates=m.gap_candidates,
            gap_active_fraction=m.gap_active_fraction,
        ),
        ObservationSpec(
            obs_fraction=o.obs_fraction,
            obs_classes=o.obs_classes,
            bin_ms=o.bin_ms,
            noise_sd=o.noise_sd,
        ),
        SysIDTrainConfig(
            dt_ms=d.dt_ms,
            sim_ms=d.sim_ms,
            n_train_stimuli=d.n_train_stimuli,
            n_holdout_stimuli=d.n_holdout_stimuli,
            epochs=exp.runner.max_steps,
            lr=d.lr,
            l1_gap=d.l1_gap,
            l2_reg=d.l2_reg,
            n_restarts=1,
            device=device,
        ),
    )


def _colormap_fire(v: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Map [0, 1] firing intensity to crisp dark-indigo -> cyan -> warm-gold RGB."""
    x = np.clip(v, 0.0, 1.0)
    r = np.clip(np.where(x < 0.5, x * 0.4, 0.2 + (x - 0.5) * 1.6), 0.0, 1.0)
    g = np.clip(np.where(x < 0.5, x * 1.4, 0.7 + (x - 0.5) * 0.6), 0.0, 1.0)
    b = np.clip(np.where(x < 0.5, 0.12 + x * 1.6, 0.92 - (x - 0.5) * 1.5), 0.0, 1.0)
    return np.rint(np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _render_sysid_rgb(
    ho_true: npt.NDArray[np.float32],
    ho_pred: npt.NDArray[np.float32] | None,
    metrics: dict[str, float],
) -> npt.NDArray[np.uint8]:
    """Deterministic 160x120 RGB diagnostic panel (no matplotlib dependency).

    Quadrants:
      Top-left  (76x56): Hidden teacher firing heatmap on unseen stimulus 0
      Top-right (76x56): Student predicted firing heatmap on unseen stimulus 0
      Bot-left  (76x56): |Teacher - Student| error heatmap (darker = better fit)
      Bot-right (76x56): Horizontal bars for heldout_unobs_r2, tau_r, sign_auroc, nm_r, gap_auroc
    """
    H, W = 120, 160
    img = np.full((H, W, 3), 16, dtype=np.uint8)
    # Thin slate divider lines
    img[59:61, :, :] = 46
    img[:, 79:81, :] = 46

    pred = ho_pred if ho_pred is not None else np.zeros_like(ho_true)
    # Sample up to 56 neurons evenly across the circuit and 76 time columns via nearest index.
    t_steps, n_cells, _ = ho_true.shape
    row_idx = np.linspace(0, n_cells - 1, 56).astype(np.int32)
    col_idx = np.linspace(0, t_steps - 1, 76).astype(np.int32)
    t_mat = ho_true[:, :, 0][np.ix_(col_idx, row_idx)].T  # (56, 76)
    p_mat = pred[:, :, 0][np.ix_(col_idx, row_idx)].T  # (56, 76)
    e_mat = np.clip(np.abs(t_mat - p_mat) * 2.5, 0.0, 1.0)

    img[2:58, 2:78, :] = _colormap_fire(t_mat)
    img[2:58, 82:158, :] = _colormap_fire(p_mat)
    # Error map in crimson/amber
    err_rgb = np.stack(
        [
            np.rint(np.clip(e_mat * 255.0, 0, 255)),
            np.rint(np.clip((e_mat**2) * 180.0, 0, 255)),
            np.rint(np.clip(e_mat * 40.0 + 18.0, 0, 255)),
        ],
        axis=-1,
    ).astype(np.uint8)
    img[62:118, 2:78, :] = err_rgb

    # Bottom-right: 5 horizontal recovery bars in [0, 1]
    bars = [
        (metrics.get("heldout_unobs_r2", 0.0), (56, 189, 248)),  # sky
        (metrics.get("tau_pearson_r", 0.0), (74, 222, 128)),  # emerald
        (metrics.get("sign_flip_auroc", 0.0), (251, 191, 36)),  # amber
        (metrics.get("neuromod_pearson_r", 0.0), (192, 132, 252)),  # violet
        (metrics.get("gap_auroc", 0.0), (244, 114, 182)),  # pink
    ]
    for bi, (val, color) in enumerate(bars):
        y0 = 66 + bi * 10
        y1 = y0 + 6
        img[y0:y1, 84:156, :] = (30, 36, 48)
        v = 0.0 if (val is None or math.isnan(val) or math.isinf(val)) else max(0.0, min(1.0, val))
        fill_w = round(v * 72)
        if fill_w > 0:
            img[y0:y1, 84 : 84 + fill_w, :] = color
    return img


@dataclass
class SysIDWorld:
    session: SysIDSession
    backend: str
    last_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def tick(self) -> int:
        return self.session.epoch


_M = MetricDefinition
_METRICS = (
    _M(
        "loss",
        "metric.loss",
        "MSE",
        "epoch",
        "Observed binned trajectory MSE.",
        "none",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "train_obs_r2",
        "metric.train_obs_r2",
        "R2",
        "epoch",
        "R^2 fit on noisy partial training recordings.",
        "max",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "heldout_all_r2",
        "metric.heldout_all_r2",
        "R2",
        "epoch",
        "R^2 on unseen holdout stimuli across all neurons.",
        "max",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "heldout_obs_r2",
        "metric.heldout_obs_r2",
        "R2",
        "epoch",
        "R^2 on unseen holdout stimuli (observed neurons).",
        "max",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "heldout_unobs_r2",
        "metric.heldout_unobs_r2",
        "R2",
        "epoch",
        "R^2 on unseen holdout stimuli (unobserved neurons).",
        "max",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "tau_pearson_r",
        "metric.tau_pearson_r",
        "r",
        "epoch",
        "Pearson correlation between recovered and hidden per-type tau_m.",
        "max",
        "every epoch",
        1,
        "null if tau disabled",
    ),
    _M(
        "tau_mae_ms",
        "metric.tau_mae_ms",
        "ms",
        "epoch",
        "Mean absolute error of recovered per-type tau_m (ms).",
        "min",
        "every epoch",
        1,
        "never null",
    ),
    _M(
        "sign_flip_auroc",
        "metric.sign_flip_auroc",
        "AUROC",
        "epoch",
        "AUROC for detecting hidden excitatory/inhibitory sign flips.",
        "max",
        "every epoch",
        1,
        "null if no flips",
    ),
    _M(
        "sign_flip_accuracy",
        "metric.sign_flip_accuracy",
        "fraction",
        "epoch",
        "Classification accuracy for hidden sign flips.",
        "max",
        "every epoch",
        1,
        "null if no flips",
    ),
    _M(
        "neuromod_pearson_r",
        "metric.neuromod_pearson_r",
        "r",
        "epoch",
        "Pearson correlation for hidden per-class neuromodulation.",
        "max",
        "every epoch",
        1,
        "null if disabled",
    ),
    _M(
        "gap_auroc",
        "metric.gap_auroc",
        "AUROC",
        "epoch",
        "AUROC for discovering hidden electrical gap junctions.",
        "max",
        "every epoch",
        1,
        "null if disabled",
    ),
    _M(
        "gap_pearson_r",
        "metric.gap_pearson_r",
        "r",
        "epoch",
        "Pearson correlation of recovered gap-junction conductances.",
        "max",
        "every epoch",
        1,
        "null if disabled",
    ),
)


CONNECTOME_SYSID_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.connectome_sysid",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="connectome_sysid",
    display_name_key="substrate.connectome_sysid",
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
            "Phase F5 hidden-truth connectome system identification (external optimization layer). "
            "Tests recovery of hidden biological mechanisms from limited noisy recordings."
        ),
        "paper_urls": [],
        "implementation": "this repository (PyTorch sparse rate-based connectome)",
    },
)


class ConnectomeSysIDAdapter:
    manifest = CONNECTOME_SYSID_MANIFEST

    def __init__(self) -> None:
        self.backend = "torch"

    def set_backend(self, backend: str) -> None:
        if backend not in ("torch", "torch_cuda"):
            raise ValueError(f"unsupported backend for connectome_sysid: {backend!r}")
        self.backend = backend

    def initialize(self, spec: ExperimentSpec) -> SysIDWorld:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed for connectome_sysid")
        exp = _cs(spec)
        sub_ref, sub_stu = _get_subcircuits(exp)
        mech, obs, cfg = _session_specs(exp, self.backend)
        sess = SysIDSession(
            sub_ref,
            sub_stu,
            mech,
            obs,
            cfg,
            seed=int(exp.seed),
            restart_index=int(exp.dynamics.restart_index),
        )
        m0 = sess.evaluate()
        return SysIDWorld(session=sess, backend=self.backend, last_metrics=m0)

    def step(self, state: SysIDWorld, spec: ExperimentSpec) -> SysIDWorld:
        state.session.step_epoch()
        state.last_metrics = state.session.evaluate()
        return state

    def metrics(self, state: SysIDWorld, spec: ExperimentSpec) -> dict[str, float]:
        wanted = set(_cs(spec).observations.metrics)
        out: dict[str, float] = {}
        for k, v in state.last_metrics.items():
            if k in wanted and not (math.isnan(v) or math.isinf(v)):
                out[k] = float(v)
        return out

    def render(self, state: SysIDWorld) -> tuple[dict[str, Any], bytes]:
        rgb = _render_sysid_rgb(
            state.session.ho_true_np, state.session.last_ho_pred_np, state.last_metrics
        )
        h, w = rgb.shape[:2]
        return (
            {
                "codec": "field_u8",
                "width": w,
                "height": h,
                "channels": 3,
                "scale": "F5 Teacher/Student unseen-stimulus heatmaps + mechanism recovery bars",
            },
            rgb.tobytes(),
        )

    def apply_intervention(
        self, state: SysIDWorld, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[SysIDWorld, dict[str, Any]]:
        raise ValueError("connectome_sysid runs accept no live interventions (external fitting)")

    def inspect(
        self, state: SysIDWorld, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        sess = state.session
        return {
            "substrate": "connectome_sysid",
            "epoch": sess.epoch,
            "dataset_id": sess.sub_ref.dataset_id,
            "subcircuit_id": sess.sub_ref.subcircuit_id,
            "wiring": sess.sub_student.wiring,
            "num_cells": sess.sub_ref.num_cells,
            "num_edges": sess.sub_ref.num_edges,
            "observed_cells": int(sess.obs_idx.size),
            "unobserved_cells": int(sess.unobs_idx.size),
            "device": sess.device,
            "metrics": {
                k: (None if math.isnan(v) or math.isinf(v) else round(v, 4))
                for k, v in state.last_metrics.items()
            },
        }

    def state_distance(self, a: SysIDWorld, b: SysIDWorld) -> dict[str, Any]:
        ra = a.session.export_raw_state()
        rb = b.session.export_raw_state()
        max_diff = max((float(np.abs(ra[k] - rb[k]).max()) if ra[k].size else 0.0) for k in ra)
        exact = all(np.array_equal(ra[k], rb[k]) for k in ra)
        return {"metric": "params_max_abs_difference", "value": max_diff, "exact_equal": exact}

    def serialize_state(self, state: SysIDWorld) -> dict[str, Any]:
        raw = state.session.export_raw_state()
        return {
            "arrays": raw,
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": int(state.session.epoch),
                "dataset_id": state.session.sub_ref.dataset_id,
                "subcircuit_id": state.session.sub_ref.subcircuit_id,
                "wiring": state.session.sub_student.wiring,
            },
        }

    def deserialize_state(self, data: dict[str, Any], spec: ExperimentSpec) -> SysIDWorld:
        meta = data["meta"]
        if int(meta["serialize_version"]) != SERIALIZE_VERSION:
            raise ValueError("incompatible connectome_sysid checkpoint serialize_version")
        arr = data["arrays"]
        world = self.initialize(spec)
        raw = {
            k: np.asarray(arr[k], dtype=np.float32)
            for k in ("tau_logit", "sign_mult", "nm_log_gain", "nm_bias", "gap_raw")
        }
        epoch = int(meta["tick"])
        world.session.load_raw_state(raw, epoch)
        world.last_metrics = world.session.evaluate()
        return world
