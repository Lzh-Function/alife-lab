"""Phase F10-A: Connectome-Constrained Drosophila Visual System (``flyvis``, Lappalainen 2024).

Integrates TuragaLab's pretrained 45,669-neuron, 65-cell-type optic-lobe deep mechanistic network
(``flyvis``, MIT, PyTorch) as an optional vision front-end (``uv sync --extra vision``), while
keeping CPU-only CI 100% runnable via a lightweight pure-NumPy Hassenstein-Reichardt fallback for
offline unit checks.

Verifies two canonical Drosophila visual reflexes + negative controls:
1. **Optomotor direction selectivity (T4a/T4b, T5a/T5b):** drifting sinusoidal gratings moving
   rightward (+X) vs leftward (-X) produce opposite-sign opponent indices ``(T4a+T5a)-(T4b+T5b)``.
2. **Looming predator response:** an expanding dark disc elicits higher transient T4/T5/Tm/Mi
   modulation than a static control disc or uniform gray field.
3. **Visual-to-Descending Bridge:** maps real optic-lobe optomotor & looming outputs into bounded
   ``turn_left_dn`` / ``turn_right_dn`` / ``backward_dn`` (MDN) stimulation rates for F8/F11.
"""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]


def _prepare_flyvis_env(datasets_dir: Path | None) -> Path:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), f"alife_mpl_{os.getuid()}")
    root = (
        (datasets_dir / "flyvis").resolve()
        if datasets_dir is not None
        else Path(".alife/datasets/flyvis").resolve()
    )
    os.environ["FLYVIS_ROOT_DIR"] = str(root)
    return root


def flyvis_available(datasets_dir: Path | None = None) -> bool:
    """True if ``flyvis`` is installed and pretrained weights exist in ``datasets_dir/flyvis``."""
    if importlib.util.find_spec("flyvis") is None or importlib.util.find_spec("torch") is None:
        return False
    if datasets_dir is None:
        return True
    model_dir = datasets_dir / "flyvis" / "results" / "flow" / "0000" / "000"
    return model_dir.is_dir()


def make_drifting_grating(
    n_frames: int = 30,
    size: int = 64,
    *,
    direction: float = 1.0,
    spatial_cycles: float = 4.0,
    temporal_cycles: float = 2.0,
) -> F32:
    """Generate a ``(n_frames, size, size)`` float32 drifting vertical sinusoidal grating [0, 1]."""
    xs = np.linspace(0.0, 2.0 * math.pi * spatial_cycles, size, dtype=np.float32)
    ts = np.linspace(0.0, 2.0 * math.pi * temporal_cycles, n_frames, dtype=np.float32)
    phase = xs[None, None, :] - np.float32(direction) * ts[:, None, None]
    frame = 0.5 + 0.45 * np.sin(phase)
    return np.broadcast_to(frame, (n_frames, size, size)).astype(np.float32, copy=True)


def make_looming_disc(
    n_frames: int = 30,
    size: int = 64,
    *,
    expanding: bool = True,
) -> F32:
    """Generate an expanding dark looming shadow (or static dark disc control) in [0, 1]."""
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - (size - 1) * 0.5, yy - (size - 1) * 0.5)
    out = np.full((n_frames, size, size), 0.85, dtype=np.float32)
    for t in range(n_frames):
        frac = t / max(1, n_frames - 1)
        rad = (3.0 + 23.0 * (frac**1.8)) if expanding else 12.0
        out[t, r <= rad] = np.float32(0.08)
    return out


def make_static_gray(n_frames: int = 30, size: int = 64) -> F32:
    return np.full((n_frames, size, size), 0.5, dtype=np.float32)


def visual_to_descending_hz(
    optomotor_opponent: float,
    looming_contrast: float,
    *,
    max_turn_hz: float = 160.0,
    max_escape_hz: float = 180.0,
) -> dict[str, float]:
    """Map optic-lobe opponent motion & looming contrast to bounded descending neuron rates (Hz)."""
    turn_l = max(0.0, min(max_turn_hz, optomotor_opponent * max_turn_hz))
    turn_r = max(0.0, min(max_turn_hz, -optomotor_opponent * max_turn_hz))
    escape = max(0.0, min(max_escape_hz, looming_contrast * max_escape_hz))
    return {
        "turn_left_dn_hz": round(float(turn_l), 2),
        "turn_right_dn_hz": round(float(turn_r), 2),
        "backward_dn_hz": round(float(escape), 2),
    }


def run_synthetic_visual_assay(n_frames: int = 24, size: int = 48) -> dict[str, Any]:
    """Offline CPU-only Hassenstein-Reichardt & looming assay for CI without ``flyvis``."""
    t0 = time.perf_counter()

    def _hr_index(mov: F32) -> float:
        # 1-frame, 1-pixel classic Hassenstein-Reichardt correlator along +X vs -X
        a = mov[:-1, :, :-1] - 0.5
        b = mov[1:, :, 1:] - 0.5
        c = mov[:-1, :, 1:] - 0.5
        d = mov[1:, :, :-1] - 0.5
        right = float(np.mean(a * b))
        left = float(np.mean(c * d))
        denom = abs(right) + abs(left) + 1e-6
        return (right - left) / denom

    g_r = make_drifting_grating(n_frames, size, direction=+1.0)
    g_l = make_drifting_grating(n_frames, size, direction=-1.0)
    gray = make_static_gray(n_frames, size)
    loom = make_looming_disc(n_frames, size, expanding=True)
    stat = make_looming_disc(n_frames, size, expanding=False)

    idx_r = _hr_index(g_r)
    idx_l = _hr_index(g_l)
    idx_0 = _hr_index(gray)
    loom_act = float(np.mean(np.abs(np.diff(loom, axis=0))))
    stat_act = float(np.mean(np.abs(np.diff(stat, axis=0))))

    checks = {
        "optomotor_reverses_with_direction": bool(idx_r > 0.2 and idx_l < -0.2),
        "static_gray_near_zero": bool(abs(idx_0) < 0.05),
        "looming_exceeds_static": bool(loom_act > stat_act + 0.01),
    }
    return {
        "backend": "synthetic_hr_cpu",
        "num_cells": 0,
        "unique_cell_types": 0,
        "wall_seconds": round(time.perf_counter() - t0, 4),
        "stimuli": {
            "grating_right": {"opponent_index": round(idx_r, 4)},
            "grating_left": {"opponent_index": round(idx_l, 4)},
            "static_gray": {"opponent_index": round(idx_0, 4)},
            "looming_disc": {"transient_activity": round(loom_act, 4)},
            "static_disc": {"transient_activity": round(stat_act, 4)},
        },
        "bridge_example": visual_to_descending_hz(idx_r, min(1.0, loom_act * 10.0)),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def run_flyvis_visual_assay(
    datasets_dir: Path,
    *,
    model_id: str = "000",
    dt: float = 0.02,
    n_frames: int = 30,
    size: int = 64,
) -> dict[str, Any]:
    """Run real pretrained ``flyvis`` (45,669-cell optic lobe) optomotor & looming verification."""
    t0 = time.perf_counter()
    root = _prepare_flyvis_env(datasets_dir)
    model_dir = root / "results" / "flow" / "0000" / model_id
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"flyvis pretrained model not found at {model_dir}; "
            "run `FLYVIS_ROOT_DIR=.alife/datasets/flyvis uv run flyvis download-pretrained`"
        )

    import flyvis
    import torch

    nv = flyvis.NetworkView(model_dir)
    net = nv.init_network()
    dev = next(net.parameters()).device
    eye = flyvis.rendering.BoxEye(extent=15, kernel_size=13)

    node_types = [
        t.decode() if isinstance(t, bytes) else str(t) for t in net.connectome.nodes.type[:]
    ]
    n_nodes = len(node_types)
    type_indices: dict[str, np.ndarray[Any, np.dtype[np.int64]]] = {}
    for ct in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "Mi1", "Tm9"):
        idx = np.asarray([i for i, t in enumerate(node_types) if t == ct], dtype=np.int64)
        if idx.size > 0:
            type_indices[ct] = idx

    def _eval_movie(mov: F32) -> dict[str, float]:
        with torch.no_grad():
            hex_in = eye(torch.from_numpy(mov[None, ...]).to(dev))
            resp = net.simulate(hex_in, dt=dt)[0].detach().cpu().numpy()  # (T, N)
        steady = resp[max(1, n_frames // 3) :]
        means = {ct: float(steady[:, idx].mean()) for ct, idx in type_indices.items()}
        stds = {
            ct: float(np.std(steady[:, idx], axis=0).mean()) for ct, idx in type_indices.items()
        }
        # T4a/T5a vs T4b/T5b horizontal motion opponency
        pos_h = 0.5 * (means.get("T4a", 0.0) + means.get("T5a", 0.0))
        neg_h = 0.5 * (means.get("T4b", 0.0) + means.get("T5b", 0.0))
        opp_h = pos_h - neg_h
        t5_dyn = float(np.mean([stds.get(k, 0.0) for k in ("T5a", "T5b", "T5c", "T5d")]))
        t4_dyn = float(np.mean([stds.get(k, 0.0) for k in ("T4a", "T4b", "T4c", "T4d")]))
        return {
            "T4a_mean": round(means.get("T4a", 0.0), 4),
            "T4b_mean": round(means.get("T4b", 0.0), 4),
            "T5a_mean": round(means.get("T5a", 0.0), 4),
            "T5b_mean": round(means.get("T5b", 0.0), 4),
            "horizontal_opponency": round(opp_h, 4),
            "t4_temporal_std": round(t4_dyn, 4),
            "t5_temporal_std": round(t5_dyn, 4),
        }

    r_right = _eval_movie(make_drifting_grating(n_frames, size, direction=+1.0))
    r_left = _eval_movie(make_drifting_grating(n_frames, size, direction=-1.0))
    r_gray = _eval_movie(make_static_gray(n_frames, size))
    r_loom = _eval_movie(make_looming_disc(n_frames, size, expanding=True))
    r_stat = _eval_movie(make_looming_disc(n_frames, size, expanding=False))

    opp_0 = r_gray["horizontal_opponency"]
    opp_r = r_right["horizontal_opponency"] - opp_0
    opp_l = r_left["horizontal_opponency"] - opp_0
    gray_mod = r_gray["t5_temporal_std"] + r_gray["t4_temporal_std"]
    loom_mod = r_loom["t5_temporal_std"] + r_loom["t4_temporal_std"]
    stat_mod = r_stat["t5_temporal_std"] + r_stat["t4_temporal_std"]

    # Baseline-corrected opposite signs for rightward vs leftward grating motion,
    # near-zero temporal modulation on uniform gray, and strong looming modulation vs static disc.
    checks = {
        "optomotor_reverses_with_direction": bool(
            opp_r * opp_l < 0.0 and abs(opp_r - opp_l) > 0.10
        ),
        "static_gray_quiescent": bool(gray_mod < 0.02),
        "looming_exceeds_static": bool(loom_mod > stat_mod * 1.5 and loom_mod > 0.01),
    }
    return {
        "backend": f"flyvis_{flyvis.__version__}",
        "model_id": model_id,
        "device": str(next(net.parameters()).device),
        "num_cells": n_nodes,
        "unique_cell_types": len(set(node_types)),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "stimuli": {
            "grating_right": r_right,
            "grating_left": r_left,
            "static_gray": r_gray,
            "looming_disc": r_loom,
            "static_disc": r_stat,
        },
        "bridge_example": visual_to_descending_hz(
            max(-1.0, min(1.0, opp_r * 2.0)),
            max(0.0, min(1.0, (loom_mod - stat_mod) * 2.0)),
        ),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
