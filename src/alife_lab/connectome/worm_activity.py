"""Whole-brain calcium fluorescence forward model & C. elegans activity fitting harness (S1).

Includes pure-CSV/JSON trace ingestion (ready for real Flavell/Kato calcium recordings once user
approves dataset downloads) and an offline 302-cell synthetic benchmark evaluating intact
connectome vs no-gap-junctions and degree-preserving shuffled controls on held-out R^2 & r.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.celegans import make_synthetic_celegans_302_pack
from alife_lab.connectome.graded import (
    GradedCircuit,
    GradedParams,
    make_initial_graded_state,
    step_graded_tick_cpu,
)
from alife_lab.connectome.lif import make_shuffled_csr
from alife_lab.connectome.sysid import _pearson_r, _r2_score

F64 = npt.NDArray[np.float64]


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
