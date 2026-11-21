"""Systematic single-cell-type silencing screen for connectome circuits (F3).

Runs an intact baseline plus one silencing condition per candidate ``cell_type`` across
multiple seeds, using either the exact CPU oracle (``numpy``) or batched CUDA graphs
(``cupy``, ``CupyBatchedLIFRunner``) with bitwise-identical spike counts. Evaluates each
silencing condition against the intact baseline using the platform's seed-level permutation
test, Cliff's delta and Holm-Bonferroni correction (``alife_lab.analysis.stats``).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

from alife_lab.analysis.stats import compare, describe, holm
from alife_lab.connectome.lif import LIFParams, make_initial_state, step_tick_cpu
from alife_lab.connectome.schema import ConnectomePack, CSRGraph


def _build_stim_array(
    pack: ConnectomePack, stimuli: list[tuple[str, float]]
) -> np.ndarray[Any, Any]:
    stim = np.zeros(pack.cells.num_cells, dtype=np.float32)
    for port_id, rate_hz in stimuli:
        if port_id not in pack.io_ports:
            known = ", ".join(sorted(pack.io_ports))
            raise ValueError(f"unknown stimulus port {port_id!r} (available: {known})")
        stim[pack.io_ports[port_id].cell_indices] = np.float32(rate_hz)
    return stim


def _cell_type_indices(pack: ConnectomePack) -> dict[str, np.ndarray[Any, np.dtype[np.int32]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, ct in enumerate(pack.cells.cell_type):
        if ct:
            groups[ct].append(i)
    return {ct: np.asarray(idxs, dtype=np.int32) for ct, idxs in groups.items()}


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def _run_batch_total_spikes(
    csr: CSRGraph,
    params: LIFParams,
    stim_rows: np.ndarray[Any, Any],
    silenced_rows: np.ndarray[Any, Any],
    seeds: list[int],
    ticks: int,
    backend: str,
    batch_size: int = 8,
) -> np.ndarray[Any, np.dtype[np.int32]]:
    """Run ``M`` independent brain evaluations and return ``(M, N)`` int32 total_spikes."""
    m = len(seeds)
    n = csr.num_cells
    out = np.empty((m, n), dtype=np.int32)
    if m == 0:
        return out

    if backend == "cupy":
        from alife_lab.connectome.lif_cupy import CupyBatchedLIFRunner

        n_substeps = ticks * params.substeps_per_tick
        # Chunk at batch_size (default 8, peak throughput on RTX 5070) to keep VRAM bounded.
        chunk_steps = min(n_substeps, 100)
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            runner = CupyBatchedLIFRunner(
                csr,
                params,
                seeds[start:end],
                stim_rows[start:end],
            )
            runner.set_silenced(silenced_rows[start:end])
            rem = n_substeps
            while rem > 0:
                step_now = chunk_steps if rem >= chunk_steps else rem
                runner.step_substeps(step_now, use_graph=True)
                rem -= step_now
            out[start:end] = runner.total_spikes_cpu()
        return out

    if backend != "numpy":
        raise ValueError(f"unsupported backend {backend!r} (choose 'auto', 'numpy' or 'cupy')")

    for i in range(m):
        lif = make_initial_state(n, params)
        lif.stim_rate_hz[:] = stim_rows[i]
        lif.silenced[:] = silenced_rows[i]
        seed = int(seeds[i])
        for _ in range(ticks):
            step_tick_cpu(lif, csr, params, seed)
        out[i] = lif.total_spikes
    return out


def discover_active_cell_types(
    pack: ConnectomePack,
    csr: CSRGraph,
    params: LIFParams,
    *,
    stimuli: list[tuple[str, float]],
    seed: int = 1,
    ticks: int = 500,
    top_k: int = 24,
    exclude_ports: tuple[str, ...] = ("mn9_proboscis_extension",),
    backend: str = "auto",
) -> list[str]:
    """Run one baseline pass and return the ``top_k`` most active non-excluded ``cell_type``s."""
    eff_backend = ("cupy" if _has_cupy() else "numpy") if backend == "auto" else backend
    n = pack.cells.num_cells
    stim = _build_stim_array(pack, stimuli)
    sil = np.zeros((1, n), dtype=np.uint8)
    totals = _run_batch_total_spikes(
        csr, params, stim[None, :], sil, [seed], ticks, eff_backend, batch_size=1
    )[0]

    excluded_cells: set[int] = set()
    for pid, _ in stimuli:
        if pid in pack.io_ports:
            excluded_cells.update(int(i) for i in pack.io_ports[pid].cell_indices.tolist())
    for pid in exclude_ports:
        if pid in pack.io_ports:
            excluded_cells.update(int(i) for i in pack.io_ports[pid].cell_indices.tolist())

    type_spikes: dict[str, int] = defaultdict(int)
    for i in np.flatnonzero(totals > 0).tolist():
        if i in excluded_cells:
            continue
        ct = pack.cells.cell_type[i]
        if ct:
            type_spikes[ct] += int(totals[i])

    ordered = sorted(type_spikes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ct for ct, _ in ordered[:top_k]]


def run_silencing_screen(
    pack: ConnectomePack,
    csr: CSRGraph,
    params: LIFParams,
    *,
    stimuli: list[tuple[str, float]],
    cell_types: list[str] | None = None,
    top_k: int = 24,
    target_port: str = "mn9_proboscis_extension",
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    ticks: int = 1000,
    wiring: str = "reference",
    backend: str = "auto",
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run baseline + single-cell-type silencing screen across multiple seeds."""
    if target_port not in pack.io_ports:
        known = ", ".join(sorted(pack.io_ports))
        raise ValueError(f"unknown target port {target_port!r} (available: {known})")
    if not seeds:
        raise ValueError("seeds must be non-empty")

    eff_backend = ("cupy" if _has_cupy() else "numpy") if backend == "auto" else backend
    t0 = time.perf_counter()

    if cell_types is None:
        selected_types = discover_active_cell_types(
            pack,
            csr,
            params,
            stimuli=stimuli,
            seed=int(seeds[0]),
            ticks=min(ticks, 500),
            top_k=top_k,
            exclude_ports=(target_port,),
            backend=eff_backend,
        )
    else:
        selected_types = list(dict.fromkeys(cell_types))

    ct_map = _cell_type_indices(pack)
    for ct in selected_types:
        if ct not in ct_map:
            raise ValueError(f"unknown cell_type {ct!r} in dataset {pack.dataset_id!r}")

    n = pack.cells.num_cells
    base_stim = _build_stim_array(pack, stimuli)
    target_idx = pack.io_ports[target_port].cell_indices
    n_target = max(1, int(target_idx.size))
    elapsed_s = max(ticks * params.substeps_per_tick * params.dt_s, 1e-9)

    # Condition 0 = intact baseline; conditions 1..K = each single silenced cell_type.
    conditions: list[tuple[str, np.ndarray[Any, np.dtype[np.int32]]]] = [
        ("baseline", np.zeros(0, dtype=np.int32))
    ]
    for ct in selected_types:
        conditions.append((ct, ct_map[ct]))

    n_cond = len(conditions)
    n_seeds = len(seeds)
    total_evals = n_cond * n_seeds

    stim_rows = np.broadcast_to(base_stim[None, :], (total_evals, n)).copy()
    sil_rows = np.zeros((total_evals, n), dtype=np.uint8)
    seed_rows: list[int] = []
    for ci, (_, idx) in enumerate(conditions):
        for si, sd in enumerate(seeds):
            row = ci * n_seeds + si
            if idx.size > 0:
                sil_rows[row, idx] = np.uint8(1)
            seed_rows.append(int(sd))

    totals = _run_batch_total_spikes(
        csr,
        params,
        stim_rows,
        sil_rows,
        seed_rows,
        ticks,
        eff_backend,
        batch_size=batch_size,
    )
    wall_s = time.perf_counter() - t0

    rates_by_cond: list[list[float]] = []
    for ci in range(n_cond):
        vals: list[float] = []
        for si in range(n_seeds):
            row = ci * n_seeds + si
            sp = int(totals[row, target_idx].sum())
            vals.append(sp / (n_target * elapsed_s))
        rates_by_cond.append(vals)

    base_vals = rates_by_cond[0]
    base_desc = describe(base_vals)
    base_mean = base_desc["mean"] or 0.0

    raw_comps = [
        compare(rates_by_cond[ci], base_vals, seed=10_000 + ci) for ci in range(1, n_cond)
    ]
    raw_p = [c["permutation"]["p_value"] for c in raw_comps]
    adj_p = holm(raw_p)

    results: list[dict[str, Any]] = []
    for k, (ct, idx) in enumerate(conditions[1:]):
        vals = rates_by_cond[k + 1]
        desc = describe(vals)
        mean_hz = desc["mean"] or 0.0
        comp = raw_comps[k]
        delta_hz = mean_hz - base_mean
        suppression_pct = (
            ((base_mean - mean_hz) / base_mean * 100.0) if base_mean > 1e-9 else 0.0
        )
        results.append(
            {
                "cell_type": ct,
                "cell_count": int(idx.size),
                "mean_hz": round(mean_hz, 3),
                "sd_hz": round(desc["sd"] or 0.0, 3),
                "delta_hz": round(delta_hz, 3),
                "suppression_pct": round(suppression_pct, 2),
                "cliffs_delta": comp["cliffs_delta"],
                "p_value": comp["permutation"]["p_value"],
                "p_value_holm": adj_p[k],
                "per_seed_hz": [round(v, 3) for v in vals],
            }
        )

    # Sort by largest reduction in target firing rate first.
    results.sort(key=lambda r: (r["delta_hz"], r["cell_type"]))
    brain_seconds = total_evals * elapsed_s
    return {
        "dataset_id": pack.dataset_id,
        "pack_digest": pack.pack_digest,
        "wiring": wiring,
        "backend": eff_backend,
        "batch_size": batch_size if eff_backend == "cupy" else 1,
        "stimuli": [{"port": p, "rate_hz": r} for p, r in stimuli],
        "target_port": target_port,
        "seeds": [int(s) for s in seeds],
        "ticks": ticks,
        "simulated_seconds_per_eval": round(elapsed_s, 4),
        "total_evaluations": total_evals,
        "wall_seconds": round(wall_s, 3),
        "brain_seconds_per_wall_second": round(brain_seconds / max(wall_s, 1e-9), 2),
        "baseline": {
            "mean_hz": round(base_mean, 3),
            "sd_hz": round(base_desc["sd"] or 0.0, 3),
            "per_seed_hz": [round(v, 3) for v in base_vals],
        },
        "results": results,
    }
