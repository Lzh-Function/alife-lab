"""Phase F10-B: Cross-Sex Comparative Connectome Analysis (Female FlyWire/BANC vs Male MaleCNS).

Compares anatomical cell-type homology, sexually dimorphic courtship/song/receptivity circuits
(male ``pIP10 / pMP2 / dPR1 / vPR6`` vs female ``pC1a-e``), and identical-stimulus spiking LIF
responses across female (``flywire_783``, ``banc_888``) and male (``male_cns_v1``) connectomes
alongside degree-preserving ``shuffled_control`` arms.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np

from alife_lab.connectome.embodied import resolve_embodied_ports
from alife_lab.connectome.lif import (
    LIFParams,
    make_initial_state,
    make_shuffled_csr,
    step_tick_cpu,
)
from alife_lab.connectome.schema import ConnectomePack, CSRGraph

DIMORPHIC_MARKERS: tuple[str, ...] = (
    # Male-specific courtship song pathway (Janelia / MaleCNS)
    "pIP10",
    "pMP2",
    "dPR1",
    "vPR6",
    "vMS11",
    # Female-specific courtship receptivity / aggression cluster (FlyWire / BANC)
    "pC1a",
    "pC1b",
    "pC1c",
    "pC1d",
    "pC1e",
    # Shared descending & motor control anchors
    "DNp09",
    "DNa01",
    "DNa02",
    "DNg84",
    "MDN",
    "CB0701",
    "MN9",
)


def _run_stim_rates(
    pack: ConnectomePack,
    csr: CSRGraph,
    stim_ports: list[tuple[str, float]],
    *,
    backend: str,
    seed: int,
    ticks: int,
) -> dict[str, float]:
    """Run one identical-stimulus LIF trial and return mean rates (Hz) at canonical readouts."""
    params = LIFParams()
    n = pack.cells.num_cells
    ports = resolve_embodied_ports(pack)
    stim = np.zeros(n, dtype=np.float32)
    for pid, hz in stim_ports:
        if pid in pack.io_ports and pack.io_ports[pid].cell_indices.size > 0:
            stim[pack.io_ports[pid].cell_indices] = np.float32(hz)

    if backend == "cupy":
        from alife_lab.connectome.lif_cupy import CupyLIFRunner

        lif = make_initial_state(n, params)
        lif.stim_rate_hz[...] = stim
        runner = CupyLIFRunner(csr, params, seed=seed)
        runner.upload_state(lif)
        runner.step_substeps(ticks * params.substeps_per_tick, use_graph=True)
        runner.download_state(lif)
        totals = lif.total_spikes
    else:
        lif = make_initial_state(n, params)
        lif.stim_rate_hz[...] = stim
        for _ in range(ticks):
            step_tick_cpu(lif, csr, params, seed)
        totals = lif.total_spikes

    dur_s = max(1e-3, ticks * 1e-3)

    def _rate(idx: np.ndarray[Any, np.dtype[np.int32]]) -> float:
        if idx.size == 0:
            return 0.0
        return round(float(totals[idx].mean()) / dur_s, 2)

    out: dict[str, float] = {
        "mn9_hz": _rate(ports.proboscis_mn),
        "dng84_hz": _rate(ports.grooming_dn),
        "dnp09_hz": _rate(ports.forward_dn),
        "total_mean_hz": round(float(totals.mean()) / dur_s, 3),
    }
    if ports.has_vnc_ports:
        leg_idx = np.concatenate(
            [ports.all_command_union[lo:hi] for (lo, hi) in ports.leg_mn_slices.values() if hi > lo]
        )
        out["vnc_leg_mn_hz"] = _rate(leg_idx)
    else:
        out["vnc_leg_mn_hz"] = -1.0
    return out


def compare_sex_connectomes(
    packs: dict[str, ConnectomePack],
    *,
    backend: str = "auto",
    seed: int = 1,
    ticks: int = 300,
) -> dict[str, Any]:
    """Compare anatomy & identical-stimulus physiology across female and male connectomes."""
    t0 = time.perf_counter()
    if backend == "auto":
        from alife_lab.connectome.lif_cupy import available as cupy_ok

        use_gpu = cupy_ok()
    else:
        use_gpu = backend == "cupy"

    # 1. Cell-type homology & sexually dimorphic marker inventory
    counters = {name: Counter(p.cells.cell_type) for name, p in packs.items()}
    type_sets = {name: {ct for ct in c if ct} for name, c in counters.items()}
    names = list(packs.keys())
    pairwise_overlap: dict[str, int] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise_overlap[f"{a}__x__{b}"] = len(type_sets[a] & type_sets[b])
    all_shared = len(set.intersection(*type_sets.values())) if type_sets else 0

    anatomy: dict[str, Any] = {}
    for name, p in packs.items():
        exc = int((p.edges.sign > 0).sum())
        inh = int((p.edges.sign < 0).sum())
        anatomy[name] = {
            "num_cells": p.cells.num_cells,
            "num_edges": p.edges.num_edges,
            "unique_cell_types": len(type_sets[name]),
            "excitatory_edge_ratio": round(exc / max(1, exc + inh), 4),
            "dimorphic_markers": {m: int(counters[name].get(m, 0)) for m in DIMORPHIC_MARKERS},
            "jo_auditory_cells": int(
                sum(cnt for ct, cnt in counters[name].items() if ct.startswith("JO-B"))
            ),
        }

    # 2. Identical-stimulus physiological comparison (reference + shuffled_control)
    protocols: list[tuple[str, list[tuple[str, float]]]] = [
        ("sugar_150hz", [("sugar_grn", 150.0)]),
        ("sugar_plus_bitter", [("sugar_grn", 150.0), ("bitter_grn", 200.0)]),
        ("antennal_poke_180hz", [("antennal_grooming_mechanosensory", 180.0)]),
        ("forward_dn_150hz", [("forward_dn", 150.0)]),
    ]
    physiology: dict[str, Any] = {}
    for name, p in packs.items():
        eff_b = "cupy" if (use_gpu and p.cells.num_cells > 500) else "numpy"
        csr_ref = p.edges.to_csr(p.cells.num_cells)
        csr_shuf = make_shuffled_csr(csr_ref, seed)
        pack_res: dict[str, Any] = {"backend": eff_b, "reference": {}, "shuffled_control": {}}
        for proto_name, stim_ports in protocols:
            pack_res["reference"][proto_name] = _run_stim_rates(
                p, csr_ref, stim_ports, backend=eff_b, seed=seed, ticks=ticks
            )
            pack_res["shuffled_control"][proto_name] = _run_stim_rates(
                p, csr_shuf, stim_ports, backend=eff_b, seed=seed, ticks=ticks
            )
        sug = pack_res["reference"]["sugar_150hz"]["mn9_hz"]
        bit = pack_res["reference"]["sugar_plus_bitter"]["mn9_hz"]
        pack_res["bitter_suppression_pct"] = round(100.0 * (sug - bit) / sug, 1) if sug > 0 else 0.0
        physiology[name] = pack_res

    return {
        "datasets": names,
        "ticks": ticks,
        "seed": seed,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "shared_cell_types_all": all_shared,
        "pairwise_shared_cell_types": pairwise_overlap,
        "anatomy": anatomy,
        "physiology": physiology,
    }
