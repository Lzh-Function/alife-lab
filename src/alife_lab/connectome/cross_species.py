"""Stage 4 / S2-S4 Cross-Species Comparative Connectome Statistics & Sensorimotor Probe.

Compares biological connectomes across phylogenetic scales (Ciona 205 -> C. elegans 302 ->
Platynereis 1,720 -> Fly Larva L1 2,956 -> Adult Fly 138k-165k) using identical structural
metrics and paired degree-preserving ``shuffled_control`` physiological probes.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from alife_lab.connectome.graded import (
    GradedCircuit,
    GradedParams,
    has_graded_or_gap,
    make_initial_graded_state,
    make_initial_hybrid_state,
    step_graded_tick_cpu,
    step_hybrid_tick_cpu,
)
from alife_lab.connectome.lif import (
    LIFParams,
    make_initial_state,
    make_shuffled_csr,
    step_tick_cpu,
)
from alife_lab.connectome.schema import (
    CELL_MODEL_NAMES,
    CELL_ROLE_NAMES,
    MODEL_GRADED,
    MODEL_SPIKING_LIF,
    ROLE_MOTOR,
    ROLE_SENSORY,
    SYNAPSE_CHEMICAL,
    SYNAPSE_GAP_JUNCTION,
    ConnectomePack,
)


def _pick_ports(
    pack: ConnectomePack,
) -> tuple[str, np.ndarray[Any, np.dtype[np.int32]], str, np.ndarray[Any, np.dtype[np.int32]]]:
    """Select canonical (sensory_port, sensory_idx, motor_port, motor_idx) for a pack."""
    sensory_order = (
        "posterior_touch_sensory",
        "larva_olfactory_sn",
        "platy_photoreceptor_prc",
        "ciona_photoreceptors",
        "sugar_grn",
        "primary_sensory_in",
        "olfactory_sn",
    )
    motor_order = (
        "forward_motor_vnc",
        "larva_dn_vnc",
        "platy_ciliomotor_mn",
        "ciona_motor_mn",
        "mn9_proboscis_extension",
        "primary_motor_out",
    )
    s_name, s_idx = "", np.zeros(0, dtype=np.int32)
    for pid in sensory_order:
        p = pack.io_ports.get(pid)
        if p is not None and p.cell_indices.size > 0:
            s_name, s_idx = pid, p.cell_indices
            break
    if s_idx.size == 0:
        s_idx = np.flatnonzero(pack.cells.role == ROLE_SENSORY).astype(np.int32)
        s_name = "role_sensory"

    m_name, m_idx = "", np.zeros(0, dtype=np.int32)
    for pid in motor_order:
        p = pack.io_ports.get(pid)
        if p is not None and p.cell_indices.size > 0:
            m_name, m_idx = pid, p.cell_indices
            break
    if m_idx.size == 0:
        m_idx = np.flatnonzero(pack.cells.role == ROLE_MOTOR).astype(np.int32)
        m_name = "role_motor"
    return s_name, s_idx, m_name, m_idx


def analyze_connectome_topology(pack: ConnectomePack) -> dict[str, Any]:
    """Compute species-agnostic structural graph metrics on any ``ConnectomePack``."""
    n = pack.cells.num_cells
    e = pack.edges.num_edges
    pre = pack.edges.pre.astype(np.int64)
    post = pack.edges.post.astype(np.int64)
    cnt = pack.edges.count.astype(np.int64)
    total_syn = int(cnt.sum()) if e else 0

    density = float(e) / float(n * (n - 1)) if n > 1 else 0.0
    out_deg = np.bincount(pre, minlength=n) if e else np.zeros(n, dtype=np.int64)
    in_deg = np.bincount(post, minlength=n) if e else np.zeros(n, dtype=np.int64)
    out_syn = np.bincount(pre, weights=cnt, minlength=n) if e else np.zeros(n, dtype=np.float64)
    in_syn = np.bincount(post, weights=cnt, minlength=n) if e else np.zeros(n, dtype=np.float64)
    tot_cell_syn = out_syn + in_syn
    k_hub = max(1, int(np.ceil(0.01 * n)))
    top1_share = (
        float(np.sort(tot_cell_syn)[-k_hub:].sum()) / float(max(1.0, tot_cell_syn.sum()))
        if n > 0
        else 0.0
    )

    # Exact directed reciprocity: fraction of edges (u, v) whose reverse (v, u) also exists.
    if e > 0:
        fwd = (pre << np.int64(32)) | post
        rev = (post << np.int64(32)) | pre
        fwd_sorted = np.sort(fwd)
        pos = np.searchsorted(fwd_sorted, rev)
        valid = pos < fwd_sorted.size
        recip_mask = np.zeros(e, dtype=bool)
        recip_mask[valid] = fwd_sorted[pos[valid]] == rev[valid]
        reciprocity = float(recip_mask.mean())
    else:
        reciprocity = 0.0

    chem_mask = pack.edges.kind == SYNAPSE_CHEMICAL
    gap_mask = pack.edges.kind == SYNAPSE_GAP_JUNCTION
    n_chem = int(chem_mask.sum())
    n_exc = int(((pack.edges.sign > 0) & chem_mask).sum())
    exc_frac = float(n_exc) / float(max(1, n_chem))
    gap_frac = float(int(gap_mask.sum())) / float(max(1, e))

    role_fracs = {
        name: round(float((pack.cells.role == i).mean()), 4) if n else 0.0
        for i, name in enumerate(CELL_ROLE_NAMES)
    }
    model_fracs = {
        name: round(float((pack.cells.cell_model == i).mean()), 4) if n else 0.0
        for i, name in enumerate(CELL_MODEL_NAMES)
    }
    species_fracs = {
        sp: round(float((pack.cells.species_idx == i).mean()), 4) if n else 0.0
        for i, sp in enumerate(pack.cells.species_names)
    }

    return {
        "dataset_id": pack.dataset_id,
        "species": pack.species,
        "num_cells": n,
        "num_edges": e,
        "total_synapses": total_syn,
        "density": round(density, 6),
        "mean_out_degree": round(float(out_deg.mean()), 3) if n else 0.0,
        "std_out_degree": round(float(out_deg.std()), 3) if n else 0.0,
        "max_in_degree": int(in_deg.max()) if n else 0,
        "max_out_degree": int(out_deg.max()) if n else 0,
        "top1pct_hub_synapse_share": round(top1_share, 4),
        "reciprocity": round(reciprocity, 4),
        "excitatory_chem_fraction": round(exc_frac, 4),
        "gap_junction_edge_fraction": round(gap_frac, 4),
        "role_fractions": role_fracs,
        "cell_model_fractions": model_fracs,
        "species_provenance_fractions": species_fracs,
    }


def recommended_lif_params(pack: ConnectomePack) -> LIFParams:
    """Return species-scaled LIF parameters (adult fly keeps exact Shiu et al. 2024 default).

    Small nervous systems (Ciona 205 cells, Fly Larva 2,956 cells) have fewer converging inputs
    per neuron than the 138k-cell adult fly brain, so single-synapse voltage gain ``w_syn_mv`` is
    explicitly scaled for excitability while applied identically to reference and shuffled arms.
    """
    did = pack.dataset_id
    if did in ("ciona_larva", "synthetic_ciona"):
        return LIFParams(w_syn_mv=1.45)
    if did in ("fly_larva_l1", "synthetic_fly_larva"):
        return LIFParams(w_syn_mv=0.85)
    return LIFParams()


def probe_sensorimotor_selectivity(
    pack: ConnectomePack,
    *,
    seed: int = 1,
    ticks: int = 150,
    stim_hz: float = 150.0,
) -> dict[str, Any]:
    """Compare sensory->motor propagation on reference wiring vs degree-preserving shuffle."""
    n = pack.cells.num_cells
    s_name, s_idx, m_name, m_idx = _pick_ports(pack)
    csr_ref = pack.edges.to_csr(n)
    csr_shuf = make_shuffled_csr(csr_ref, seed)

    use_graded = has_graded_or_gap(pack, csr_ref)
    has_lif = bool(np.any(pack.cells.cell_model == MODEL_SPIKING_LIF))
    has_grad = bool(np.any(pack.cells.cell_model == MODEL_GRADED))
    lp = recommended_lif_params(pack)

    def _run(csr: Any) -> tuple[float, float]:
        if use_graded:
            gp = GradedParams()
            st = (
                make_initial_hybrid_state(pack, lp, gp)
                if (has_lif and has_grad)
                else make_initial_graded_state(n, gp)
            )
            if s_idx.size:
                st.stim_rate_hz[s_idx] = np.float32(stim_hz)
            circ = GradedCircuit.from_csr(csr)
            for _ in range(ticks):
                if has_lif and has_grad:
                    step_hybrid_tick_cpu(st, pack, csr, circ, lp, gp, seed)
                else:
                    step_graded_tick_cpu(st, circ, gp)
            motor_dv = float(st.v[m_idx].mean() - gp.e_leak_mv) if m_idx.size else 0.0
            pop_dv = float(st.v.mean() - gp.e_leak_mv) if n else 0.0
            return motor_dv, pop_dv

        st = make_initial_state(n, lp)
        if s_idx.size:
            st.stim_rate_hz[s_idx] = np.float32(stim_hz)
        for _ in range(ticks):
            step_tick_cpu(st, csr, lp, seed)
        dur_s = max(1e-3, ticks * 1e-3)
        motor_hz = float(st.total_spikes[m_idx].mean()) / dur_s if m_idx.size else 0.0
        pop_hz = float(st.total_spikes.mean()) / dur_s if n else 0.0
        return motor_hz, pop_hz

    ref_m, ref_pop = _run(csr_ref)
    shuf_m, shuf_pop = _run(csr_shuf)
    unit = "delta_mV" if use_graded else "Hz"
    return {
        "dataset_id": pack.dataset_id,
        "dynamics_mode": (
            "hybrid" if (has_lif and has_grad) else ("graded" if use_graded else "spiking_lif")
        ),
        "w_syn_mv": lp.w_syn_mv,
        "unit": unit,
        "sensory_port": s_name,
        "sensory_count": int(s_idx.size),
        "motor_port": m_name,
        "motor_count": int(m_idx.size),
        "ticks": ticks,
        "reference_motor_response": round(ref_m, 4),
        "shuffled_motor_response": round(shuf_m, 4),
        "reference_pop_response": round(ref_pop, 4),
        "shuffled_pop_response": round(shuf_pop, 4),
        "wiring_delta": round(ref_m - shuf_m, 4),
        "selectivity_ratio": round(ref_m / max(1e-4, abs(shuf_m)), 3) if abs(ref_m) > 0 else 0.0,
    }


def compare_species_connectomes(
    packs: dict[str, ConnectomePack],
    *,
    seed: int = 1,
    ticks: int = 150,
    run_physiology: bool = True,
) -> dict[str, Any]:
    """Run uniform structural and physiological comparison across multiple species packs."""
    t0 = time.perf_counter()
    ordered = sorted(packs.items(), key=lambda kv: kv[1].cells.num_cells)
    topo_rows = [analyze_connectome_topology(p) for _, p in ordered]
    phys_rows: list[dict[str, Any]] = []
    if run_physiology:
        for _, p in ordered:
            # Cap CPU probe ticks cleanly for huge adult CNS packs (>50k cells) unless small
            eff_ticks = min(ticks, 50) if p.cells.num_cells > 50_000 else ticks
            phys_rows.append(probe_sensorimotor_selectivity(p, seed=seed, ticks=eff_ticks))

    return {
        "datasets": [name for name, _ in ordered],
        "seed": seed,
        "ticks": ticks,
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "topology": topo_rows,
        "sensorimotor_probe": phys_rows,
    }
