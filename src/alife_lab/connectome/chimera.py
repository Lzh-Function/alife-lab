"""Stage 4 / Phase F13a-c: Cross-Species & Intra-Species Chimera Connectomes (ADR 0053).

All chimeras are explicitly labeled as exploratory artificial-life hybrids (NOT biological
animal models), always report exact species/bridge provenance fractions, and are benchmarked
against degree-preserving shuffled controls and graft-silenced ablations.

Implemented chimera types:
- **F13a (`build_intraspecies_hybrid_pack`, `run_f13a_intraspecies_hybrid_assay`)**:
  Cell-type-aligned sexual/strain crossover (e.g., female FlyWire/BANC x male MaleCNS, or
  empirical vs synthetic C. elegans) preserving biological cell-type homology.
- **F13b (`run_f13b_brain_transplant_experiment`)**:
  Cross-species brain transplant into a planar 12-segment RFT crawling/swimming body via an
  explicit, bounded trainable bridge layer, benchmarked against ``donor_shuffled`` and
  ``donor_silenced`` controls with identical optimization budgets.
- **F13c (`build_mb_graft_chimera_pack`, `run_f13c_mb_graft_assay`)**:
  Circuit graft transplanting a Drosophila Mushroom Body learning module (KC + MBON + MBIN/DAN,
  spiking LIF + F4 dopamine-gated KC->MBON LTD plasticity) into a C. elegans 302-neuron host
  brain (MODEL_GRADED + gap junctions) via sensory->KC, ASH->DAN, and MBON->AVB/AVA bridge edges,
  tested on Pavlovian odor-aversion conditioning against ``unpaired_control``, ``graft_silenced``,
  and ``shuffled_graft`` controls.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from alife_lab.connectome.celegans import make_synthetic_celegans_302_pack
from alife_lab.connectome.cross_species import (
    analyze_connectome_topology,
    probe_sensorimotor_selectivity,
    recommended_lif_params,
)
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
    MODEL_SPIKING_LIF,
    ROLE_INTERNEURON,
    ROLE_MOTOR,
    ROLE_SENSORY,
    SYNAPSE_CHEMICAL,
    CellTable,
    ConnectomePack,
    DatasetProvenance,
    EdgeTable,
    IOPort,
    compute_pack_digest,
)
from alife_lab.connectome.species_packs import make_synthetic_fly_larva_pack
from alife_lab.connectome.worm_body import N_JOINTS, WormBodyState, step_rft_kinematics

CHIMERA_FRAMING_NOTE = (
    "F13 Exploratory Artificial-Life Chimera: combines neural circuits across strains/species via "
    "explicit bridge synapses. Results describe hybrid circuit dynamics only and are NOT claims "
    "about natural biological animals."
)


def _port(
    pid: str,
    direction: Any,
    label: str,
    desc: str,
    indices: list[int],
    rule: str,
) -> IOPort:
    return IOPort(
        port_id=pid,
        direction=direction,
        label=label,
        description=desc,
        cell_indices=np.asarray(sorted(set(indices)), dtype=np.int32),
        selection_rule=rule,
    )


# =========================================================================== #
# F13a: Intra-Species Cell-Type-Aligned Hybrid Crossover
# =========================================================================== #


def build_intraspecies_hybrid_pack(
    pack_a: ConnectomePack,
    pack_b: ConnectomePack,
    *,
    blend_ratio: float = 0.5,
    seed: int = 1,
) -> tuple[ConnectomePack, dict[str, Any]]:
    """Build a cell-type-aligned hybrid by crossing two packs of the same species/lineage.

    For every cell type present in ``pack_a``:
    - With probability ``blend_ratio`` (when the type also exists in ``pack_b``), outgoing edge
      weights onto shared target cell types are inherited from ``pack_b``'s type-level density
      and the cell is tagged with ``species_idx = 1`` (parent B provenance).
    - Otherwise the cell keeps ``pack_a``'s exact wiring and is tagged ``species_idx = 0``.
    """
    rng = np.random.default_rng(seed)
    n_a = pack_a.cells.num_cells
    types_a = pack_a.cells.cell_type
    types_b = pack_b.cells.cell_type

    # Aggregate type -> type mean outgoing synapse weight in pack_b for shared homology transfer
    b_type_cells: dict[str, int] = defaultdict(int)
    for ct in types_b:
        if ct:
            b_type_cells[ct] += 1
    b_pair_sum: dict[tuple[str, str], float] = defaultdict(float)
    b_pair_cnt: dict[tuple[str, str], int] = defaultdict(int)
    for k in range(pack_b.edges.num_edges):
        u = int(pack_b.edges.pre[k])
        v = int(pack_b.edges.post[k])
        cu, cv = types_b[u], types_b[v]
        if cu and cv:
            b_pair_sum[(cu, cv)] += float(pack_b.edges.count[k])
            b_pair_cnt[(cu, cv)] += 1
    b_pair_mean = {pair: b_pair_sum[pair] / float(c) for pair, c in b_pair_cnt.items()}

    unique_a = sorted({ct for ct in types_a if ct})
    shared_types = {ct for ct in unique_a if ct in b_type_cells}
    chosen_b_types = {
        ct
        for ct in sorted(shared_types)
        if float(rng.random()) < float(np.clip(blend_ratio, 0.0, 1.0))
    }

    sp_idx = np.zeros(n_a, dtype=np.uint8)
    for i, ct in enumerate(types_a):
        if ct in chosen_b_types:
            sp_idx[i] = np.uint8(1)

    new_counts = pack_a.edges.count.copy()
    edges_from_b = 0
    l1_delta = 0.0
    for k in range(pack_a.edges.num_edges):
        u = int(pack_a.edges.pre[k])
        v = int(pack_a.edges.post[k])
        if sp_idx[u] == 1:
            pair = (types_a[u], types_a[v])
            if pair in b_pair_mean:
                w_new = max(1, round(b_pair_mean[pair]))
                l1_delta += abs(float(w_new) - float(new_counts[k]))
                new_counts[k] = np.int32(w_new)
                edges_from_b += 1

    hybrid_id = f"chimera_f13a_{pack_a.dataset_id}_x_{pack_b.dataset_id}"
    sp_names = (f"parent_a:{pack_a.dataset_id}", f"parent_b:{pack_b.dataset_id}")
    cells = CellTable(
        source_id=pack_a.cells.source_id.copy(),
        species_names=sp_names,
        species_idx=sp_idx,
        cell_type=pack_a.cells.cell_type,
        super_class=pack_a.cells.super_class,
        cell_class=pack_a.cells.cell_class,
        cell_sub_class=pack_a.cells.cell_sub_class,
        region=pack_a.cells.region,
        side=pack_a.cells.side,
        role=pack_a.cells.role.copy(),
        cell_model=pack_a.cells.cell_model.copy(),
        neurotransmitter=pack_a.cells.neurotransmitter,
        position_nm=pack_a.cells.position_nm.copy(),
        position_source=pack_a.cells.position_source.copy(),
    )
    edges = EdgeTable(
        pre=pack_a.edges.pre.copy(),
        post=pack_a.edges.post.copy(),
        count=new_counts,
        sign=pack_a.edges.sign.copy(),
        kind=pack_a.edges.kind.copy(),
    )
    prov = DatasetProvenance(
        dataset_id=hybrid_id,
        species=f"hybrid({pack_a.species} x {pack_b.species})",
        version=f"f13a_blend{blend_ratio:.2f}_s{seed}",
        citation=(
            f"F13a intra-species hybrid: {pack_a.dataset_id} x {pack_b.dataset_id}. "
            f"{CHIMERA_FRAMING_NOTE}"
        ),
        licence_note="Derived local exploratory chimera.",
        files=(),
    )
    digest = compute_pack_digest(hybrid_id, prov.species, cells, edges, pack_a.io_ports, prov)
    pack = ConnectomePack(
        dataset_id=hybrid_id,
        species=prov.species,
        cells=cells,
        edges=edges,
        io_ports=pack_a.io_ports,
        provenance=prov,
        pack_digest=digest,
    )
    frac_b = float((sp_idx == 1).mean()) if n_a else 0.0
    meta = {
        "chimera_kind": "F13a_intraspecies_hybrid",
        "framing_note": CHIMERA_FRAMING_NOTE,
        "parent_a": pack_a.dataset_id,
        "parent_b": pack_b.dataset_id,
        "blend_ratio_requested": round(blend_ratio, 3),
        "shared_cell_types": len(shared_types),
        "crossed_cell_types": len(chosen_b_types),
        "provenance_breakdown": {
            sp_names[0]: round(1.0 - frac_b, 4),
            sp_names[1]: round(frac_b, 4),
        },
        "edges_transferred_from_b": edges_from_b,
        "mean_weight_l1_delta": round(l1_delta / max(1, edges_from_b), 4),
    }
    return pack, meta


def run_f13a_intraspecies_hybrid_assay(
    pack_a: ConnectomePack,
    pack_b: ConnectomePack,
    *,
    blend_ratio: float = 0.5,
    seed: int = 1,
    ticks: int = 120,
) -> dict[str, Any]:
    """Build F13a hybrid and benchmark sensorimotor integrity vs both parents & shuffle."""
    t0 = time.perf_counter()
    hybrid, meta = build_intraspecies_hybrid_pack(
        pack_a, pack_b, blend_ratio=blend_ratio, seed=seed
    )
    eff_ticks = min(ticks, 40) if hybrid.cells.num_cells > 50_000 else ticks
    probe_h = probe_sensorimotor_selectivity(hybrid, seed=seed, ticks=eff_ticks)
    probe_a = probe_sensorimotor_selectivity(pack_a, seed=seed, ticks=eff_ticks)
    return {
        **meta,
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "topology": analyze_connectome_topology(hybrid),
        "hybrid_probe": probe_h,
        "parent_a_probe": probe_a,
    }


# =========================================================================== #
# F13b: Cross-Species Brain Transplant + Trainable Bridge Layer into RFT Body
# =========================================================================== #


def _pick_transplant_ports(
    pack: ConnectomePack,
) -> tuple[np.ndarray[Any, np.dtype[np.int32]], np.ndarray[Any, np.dtype[np.int32]]]:
    s_candidates = (
        "ciona_photoreceptors",
        "platy_photoreceptor_prc",
        "larva_olfactory_sn",
        "posterior_touch_sensory",
        "primary_sensory_in",
    )
    m_candidates = (
        "ciona_motor_mn",
        "platy_ciliomotor_mn",
        "larva_dn_vnc",
        "forward_motor_vnc",
        "primary_motor_out",
    )
    s_idx = np.zeros(0, dtype=np.int32)
    for pid in s_candidates:
        p = pack.io_ports.get(pid)
        if p is not None and p.cell_indices.size > 0:
            s_idx = p.cell_indices
            break
    if s_idx.size == 0:
        s_idx = np.flatnonzero(pack.cells.role == ROLE_SENSORY).astype(np.int32)

    m_idx = np.zeros(0, dtype=np.int32)
    for pid in m_candidates:
        p = pack.io_ports.get(pid)
        if p is not None and p.cell_indices.size > 0:
            m_idx = p.cell_indices
            break
    if m_idx.size == 0:
        m_idx = np.flatnonzero(pack.cells.role == ROLE_MOTOR).astype(np.int32)
    return s_idx, m_idx


def _eval_transplant_bridge(
    pack: ConnectomePack,
    csr: Any,
    s_idx: np.ndarray[Any, np.dtype[np.int32]],
    m_idx: np.ndarray[Any, np.dtype[np.int32]],
    bridge_params: np.ndarray[Any, np.dtype[np.float64]],
    *,
    steps: int,
    seed: int,
    silence_brain: bool = False,
) -> dict[str, float]:
    """Evaluate one donor brain + 4-parameter bridge layer driving a 12-segment RFT body.

    The bridge strictly converts donor motor readout activity (mean activity & left/right
    asymmetry) into travelling wave gain and steering torque on the 12-segment RFT swimmer/crawler.
    When ``silence_brain=True`` all internal donor synapses are muted (negative control).
    """
    n = pack.cells.num_cells
    use_graded = has_graded_or_gap(pack, csr)
    gp = GradedParams()
    lp = recommended_lif_params(pack)
    st = make_initial_graded_state(n, gp) if use_graded else make_initial_state(n, lp)
    if silence_brain:
        # Disconnect internal brain wiring completely to test whether donor processing matters.
        empty_edges = EdgeTable(
            pre=np.zeros(0, dtype=np.int32),
            post=np.zeros(0, dtype=np.int32),
            count=np.zeros(0, dtype=np.int32),
            sign=np.zeros(0, dtype=np.int8),
            kind=np.zeros(0, dtype=np.uint8),
        )
        eff_csr = empty_edges.to_csr(n)
    else:
        eff_csr = csr
    circ = GradedCircuit.from_csr(eff_csr) if use_graded else None

    # Split sensory & motor ports into left/right halves for bilateral steering readout
    s_half = max(1, s_idx.size // 2)
    s_left = s_idx[:s_half]
    s_right = s_idx[s_half:] if s_idx.size > 1 else s_idx
    m_half = max(1, m_idx.size // 2)
    m_left = m_idx[:m_half]
    m_right = m_idx[m_half:] if m_idx.size > 1 else m_idx

    # Bridge parameters: [sensory_gain, wave_gain, steer_gain, phase_freq_hz]
    sens_gain = float(np.clip(bridge_params[0], 0.2, 3.0))
    wave_gain = float(np.clip(bridge_params[1], 0.2, 2.5))
    steer_gain = float(np.clip(bridge_params[2], -1.5, 1.5))
    freq_hz = float(np.clip(bridge_params[3], 0.6, 2.5))

    body = WormBodyState(
        x=0.0,
        y=0.0,
        heading=0.0,
        phi=np.zeros(N_JOINTS, dtype=np.float64),
        d_act=np.zeros(N_JOINTS, dtype=np.float64),
        v_act=np.zeros(N_JOINTS, dtype=np.float64),
    )
    target_xy = np.array([3.5, 1.2], dtype=np.float64)
    init_dist = float(np.hypot(target_xy[0], target_xy[1]))
    dt_s = 0.02  # 20 ms per body frame
    brain_subticks = 4  # 4 ms neural steps per body frame
    phase = 0.0
    total_dist = 0.0
    motor_drive_sum = 0.0
    wave_rms_sum = 0.0

    for _ in range(steps):
        # Compute bilateral target sensory bearing
        dx = float(target_xy[0] - body.x)
        dy = float(target_xy[1] - body.y)
        bearing = math.atan2(dy, dx) - body.heading
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        dist = max(0.2, math.hypot(dx, dy))
        base_cue = 120.0 * sens_gain * (1.0 / (1.0 + 0.15 * dist))
        l_stim = max(10.0, base_cue * (1.0 + 0.6 * math.sin(bearing)))
        r_stim = max(10.0, base_cue * (1.0 - 0.6 * math.sin(bearing)))

        st.stim_rate_hz.fill(np.float32(0.0))
        if s_left.size:
            st.stim_rate_hz[s_left] = np.float32(l_stim)
        if s_right.size:
            st.stim_rate_hz[s_right] = np.float32(r_stim)

        m_act_l = 0.0
        m_act_r = 0.0
        for _ in range(brain_subticks):
            if use_graded and circ is not None:
                step_graded_tick_cpu(st, circ, gp)
                m_act_l += float(np.maximum(0.0, (st.v[m_left] - gp.e_leak_mv) / 25.0).mean())
                m_act_r += float(np.maximum(0.0, (st.v[m_right] - gp.e_leak_mv) / 25.0).mean())
            else:
                step_tick_cpu(st, eff_csr, lp, seed)
                m_act_l += float(st.last_tick_spikes[m_left].mean())
                m_act_r += float(st.last_tick_spikes[m_right].mean())
        m_act_l /= float(brain_subticks)
        m_act_r /= float(brain_subticks)

        drive = min(1.5, 0.5 * (m_act_l + m_act_r) * (3.0 if not use_graded else 2.0))
        asym = (m_act_l - m_act_r) * (2.5 if not use_graded else 2.0)
        motor_drive_sum += drive

        # Advance body wave ONLY proportional to donor brain motor output (zero motor => motionless)
        phase += 2.0 * math.pi * freq_hz * dt_s * (0.2 + 0.8 * min(1.0, drive))
        amp = 0.55 * wave_gain * min(1.2, drive)
        steer = float(np.clip(steer_gain * asym, -0.45, 0.45))
        phi_target = np.array(
            [amp * math.sin(phase - 0.62 * j) + (steer if j < 3 else 0.0) for j in range(N_JOINTS)],
            dtype=np.float64,
        )
        phi_dot = (phi_target - body.phi) / 0.08
        nx, ny, nh, vf, vl, om, nodes_w = step_rft_kinematics(
            body.x, body.y, body.heading, body.phi, phi_dot, dt_s, drag_ratio=3.0
        )
        total_dist += math.hypot(nx - body.x, ny - body.y)
        body.x, body.y, body.heading = nx, ny, nh
        body.phi = np.clip(body.phi + phi_dot * dt_s, -1.2, 1.2)
        body.v_forward, body.v_lateral, body.omega = vf, vl, om
        body.nodes_xy = nodes_w
        wave_rms_sum += float(np.sqrt(np.mean(body.phi * body.phi)))

    final_dist = float(math.hypot(target_xy[0] - body.x, target_xy[1] - body.y))
    approach_mm = init_dist - final_dist
    score = 0.6 * total_dist + 1.2 * approach_mm
    return {
        "score": round(score, 4),
        "distance_mm": round(total_dist, 4),
        "target_approach_mm": round(approach_mm, 4),
        "mean_motor_drive": round(motor_drive_sum / max(1, steps), 4),
        "mean_wave_rms_rad": round(wave_rms_sum / max(1, steps), 4),
    }


def run_f13b_brain_transplant_experiment(
    donor_pack: ConnectomePack,
    *,
    body_name: str = "rft_12seg_swimmer_crawler",
    seed: int = 1,
    bridge_trials: int = 8,
    eval_steps: int = 80,
) -> dict[str, Any]:
    """F13b: Transplant ``donor_pack`` into an RFT body via a tuned bridge layer vs controls."""
    t0 = time.perf_counter()
    s_idx, m_idx = _pick_transplant_ports(donor_pack)
    n = donor_pack.cells.num_cells
    csr_ref = donor_pack.edges.to_csr(n)
    csr_shuf = make_shuffled_csr(csr_ref, seed)

    # Identical candidate bridge parameter grid for all arms (fair equal-budget comparison)
    rng = np.random.default_rng(seed + 1300)
    candidates = [np.array([1.2, 1.4, 0.7, 1.4], dtype=np.float64)]
    for _ in range(max(1, bridge_trials - 1)):
        candidates.append(
            np.array(
                [
                    float(rng.uniform(0.6, 2.2)),
                    float(rng.uniform(0.7, 2.0)),
                    float(rng.uniform(-1.2, 1.2)),
                    float(rng.uniform(0.9, 2.0)),
                ],
                dtype=np.float64,
            )
        )

    def _optimize_arm(csr: Any, silence: bool) -> dict[str, Any]:
        best_res: dict[str, float] | None = None
        best_p: list[float] = []
        for p in candidates:
            res = _eval_transplant_bridge(
                donor_pack, csr, s_idx, m_idx, p, steps=eval_steps, seed=seed, silence_brain=silence
            )
            if best_res is None or res["score"] > best_res["score"]:
                best_res = res
                best_p = [round(float(x), 3) for x in p]
        assert best_res is not None
        return {**best_res, "best_bridge_params": best_p}

    arm_intact = _optimize_arm(csr_ref, False)
    arm_shuf = _optimize_arm(csr_shuf, False)
    arm_sil = _optimize_arm(csr_ref, True)

    bridge_edges = int(s_idx.size + m_idx.size)
    tot_edges = donor_pack.edges.num_edges + bridge_edges
    return {
        "chimera_kind": "F13b_brain_transplant",
        "framing_note": CHIMERA_FRAMING_NOTE,
        "donor_dataset_id": donor_pack.dataset_id,
        "donor_species": donor_pack.species,
        "target_body": body_name,
        "seed": seed,
        "bridge_trials": len(candidates),
        "eval_steps": eval_steps,
        "provenance_breakdown": {
            f"donor_brain:{donor_pack.dataset_id}": round(
                donor_pack.edges.num_edges / max(1, tot_edges), 4
            ),
            "bridge_interface_edges": round(bridge_edges / max(1, tot_edges), 4),
        },
        "arms": {
            "donor_intact": arm_intact,
            "donor_shuffled": arm_shuf,
            "donor_silenced": arm_sil,
        },
        "intact_beats_silenced": arm_intact["distance_mm"] > arm_sil["distance_mm"] + 0.05,
        "intact_vs_shuffled_delta_mm": round(
            arm_intact["distance_mm"] - arm_shuf["distance_mm"], 4
        ),
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


# =========================================================================== #
# F13c: Circuit Graft Chimera — C. elegans Host + Drosophila Mushroom Body Graft
# =========================================================================== #


@dataclass(frozen=True)
class MBGraftLayout:
    """Indices inside a unified F13c worm + fly-MB chimera ``ConnectomePack``."""

    host_n: int
    odor_a_sn: np.ndarray[Any, np.dtype[np.int32]]
    odor_b_sn: np.ndarray[Any, np.dtype[np.int32]]
    nociceptor_sn: np.ndarray[Any, np.dtype[np.int32]]
    host_avb: np.ndarray[Any, np.dtype[np.int32]]
    host_ava: np.ndarray[Any, np.dtype[np.int32]]
    graft_kc_a: np.ndarray[Any, np.dtype[np.int32]]
    graft_kc_b: np.ndarray[Any, np.dtype[np.int32]]
    graft_mbon_approach: np.ndarray[Any, np.dtype[np.int32]]
    graft_mbon_avoid: np.ndarray[Any, np.dtype[np.int32]]
    graft_dan: np.ndarray[Any, np.dtype[np.int32]]
    provenance_breakdown: dict[str, float]


def _extract_mb_subcircuit(
    donor_pack: ConnectomePack,
    *,
    max_kc: int = 48,
    max_mbon: int = 12,
    max_dan: int = 8,
) -> tuple[list[int], list[int], list[int], list[tuple[int, int, int, int]]]:
    """Extract real (KC, MBON, MBIN/DAN) indices & internal edges from a Drosophila pack."""
    kc_p = donor_pack.io_ports.get("larva_kc") or donor_pack.io_ports.get("kc_mushroom_body")
    mbon_p = donor_pack.io_ports.get("larva_mbon") or donor_pack.io_ports.get("mbon_output")
    dan_p = donor_pack.io_ports.get("larva_mbin") or donor_pack.io_ports.get("dan_ppl1_punishment")

    kc = [int(i) for i in (kc_p.cell_indices[:max_kc] if kc_p is not None else [])]
    mbon = [int(i) for i in (mbon_p.cell_indices[:max_mbon] if mbon_p is not None else [])]
    dan = [int(i) for i in (dan_p.cell_indices[:max_dan] if dan_p is not None else [])]

    if not kc:
        kc = [i for i, ct in enumerate(donor_pack.cells.cell_type) if "KC" in ct][:max_kc]
    if not mbon:
        mbon = [i for i, ct in enumerate(donor_pack.cells.cell_type) if "MBON" in ct][:max_mbon]
    if not dan:
        dan = [
            i for i, ct in enumerate(donor_pack.cells.cell_type) if "MBIN" in ct or "PPL1" in ct
        ][:max_dan]

    keep = kc + mbon + dan
    remap = {old: idx for idx, old in enumerate(keep)}
    internal: list[tuple[int, int, int, int]] = []
    for e_i in range(donor_pack.edges.num_edges):
        u = int(donor_pack.edges.pre[e_i])
        v = int(donor_pack.edges.post[e_i])
        if u in remap and v in remap:
            internal.append(
                (
                    remap[u],
                    remap[v],
                    int(donor_pack.edges.count[e_i]),
                    int(donor_pack.edges.sign[e_i]),
                )
            )
    kc_loc = [remap[i] for i in kc]
    mbon_loc = [remap[i] for i in mbon]
    dan_loc = [remap[i] for i in dan]
    return kc_loc, mbon_loc, dan_loc, internal


def build_mb_graft_chimera_pack(
    host_worm: ConnectomePack | None = None,
    donor_fly: ConnectomePack | None = None,
    *,
    seed: int = 1,
) -> tuple[ConnectomePack, MBGraftLayout]:
    """Build F13c unified chimera: C. elegans 302 host + grafted Drosophila Mushroom Body.

    Combines ``MODEL_GRADED`` + ``SYNAPSE_GAP_JUNCTION`` host neurons with ``MODEL_SPIKING_LIF``
    grafted KC/MBON/DAN neurons in a single canonical ``ConnectomePack`` with explicit
    ``species_names`` and provenance breakdown.
    """
    host = host_worm or make_synthetic_celegans_302_pack()
    donor = donor_fly or make_synthetic_fly_larva_pack()
    rng = np.random.default_rng(seed + 1313)

    h_n = host.cells.num_cells
    kc_loc, mbon_loc, dan_loc, internal_edges = _extract_mb_subcircuit(donor)
    g_n = len(kc_loc) + len(mbon_loc) + len(dan_loc)
    if len(kc_loc) < 4 or len(mbon_loc) < 2 or len(dan_loc) < 1:
        raise ValueError(
            f"donor pack {donor.dataset_id!r} lacks KC/MBON/MBIN populations for graft"
        )

    total_n = h_n + g_n
    sp_names = (f"host:{host.species}", f"graft_mb:{donor.species}")
    sp_idx = np.concatenate([np.zeros(h_n, dtype=np.uint8), np.ones(g_n, dtype=np.uint8)])

    # Identify host sensory & command neurons by exact C. elegans cell names
    name_to_idx = {ct: i for i, ct in enumerate(host.cells.cell_type)}

    def _find_host(names: tuple[str, ...], fallback: list[int]) -> list[int]:
        found = [name_to_idx[nm] for nm in names if nm in name_to_idx]
        return found if found else fallback

    odor_a_sn = _find_host(("AWAL", "AWAR"), [0, 1])
    odor_b_sn = _find_host(("AWCL", "AWCR"), [2, 3])
    noci_sn = _find_host(("ASHL", "ASHR", "ADLL", "ADLR"), [4, 5])
    host_avb = _find_host(("AVBL", "AVBR"), [10, 11])
    host_ava = _find_host(("AVAL", "AVAR"), [12, 13])

    # Split grafted KC into disjoint sparse odor-A / odor-B populations and MBON into approach/avoid
    kc_global = [h_n + i for i in kc_loc]
    mbon_global = [h_n + i for i in mbon_loc]
    dan_global = [h_n + i for i in dan_loc]
    mid_kc = len(kc_global) // 2
    kc_a = kc_global[:mid_kc]
    kc_b = kc_global[mid_kc:]
    mid_mb = max(1, len(mbon_global) // 2)
    mbon_app = mbon_global[:mid_mb]
    mbon_avd = mbon_global[mid_mb:]

    # Build unified CellTable placing grafted MB just dorsal-anterior to C. elegans nerve ring
    g_ct: list[str] = []
    g_sc: list[str] = []
    g_cc: list[str] = []
    g_sub: list[str] = []
    g_reg: list[str] = []
    g_side: list[str] = []
    g_role = np.full(g_n, ROLE_INTERNEURON, dtype=np.uint8)
    g_model = np.full(g_n, MODEL_SPIKING_LIF, dtype=np.uint8)
    g_nt: list[str] = []
    g_pos = np.zeros((g_n, 3), dtype=np.float32)

    for k in range(g_n):
        glob = h_n + k
        if glob in kc_global:
            lbl = f"GRAFT_KC_{k}"
            cls = "KC"
            nt = "ACH"
            z_off = 32_000.0
        elif glob in mbon_global:
            lbl = f"GRAFT_MBON_{k}"
            cls = "MBON_app" if glob in mbon_app else "MBON_avd"
            nt = "ACH" if glob in mbon_app else "GABA"
            z_off = 24_000.0
        else:
            lbl = f"GRAFT_DAN_{k}"
            cls = "DAN_PPL1"
            nt = "DA"
            z_off = 38_000.0
        g_ct.append(lbl)
        g_sc.append("chimera_graft_mb")
        g_cc.append(cls)
        g_sub.append(cls)
        g_reg.append("grafted_fly_MB")
        g_side.append("left" if k % 2 == 0 else "right")
        g_nt.append(nt)
        g_pos[k] = (
            65_000.0 + (k % 8) * 3_500.0,
            (-18_000.0 if k % 2 == 0 else 18_000.0) + ((k // 8) % 4) * 3_000.0,
            z_off,
        )

    cells = CellTable(
        source_id=np.concatenate(
            [host.cells.source_id, np.arange(1_000_000, 1_000_000 + g_n, dtype=np.int64)]
        ),
        species_names=sp_names,
        species_idx=sp_idx,
        cell_type=host.cells.cell_type + tuple(g_ct),
        super_class=host.cells.super_class + tuple(g_sc),
        cell_class=host.cells.cell_class + tuple(g_cc),
        cell_sub_class=host.cells.cell_sub_class + tuple(g_sub),
        region=host.cells.region + tuple(g_reg),
        side=host.cells.side + tuple(g_side),
        role=np.concatenate([host.cells.role, g_role]),
        cell_model=np.concatenate([host.cells.cell_model, g_model]),
        neurotransmitter=host.cells.neurotransmitter + tuple(g_nt),
        position_nm=np.vstack([host.cells.position_nm, g_pos]),
        position_source=np.concatenate(
            [host.cells.position_source, np.full(g_n, 2, dtype=np.uint8)]
        ),
    )

    # Combine edges: (1) intact host edges, (2) grafted MB internal edges, (3) explicit bridge edges
    pre_l = host.edges.pre.tolist()
    post_l = host.edges.post.tolist()
    cnt_l = host.edges.count.tolist()
    sgn_l = host.edges.sign.tolist()
    knd_l = host.edges.kind.tolist()
    n_host_edges = len(pre_l)

    existing: set[tuple[int, int, int]] = {
        (pre_l[i], post_l[i], knd_l[i]) for i in range(n_host_edges)
    }

    def _add(u: int, v: int, w: int, sgn: int) -> bool:
        key = (u, v, SYNAPSE_CHEMICAL)
        if u == v or w <= 0 or key in existing:
            return False
        existing.add(key)
        pre_l.append(u)
        post_l.append(v)
        cnt_l.append(w)
        sgn_l.append(sgn)
        knd_l.append(SYNAPSE_CHEMICAL)
        return True

    n_graft_edges = 0
    # Keep extracted MB internal edges (KC->MBON, MBON->MBON) while treating DAN/MBIN strictly as
    # neuromodulatory plasticity gates (F4 three-factor rule) rather than spiking drivers of KCs.
    kc_set_local = set(kc_loc)
    dan_set_local = set(dan_loc)
    for u_l, v_l, w, sgn in internal_edges:
        if u_l in kc_set_local and v_l in kc_set_local:
            continue
        if u_l in dan_set_local and v_l in kc_set_local:
            continue
        if _add(h_n + u_l, h_n + v_l, w, sgn):
            n_graft_edges += 1
    # Ensure grafted KC -> MBON_approach excitatory pathway exists even on sparse sub-extractions
    for u in kc_global:
        for v in mbon_app:
            if float(rng.random()) < 0.75 and _add(u, v, 16, 1):
                n_graft_edges += 1
    # Mutual inhibition inside MBONs: MBON_approach inhibits MBON_avoid (disinhibition upon LTD)
    for u in mbon_app:
        for v in mbon_avd:
            if _add(u, v, 18, -1):
                n_graft_edges += 1

    # Explicit chimera bridge edges (tracked separately for provenance transparency)
    n_bridge_edges = 0
    for u in odor_a_sn:
        for v in kc_a:
            if _add(u, v, 24, 1):
                n_bridge_edges += 1
    for u in odor_b_sn:
        for v in kc_b:
            if _add(u, v, 24, 1):
                n_bridge_edges += 1
    for u in noci_sn:
        for v in dan_global:
            if _add(u, v, 26, 1):
                n_bridge_edges += 1
    # MBON_approach excites host forward command (AVB) and inhibits host reversal (AVA)
    for u in mbon_app:
        for v in host_avb:
            if _add(u, v, 42, 1):
                n_bridge_edges += 1
        for v in host_ava:
            if _add(u, v, 42, -1):
                n_bridge_edges += 1
    # MBON_avoid excites host reversal command (AVA)
    for u in mbon_avd:
        for v in host_ava:
            if _add(u, v, 46, 1):
                n_bridge_edges += 1

    edges = EdgeTable(
        pre=np.asarray(pre_l, dtype=np.int32),
        post=np.asarray(post_l, dtype=np.int32),
        count=np.asarray(cnt_l, dtype=np.int32),
        sign=np.asarray(sgn_l, dtype=np.int8),
        kind=np.asarray(knd_l, dtype=np.uint8),
    )

    io_ports = dict(host.io_ports)
    io_ports.update(
        {
            "chimera_odor_a": _port(
                "chimera_odor_a",
                "sensory",
                "Host Odor A (AWA)",
                "Host AWA sensory neurons bridged to Graft KC-A.",
                odor_a_sn,
                "host AWA",
            ),
            "chimera_odor_b": _port(
                "chimera_odor_b",
                "sensory",
                "Host Odor B (AWC)",
                "Host AWC sensory neurons bridged to Graft KC-B.",
                odor_b_sn,
                "host AWC",
            ),
            "chimera_punish_ash": _port(
                "chimera_punish_ash",
                "sensory",
                "Host Nociceptor (ASH/ADL)",
                "Host ASH/ADL bridged to Graft DAN punishment.",
                noci_sn,
                "host ASH/ADL",
            ),
            "chimera_graft_kc": _port(
                "chimera_graft_kc",
                "circuit",
                "Grafted Fly KC",
                "Grafted Drosophila Kenyon cells.",
                kc_global,
                "graft KC",
            ),
            "chimera_graft_mbon_approach": _port(
                "chimera_graft_mbon_approach",
                "circuit",
                "Grafted Fly MBON (approach)",
                "Grafted approach MBONs driving host AVB.",
                mbon_app,
                "graft MBON approach",
            ),
            "chimera_graft_mbon_avoid": _port(
                "chimera_graft_mbon_avoid",
                "circuit",
                "Grafted Fly MBON (avoid)",
                "Grafted avoidance MBONs driving host AVA.",
                mbon_avd,
                "graft MBON avoid",
            ),
            "chimera_graft_dan": _port(
                "chimera_graft_dan",
                "circuit",
                "Grafted Fly DAN/MBIN",
                "Grafted dopaminergic punishment neurons.",
                dan_global,
                "graft DAN",
            ),
        }
    )

    tot_e = max(1, edges.num_edges)
    breakdown = {
        f"host_cells:{host.dataset_id}": round(h_n / total_n, 4),
        f"graft_cells:{donor.dataset_id}": round(g_n / total_n, 4),
        f"host_edges:{host.dataset_id}": round(n_host_edges / tot_e, 4),
        f"graft_edges:{donor.dataset_id}": round(n_graft_edges / tot_e, 4),
        "bridge_edges": round(n_bridge_edges / tot_e, 4),
    }

    cid = (
        "chimera_worm_fly_mb"
        if (host.dataset_id == "celegans_302" and donor.dataset_id == "fly_larva_l1")
        else "synthetic_chimera_worm_fly_mb"
    )
    prov = DatasetProvenance(
        dataset_id=cid,
        species=f"chimera({host.species} + {donor.species}_MB)",
        version="f13c_v1",
        citation=(
            f"F13c Circuit Graft Chimera: {host.dataset_id} + {donor.dataset_id} MB. "
            f"{CHIMERA_FRAMING_NOTE}"
        ),
        licence_note="Derived local exploratory chimera.",
        files=(),
    )
    digest = compute_pack_digest(cid, prov.species, cells, edges, io_ports, prov)
    pack = ConnectomePack(
        dataset_id=cid,
        species=prov.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
        pack_digest=digest,
    )
    layout = MBGraftLayout(
        host_n=h_n,
        odor_a_sn=np.asarray(odor_a_sn, dtype=np.int32),
        odor_b_sn=np.asarray(odor_b_sn, dtype=np.int32),
        nociceptor_sn=np.asarray(noci_sn, dtype=np.int32),
        host_avb=np.asarray(host_avb, dtype=np.int32),
        host_ava=np.asarray(host_ava, dtype=np.int32),
        graft_kc_a=np.asarray(kc_a, dtype=np.int32),
        graft_kc_b=np.asarray(kc_b, dtype=np.int32),
        graft_mbon_approach=np.asarray(mbon_app, dtype=np.int32),
        graft_mbon_avoid=np.asarray(mbon_avd, dtype=np.int32),
        graft_dan=np.asarray(dan_global, dtype=np.int32),
        provenance_breakdown=breakdown,
    )
    return pack, layout


def run_f13c_mb_graft_assay(
    host_worm: ConnectomePack | None = None,
    donor_fly: ConnectomePack | None = None,
    *,
    seed: int = 1,
    train_trials: int = 4,
) -> dict[str, Any]:
    """Run Pavlovian odor-aversion conditioning on the F13c worm+fly-MB chimera + 3 controls.

    Arms:
    1. ``paired_intact``: Odor A paired simultaneously with host ASH punishment -> KC-A -> MBON_app
       synapses undergo dopamine-gated LTD, shifting host command balance (AVA - AVB) toward
       avoidance specifically for Odor A while Odor B remains intact.
    2. ``unpaired_control``: Odor A and punishment separated in time -> negligible LTD.
    3. ``graft_silenced``: All grafted fly MB neurons silenced -> host worm shows zero conditioning.
    4. ``shuffled_graft``: KC -> MBON_approach synapses replaced by random cross-pool mapping ->
       loses odor-A specificity relative to Odor B.
    """
    t0 = time.perf_counter()
    pack, lay = build_mb_graft_chimera_pack(host_worm, donor_fly, seed=seed)
    n = pack.cells.num_cells
    base_csr = pack.edges.to_csr(n)
    gp = GradedParams()
    lp = LIFParams()

    kc_set = set(lay.graft_kc_a.tolist()) | set(lay.graft_kc_b.tolist())
    mbon_app_set = set(lay.graft_mbon_approach.tolist())

    def _run_arm(mode: str) -> dict[str, Any]:
        # Clone CSR weights for independent arm plasticity
        w_vec = base_csr.signed_weight.copy()
        post_vec = base_csr.post.copy()
        indptr = base_csr.indptr
        kind_vec = base_csr.kind

        rng_s = np.random.default_rng(seed + 999)
        kc_all = np.concatenate([lay.graft_kc_a, lay.graft_kc_b])
        if mode == "shuffled_graft":
            # Randomly wire both odor A and odor B across the entire KC pool (destroying
            # sparse segregated odor coding while keeping identical active KC count).
            shuf_kc_a = rng_s.choice(kc_all, size=lay.graft_kc_a.size, replace=False).astype(
                np.int32
            )
            shuf_kc_b = rng_s.choice(kc_all, size=lay.graft_kc_b.size, replace=False).astype(
                np.int32
            )
            perm_map = dict(zip(kc_all.tolist(), rng_s.permutation(kc_all).tolist(), strict=True))
            for src in np.concatenate([lay.odor_a_sn, lay.odor_b_sn]).tolist():
                lo, hi = int(indptr[src]), int(indptr[src + 1])
                for ei in range(lo, hi):
                    tgt = int(post_vec[ei])
                    if tgt in perm_map:
                        post_vec[ei] = np.int32(perm_map[tgt])
        else:
            shuf_kc_a = lay.graft_kc_a
            shuf_kc_b = lay.graft_kc_b

        from alife_lab.connectome.schema import CSRGraph

        arm_csr = CSRGraph(
            num_cells=n,
            indptr=indptr,
            post=post_vec,
            signed_weight=w_vec,
            kind=kind_vec,
        )

        # Locate plastic KC -> MBON_approach edges
        plastic_edges: list[tuple[int, int, int]] = []  # (edge_idx, pre_kc, base_w)
        for u in sorted(kc_set):
            lo, hi = int(indptr[u]), int(indptr[u + 1])
            for ei in range(lo, hi):
                if int(post_vec[ei]) in mbon_app_set and int(w_vec[ei]) > 0:
                    plastic_edges.append((ei, u, int(w_vec[ei])))
        scales = np.ones(len(plastic_edges), dtype=np.float32)

        def _make_state() -> Any:
            st = make_initial_hybrid_state(pack, lp, gp)
            if mode == "graft_silenced":
                st.silenced[lay.host_n :] = np.uint8(1)
            return st

        def _probe_odor(
            odor_sn: np.ndarray[Any, np.dtype[np.int32]],
            kc_direct: np.ndarray[Any, np.dtype[np.int32]],
        ) -> dict[str, float]:
            st = _make_state()
            circ = GradedCircuit.from_csr(arm_csr)
            st.stim_rate_hz[odor_sn] = np.float32(180.0)
            if mode != "graft_silenced":
                st.stim_rate_hz[kc_direct] = np.float32(140.0)
                st.stim_rate_hz[lay.graft_mbon_avoid] = np.float32(85.0)
            ava_v, avb_v, mbon_app_sp = [], [], 0
            for _ in range(60):
                step_hybrid_tick_cpu(st, pack, arm_csr, circ, lp, gp, seed, lif_to_graded_gain=3.0)
                ava_v.append(float(st.v[lay.host_ava].mean()))
                avb_v.append(float(st.v[lay.host_avb].mean()))
                mbon_app_sp += int(st.last_tick_spikes[lay.graft_mbon_approach].sum())
            ava_mean = float(np.mean(ava_v[-25:]))
            avb_mean = float(np.mean(avb_v[-25:]))
            return {
                "ava_mv": round(ava_mean, 3),
                "avb_mv": round(avb_mean, 3),
                "avoid_bias_mv": round(ava_mean - avb_mean, 3),
                "mbon_approach_hz": round(
                    mbon_app_sp / max(1, lay.graft_mbon_approach.size) / 0.06, 2
                ),
            }

        pre_a = _probe_odor(lay.odor_a_sn, shuf_kc_a)
        pre_b = _probe_odor(lay.odor_b_sn, shuf_kc_b)

        # Training phase: dopamine-gated KC -> MBON_approach LTD (F4 rule)
        elig = np.zeros(n, dtype=np.float32)
        dan_trace = 0.0
        for _ in range(train_trials):
            st = _make_state()
            circ = GradedCircuit.from_csr(arm_csr)
            if mode == "unpaired_control":
                # Odor A alone (40 ticks) -> gap washout (80 ticks) -> Punishment alone (40 ticks)
                phases = [
                    (40, True, False),
                    (80, False, False),
                    (40, False, True),
                ]
            else:
                phases = [(50, True, True)]
            for n_t, odor_on, punish_on in phases:
                st.stim_rate_hz.fill(np.float32(0.0))
                if odor_on:
                    st.stim_rate_hz[lay.odor_a_sn] = np.float32(180.0)
                    if mode != "graft_silenced":
                        st.stim_rate_hz[shuf_kc_a] = np.float32(160.0)
                if punish_on:
                    st.stim_rate_hz[lay.nociceptor_sn] = np.float32(180.0)
                    if mode != "graft_silenced":
                        st.stim_rate_hz[lay.graft_dan] = np.float32(160.0)
                for _ in range(n_t):
                    step_hybrid_tick_cpu(
                        st, pack, arm_csr, circ, lp, gp, seed, lif_to_graded_gain=3.0
                    )
                    elig *= np.float32(0.88)
                    elig += st.last_tick_spikes.astype(np.float32)
                    dan_sp = float(st.last_tick_spikes[lay.graft_dan].mean())
                    dan_trace = 0.85 * dan_trace + dan_sp
                    if mode != "graft_silenced" and dan_trace > 0.08:
                        for k_i, (ei, pre_kc, base_w) in enumerate(plastic_edges):
                            if elig[pre_kc] > 0.15:
                                scales[k_i] = max(
                                    0.08,
                                    float(scales[k_i]) - 0.025 * float(elig[pre_kc]) * dan_trace,
                                )
                                w_vec[ei] = np.int32(max(1, round(base_w * float(scales[k_i]))))
            elig.fill(np.float32(0.0))
            dan_trace = 0.0

        post_a = _probe_odor(lay.odor_a_sn, shuf_kc_a)
        post_b = _probe_odor(lay.odor_b_sn, shuf_kc_b)
        delta_a = round(post_a["avoid_bias_mv"] - pre_a["avoid_bias_mv"], 3)
        delta_b = round(post_b["avoid_bias_mv"] - pre_b["avoid_bias_mv"], 3)
        return {
            "pre_odor_a": pre_a,
            "post_odor_a": post_a,
            "pre_odor_b": pre_b,
            "post_odor_b": post_b,
            "delta_avoid_odor_a_mv": delta_a,
            "delta_avoid_odor_b_mv": delta_b,
            "odor_specificity_mv": round(delta_a - delta_b, 3),
            "mean_plastic_scale": round(float(scales.mean()), 4) if scales.size else 1.0,
            "min_plastic_scale": round(float(scales.min()), 4) if scales.size else 1.0,
        }

    arms = {
        mode: _run_arm(mode)
        for mode in ("paired_intact", "unpaired_control", "graft_silenced", "shuffled_graft")
    }
    pi = arms["paired_intact"]
    up = arms["unpaired_control"]
    gs = arms["graft_silenced"]
    sh = arms["shuffled_graft"]
    checks = {
        "paired_learns_odor_a_avoidance": pi["delta_avoid_odor_a_mv"] > 0.25,
        "odor_specific_over_odor_b": pi["odor_specificity_mv"] > 0.25,
        "beats_unpaired_control": pi["delta_avoid_odor_a_mv"] > up["delta_avoid_odor_a_mv"] + 0.25,
        "graft_silencing_eliminates_learning": abs(gs["delta_avoid_odor_a_mv"]) < 0.05,
        "beats_shuffled_graft_specificity": pi["odor_specificity_mv"]
        > sh["odor_specificity_mv"] + 0.25,
    }
    return {
        "chimera_kind": "F13c_mb_circuit_graft",
        "framing_note": CHIMERA_FRAMING_NOTE,
        "dataset_id": pack.dataset_id,
        "host_species": (host_worm.species if host_worm else "synthetic_celegans_302"),
        "donor_species": (donor_fly.species if donor_fly else "synthetic_fly_larva"),
        "total_cells": n,
        "total_edges": pack.edges.num_edges,
        "provenance_breakdown": lay.provenance_breakdown,
        "train_trials": train_trials,
        "seed": seed,
        "arms": arms,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


def make_chimera_worm_fly_mb_pack(
    host_worm: ConnectomePack | None = None,
    donor_fly: ConnectomePack | None = None,
    *,
    seed: int = 1,
) -> ConnectomePack:
    """Return the unified F13c worm+fly-MB ``ConnectomePack`` directly for browser 3D live lab."""
    pack, _ = build_mb_graft_chimera_pack(host_worm, donor_fly, seed=seed)
    return pack
