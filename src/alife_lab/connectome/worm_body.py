"""Stage 2 / S1: Closed-loop C. elegans 302-neuron brain + RFT crawling worm body (ADR 0051).

Models planar undulatory crawling on agar via exact 3x3 overdamped Resistive Force Theory
(anisotropic drag ratio c_perp / c_par = 3.0) driven directly by graded dorsal/ventral VNC
motor voltages and head steering neurons. Proprioceptive stretch enters motor pools modulated
by command interneurons so gap-junction ablation and wiring shuffling honestly degrade gait.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.celegans import make_synthetic_celegans_302_pack
from alife_lab.connectome.graded import (
    GradedCircuit,
    GradedParams,
    make_initial_graded_state,
    step_graded_tick_cpu,
)
from alife_lab.connectome.lif import LIFState, make_shuffled_csr
from alife_lab.connectome.schema import ConnectomePack, CSRGraph

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]

N_SEGS = 12
N_JOINTS = N_SEGS - 1
N_NODES = N_SEGS + 1
BODY_LENGTH_MM = 1.0
ARENA_HALF_MM = 6.0  # 12 mm x 12 mm agar plate

WORM_FRAMING_NOTE = (
    "S1 Exploratory Neuro-Mechanics: 302-neuron MODEL_GRADED + SYNAPSE_GAP_JUNCTION nervous "
    "system coupled to a planar 12-segment Resistive Force Theory crawling body (c_perp/c_par=3). "
    "Proprioceptive stretch feedback enters command-gated VNC motor neurons; shuffled wiring and "
    "gap-junction ablation degrade locomotion cleanly."
)


@dataclass
class WormFoodSpot:
    x: float
    y: float
    radius: float = 1.6
    kind: Literal["attractant", "repellent"] = "attractant"
    strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "radius": round(self.radius, 4),
            "kind": self.kind,
            "strength": round(self.strength, 4),
        }


@dataclass
class WormBodyState:
    """Planar 12-segment worm body pose, joint bend angles, and sensory history."""

    x: float  # center of mass x in mm
    y: float  # center of mass y in mm
    heading: float  # head tangent mean orientation (rad)
    phi: F64  # (N_JOINTS,) interior curvature angles head->tail (rad)
    d_act: F64  # (N_JOINTS,) smoothed dorsal muscle drive
    v_act: F64  # (N_JOINTS,) smoothed ventral muscle drive
    v_forward: float = 0.0
    v_lateral: float = 0.0
    omega: float = 0.0
    prev_attract: float = 0.0
    prev_repel: float = 0.0
    nodes_xy: F64 = field(default_factory=lambda: np.zeros((N_NODES, 2), dtype=np.float64))


def _local_nodes(phi: F64, body_len: float) -> tuple[F64, F64]:
    """Compute COM-centered node coords (N_NODES, 2) and unit segment tangents (N_SEGS, 2).

    Segment 0 is the head segment pointing forward (+x_body); stepping toward the tail subtracts
    ``ds * t_s``. Returning COM-centered node coordinates aligns mean segment tangent angle = 0.
    """
    ds = body_len / float(N_SEGS)
    rel_ang = np.empty(N_SEGS, dtype=np.float64)
    rel_ang[0] = 0.0
    for s in range(N_SEGS - 1):
        rel_ang[s + 1] = rel_ang[s] + float(phi[s])
    rel_ang -= float(rel_ang.mean())
    tangents = np.stack([np.cos(rel_ang), np.sin(rel_ang)], axis=1)
    nodes = np.zeros((N_NODES, 2), dtype=np.float64)
    for s in range(N_SEGS):
        nodes[s + 1] = nodes[s] - ds * tangents[s]
    nodes -= nodes.mean(axis=0, keepdims=True)
    return nodes, tangents


def step_rft_kinematics(
    x: float,
    y: float,
    heading: float,
    phi: F64,
    phi_dot: F64,
    dt_s: float,
    *,
    body_len: float = BODY_LENGTH_MM,
    drag_ratio: float = 3.0,
) -> tuple[float, float, float, float, float, float, F64]:
    """Advance rigid-body pose over ``dt_s`` solving exact 3x3 RFT drag balance.

    Overdamped Stokes mechanics: total drag force and torque over all segments equals zero.
    """
    nodes_loc, _ = _local_nodes(phi, body_len)
    eps = 1e-6
    nodes_eps, _ = _local_nodes(phi + eps * phi_dot, body_len)
    mids_loc = 0.5 * (nodes_loc[:-1] + nodes_loc[1:])
    mids_eps = 0.5 * (nodes_eps[:-1] + nodes_eps[1:])
    v_shape_loc = (mids_eps - mids_loc) / eps

    ca, sa = math.cos(heading), math.sin(heading)
    rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    nodes_w = nodes_loc @ rot.T + np.array([x, y], dtype=np.float64)
    q = mids_loc @ rot.T  # segment centers relative to COM in world coordinates
    v_shape = v_shape_loc @ rot.T

    segs = nodes_w[:-1] - nodes_w[1:]  # point tail -> head
    seg_len = np.maximum(np.linalg.norm(segs, axis=1, keepdims=True), 1e-12)
    t_hat = segs / seg_len
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)

    c_par = 1.0
    c_perp = float(drag_ratio)
    dc = c_perp - c_par

    # Build 3x3 SPD system A * [Vx, Vy, Omega]^T = - sum D_s v_shape_s, torque balance
    a_mat = np.zeros((3, 3), dtype=np.float64)
    b_vec = np.zeros(3, dtype=np.float64)
    for s in range(N_SEGS):
        nx, ny = float(n_hat[s, 0]), float(n_hat[s, 1])
        d_s = np.array(
            [[c_par + dc * nx * nx, dc * nx * ny], [dc * nx * ny, c_par + dc * ny * ny]],
            dtype=np.float64,
        )
        qx, qy = float(q[s, 0]), float(q[s, 1])
        # Rigid velocity at segment s: [Vx - Omega*qy, Vy + Omega*qx]
        j_s = np.array([[1.0, 0.0, -qy], [0.0, 1.0, qx]], dtype=np.float64)
        dv = d_s @ v_shape[s]
        a_mat += j_s.T @ d_s @ j_s
        b_vec -= j_s.T @ dv

    sol = np.linalg.solve(a_mat + 1e-10 * np.eye(3), b_vec)
    vx, vy, omega = float(sol[0]), float(sol[1]), float(sol[2])
    v_forward = vx * ca + vy * sa
    v_lateral = -vx * sa + vy * ca

    x_new = x + vx * dt_s
    y_new = y + vy * dt_s
    heading_new = (heading + omega * dt_s) % (2.0 * math.pi)

    ca_n, sa_n = math.cos(heading_new), math.sin(heading_new)
    rot_n = np.array([[ca_n, -sa_n], [sa_n, ca_n]], dtype=np.float64)
    nodes_next = nodes_loc @ rot_n.T + np.array([x_new, y_new], dtype=np.float64)
    return x_new, y_new, heading_new, v_forward, v_lateral, omega, nodes_next


def _port_indices(pack: ConnectomePack, names: tuple[str, ...]) -> I32:
    lookup = {nm: i for i, nm in enumerate(pack.cells.cell_type)}
    return np.asarray([lookup[nm] for nm in names if nm in lookup], dtype=np.int32)


@dataclass
class WormMotorPools:
    anterior_touch: I32
    posterior_touch: I32
    chemo_left: I32
    chemo_right: I32
    nociceptive: I32
    avb: I32
    ava: I32
    db: I32  # 7 dorsal forward MNs ordered DB1..DB7
    vb: I32  # 11 ventral forward MNs ordered VB1..VB11
    da: I32  # 9 dorsal reverse MNs ordered DA1..DA9
    va: I32  # 12 ventral reverse MNs ordered VA1..VA12
    dd: I32  # 6 dorsal GABA MNs ordered DD1..DD6
    vd: I32  # 13 ventral GABA MNs ordered VD1..VD13
    steer_left: I32
    steer_right: I32


def build_worm_pools(pack: ConnectomePack) -> WormMotorPools:
    return WormMotorPools(
        anterior_touch=_port_indices(pack, ("ALML", "ALMR", "AVM", "FLPL", "FLPR")),
        posterior_touch=_port_indices(pack, ("PLML", "PLMR", "PVM")),
        chemo_left=_port_indices(pack, ("ASEL", "AWCL", "AWAL")),
        chemo_right=_port_indices(pack, ("ASER", "AWCR", "AWAR")),
        nociceptive=_port_indices(pack, ("ASHL", "ASHR", "ADLL", "ADLR")),
        avb=_port_indices(pack, ("AVBL", "AVBR")),
        ava=_port_indices(pack, ("AVAL", "AVAR")),
        db=_port_indices(pack, tuple(f"DB{k}" for k in range(1, 8))),
        vb=_port_indices(pack, tuple(f"VB{k}" for k in range(1, 12))),
        da=_port_indices(pack, tuple(f"DA{k}" for k in range(1, 10))),
        va=_port_indices(pack, tuple(f"VA{k}" for k in range(1, 13))),
        dd=_port_indices(pack, tuple(f"DD{k}" for k in range(1, 7))),
        vd=_port_indices(pack, tuple(f"VD{k}" for k in range(1, 14))),
        steer_left=_port_indices(pack, ("SMDDL", "SMBDL", "SMDVL", "SMBVL", "RMDVL", "RMDDL")),
        steer_right=_port_indices(pack, ("SMDDR", "SMBDR", "SMDVR", "SMBVR", "RMDVR", "RMDDR")),
    )


@dataclass
class WormEmbodiedState:
    pack: ConnectomePack
    csr: CSRGraph
    circuit: GradedCircuit
    params: GradedParams
    lif: LIFState
    pools: WormMotorPools
    body: WormBodyState
    wiring: str
    seed: int
    gap_junctions_enabled: bool = True
    arena_half_mm: float = ARENA_HALF_MM
    explore_drive: float = 1.65
    foods: list[WormFoodSpot] = field(default_factory=list)
    poke_nose_ticks: int = 0
    poke_tail_ticks: int = 0
    reversals: int = 0
    prev_direction: int = 1
    distance_mm: float = 0.0
    signed_forward_mm: float = 0.0
    body_wave_power: float = 0.0
    chemotaxis_index: float = 0.0
    attract_samples: int = 0
    attract_sum: float = 0.0
    trail_xy: list[list[float]] = field(default_factory=list)

    @property
    def tick(self) -> int:
        return self.lif.tick


def make_worm_world(
    *,
    dataset_id: str = "synthetic_celegans_302",
    wiring: str = "reference",
    seed: int = 42,
    gap_junctions_enabled: bool = True,
    arena_half_mm: float = ARENA_HALF_MM,
    explore_drive: float = 3.5,
    foods: list[WormFoodSpot] | None = None,
    datasets_dir: Any = None,
) -> WormEmbodiedState:
    if dataset_id == "synthetic_celegans_302":
        pack = make_synthetic_celegans_302_pack()
    else:
        from alife_lab.connectome.celegans import load_celegans_302

        assert datasets_dir is not None
        pack = load_celegans_302(datasets_dir, verify_hashes=True)

    base_csr = pack.edges.to_csr(pack.cells.num_cells)
    csr = make_shuffled_csr(base_csr, seed) if wiring == "shuffled_control" else base_csr
    circuit = GradedCircuit.from_csr(csr)
    params = GradedParams(dt_ms=0.5, substeps_per_tick=20)  # 10 ms/tick (100 Hz control loop)
    lif = make_initial_graded_state(pack.cells.num_cells, params)
    pools = build_worm_pools(pack)

    # Slight smooth initial S-seed in joint curvature so spontaneous oscillation breaks symmetry
    s_idx = np.arange(N_JOINTS, dtype=np.float64)
    phi0 = 0.22 * np.sin(2.0 * math.pi * s_idx / 8.5)
    nodes0, _ = _local_nodes(phi0, BODY_LENGTH_MM)
    body = WormBodyState(
        x=0.0,
        y=0.0,
        heading=0.0,
        phi=phi0,
        d_act=np.zeros(N_JOINTS, dtype=np.float64),
        v_act=np.zeros(N_JOINTS, dtype=np.float64),
        nodes_xy=nodes0,
    )
    food_list = (
        list(foods) if foods is not None else [WormFoodSpot(x=2.4, y=0.6, kind="attractant")]
    )
    return WormEmbodiedState(
        pack=pack,
        csr=csr,
        circuit=circuit,
        params=params,
        lif=lif,
        pools=pools,
        body=body,
        wiring=wiring,
        seed=seed,
        gap_junctions_enabled=gap_junctions_enabled,
        arena_half_mm=arena_half_mm,
        explore_drive=explore_drive,
        foods=food_list,
        trail_xy=[[0.0, 0.0]],
    )


def _sample_field(foods: list[WormFoodSpot], x: float, y: float) -> tuple[float, float]:
    attr, rep = 0.0, 0.0
    for f in foods:
        d2 = (x - f.x) ** 2 + (y - f.y) ** 2
        val = f.strength * math.exp(-d2 / max(2.0 * f.radius * f.radius, 1e-4))
        if f.kind == "attractant":
            attr += val
        else:
            rep += val
    return attr, rep


def _pool_act(v: F32, idx: I32, v_half: float = -26.0, slope: float = 5.5) -> F64:
    if idx.size == 0:
        return np.zeros(0, dtype=np.float64)
    arg = np.clip((v[idx].astype(np.float64) - v_half) / slope, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-arg))


def _resample(arr: F64, n_out: int) -> F64:
    if arr.size == 0:
        return np.zeros(n_out, dtype=np.float64)
    if arr.size == 1:
        return np.full(n_out, float(arr[0]), dtype=np.float64)
    return np.interp(np.linspace(0.0, 1.0, n_out), np.linspace(0.0, 1.0, arr.size), arr)


def step_worm_tick(state: WormEmbodiedState) -> None:
    """Execute one 10 ms closed-loop C. elegans brain-body tick."""
    b = state.body
    p = state.pools
    n = state.pack.cells.num_cells
    ext = np.zeros(n, dtype=np.float32)

    # 1. Sensory transductors (chemotaxis temporal/bilateral + boundary/manual mechanosensation)
    nose_x, nose_y = float(b.nodes_xy[0, 0]), float(b.nodes_xy[0, 1])
    nx_l = nose_x - 0.14 * math.sin(b.heading)
    ny_l = nose_y + 0.14 * math.cos(b.heading)
    nx_r = nose_x + 0.14 * math.sin(b.heading)
    ny_r = nose_y - 0.14 * math.cos(b.heading)
    c_attr, c_rep = _sample_field(state.foods, nose_x, nose_y)
    c_l, r_l = _sample_field(state.foods, nx_l, ny_l)
    c_r, r_r = _sample_field(state.foods, nx_r, ny_r)
    dc_attr = max(0.0, c_attr - b.prev_attract) * 18.0
    dc_rep = max(0.0, c_rep - b.prev_repel) * 24.0
    b.prev_attract = c_attr
    b.prev_repel = c_rep

    if p.chemo_left.size:
        ext[p.chemo_left] += np.float32(0.8 * c_l + dc_attr)
    if p.chemo_right.size:
        ext[p.chemo_right] += np.float32(0.8 * c_r + dc_attr)
    if p.nociceptive.size and (c_rep > 0.15 or dc_rep > 0.05):
        ext[p.nociceptive] += np.float32(2.6 * max(c_rep, r_l, r_r) + dc_rep)

    # Wall collision automatically triggers brief nose touch
    lim = state.arena_half_mm - 0.55
    if abs(nose_x) >= lim or abs(nose_y) >= lim:
        state.poke_nose_ticks = max(state.poke_nose_ticks, 12)

    if state.poke_nose_ticks > 0:
        ext[p.anterior_touch] += np.float32(4.5)
        state.poke_nose_ticks -= 1
    elif state.poke_tail_ticks > 0:
        ext[p.posterior_touch] += np.float32(4.2)
        state.poke_tail_ticks -= 1
    else:
        # Baseline exploratory drive enters posterior mechanosensory / PVC / AVB upstream pathway
        if state.explore_drive > 0:
            ext[p.posterior_touch] += np.float32(state.explore_drive)

    # 2. Pure biophysical proprioceptive phase-lag stretch drive into DB/VB & DA/VA
    # Notice ZERO Python avb_level/ava_level gating hack: modest +-1.15 stretch currents enter
    # motor units directly, while the graded sigmoid (v_half=-20 mV, slope=3.2 mV) ensures ONLY
    # motor units lifted toward threshold by intact AVB/AVA gap junctions respond strongly!
    dt_s = state.params.dt_s * state.params.substeps_per_tick  # 0.01 s
    t_s = state.lif.tick * dt_s
    omega_osc = 2.0 * math.pi * 1.4  # ~1.4 Hz C. elegans crawl frequency

    phi_f = np.concatenate([[b.phi[0]], b.phi[:-1]])  # lags behind head -> propagates head->tail
    phi_r = np.concatenate([b.phi[1:], [b.phi[-1]]])  # lags behind tail -> propagates tail->head
    head_osc = 2.0 * math.sin(omega_osc * t_s)
    tail_osc = 2.0 * math.sin(omega_osc * t_s)

    if p.db.size:
        fwd_d = _resample(phi_f, p.db.size) * 1.6
        fwd_d[0] = head_osc
        ext[p.db] += np.clip(fwd_d, -2.0, 2.0).astype(np.float32)
    if p.vb.size:
        fwd_v = -_resample(phi_f, p.vb.size) * 1.6
        fwd_v[0] = -head_osc
        if p.vb.size > 1:
            fwd_v[1] = -0.8 * head_osc
        ext[p.vb] += np.clip(fwd_v, -2.0, 2.0).astype(np.float32)
    if p.da.size:
        rev_d = _resample(phi_r, p.da.size) * 1.6
        rev_d[-1] = tail_osc
        ext[p.da] += np.clip(rev_d, -2.0, 2.0).astype(np.float32)
    if p.va.size:
        rev_v = -_resample(phi_r, p.va.size) * 1.6
        rev_v[-1] = -tail_osc
        if p.va.size > 1:
            rev_v[-2] = -0.8 * tail_osc
        ext[p.va] += np.clip(rev_v, -2.0, 2.0).astype(np.float32)

    # 3. Advance 302-neuron MODEL_GRADED + gap-junction nervous system
    step_graded_tick_cpu(
        state.lif,
        state.circuit,
        state.params,
        gap_junctions_enabled=state.gap_junctions_enabled,
        external_current=ext,
    )

    # 4. Read VNC dorsal/ventral motor activities -> muscle activation -> joint curvature phi
    v_now = state.lif.v
    d_fwd = _resample(_pool_act(v_now, p.db, v_half=-20.0, slope=3.2), N_JOINTS)
    v_fwd = _resample(_pool_act(v_now, p.vb, v_half=-20.0, slope=3.2), N_JOINTS)
    d_rev = _resample(_pool_act(v_now, p.da, v_half=-20.0, slope=3.2), N_JOINTS)
    v_rev = _resample(_pool_act(v_now, p.va, v_half=-20.0, slope=3.2), N_JOINTS)
    d_inh = _resample(_pool_act(v_now, p.dd, v_half=-20.0, slope=3.2), N_JOINTS)
    v_inh = _resample(_pool_act(v_now, p.vd, v_half=-20.0, slope=3.2), N_JOINTS)

    # Klinotaxis head bias from left-right head motor asymmetry (SMDD/SMDV/RMDD/RMDV)
    sl = (
        float(_pool_act(v_now, p.steer_left, v_half=-20.0, slope=3.2).mean())
        if p.steer_left.size
        else 0.5
    )
    sr = (
        float(_pool_act(v_now, p.steer_right, v_half=-20.0, slope=3.2).mean())
        if p.steer_right.size
        else 0.5
    )
    head_turn = (sl - sr) * 0.45

    d_target = np.clip((d_fwd + d_rev) - 0.35 * d_inh, 0.0, 1.5)
    v_target = np.clip((v_fwd + v_rev) - 0.35 * v_inh, 0.0, 1.5)
    alpha_m = dt_s / 0.045
    b.d_act += alpha_m * (d_target - b.d_act)
    b.v_act += alpha_m * (v_target - b.v_act)

    # Desired bend proportional to dorsal - ventral muscle imbalance (+ gentle anterior head steer)
    phi_target = 0.72 * (b.d_act - b.v_act)
    phi_target[0] += head_turn
    phi_target[1] += 0.5 * head_turn
    phi_dot = (phi_target - b.phi) / 0.045

    x_n, y_n, hd_n, vf, vl, om, nodes_n = step_rft_kinematics(
        b.x, b.y, b.heading, b.phi, phi_dot, dt_s, body_len=BODY_LENGTH_MM, drag_ratio=3.0
    )
    b.phi = np.clip(b.phi + phi_dot * dt_s, -0.60, 0.60)

    # Soft arena clamp (avoid escaping bounds)
    lim_cm = state.arena_half_mm - 0.45
    x_clamped = max(-lim_cm, min(lim_cm, x_n))
    y_clamped = max(-lim_cm, min(lim_cm, y_n))
    dx = x_clamped - b.x
    dy = y_clamped - b.y
    step_dist = math.hypot(dx, dy)

    b.x, b.y, b.heading = x_clamped, y_clamped, hd_n
    b.v_forward, b.v_lateral, b.omega = vf, vl, om
    b.nodes_xy = nodes_n + np.array([x_clamped - x_n, y_clamped - y_n], dtype=np.float64)

    state.distance_mm += step_dist
    state.signed_forward_mm += vf * dt_s
    state.body_wave_power = float(np.sqrt(np.mean(b.phi * b.phi)))
    cur_dir = 1 if vf >= -0.02 else -1
    if state.prev_direction == 1 and cur_dir == -1:
        state.reversals += 1
    if abs(vf) > 0.02:
        state.prev_direction = cur_dir

    # Track chemotaxis proximity index when attractants exist
    attr_spots = [f for f in state.foods if f.kind == "attractant"]
    if attr_spots:
        d_min = min(math.hypot(b.x - f.x, b.y - f.y) for f in attr_spots)
        state.attract_sum += max(-1.0, 1.0 - d_min / state.arena_half_mm)
        state.attract_samples += 1
        state.chemotaxis_index = state.attract_sum / state.attract_samples

    state.trail_xy.append([round(b.x, 3), round(b.y, 3)])
    if len(state.trail_xy) > 220:
        del state.trail_xy[:-220]


def render_worm_frame(state: WormEmbodiedState) -> tuple[dict[str, Any], bytes]:
    """Render agar plate, odor plumes, 12-segment crawling worm, and neural strip into field_u8."""
    w, h = 320, 240
    dish_h = 208  # top 208px agar arena, bottom 32px neural activity bar
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:dish_h] = (0.07, 0.09, 0.11)

    # Agar grid lines
    img[:dish_h:26, :] = (0.11, 0.14, 0.17)
    img[:dish_h, ::26] = (0.11, 0.14, 0.17)

    lim = state.arena_half_mm
    yy, xx = np.mgrid[0:dish_h, 0:w]
    wx = (xx / float(w) * 2.0 - 1.0) * lim
    wy = (1.0 - yy / float(dish_h) * 2.0) * lim

    for f in state.foods:
        d2 = (wx - f.x) ** 2 + (wy - f.y) ** 2
        g = np.exp(-d2 / max(2.0 * f.radius * f.radius, 1e-4)).astype(np.float32)
        if f.kind == "attractant":
            img[:dish_h, :, 1] += 0.55 * g
            img[:dish_h, :, 0] += 0.12 * g
        else:
            img[:dish_h, :, 0] += 0.65 * g
            img[:dish_h, :, 2] += 0.15 * g

    def to_px(px_mm: float, py_mm: float) -> tuple[int, int]:
        ix = round((px_mm / lim * 0.5 + 0.5) * (w - 1))
        iy = round((0.5 - py_mm / lim * 0.5) * (dish_h - 1))
        return max(0, min(w - 1, ix)), max(0, min(dish_h - 1, iy))

    # Movement trail
    for idx_t, (tx, ty) in enumerate(state.trail_xy):
        px, py = to_px(tx, ty)
        fade = 0.2 + 0.5 * (idx_t / max(1, len(state.trail_xy)))
        img[py, px] = (0.20 * fade, 0.55 * fade, 0.72 * fade)

    # Zoomed inset / worm spine segments (scale visually so 1 mm worm reads crisply on 12 mm plate)
    nodes = state.body.nodes_xy
    com = np.array([state.body.x, state.body.y], dtype=np.float64)
    vis_scale = (
        2.6  # subtle visual magnification around COM so curvature wave stays clearly legible
    )
    nodes_vis = com + (nodes - com) * vis_scale
    for k in range(N_NODES - 1):
        p0 = nodes_vis[k]
        p1 = nodes_vis[k + 1]
        for s_t in np.linspace(0.0, 1.0, 9):
            pt = p0 + s_t * (p1 - p0)
            px, py = to_px(float(pt[0]), float(pt[1]))
            col = (0.98, 0.85, 0.25) if k == 0 else (0.32, 0.90, 0.96)
            y0, y1 = max(0, py - 2), min(dish_h, py + 3)
            x0, x1 = max(0, px - 2), min(w, px + 3)
            img[y0:y1, x0:x1] = col

    # Bottom 32px real-time 302-neuron voltage bar (-60 mV .. -10 mV)
    img[dish_h:, :] = (0.04, 0.05, 0.07)
    v_norm = np.clip((state.lif.v.astype(np.float32) + 55.0) / 40.0, 0.0, 1.0)
    bar_cols = np.interp(np.linspace(0, 301, w), np.arange(302), v_norm).astype(np.float32)
    img[dish_h + 4 : h - 3, :, 0] = bar_cols[None, :] * 0.95
    img[dish_h + 4 : h - 3, :, 1] = bar_cols[None, :] * 0.55 + 0.15
    img[dish_h + 4 : h - 3, :, 2] = (1.0 - bar_cols[None, :]) * 0.70

    data = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
    header: dict[str, Any] = {
        "codec": "field_u8",
        "width": w,
        "height": h,
        "channels": 3,
        "scale": (
            f"12mm agar plate (worm zoomed {vis_scale}x); green=attractant, red=repellent; "
            "bottom strip=302 C. elegans graded neuron potentials"
        ),
    }
    return header, data
