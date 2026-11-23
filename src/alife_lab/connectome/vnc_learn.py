"""Phase F8 Level 2 Deepening: Biophysical ALIF + Per-Hemilineage Fixed-Topology VNC Learning.

Tests the core scientific question of F8 Level 2 (`connectome_plan.md §4 & §8`):
Given a **fixed biological Brain + Ventral Nerve Cord (VNC) wiring topology** (BANC v888 female or
MaleCNS v1.0 male) — with ZERO hand-crafted tripod CPGs, ZERO invented synapses, ZERO Dale's-law
sign flips, and ZERO artificial velocity injection (`d.qvel` untouched) — can learning ONLY
biophysically interpretable per-hemilineage parameters (synaptic conductance multipliers, tonic
aminergic excitability bias, and neuron-local spike-frequency adaptation ALIF) produce coordinated
pure-MuJoCo leg stepping and forward locomotion that outperforms a degree-preserving shuffled VNC
trained with the exact same model and optimization budget?
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.cns_packs import LEG_NAMES, MUSCLE_CHANNELS
from alife_lab.connectome.embodied import (
    EmbodiedFlyWorld,
    make_embodied_world,
    resolve_embodied_ports,
    step_embodied_tick,
)
from alife_lab.connectome.fly_body import ArenaState, fly_pose_xy_yaw, make_fly_body
from alife_lab.connectome.lif import LIFParams, make_initial_state, make_shuffled_csr
from alife_lab.connectome.schema import (
    CellTable,
    ConnectomePack,
    CSRGraph,
    DatasetProvenance,
    EdgeTable,
    IOPort,
    compute_pack_digest,
)

I32 = npt.NDArray[np.int32]
F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

# Fixed-point quantization factor for exact int32 associative/commutative GPU atomicAdd.
VNC_WEIGHT_QUANT = 16

# 16 canonical biological VNC hemilineage & functional groups shared verbatim across
# BANC v888 (female) and MaleCNS v1.0 (male) — Lesser et al. (2024) & Cheong et al. (2024).
VNC_GROUP_NAMES: tuple[str, ...] = (
    "hl_20A_22A",  # 0: cholinergic excitatory leg premotor
    "hl_03A",  # 1: cholinergic leg premotor
    "hl_19A",  # 2: GABAergic local leg flexor/extensor inhibition
    "hl_04B",  # 3: cholinergic sensory-motor relay
    "hl_13A",  # 4: GABAergic leg flexor/extensor inhibition
    "hl_12B",  # 5: cholinergic leg premotor
    "hl_21A",  # 6: cholinergic leg premotor
    "hl_16B",  # 7: cholinergic leg premotor
    "hl_13B",  # 8: GABAergic contralateral commissural
    "hl_08A_09A",  # 9: glutamatergic 08A + GABAergic 09A proprioceptive/premotor inhibition
    "hl_other_exc",  # 10: other excitatory VNC interneurons (01A, 14A, 06A, 07B, 23B, ...)
    "hl_other_inh",  # 11: other inhibitory VNC interneurons (GABA/Glu)
    "descending",  # 12: brain-to-VNC descending neurons (DNp09, DNa01/02, MDN, ...)
    "proprioceptive",  # 13: 6-leg chordotonal / campaniform / hair plate feedback
    "leg_motor_swing",  # 14: leg swing antagonist pools (coxa_pro, fem_flex, tib_ext)
    "leg_motor_stance",  # 15: leg stance antagonist pools (coxa_ret, fem_ext, tib_flex)
)
NUM_VNC_GROUPS = len(VNC_GROUP_NAMES)
GROUP_NONE = -1  # Non-VNC brain neurons stay untouched (scale=1.0, bias=0.0, adapt=0.0)

_HL_TO_GROUP: dict[str, int] = {
    "20A.22A": 0,
    "20A/22A": 0,
    "03A": 1,
    "3A": 1,
    "19A": 2,
    "04B": 3,
    "4B": 3,
    "13A": 4,
    "12B": 5,
    "21A": 6,
    "16B": 7,
    "13B": 8,
    "08A": 9,
    "8A": 9,
    "09A": 9,
    "9A": 9,
}

_SWING_CHANNELS = {"coxa_pro", "fem_flex", "tib_ext"}
_STANCE_CHANNELS = {"coxa_ret", "fem_ext", "tib_flex"}
_INH_NTS = {"gaba", "glutamate", "histamine"}

# Cache hemilineage source_id -> name maps per dataset_dir to avoid re-reading feather metadata.
_HL_MAP_CACHE: dict[tuple[str, str], dict[int, str]] = {}


def _load_hemilineage_map(dataset_id: str, datasets_dir: Path | None) -> dict[int, str]:
    if datasets_dir is None or dataset_id not in ("banc_888", "male_cns_v1"):
        return {}
    key = (dataset_id, str(datasets_dir.resolve()))
    cached = _HL_MAP_CACHE.get(key)
    if cached is not None:
        return cached
    import pyarrow.feather as pf

    out: dict[int, str] = {}
    if dataset_id == "banc_888":
        p = datasets_dir / "banc_888" / "raw" / "banc_888_meta.feather"
        if p.is_file():
            tbl = pf.read_table(p, columns=["banc_888_id", "hemilineage"])
            for rid, hl in zip(
                tbl["banc_888_id"].to_pylist(), tbl["hemilineage"].to_pylist(), strict=False
            ):
                if rid and hl:
                    out[int(rid)] = str(hl).strip()
    elif dataset_id == "male_cns_v1":
        p = (
            datasets_dir
            / "male_cns_v1"
            / "raw"
            / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        )
        if p.is_file():
            tbl = pf.read_table(p, columns=["bodyId", "trumanHl", "itoleeHl"])
            for bid, th, ih in zip(
                tbl["bodyId"].to_pylist(),
                tbl["trumanHl"].to_pylist(),
                tbl["itoleeHl"].to_pylist(),
                strict=False,
            ):
                hl = th or ih
                if bid is not None and hl:
                    out[int(bid)] = str(hl).strip()
    _HL_MAP_CACHE[key] = out
    return out


def assign_vnc_groups(pack: ConnectomePack, datasets_dir: Path | None = None) -> I32:
    """Assign each cell in ``pack`` to a biological VNC group 0..15 (or -1 for brain-only)."""
    cells = pack.cells
    n = cells.num_cells
    grp = np.full(n, GROUP_NONE, dtype=np.int32)
    hl_map = _load_hemilineage_map(pack.dataset_id, datasets_dir)

    # 1. Tag VNC intrinsic interneurons by developmental hemilineage & Dale's-law NT sign.
    for i in range(n):
        sc = cells.super_class[i].lower()
        reg = cells.region[i].lower()
        nt = cells.neurotransmitter[i].lower()
        if "descending" in sc:
            grp[i] = 12
            continue
        if "ventral_nerve_cord" in reg or sc in ("ventral_nerve_cord_intrinsic", "vnc_intrinsic"):
            hl = hl_map.get(int(cells.source_id[i]), "")
            if hl in _HL_TO_GROUP:
                grp[i] = _HL_TO_GROUP[hl]
            elif cells.cell_sub_class[i] == "premotor_cpg":
                grp[i] = 0 if nt not in _INH_NTS else 2
            elif nt in _INH_NTS:
                grp[i] = 11
            else:
                grp[i] = 10

    # 2. Override exact anatomical I/O ports (proprioceptors, swing MNs, stance MNs, DNs).
    for leg in LEG_NAMES:
        pid = f"proprio_{leg}"
        if pid in pack.io_ports:
            grp[pack.io_ports[pid].cell_indices] = 13
        for ch in MUSCLE_CHANNELS:
            mpid = f"mn_{leg}_{ch}"
            if mpid in pack.io_ports:
                grp[pack.io_ports[mpid].cell_indices] = 14 if ch in _SWING_CHANNELS else 15
    for dpid in ("forward_dn", "turn_left_dn", "turn_right_dn", "backward_dn", "grooming_dn"):
        if dpid in pack.io_ports:
            grp[pack.io_ports[dpid].cell_indices] = 12

    return grp


@dataclass(frozen=True)
class VNCBiophysicalProfile:
    """Compact, biologically interpretable fixed-topology VNC parameter profile (36 floats).

    Every parameter maps to a real biophysical mechanism missing from uniform-weight LIF:
    - ``group_syn_scale[16]``: positive synaptic conductance multiplier per VNC hemilineage/group
      (strictly > 0; Dale's-law signs and wiring topology are NEVER altered).
    - ``group_bias_mv[12]``: tonic aminergic arousal shift (mV) for VNC interneuron groups 0..11.
    - ``adapt_exc_mv``, ``adapt_inh_mv``, ``adapt_mn_mv``: neuron-local K+ spike-frequency
      adaptation increment per spike (mV) for excitatory VNC INs, inhibitory VNC INs, and leg MNs.
    - ``vnc_proprio_gain_hz``, ``explore_drive_hz``: sensory proprioceptive & descending arousal.
    - ``vnc_coxa_gain``, ``vnc_femur_gain``, ``vnc_tibia_gain``, ``vnc_mn_norm_hz``: shared
      biomechanical muscle-force scaling (identical across all 6 legs — zero per-leg tuning).
    """

    group_syn_scale: tuple[float, ...] = field(default_factory=lambda: (1.0,) * NUM_VNC_GROUPS)
    group_bias_mv: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 12)
    adapt_exc_mv: float = 0.8
    adapt_inh_mv: float = 0.8
    adapt_mn_mv: float = 0.5
    tau_adapt_ms: float = 80.0
    vnc_proprio_gain_hz: float = 110.0
    explore_drive_hz: float = 160.0
    vnc_coxa_gain: float = 0.55
    vnc_femur_gain: float = 0.55
    vnc_tibia_gain: float = 0.45
    vnc_mn_norm_hz: float = 22.0
    source_dataset_id: str = ""
    notes: str = ""

    @staticmethod
    def vector_dim() -> int:
        # 16 syn_scales + 12 biases + 3 adapt + 1 proprio + 1 explore + 3 joint gains + 1 mn_norm
        return 37

    @staticmethod
    def from_vector(x: F64, *, source_dataset_id: str = "") -> VNCBiophysicalProfile:
        """Decode an unconstrained CMA-ES vector ``x`` (in ~[-2, +2]) into bounded biophysics."""

        def _sig(v: float, lo: float, hi: float) -> float:
            s = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, float(v)))))
            return lo + (hi - lo) * s

        syn = tuple(round(_sig(float(x[i]), 0.15, 5.5), 4) for i in range(16))
        # Strictly subthreshold aminergic resting bias ([-1.5, +2.4] mV vs 7.0 mV threshold gap)
        # so VNC interneurons NEVER self-excite without descending DNp09 drive from the brain!
        bias = tuple(round(_sig(float(x[16 + i]), -1.5, 2.4), 4) for i in range(12))
        a_exc = round(_sig(float(x[28]), 0.0, 4.5), 4)
        a_inh = round(_sig(float(x[29]), 0.0, 4.5), 4)
        a_mn = round(_sig(float(x[30]), 0.0, 3.5), 4)
        prop_hz = round(_sig(float(x[31]), 20.0, 220.0), 2)
        exp_hz = round(_sig(float(x[32]), 90.0, 240.0), 2)
        coxa_g = round(_sig(float(x[33]), 0.25, 0.72), 4)
        fem_g = round(_sig(float(x[34]), 0.25, 0.72), 4)
        tib_g = round(_sig(float(x[35]), 0.15, 0.65), 4)
        mn_norm = round(_sig(float(x[36]), 6.0, 60.0), 2)
        return VNCBiophysicalProfile(
            group_syn_scale=syn,
            group_bias_mv=bias,
            adapt_exc_mv=a_exc,
            adapt_inh_mv=a_inh,
            adapt_mn_mv=a_mn,
            tau_adapt_ms=60.0,
            vnc_proprio_gain_hz=prop_hz,
            explore_drive_hz=exp_hz,
            vnc_coxa_gain=coxa_g,
            vnc_femur_gain=fem_g,
            vnc_tibia_gain=tib_g,
            vnc_mn_norm_hz=mn_norm,
            source_dataset_id=source_dataset_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source_dataset_id": self.source_dataset_id,
            "notes": self.notes,
            "group_names": list(VNC_GROUP_NAMES),
            "group_syn_scale": {
                VNC_GROUP_NAMES[i]: self.group_syn_scale[i] for i in range(NUM_VNC_GROUPS)
            },
            "group_bias_mv": {VNC_GROUP_NAMES[i]: self.group_bias_mv[i] for i in range(12)},
            "adapt_exc_mv": self.adapt_exc_mv,
            "adapt_inh_mv": self.adapt_inh_mv,
            "adapt_mn_mv": self.adapt_mn_mv,
            "tau_adapt_ms": self.tau_adapt_ms,
            "vnc_proprio_gain_hz": self.vnc_proprio_gain_hz,
            "explore_drive_hz": self.explore_drive_hz,
            "vnc_coxa_gain": self.vnc_coxa_gain,
            "vnc_femur_gain": self.vnc_femur_gain,
            "vnc_tibia_gain": self.vnc_tibia_gain,
            "vnc_mn_norm_hz": self.vnc_mn_norm_hz,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> VNCBiophysicalProfile:
        syn_map = data.get("group_syn_scale", {})
        bias_map = data.get("group_bias_mv", {})
        syn = tuple(float(syn_map.get(name, 1.0)) for name in VNC_GROUP_NAMES)
        bias = tuple(float(bias_map.get(VNC_GROUP_NAMES[i], 0.0)) for i in range(12))
        return VNCBiophysicalProfile(
            group_syn_scale=syn,
            group_bias_mv=bias,
            adapt_exc_mv=float(data.get("adapt_exc_mv", 0.8)),
            adapt_inh_mv=float(data.get("adapt_inh_mv", 0.8)),
            adapt_mn_mv=float(data.get("adapt_mn_mv", 0.5)),
            tau_adapt_ms=float(data.get("tau_adapt_ms", 80.0)),
            vnc_proprio_gain_hz=float(data.get("vnc_proprio_gain_hz", 110.0)),
            explore_drive_hz=float(data.get("explore_drive_hz", 160.0)),
            vnc_coxa_gain=float(data.get("vnc_coxa_gain", 0.55)),
            vnc_femur_gain=float(data.get("vnc_femur_gain", 0.55)),
            vnc_tibia_gain=float(data.get("vnc_tibia_gain", 0.45)),
            vnc_mn_norm_hz=float(data.get("vnc_mn_norm_hz", 22.0)),
            source_dataset_id=str(data.get("source_dataset_id", "")),
            notes=str(data.get("notes", "")),
        )


_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


def resolve_vnc_profile_path(name: str) -> Path:
    """Resolve a validated VNC profile basename strictly inside ``configs/vnc_profiles/``."""
    if not _PROFILE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid vnc_profile name {name!r}: must match ^[a-z0-9][a-z0-9_-]{{1,63}}$"
        )
    profiles_dir = (Path(__file__).resolve().parents[3] / "configs" / "vnc_profiles").resolve()
    candidate = (profiles_dir / f"{name}.json").resolve()
    if candidate.parent != profiles_dir or not candidate.is_file():
        raise FileNotFoundError(f"VNC profile {name!r} not found in {profiles_dir}")
    return candidate


def load_vnc_profile(path: Path) -> VNCBiophysicalProfile:
    return VNCBiophysicalProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def materialize_alif_arrays(
    csr: CSRGraph,
    groups: I32,
    profile: VNCBiophysicalProfile,
) -> tuple[I32, F32, F32]:
    """Expand a 37-float ``VNCBiophysicalProfile`` to fixed-point int32 CSR weights + ALIF arrays.

    Strictly preserves biological edge mask and Dale's-law signs: every nonzero edge keeps its
    original sign with magnitude ``max(1, round(|w| * scale * 16))``.
    """
    n = csr.num_cells
    pre_scale = np.ones(n, dtype=np.float32)
    post_scale = np.ones(n, dtype=np.float32)
    bias_mv = np.zeros(n, dtype=np.float32)
    adapt_inc_mv = np.zeros(n, dtype=np.float32)

    inh_groups = {2, 4, 8, 9, 11}
    exc_groups = {0, 1, 3, 5, 6, 7, 10}

    for g_id in range(NUM_VNC_GROUPS):
        mask = groups == g_id
        if not bool(mask.any()):
            continue
        if g_id in (14, 15):
            # Leg MNs are output sinks: scale their POSTSYNAPTIC dendritic input resistance
            # (biological motor-unit size principle, strictly preserving Dale's-law signs).
            post_scale[mask] = np.float32(profile.group_syn_scale[g_id])
            adapt_inc_mv[mask] = np.float32(profile.adapt_mn_mv)
        else:
            pre_scale[mask] = np.float32(profile.group_syn_scale[g_id])
            if g_id < 12:
                bias_mv[mask] = np.float32(profile.group_bias_mv[g_id])
            if g_id in exc_groups:
                adapt_inc_mv[mask] = np.float32(profile.adapt_exc_mv)
            elif g_id in inh_groups:
                adapt_inc_mv[mask] = np.float32(profile.adapt_inh_mv)

    # Expand presynaptic + postsynaptic MN input-resistance scales across CSR edges.
    counts = np.diff(csr.indptr)
    edge_scale = np.repeat(pre_scale, counts) * post_scale[csr.post]
    raw_w = csr.signed_weight.astype(np.float32)
    scaled = np.rint(raw_w * edge_scale * np.float32(VNC_WEIGHT_QUANT)).astype(np.int32)
    # Never flip sign or zero out an existing biological synapse.
    pos_mask = csr.signed_weight > 0
    neg_mask = csr.signed_weight < 0
    scaled = np.where(pos_mask, np.maximum(1, scaled), scaled)
    scaled = np.where(neg_mask, np.minimum(-1, scaled), scaled)
    return (
        np.ascontiguousarray(scaled, dtype=np.int32),
        np.ascontiguousarray(bias_mv, dtype=np.float32),
        np.ascontiguousarray(adapt_inc_mv, dtype=np.float32),
    )


def apply_vnc_profile_to_world(
    world: EmbodiedFlyWorld,
    profile: VNCBiophysicalProfile,
    *,
    datasets_dir: Path | None = None,
) -> None:
    """Install a fixed-topology ALIF ``VNCBiophysicalProfile`` onto an ``EmbodiedFlyWorld``."""
    groups = assign_vnc_groups(world.pack, datasets_dir)
    # Reconstruct quantized CSR from world.csr (preserving reference or shuffled_control topology!)
    w_q, bias_mv, adapt_inc_mv = materialize_alif_arrays(world.csr, groups, profile)
    world.csr = CSRGraph(
        num_cells=world.csr.num_cells,
        indptr=world.csr.indptr,
        post=world.csr.post,
        signed_weight=w_q,
        kind=world.csr.kind,
    )
    base_p = LIFParams(
        tau_adapt_ms=profile.tau_adapt_ms,
        weight_quant_div=float(VNC_WEIGHT_QUANT),
    )
    world.params = base_p
    world.lif.enable_alif(bias_mv=bias_mv, adapt_inc_mv=adapt_inc_mv)
    world.vnc_proprio_gain_hz = float(profile.vnc_proprio_gain_hz)
    world.explore_drive_hz = float(profile.explore_drive_hz)
    world.vnc_coxa_gain = float(profile.vnc_coxa_gain)
    world.vnc_femur_gain = float(profile.vnc_femur_gain)
    world.vnc_tibia_gain = float(profile.vnc_tibia_gain)
    world.vnc_mn_norm_hz = float(profile.vnc_mn_norm_hz)
    if world.gpu_runner is not None:
        from alife_lab.connectome.lif_cupy import CupyLIFRunner

        world.gpu_runner = CupyLIFRunner(world.csr, world.params, seed=world.seed)
        world.gpu_runner.upload_state(world.lif)
        world.gpu_synced_to_host = True


def extract_vnc_subcircuit(
    pack: ConnectomePack,
    datasets_dir: Path | None = None,
) -> tuple[ConnectomePack, I32]:
    """Extract the exact closed VNC + Descending + Proprioceptor + Leg MN subcircuit.

    Preserves 100% of VNC neurons, descending neurons, proprioceptors, and 36 leg motor pools plus
    every biological edge between them, omitting only unconnected optic-lobe / non-VNC brain cells
    so CMA-ES inner-loop rollouts run ~4x faster on GPU before full whole-CNS verification!
    """
    full_groups = assign_vnc_groups(pack, datasets_dir)
    keep_idx = np.flatnonzero(full_groups != GROUP_NONE).astype(np.int32)
    if keep_idx.size == pack.cells.num_cells:
        return pack, full_groups

    n_sub = int(keep_idx.size)
    old_to_new = np.full(pack.cells.num_cells, -1, dtype=np.int32)
    old_to_new[keep_idx] = np.arange(n_sub, dtype=np.int32)

    c = pack.cells
    sub_cells = CellTable(
        source_id=c.source_id[keep_idx].copy(),
        species_names=c.species_names,
        species_idx=c.species_idx[keep_idx].copy(),
        cell_type=tuple(c.cell_type[int(i)] for i in keep_idx),
        super_class=tuple(c.super_class[int(i)] for i in keep_idx),
        cell_class=tuple(c.cell_class[int(i)] for i in keep_idx),
        cell_sub_class=tuple(c.cell_sub_class[int(i)] for i in keep_idx),
        region=tuple(c.region[int(i)] for i in keep_idx),
        side=tuple(c.side[int(i)] for i in keep_idx),
        role=c.role[keep_idx].copy(),
        cell_model=c.cell_model[keep_idx].copy(),
        neurotransmitter=tuple(c.neurotransmitter[int(i)] for i in keep_idx),
        position_nm=c.position_nm[keep_idx].copy(),
        position_source=c.position_source[keep_idx].copy(),
    )

    e = pack.edges
    np_pre = old_to_new[e.pre]
    np_post = old_to_new[e.post]
    emask = (np_pre >= 0) & (np_post >= 0)
    sub_edges = EdgeTable(
        pre=np.ascontiguousarray(np_pre[emask], dtype=np.int32),
        post=np.ascontiguousarray(np_post[emask], dtype=np.int32),
        count=np.ascontiguousarray(e.count[emask], dtype=np.int32),
        sign=np.ascontiguousarray(e.sign[emask], dtype=np.int8),
        kind=np.ascontiguousarray(e.kind[emask], dtype=np.uint8),
    )

    sub_ports: dict[str, IOPort] = {}
    for pid, port in pack.io_ports.items():
        mapped = old_to_new[port.cell_indices]
        valid = mapped[mapped >= 0]
        if valid.size > 0:
            sub_ports[pid] = IOPort(
                port_id=pid,
                direction=port.direction,
                label=port.label,
                description=port.description,
                cell_indices=np.ascontiguousarray(valid, dtype=np.int32),
                selection_rule=port.selection_rule,
            )

    prov = DatasetProvenance(
        dataset_id=f"{pack.dataset_id}_vnc_sub",
        species=pack.species,
        version=pack.provenance.version,
        citation=pack.provenance.citation,
        licence_note=pack.provenance.licence_note,
        files=pack.provenance.files,
    )
    digest = compute_pack_digest(
        prov.dataset_id, pack.species, sub_cells, sub_edges, sub_ports, prov
    )
    sub_pack = ConnectomePack(
        dataset_id=prov.dataset_id,
        species=pack.species,
        cells=sub_cells,
        edges=sub_edges,
        io_ports=sub_ports,
        provenance=prov,
        pack_digest=digest,
    )
    return sub_pack, full_groups[keep_idx].copy()


def _evaluate_profile_fast(
    sub_pack: ConnectomePack,
    base_csr: CSRGraph,
    sub_groups: I32,
    ports: Any,
    profile: VNCBiophysicalProfile,
    *,
    runner: Any,
    backend: str,
    seed: int,
    ticks: int,
    datasets_dir: Path | None,
) -> dict[str, Any]:
    """Run one pure-MuJoCo Level 2 rollout reusing a resident GPU runner (zero VRAM re-alloc)."""
    w_q, bias_mv, adapt_inc_mv = materialize_alif_arrays(base_csr, sub_groups, profile)
    params = LIFParams(tau_adapt_ms=profile.tau_adapt_ms, weight_quant_div=float(VNC_WEIGHT_QUANT))
    lif = make_initial_state(sub_pack.cells.num_cells, params)
    lif.enable_alif(bias_mv=bias_mv, adapt_inc_mv=adapt_inc_mv)

    csr_q = CSRGraph(base_csr.num_cells, base_csr.indptr, base_csr.post, w_q, base_csr.kind)
    if runner is not None:
        runner.upload_alif_weights_and_params(w_q, bias_mv, adapt_inc_mv)
        runner.upload_state(lif)

    body = make_fly_body(datasets_dir=datasets_dir)
    arena = ArenaState(arena_half_size=6.0, foods=[])
    x0, y0, _ = fly_pose_xy_yaw(body)
    world = EmbodiedFlyWorld(
        pack=sub_pack,
        csr=csr_q,
        ports=ports,
        params=params,
        lif=lif,
        body=body,
        arena=arena,
        seed=seed,
        backend=backend,
        wiring="reference",
        control_mode="level2_vnc_tuned",
        explore_drive_hz=profile.explore_drive_hz,
        vnc_proprio_gain_hz=profile.vnc_proprio_gain_hz,
        vnc_mn_norm_hz=profile.vnc_mn_norm_hz,
        vnc_coxa_gain=profile.vnc_coxa_gain,
        vnc_femur_gain=profile.vnc_femur_gain,
        vnc_tibia_gain=profile.vnc_tibia_gain,
        gpu_runner=runner,
        gpu_synced_to_host=True,
        manual_stim_hz=np.zeros(sub_pack.cells.num_cells, dtype=np.float32),
        prev_xy=(x0, y0),
    )

    trip_a_series = np.empty(ticks, dtype=np.float32)
    trip_b_series = np.empty(ticks, dtype=np.float32)
    mn_sum = 0.0
    transitions = 0
    prev_c: tuple[bool, ...] = (True,) * 6
    z_min = 999.0

    for t in range(ticks):
        step_embodied_tick(world)
        lt = world.leg_mn_rates.mean(axis=1)
        trip_a_series[t] = (lt[0] + lt[4] + lt[2]) / 3.0
        trip_b_series[t] = (lt[3] + lt[1] + lt[5]) / 3.0
        mn_sum += float(lt.mean())
        cur_c = world.last_sensors.tarsus_contacts
        transitions += sum(1 for a, b in zip(prev_c, cur_c, strict=True) if a != b)
        prev_c = cur_c
        z_min = min(z_min, float(world.body.data.qpos[2]))

    x1, y1, yaw1 = fly_pose_xy_yaw(world.body)
    net_x = float(x1 - x0)
    net_disp = float(math.hypot(x1 - x0, y1 - y0))
    mean_mn = mn_sum / max(1, ticks)

    # Genuine temporal oscillation & anti-phase tripod alternation across the rollout!
    std_a = float(np.std(trip_a_series[50:])) if ticks > 60 else 0.0
    std_b = float(np.std(trip_b_series[50:])) if ticks > 60 else 0.0
    osc_amp = 0.5 * (std_a + std_b)
    if std_a > 0.25 and std_b > 0.25:
        corr = float(np.corrcoef(trip_a_series[50:], trip_b_series[50:])[0, 1])
        anti_phase = max(0.0, -corr)
    else:
        corr = 0.0
        anti_phase = 0.0

    # Fitness: STRICTLY reward genuine FORWARD (+X) pure-MuJoCo walking + rhythmic stepping,
    # NEVER rewarding backward (-X) scooting or seizure-rate (>80 Hz) saturation!
    health = 1.0 if mean_mn <= 80.0 else max(0.05, (80.0 / mean_mn) ** 2)
    posture_pen = 0.0 if z_min >= 0.14 else 0.5 * (0.14 - z_min)
    fwd_score = max(0.0, net_x) * 2.5 - max(0.0, -net_x) * 0.45 - abs(y1 - y0) * 0.25
    step_bonus = min(0.45, transitions / 350.0) if net_x > 0.04 else min(0.12, transitions / 500.0)
    osc_bonus = min(0.35, 0.03 * osc_amp) + 0.45 * anti_phase
    fitness = (fwd_score + step_bonus + osc_bonus) * health - posture_pen

    return {
        "fitness": round(float(fitness), 5),
        "net_x_cm": round(net_x, 4),
        "net_disp_cm": round(net_disp, 4),
        "path_cm": round(float(world.distance_traveled), 4),
        "yaw_rad": round(float(yaw1), 3),
        "z_min": round(float(z_min), 4),
        "vnc_mn_mean_hz": round(mean_mn, 2),
        "tripod_osc_std_hz": round(osc_amp, 2),
        "tripod_corr": round(corr, 3),
        "tarsus_transitions": int(transitions),
        "realtime_factor": world.timing.realtime_factor,
    }


def _run_cmaes_vnc(
    sub_pack: ConnectomePack,
    sub_groups: I32,
    *,
    wiring: str,
    backend: str,
    seed: int,
    generations: int,
    pop_size: int,
    ticks_per_eval: int,
    datasets_dir: Path | None,
) -> tuple[VNCBiophysicalProfile, dict[str, Any], list[dict[str, Any]]]:
    """Run separable CMA-ES on fixed-topology VNC ALIF biophysics (intact or shuffled)."""
    base_csr = sub_pack.edges.to_csr(sub_pack.cells.num_cells)
    if wiring == "shuffled_control":
        base_csr = make_shuffled_csr(base_csr, seed)
    ports = resolve_embodied_ports(sub_pack)

    runner: Any = None
    if backend == "cupy":
        from alife_lab.connectome.lif_cupy import CupyLIFRunner

        init_params = LIFParams(tau_adapt_ms=60.0, weight_quant_div=float(VNC_WEIGHT_QUANT))
        runner = CupyLIFRunner(base_csr, init_params, seed=seed)

    dim = VNCBiophysicalProfile.vector_dim()
    rng = np.random.default_rng(seed + (0 if wiring == "reference" else 10_000))
    mean = np.zeros(dim, dtype=np.float64)
    # Biologically sensible initial mean: moderate syn_scale (~1.2), mild arousal bias, mild ALIF.
    mean[:16] = -1.1  # sigmoid -> ~1.4x
    mean[16:28] = -0.5  # sigmoid -> ~1.5 mV resting shift
    mean[28:31] = -0.8  # sigmoid -> ~1.4 mV adaptation
    sigma = np.full(dim, 0.85, dtype=np.float64)

    mu = max(2, pop_size // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
    weights /= weights.sum()

    best_score = -1e9
    best_prof = VNCBiophysicalProfile.from_vector(mean, source_dataset_id=sub_pack.dataset_id)
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    for gen in range(generations):
        xs: list[F64] = []
        evals: list[dict[str, Any]] = []
        for k in range(pop_size):
            x = mean.copy() if (gen == 0 and k == 0) else mean + sigma * rng.standard_normal(dim)
            prof = VNCBiophysicalProfile.from_vector(x, source_dataset_id=sub_pack.dataset_id)
            m = _evaluate_profile_fast(
                sub_pack,
                base_csr,
                sub_groups,
                ports,
                prof,
                runner=runner,
                backend=backend,
                seed=seed,
                ticks=ticks_per_eval,
                datasets_dir=datasets_dir,
            )
            xs.append(x)
            evals.append(m)
            if m["fitness"] > best_score:
                best_score = m["fitness"]
                best_prof = prof
                best_metrics = m

        order = np.argsort([-e["fitness"] for e in evals])
        elites = np.stack([xs[int(i)] for i in order[:mu]], axis=0)
        old_mean = mean.copy()
        mean = (weights[:, None] * elites).sum(axis=0)
        diff = elites - old_mean[None, :]
        var = (weights[:, None] * (diff * diff)).sum(axis=0)
        sigma = np.clip(0.65 * sigma + 0.35 * np.sqrt(np.maximum(var, 1e-4)), 0.12, 1.4)

        top = evals[int(order[0])]
        history.append(
            {
                "gen": gen,
                "best_fitness": round(best_score, 4),
                "gen_best_fitness": top["fitness"],
                "gen_best_net_x_cm": top["net_x_cm"],
                "gen_best_path_cm": top["path_cm"],
                "gen_best_mn_hz": top["vnc_mn_mean_hz"],
                "gen_best_osc_std": top["tripod_osc_std_hz"],
            }
        )

    return best_prof, best_metrics, history


def verify_profile_on_full_cns(
    pack: ConnectomePack,
    profile: VNCBiophysicalProfile,
    *,
    wiring: str = "reference",
    backend: str = "numpy",
    seed: int = 1,
    ticks: int = 1500,
    silence_dn: bool = False,
    open_loop: bool = False,
    datasets_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a learned ``VNCBiophysicalProfile`` on the FULL whole-CNS connectome + MuJoCo."""
    eff_prof = (
        VNCBiophysicalProfile(
            **{
                **profile.__dict__,
                "vnc_proprio_gain_hz": 0.0,
            }
        )
        if open_loop
        else profile
    )
    w = make_embodied_world(
        pack,
        wiring=wiring,
        seed=seed,
        backend=backend,
        foods=[],
        explore_drive_hz=eff_prof.explore_drive_hz,
        control_mode="level2_vnc_tuned",
        vnc_proprio_gain_hz=eff_prof.vnc_proprio_gain_hz,
        vnc_mn_norm_hz=eff_prof.vnc_mn_norm_hz,
        vnc_coxa_gain=eff_prof.vnc_coxa_gain,
        vnc_femur_gain=eff_prof.vnc_femur_gain,
        vnc_tibia_gain=eff_prof.vnc_tibia_gain,
        datasets_dir=datasets_dir,
    )
    apply_vnc_profile_to_world(w, eff_prof, datasets_dir=datasets_dir)
    if silence_dn and w.ports.forward_dn.size > 0:
        w.lif.silenced[w.ports.forward_dn] = np.uint8(1)
        if w.gpu_runner is not None:
            w.gpu_runner.upload_controls(w.lif)

    x0, y0, _ = fly_pose_xy_yaw(w.body)
    trip_a = np.empty(ticks, dtype=np.float32)
    trip_b = np.empty(ticks, dtype=np.float32)
    mn_sum = 0.0
    transitions = 0
    prev_c: tuple[bool, ...] = (True,) * 6
    z_min = 999.0

    for t in range(ticks):
        step_embodied_tick(w)
        lt = w.leg_mn_rates.mean(axis=1)
        trip_a[t] = (lt[0] + lt[4] + lt[2]) / 3.0
        trip_b[t] = (lt[3] + lt[1] + lt[5]) / 3.0
        mn_sum += float(lt.mean())
        cur_c = w.last_sensors.tarsus_contacts
        transitions += sum(1 for a, b in zip(prev_c, cur_c, strict=True) if a != b)
        prev_c = cur_c
        z_min = min(z_min, float(w.body.data.qpos[2]))

    x1, y1, yaw1 = fly_pose_xy_yaw(w.body)
    std_a = float(np.std(trip_a[100:])) if ticks > 120 else 0.0
    std_b = float(np.std(trip_b[100:])) if ticks > 120 else 0.0
    corr = (
        float(np.corrcoef(trip_a[100:], trip_b[100:])[0, 1])
        if (std_a > 0.25 and std_b > 0.25)
        else 0.0
    )
    return {
        "dataset_id": pack.dataset_id,
        "num_cells": pack.cells.num_cells,
        "wiring": wiring,
        "silence_dn": silence_dn,
        "open_loop": open_loop,
        "ticks": ticks,
        "net_x_cm": round(float(x1 - x0), 4),
        "net_disp_cm": round(float(math.hypot(x1 - x0, y1 - y0)), 4),
        "path_cm": round(float(w.distance_traveled), 4),
        "speed_cm_s": round(float(w.distance_traveled) / (ticks * 1e-3), 3),
        "yaw_rad": round(float(yaw1), 3),
        "z_min": round(float(z_min), 4),
        "vnc_mn_mean_hz": round(mn_sum / max(1, ticks), 2),
        "tripod_osc_std_hz": round(0.5 * (std_a + std_b), 2),
        "tripod_corr": round(corr, 3),
        "tarsus_transitions": int(transitions),
        "realtime_factor": w.timing.realtime_factor,
    }


def learn_vnc_locomotion(
    pack: ConnectomePack,
    *,
    backend: str = "numpy",
    seed: int = 1,
    generations: int = 20,
    pop_size: int = 12,
    ticks_per_eval: int = 600,
    verify_ticks: int = 1500,
    include_shuffled_cmaes: bool = True,
    datasets_dir: Path | None = None,
    save_profile_path: Path | None = None,
) -> dict[str, Any]:
    """Full F8 Level 2 deepening: hemilineage ALIF CMA-ES + full-CNS & shuffled controls."""
    t0 = time.perf_counter()
    sub_pack, sub_groups = extract_vnc_subcircuit(pack, datasets_dir)

    # 1. Untuned uniform-LIF baseline on full whole-CNS connectome
    untuned_prof = VNCBiophysicalProfile(
        adapt_exc_mv=0.0,
        adapt_inh_mv=0.0,
        adapt_mn_mv=0.0,
        vnc_proprio_gain_hz=95.0,
        explore_drive_hz=140.0,
        source_dataset_id=pack.dataset_id,
    )
    full_untuned = verify_profile_on_full_cns(
        pack,
        untuned_prof,
        wiring="reference",
        backend=backend,
        seed=seed,
        ticks=verify_ticks,
        datasets_dir=datasets_dir,
    )

    # 2. Fixed-topology CMA-ES on intact biological VNC hemilineages
    learned_prof, sub_best, intact_hist = _run_cmaes_vnc(
        sub_pack,
        sub_groups,
        wiring="reference",
        backend=backend,
        seed=seed,
        generations=generations,
        pop_size=pop_size,
        ticks_per_eval=ticks_per_eval,
        datasets_dir=datasets_dir,
    )
    learned_prof = VNCBiophysicalProfile(
        **{
            **learned_prof.__dict__,
            "source_dataset_id": pack.dataset_id,
            "notes": (
                f"Learned via fixed-topology hemilineage ALIF CMA-ES "
                f"(gens={generations}, pop={pop_size}, seed={seed}); ZERO hand-crafted CPG."
            ),
        }
    )

    # 3. Transfer learned biological profile onto FULL whole-CNS connectome + controls
    full_learned = verify_profile_on_full_cns(
        pack,
        learned_prof,
        wiring="reference",
        backend=backend,
        seed=seed,
        ticks=verify_ticks,
        datasets_dir=datasets_dir,
    )
    full_transfer_shuffled = verify_profile_on_full_cns(
        pack,
        learned_prof,
        wiring="shuffled_control",
        backend=backend,
        seed=seed,
        ticks=verify_ticks,
        datasets_dir=datasets_dir,
    )
    full_dn_silenced = verify_profile_on_full_cns(
        pack,
        learned_prof,
        wiring="reference",
        backend=backend,
        seed=seed,
        ticks=verify_ticks,
        silence_dn=True,
        datasets_dir=datasets_dir,
    )
    full_open_loop = verify_profile_on_full_cns(
        pack,
        learned_prof,
        wiring="reference",
        backend=backend,
        seed=seed,
        ticks=verify_ticks,
        open_loop=True,
        datasets_dir=datasets_dir,
    )

    # 4. Matched-budget CMA-ES trained FROM SCRATCH on degree-preserving SHUFFLED VNC!
    full_shuffled_trained: dict[str, Any] | None = None
    shuffled_hist: list[dict[str, Any]] = []
    if include_shuffled_cmaes:
        shuf_prof, _, shuffled_hist = _run_cmaes_vnc(
            sub_pack,
            sub_groups,
            wiring="shuffled_control",
            backend=backend,
            seed=seed,
            generations=generations,
            pop_size=pop_size,
            ticks_per_eval=ticks_per_eval,
            datasets_dir=datasets_dir,
        )
        full_shuffled_trained = verify_profile_on_full_cns(
            pack,
            shuf_prof,
            wiring="shuffled_control",
            backend=backend,
            seed=seed,
            ticks=verify_ticks,
            datasets_dir=datasets_dir,
        )

    if save_profile_path is not None:
        save_profile_path.parent.mkdir(parents=True, exist_ok=True)
        save_profile_path.write_text(
            json.dumps(learned_prof.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    return {
        "dataset_id": pack.dataset_id,
        "full_cells": pack.cells.num_cells,
        "vnc_subcircuit_cells": sub_pack.cells.num_cells,
        "vnc_subcircuit_edges": sub_pack.edges.num_edges,
        "backend": backend,
        "seed": seed,
        "generations": generations,
        "pop_size": pop_size,
        "ticks_per_eval": ticks_per_eval,
        "verify_ticks": verify_ticks,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "learned_profile": learned_prof.to_dict(),
        "subcircuit_best": sub_best,
        "full_cns_verification": {
            "untuned_uniform_lif": full_untuned,
            "learned_intact": full_learned,
            "learned_transferred_to_shuffled": full_transfer_shuffled,
            "learned_dn_silenced": full_dn_silenced,
            "learned_open_loop": full_open_loop,
            "shuffled_trained_from_scratch": full_shuffled_trained,
        },
        "history_intact": intact_hist,
        "history_shuffled": shuffled_hist,
    }
