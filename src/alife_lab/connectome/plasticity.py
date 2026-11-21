"""Mushroom Body KC -> MBON three-factor dopamine-modulated plasticity (F4).

Scientific framing (mandatory):
This is a **simplified phenomenological eligibility-trace plasticity rule** grounded in
Drosophila mushroom body associative learning literature (Hige et al. 2015; Cohn et al. 2015;
Handler et al. 2019; Jiang & Litwin-Kumar 2021). It does NOT model detailed cAMP/PKA/cGMP
biochemistry, bouton-level calcium heterogeneity or neuromodulatory state.

Scope:
- ONLY Kenyon cell (KC) -> Mushroom Body Output Neuron (MBON) excitatory synapses are plastic
  (62,261 of 15,091,983 edges in FlyWire v783). All other brain synapses remain strictly fixed.
- Compartmental dopaminergic reinforcement is derived directly from anatomical DAN -> MBON
  synapses in the connectome (thresholded by ``min_dan_synapses`` to ignore stray contacts).
- Updates run at each 1 ms tick in deterministic ``float32``, writing rounded ``int32`` weights
  back into the active CSR (and GPU device buffer when using CuPy) so CPU and GPU stay
  100% bit-identical and checkpoints/replays are exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.schema import ConnectomePack, CSRGraph

F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]


@dataclass(frozen=True)
class MBPlasticityParams:
    enabled: bool = False
    tau_elig_ms: float = 200.0
    tau_dan_ms: float = 150.0
    dan_threshold: float = 18.0
    lr_ltd: float = 0.00022
    lr_ext: float = 0.03
    w_min: float = 0.05
    w_max: float = 2.0
    min_dan_synapses: int = 5


@dataclass(frozen=True)
class PlasticSynapseMap:
    """Immutable anatomical index of KC -> MBON plastic edges and DAN -> MBON compartments."""

    kc_indices: I32  # global neuron indices of KCs (length K)
    mbon_indices: I32  # global neuron indices of MBONs (length M)
    dan_indices: I32  # global neuron indices of DANs (length D)
    # Plastic KC -> MBON edges (length E_p):
    csr_edge_idx: I32  # index into CSRGraph.signed_weight
    pre_local: I32  # index in 0..K-1
    post_local: I32  # index in 0..M-1
    pre_global: I32
    post_global: I32
    base_weight: I32  # original integer signed_weight (> 0)
    # Anatomical DAN -> MBON compartment coupling (length E_dm):
    dan_local: I32  # index in 0..D-1
    dan_mbon_local: I32  # index in 0..M-1
    dan_weight: F32  # normalized coupling weight in (0, 1]

    @property
    def num_edges(self) -> int:
        return int(self.csr_edge_idx.shape[0])


@dataclass
class PlasticityState:
    """Mutable, checkpointable state of KC -> MBON plasticity."""

    scale: F32  # (E_p,) float32 multiplier in [w_min, w_max], initialized to 1.0
    kc_elig: F32  # (K,) float32 presynaptic KC eligibility trace
    mbon_dan: F32  # (M,) float32 compartmental dopamine reinforcement trace
    last_eff_weight: I32  # (E_p,) last int32 effective weights written to CSR/GPU


def build_plastic_synapse_map(
    pack: ConnectomePack,
    csr: CSRGraph,
    *,
    min_dan_synapses: int = 5,
) -> PlasticSynapseMap:
    """Extract all excitatory KC -> MBON edges and anatomical DAN -> MBON compartments."""
    kc_port = pack.io_ports.get("mushroom_body_kc")
    mbon_port = pack.io_ports.get("mushroom_body_mbon")
    dan_port = pack.io_ports.get("mushroom_body_dan")
    kc_indices = (
        kc_port.cell_indices.astype(np.int32, copy=False)
        if kc_port is not None
        else np.zeros(0, dtype=np.int32)
    )
    mbon_indices = (
        mbon_port.cell_indices.astype(np.int32, copy=False)
        if mbon_port is not None
        else np.zeros(0, dtype=np.int32)
    )
    dan_indices = (
        dan_port.cell_indices.astype(np.int32, copy=False)
        if dan_port is not None
        else np.zeros(0, dtype=np.int32)
    )

    mbon_to_local = {int(g): loc for loc, g in enumerate(mbon_indices.tolist())}

    # 1. Collect all excitatory KC -> MBON edges from CSR.
    csr_e: list[int] = []
    pre_l: list[int] = []
    post_l: list[int] = []
    pre_g: list[int] = []
    post_g: list[int] = []
    base_w: list[int] = []
    for loc_k, g_k in enumerate(kc_indices.tolist()):
        start = int(csr.indptr[g_k])
        end = int(csr.indptr[g_k + 1])
        for eidx in range(start, end):
            g_post = int(csr.post[eidx])
            loc_m = mbon_to_local.get(g_post)
            if loc_m is not None:
                w = int(csr.signed_weight[eidx])
                if w > 0:
                    csr_e.append(eidx)
                    pre_l.append(loc_k)
                    post_l.append(loc_m)
                    pre_g.append(g_k)
                    post_g.append(g_post)
                    base_w.append(w)

    # 2. Collect anatomical DAN -> MBON compartment couplings, normalized by total anatomical
    # DAN synapse count per MBON so ``mbon_dan`` measures weighted mean compartmental DAN
    # spike count over ``tau_dan_ms`` on a consistent physical scale across circuits.
    dan_l: list[int] = []
    dan_m: list[int] = []
    dan_raw_w: list[float] = []
    mbon_sum_syn = np.zeros(len(mbon_indices), dtype=np.float32)
    for loc_d, g_d in enumerate(dan_indices.tolist()):
        start = int(csr.indptr[g_d])
        end = int(csr.indptr[g_d + 1])
        for eidx in range(start, end):
            g_post = int(csr.post[eidx])
            loc_m = mbon_to_local.get(g_post)
            if loc_m is not None:
                syn = abs(int(csr.signed_weight[eidx]))
                if syn >= min_dan_synapses:
                    dan_l.append(loc_d)
                    dan_m.append(loc_m)
                    dan_raw_w.append(float(syn))
                    mbon_sum_syn[loc_m] += np.float32(syn)

    dan_w = np.asarray(
        [w / max(float(mbon_sum_syn[m]), 1.0) for w, m in zip(dan_raw_w, dan_m, strict=True)],
        dtype=np.float32,
    )
    return PlasticSynapseMap(
        kc_indices=kc_indices,
        mbon_indices=mbon_indices,
        dan_indices=dan_indices,
        csr_edge_idx=np.asarray(csr_e, dtype=np.int32),
        pre_local=np.asarray(pre_l, dtype=np.int32),
        post_local=np.asarray(post_l, dtype=np.int32),
        pre_global=np.asarray(pre_g, dtype=np.int32),
        post_global=np.asarray(post_g, dtype=np.int32),
        base_weight=np.asarray(base_w, dtype=np.int32),
        dan_local=np.asarray(dan_l, dtype=np.int32),
        dan_mbon_local=np.asarray(dan_m, dtype=np.int32),
        dan_weight=dan_w,
    )


def make_initial_plasticity_state(pmap: PlasticSynapseMap) -> PlasticityState:
    return PlasticityState(
        scale=np.ones(pmap.num_edges, dtype=np.float32),
        kc_elig=np.zeros(pmap.kc_indices.shape[0], dtype=np.float32),
        mbon_dan=np.zeros(pmap.mbon_indices.shape[0], dtype=np.float32),
        last_eff_weight=pmap.base_weight.copy(),
    )


def effective_weights(pmap: PlasticSynapseMap, scale: F32) -> I32:
    """Deterministic rounded integer weights derived from float32 plastic multipliers."""
    return np.rint(pmap.base_weight.astype(np.float32) * scale).astype(np.int32)


def step_plasticity_tick(
    pstate: PlasticityState,
    pmap: PlasticSynapseMap,
    params: MBPlasticityParams,
    last_tick_spikes: I32,
    silenced: npt.NDArray[np.uint8],
    dt_tick_ms: float = 1.0,
) -> tuple[I32, I32]:
    """Advance KC eligibility, DAN reinforcement traces and KC -> MBON plastic scales by 1 tick.

    Returns ``(changed_csr_edge_indices, new_effective_weights)`` for any edges whose rounded
    integer weight changed on this tick (empty arrays when no integer weight changed).
    """
    if not params.enabled or pmap.num_edges == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty

    decay_e = np.float32(math.exp(-dt_tick_ms / max(params.tau_elig_ms, 1e-3)))
    decay_d = np.float32(math.exp(-dt_tick_ms / max(params.tau_dan_ms, 1e-3)))

    # 1. Update presynaptic KC eligibility traces (silenced KCs do not emit spikes).
    kc_sp = last_tick_spikes[pmap.kc_indices] * (silenced[pmap.kc_indices] == np.uint8(0))
    pstate.kc_elig *= decay_e
    if np.any(kc_sp > 0):
        pstate.kc_elig += kc_sp.astype(np.float32)

    # 2. Update compartmental DAN reinforcement traces (silenced DANs produce zero dopamine!).
    dan_sp = last_tick_spikes[pmap.dan_indices] * (silenced[pmap.dan_indices] == np.uint8(0))
    pstate.mbon_dan *= decay_d
    if pmap.dan_local.size > 0 and np.any(dan_sp > 0):
        contrib = dan_sp[pmap.dan_local].astype(np.float32) * pmap.dan_weight
        np.add.at(pstate.mbon_dan, pmap.dan_mbon_local, contrib)

    # 3. Three-factor LTD on eligible KCs x phasic DAN bursts above ``dan_threshold``.
    w_min = np.float32(params.w_min)
    w_max = np.float32(params.w_max)
    d_thresh = np.float32(params.dan_threshold)
    d_eff_mbon = np.maximum(pstate.mbon_dan - d_thresh, np.float32(0.0))
    active_dan_mbons = d_eff_mbon > np.float32(1e-4)
    active_elig_kcs = pstate.kc_elig > np.float32(1e-4)
    if np.any(active_dan_mbons) and np.any(active_elig_kcs):
        e_edge = pstate.kc_elig[pmap.pre_local]
        d_edge = d_eff_mbon[pmap.post_local]
        coinc = (e_edge > np.float32(1e-4)) & (d_edge > np.float32(1e-4))
        if np.any(coinc):
            idx = np.flatnonzero(coinc)
            # Saturating exponential approach toward w_min.
            delta = (
                np.float32(params.lr_ltd)
                * e_edge[idx]
                * d_edge[idx]
                * np.maximum(pstate.scale[idx] - w_min, np.float32(0.0))
            )
            pstate.scale[idx] = np.clip(pstate.scale[idx] - delta, w_min, w_max)

    # 4. Extinction / unreinforced recovery when KC fires without a phasic DAN burst.
    if params.lr_ext > 0.0 and np.any(kc_sp > 0):
        sp_edge = kc_sp[pmap.pre_local]
        d_raw = pstate.mbon_dan[pmap.post_local]
        ext_mask = (sp_edge > 0) & (d_raw <= d_thresh) & (pstate.scale < np.float32(1.0))
        if np.any(ext_mask):
            idx = np.flatnonzero(ext_mask)
            delta = (
                np.float32(params.lr_ext)
                * sp_edge[idx].astype(np.float32)
                * (np.float32(1.0) - pstate.scale[idx])
            )
            pstate.scale[idx] = np.clip(pstate.scale[idx] + delta, w_min, w_max)

    # 5. Compute updated integer weights and emit sparse diff for CSR / GPU buffer.
    new_eff = effective_weights(pmap, pstate.scale)
    diff_mask = new_eff != pstate.last_eff_weight
    if not np.any(diff_mask):
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty
    diff_idx = np.flatnonzero(diff_mask)
    pstate.last_eff_weight[diff_idx] = new_eff[diff_idx]
    return pmap.csr_edge_idx[diff_idx], new_eff[diff_idx]


def plasticity_summary(
    pack: ConnectomePack,
    pmap: PlasticSynapseMap,
    pstate: PlasticityState,
    *,
    top_n: int = 12,
) -> dict[str, Any]:
    """Compact summary of changed KC -> MBON synapses for preview frames and UI inspection."""
    n_edges = pmap.num_edges
    if n_edges == 0:
        return {
            "enabled": True,
            "total_edges": 0,
            "changed_edges": 0,
            "mean_scale": 1.0,
            "min_scale": 1.0,
            "max_scale": 1.0,
            "top_changed": [],
        }
    deltas = np.abs(pstate.scale - np.float32(1.0))
    changed = np.flatnonzero(deltas > np.float32(0.01))
    top_changed: list[dict[str, Any]] = []
    if changed.size > 0:
        order = changed[np.argsort(-deltas[changed], kind="mergesort")[:top_n]]
        cells = pack.cells
        for k in order.tolist():
            pre_g = int(pmap.pre_global[k])
            post_g = int(pmap.post_global[k])
            sc = float(pstate.scale[k])
            top_changed.append(
                {
                    "pre_index": pre_g,
                    "post_index": post_g,
                    "pre_type": cells.cell_type[pre_g],
                    "post_type": cells.cell_type[post_g],
                    "base_weight": int(pmap.base_weight[k]),
                    "eff_weight": int(pstate.last_eff_weight[k]),
                    "scale": round(sc, 3),
                    "delta_pct": round((sc - 1.0) * 100.0, 1),
                }
            )
    return {
        "enabled": True,
        "total_edges": n_edges,
        "changed_edges": int(changed.size),
        "mean_scale": round(float(pstate.scale.mean()), 4),
        "min_scale": round(float(pstate.scale.min()), 4),
        "max_scale": round(float(pstate.scale.max()), 4),
        "top_changed": top_changed,
    }
