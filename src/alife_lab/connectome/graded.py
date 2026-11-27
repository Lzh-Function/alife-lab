"""Graded membrane potential & ohmic gap-junction connectome solver (Stage 2 / S1).

Implements conductance-based non-spiking neural dynamics (``MODEL_GRADED``) paired with
symmetric electrical gap junctions (``SYNAPSE_GAP_JUNCTION``) in deterministic IEEE-754
``float32`` without modifying ``lif.py`` so existing FlyWire / spiking LIF paths remain
bit-for-bit untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.lif import LIFParams, LIFState, step_substep_cpu
from alife_lab.connectome.schema import (
    MODEL_GRADED,
    MODEL_SPIKING_LIF,
    SYNAPSE_CHEMICAL,
    SYNAPSE_GAP_JUNCTION,
    ConnectomePack,
    CSRGraph,
)

F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]


@dataclass(frozen=True)
class GradedParams:
    """Biophysical constants for conductance-based graded + gap-junction networks."""

    c_m: float = 2.0
    g_leak: float = 0.15
    e_leak_mv: float = -35.0
    e_exc_mv: float = 0.0
    e_inh_mv: float = -70.0
    g_chem: float = 0.04
    g_gap: float = 0.025
    v_half_mv: float = -20.0
    v_slope_mv: float = 3.2
    v_active_mv: float = -28.0
    stim_current_scale: float = 0.045
    dt_ms: float = 0.25
    substeps_per_tick: int = 4

    @property
    def dt_s(self) -> float:
        return self.dt_ms * 1e-3


@dataclass(frozen=True)
class GradedCircuit:
    """Pre-extracted edge arrays split by excitatory chemical, inhibitory chemical, and gap."""

    num_cells: int
    chem_exc_pre: I32
    chem_exc_post: I32
    chem_exc_w: F32
    chem_inh_pre: I32
    chem_inh_post: I32
    chem_inh_w: F32
    gap_pre: I32
    gap_post: I32
    gap_w: F32

    @property
    def num_gap_edges(self) -> int:
        return int(self.gap_pre.shape[0])

    @classmethod
    def from_csr(cls, csr: CSRGraph) -> GradedCircuit:
        n = csr.num_cells
        e = int(csr.post.shape[0])
        if e == 0:
            empty_i = np.zeros(0, dtype=np.int32)
            empty_f = np.zeros(0, dtype=np.float32)
            return cls(
                n, empty_i, empty_i, empty_f, empty_i, empty_i, empty_f, empty_i, empty_i, empty_f
            )
        pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(csr.indptr)).astype(np.int32)
        post = csr.post
        sw = csr.signed_weight
        kind = csr.kind

        chem_mask = kind == SYNAPSE_CHEMICAL
        exc_m = chem_mask & (sw > 0)
        inh_m = chem_mask & (sw < 0)
        gap_m = (kind == SYNAPSE_GAP_JUNCTION) & (sw != 0)

        return cls(
            num_cells=n,
            chem_exc_pre=np.ascontiguousarray(pre[exc_m], dtype=np.int32),
            chem_exc_post=np.ascontiguousarray(post[exc_m], dtype=np.int32),
            chem_exc_w=np.ascontiguousarray(sw[exc_m].astype(np.float32)),
            chem_inh_pre=np.ascontiguousarray(pre[inh_m], dtype=np.int32),
            chem_inh_post=np.ascontiguousarray(post[inh_m], dtype=np.int32),
            chem_inh_w=np.ascontiguousarray((-sw[inh_m]).astype(np.float32)),
            gap_pre=np.ascontiguousarray(pre[gap_m], dtype=np.int32),
            gap_post=np.ascontiguousarray(post[gap_m], dtype=np.int32),
            gap_w=np.ascontiguousarray(np.abs(sw[gap_m]).astype(np.float32)),
        )


def has_graded_or_gap(pack: ConnectomePack, csr: CSRGraph) -> bool:
    """Return True if any neuron uses MODEL_GRADED or any edge uses SYNAPSE_GAP_JUNCTION."""
    if bool(np.any(pack.cells.cell_model == MODEL_GRADED)):
        return True
    return bool(csr.kind.size > 0 and bool(np.any(csr.kind == SYNAPSE_GAP_JUNCTION)))


def make_initial_graded_state(num_cells: int, params: GradedParams) -> LIFState:
    """Create initial ``LIFState`` container resting at ``params.e_leak_mv``."""
    return LIFState(
        v=np.full(num_cells, np.float32(params.e_leak_mv), dtype=np.float32),
        g=np.zeros(num_cells, dtype=np.float32),
        refrac=np.zeros(num_cells, dtype=np.int16),
        delay_ring=np.zeros((1, num_cells), dtype=np.int32),
        substep=0,
        tick=0,
        stim_rate_hz=np.zeros(num_cells, dtype=np.float32),
        silenced=np.zeros(num_cells, dtype=np.uint8),
        last_tick_spikes=np.zeros(num_cells, dtype=np.int32),
        total_spikes=np.zeros(num_cells, dtype=np.int32),
    )


def make_initial_hybrid_state(
    pack: ConnectomePack,
    lif_params: LIFParams,
    graded_params: GradedParams,
) -> LIFState:
    """Create ``LIFState`` supporting mixed ``MODEL_SPIKING_LIF`` + ``MODEL_GRADED`` packs."""
    n = pack.cells.num_cells
    v = np.where(
        pack.cells.cell_model == MODEL_GRADED,
        np.float32(graded_params.e_leak_mv),
        np.float32(lif_params.v_0),
    ).astype(np.float32)
    ring_len = lif_params.delay_steps + 1
    return LIFState(
        v=v,
        g=np.zeros(n, dtype=np.float32),
        refrac=np.zeros(n, dtype=np.int16),
        delay_ring=np.zeros((ring_len, n), dtype=np.int32),
        substep=0,
        tick=0,
        stim_rate_hz=np.zeros(n, dtype=np.float32),
        silenced=np.zeros(n, dtype=np.uint8),
        last_tick_spikes=np.zeros(n, dtype=np.int32),
        total_spikes=np.zeros(n, dtype=np.int32),
    )


def _accumulate_conductances(
    v: F32,
    s_chem: F32,
    active: F32,
    circuit: GradedCircuit,
    params: GradedParams,
    *,
    gap_junctions_enabled: bool,
) -> tuple[F32, F32, F32, F32]:
    """Compute (G_exc, G_inh, G_gap, I_gap_nbr) using vectorized bincount."""
    n = circuit.num_cells
    if circuit.chem_exc_pre.size:
        w_exc = circuit.chem_exc_w * s_chem[circuit.chem_exc_pre]
        g_exc = (
            np.float32(params.g_chem)
            * np.bincount(circuit.chem_exc_post, weights=w_exc, minlength=n).astype(np.float32)
        ).astype(np.float32, copy=False)
    else:
        g_exc = np.zeros(n, dtype=np.float32)

    if circuit.chem_inh_pre.size:
        w_inh = circuit.chem_inh_w * s_chem[circuit.chem_inh_pre]
        g_inh = (
            np.float32(params.g_chem)
            * np.bincount(circuit.chem_inh_post, weights=w_inh, minlength=n).astype(np.float32)
        ).astype(np.float32, copy=False)
    else:
        g_inh = np.zeros(n, dtype=np.float32)

    if gap_junctions_enabled and circuit.gap_pre.size:
        eff_w = circuit.gap_w * active[circuit.gap_pre] * active[circuit.gap_post]
        g_gap_scale = np.float32(params.g_gap)
        g_gap = (
            g_gap_scale
            * np.bincount(circuit.gap_post, weights=eff_w, minlength=n).astype(np.float32)
        ).astype(np.float32, copy=False)
        i_gap_nbr = (
            g_gap_scale
            * np.bincount(circuit.gap_post, weights=eff_w * v[circuit.gap_pre], minlength=n).astype(
                np.float32
            )
        ).astype(np.float32, copy=False)
    else:
        g_gap = np.zeros(n, dtype=np.float32)
        i_gap_nbr = np.zeros(n, dtype=np.float32)

    return g_exc, g_inh, g_gap, i_gap_nbr


def step_graded_substep_cpu(
    state: LIFState,
    circuit: GradedCircuit,
    params: GradedParams,
    *,
    gap_junctions_enabled: bool = True,
    external_current: F32 | None = None,
) -> I32:
    """Advance one graded + gap-junction substep via exact diagonal-implicit exp Euler."""
    n = circuit.num_cells
    v = state.v
    active = (state.silenced == np.uint8(0)).astype(np.float32)

    slope = max(params.v_slope_mv, 1e-3)
    arg = np.clip((v.astype(np.float64) - params.v_half_mv) / slope, -20.0, 20.0)
    s_chem = ((1.0 / (1.0 + np.exp(-arg))).astype(np.float32) * active).astype(
        np.float32, copy=False
    )

    g_exc, g_inh, g_gap, i_gap_nbr = _accumulate_conductances(
        v,
        s_chem,
        active,
        circuit,
        params,
        gap_junctions_enabled=gap_junctions_enabled,
    )

    i_ext = (state.stim_rate_hz * np.float32(params.stim_current_scale)).astype(
        np.float32, copy=False
    )
    if external_current is not None:
        i_ext = (i_ext + external_current.astype(np.float32, copy=False)).astype(
            np.float32, copy=False
        )

    g_l = np.float32(params.g_leak)
    e_l = np.float32(params.e_leak_mv)
    e_exc = np.float32(params.e_exc_mv)
    e_inh = np.float32(params.e_inh_mv)

    g_tot = (g_l + g_exc + g_inh + g_gap).astype(np.float32, copy=False)
    i_drive = (g_l * e_l + g_exc * e_exc + g_inh * e_inh + i_gap_nbr + i_ext).astype(
        np.float32, copy=False
    )
    i_syn = (g_exc * (e_exc - v) + g_inh * (e_inh - v) + (i_gap_nbr - g_gap * v)).astype(
        np.float32, copy=False
    )

    inv_cm_dt = params.dt_ms / max(params.c_m, 1e-6)
    g_tot_f64 = g_tot.astype(np.float64)
    i_drive_f64 = i_drive.astype(np.float64)
    v_f64 = v.astype(np.float64)
    nonzero = g_tot_f64 > 1e-12
    v_next = np.empty(n, dtype=np.float64)
    if nonzero.any():
        v_inf = i_drive_f64[nonzero] / g_tot_f64[nonzero]
        alpha = np.exp(-g_tot_f64[nonzero] * inv_cm_dt)
        v_next[nonzero] = v_inf + alpha * (v_f64[nonzero] - v_inf)
    if (~nonzero).any():
        v_next[~nonzero] = v_f64[~nonzero] + i_drive_f64[~nonzero] * inv_cm_dt

    state.v[:] = np.clip(v_next, -90.0, 40.0).astype(np.float32)
    state.g[:] = i_syn
    state.substep += 1
    return np.flatnonzero(state.v > np.float32(params.v_active_mv)).astype(np.int32)


def step_graded_tick_cpu(
    state: LIFState,
    circuit: GradedCircuit,
    params: GradedParams,
    *,
    gap_junctions_enabled: bool = True,
    external_current: F32 | None = None,
) -> int:
    """Advance one tick (``substeps_per_tick`` substeps). Returns count of active graded cells."""
    for _ in range(params.substeps_per_tick):
        step_graded_substep_cpu(
            state,
            circuit,
            params,
            gap_junctions_enabled=gap_junctions_enabled,
            external_current=external_current,
        )
    active = (state.v > np.float32(params.v_active_mv)).astype(np.int32)
    state.last_tick_spikes[:] = active
    state.total_spikes += active
    state.tick += 1
    return int(active.sum())


def step_hybrid_tick_cpu(
    state: LIFState,
    pack: ConnectomePack,
    csr: CSRGraph,
    circuit: GradedCircuit,
    lif_params: LIFParams,
    graded_params: GradedParams,
    seed: int,
    *,
    gap_junctions_enabled: bool = True,
    lif_to_graded_gain: float = 1.0,
) -> int:
    """Advance one tick on a heterogeneous pack combining spiking LIF and ``MODEL_GRADED``.

    LIF neurons step strictly by ``step_substep_cpu`` on their indices while graded neurons evolve
    on ``MODEL_GRADED`` dynamics and also feel incoming spikes / gap coupling cleanly.
    """
    n = pack.cells.num_cells
    lif_mask = pack.cells.cell_model == MODEL_SPIKING_LIF
    graded_mask = pack.cells.cell_model == MODEL_GRADED
    if not lif_mask.any():
        return step_graded_tick_cpu(
            state, circuit, graded_params, gap_junctions_enabled=gap_junctions_enabled
        )

    # Save previous LIF voltages/g and restore selectively per population each substep.
    substeps = graded_params.substeps_per_tick
    lif_fired_any = np.zeros(n, dtype=np.int32)
    g_lif_params = LIFParams(
        v_0=lif_params.v_0,
        v_rst=lif_params.v_rst,
        v_th=lif_params.v_th,
        t_mbr_ms=lif_params.t_mbr_ms,
        tau_ms=lif_params.tau_ms,
        t_rfc_ms=lif_params.t_rfc_ms,
        t_dly_ms=lif_params.t_dly_ms,
        w_syn_mv=lif_params.w_syn_mv,
        f_poi=lif_params.f_poi,
        dt_ms=lif_params.dt_ms,
        substeps_per_tick=1,
        stim_no_refrac=lif_params.stim_no_refrac,
    )
    _ = math.isfinite(graded_params.dt_ms)
    state.last_tick_spikes.fill(np.int32(0))
    for _ in range(substeps):
        # 1. Compute graded -> LIF synaptic drive only when graded cell is strongly depolarized
        #    above reversal equilibrium (prevents tonic leakage from unstimulated equilibria).
        active = (state.silenced == np.uint8(0)).astype(np.float32)
        v_thr = max(float(graded_params.e_exc_mv), 0.0)
        s_graded = (
            np.clip((state.v - np.float32(v_thr)) / np.float32(15.0), 0.0, 1.0)
            * active
            * graded_mask
        ).astype(np.float32, copy=False)
        if circuit.chem_exc_pre.size:
            ge_src = s_graded[circuit.chem_exc_pre]
            if np.any(ge_src > 0):
                drive_lif = np.bincount(
                    circuit.chem_exc_post,
                    weights=circuit.chem_exc_w * ge_src,
                    minlength=n,
                ).astype(np.float32)
                state.g[lif_mask] += np.float32(lif_params.w_syn_mv * 0.45) * drive_lif[lif_mask]
        if circuit.chem_inh_pre.size:
            gi_src = s_graded[circuit.chem_inh_pre]
            if np.any(gi_src > 0):
                inh_lif = np.bincount(
                    circuit.chem_inh_post,
                    weights=circuit.chem_inh_w * gi_src,
                    minlength=n,
                ).astype(np.float32)
                state.g[lif_mask] -= np.float32(lif_params.w_syn_mv * 0.45) * inh_lif[lif_mask]

        v_before_lif = state.v[graded_mask].copy()
        g_before_lif = state.g[graded_mask].copy()
        # Hold graded cells at LIF resting baseline during step_substep_cpu so graded potentials
        # above v_th (-45 mV) never trigger spurious LIF spikes or delay_ring scatter.
        state.v[graded_mask] = np.float32(g_lif_params.v_0)
        state.g[graded_mask] = np.float32(0.0)
        fired_lif = step_substep_cpu(state, csr, g_lif_params, seed)
        # 2. Collect LIF -> graded synaptic pulses (both excitatory and inhibitory).
        spike_vec = np.zeros(n, dtype=np.float32)
        if fired_lif.size:
            valid_f = fired_lif[lif_mask[fired_lif]]
            spike_vec[valid_f] = np.float32(lif_to_graded_gain)
            lif_fired_any[valid_f] += 1
        # Restore graded compartment state before advancing graded ODE substep.
        state.v[graded_mask] = v_before_lif
        state.g[graded_mask] = g_before_lif

        # Snapshot LIF compartment state while running the graded substep.
        v_lif_after = state.v[lif_mask].copy()
        g_lif_after = state.g[lif_mask].copy()
        sub_before = state.substep
        pulse_cur = np.zeros(n, dtype=np.float32)
        has_pulse = False
        if np.any(spike_vec > 0):
            if circuit.chem_exc_pre.size:
                s_p = spike_vec[circuit.chem_exc_pre]
                if np.any(s_p > 0):
                    pulse_cur += np.float32(graded_params.g_chem * 12.0) * np.bincount(
                        circuit.chem_exc_post,
                        weights=circuit.chem_exc_w * s_p,
                        minlength=n,
                    ).astype(np.float32)
                    has_pulse = True
            if circuit.chem_inh_pre.size:
                s_n = spike_vec[circuit.chem_inh_pre]
                if np.any(s_n > 0):
                    pulse_cur -= np.float32(graded_params.g_chem * 12.0) * np.bincount(
                        circuit.chem_inh_post,
                        weights=circuit.chem_inh_w * s_n,
                        minlength=n,
                    ).astype(np.float32)
                    has_pulse = True
        step_graded_substep_cpu(
            state,
            circuit,
            graded_params,
            gap_junctions_enabled=gap_junctions_enabled,
            external_current=pulse_cur if has_pulse else None,
        )
        state.substep = sub_before  # keep substep matching LIF clock increment
        state.v[lif_mask] = v_lif_after
        state.g[lif_mask] = g_lif_after

    state.last_tick_spikes[lif_mask] = lif_fired_any[lif_mask]
    graded_active = (state.v > np.float32(graded_params.v_active_mv)) & graded_mask
    state.last_tick_spikes[graded_mask] = graded_active[graded_mask].astype(np.int32)
    state.total_spikes[graded_mask] += state.last_tick_spikes[graded_mask]
    state.tick += 1
    return int((state.last_tick_spikes > 0).sum())
