"""Leaky integrate-and-fire (LIF) whole-brain engine — CPU oracle and shared types (F1).

Implements the neuron and synapse dynamics of Shiu et al. (2024), Nature 634:210-219
(``philshiu/Drosophila_brain_model``), discretized with exact linear ODE integration over
``dt = 0.1 ms`` (10 substeps per 1 ms tick):

    dv/dt = (v_0 - v + g) / t_mbr     (outside refractory period)
    dg/dt = -g / tau

Design for CPU/GPU bitwise reproducibility (connectome plan section 3.3):
- Incoming delayed synaptic events are accumulated as exact ``int32`` sums of signed
  synapse counts per arrival slot in a ring buffer; integer addition is associative and
  commutative regardless of parallel thread scheduling on the GPU.
- External Poisson stimulation uses a stateless counter-based SplitMix64 hash of
  ``(seed, substep, neuron_index)``, avoiding divergent RNG streams across CPU and GPU.
- All floating-point state updates use IEEE-754 ``float32`` with explicit non-FMA operation
  order ``(v_0 + (alpha_v * (v - v_0))) + (beta_vg * g)`` matching the CUDA kernel compiled
  with ``--fmad=false`` and ``__fadd_rn`` / ``__fmul_rn``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.schema import (
    MODEL_SPIKING_LIF,
    POS_SOMA,
    ROLE_INTERNEURON,
    ROLE_MOTOR,
    ROLE_SENSORY,
    SYNAPSE_CHEMICAL,
    CellTable,
    ConnectomePack,
    CSRGraph,
    DatasetProvenance,
    EdgeTable,
    FileProvenance,
    IOPort,
    compute_pack_digest,
)

F32 = npt.NDArray[np.float32]
I16 = npt.NDArray[np.int16]
I32 = npt.NDArray[np.int32]
U8 = npt.NDArray[np.uint8]
U64 = npt.NDArray[np.uint64]

# SplitMix64 constants (shared verbatim with lif_cupy.py).
_SM_C1 = np.uint64(0x9E3779B97F4A7C15)
_SM_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SM_M2 = np.uint64(0x94D049BB133111EB)
_SM_SALT_STEP = np.uint64(0xD1B54A32D192ED03)
_SM_SALT_NEURON = np.uint64(0x94D049BB133111EB)
_INV_2_53 = 1.0 / float(1 << 53)


@dataclass(frozen=True)
class LIFParams:
    """Biophysical and integration constants (Shiu et al. 2024 defaults)."""

    v_0: float = -52.0
    v_rst: float = -52.0
    v_th: float = -45.0
    t_mbr_ms: float = 20.0
    tau_ms: float = 5.0
    t_rfc_ms: float = 2.2
    t_dly_ms: float = 1.8
    w_syn_mv: float = 0.275
    f_poi: float = 250.0
    dt_ms: float = 0.1
    substeps_per_tick: int = 10
    stim_no_refrac: bool = True
    tau_adapt_ms: float = 80.0
    weight_quant_div: float = 1.0

    @property
    def effective_w_syn_mv(self) -> float:
        return float(np.float32(self.w_syn_mv) / np.float32(self.weight_quant_div))

    @property
    def alpha_adapt_f32(self) -> np.float32:
        return np.float32(math.exp(-self.dt_ms / max(self.tau_adapt_ms, 1e-3)))

    @property
    def delay_steps(self) -> int:
        return max(1, round(self.t_dly_ms / self.dt_ms))

    @property
    def refrac_steps(self) -> int:
        return max(0, round(self.t_rfc_ms / self.dt_ms))

    @property
    def ring_len(self) -> int:
        return self.delay_steps + 1

    @property
    def stim_weight_mv(self) -> float:
        return float(np.float32(self.w_syn_mv) * np.float32(self.f_poi))

    @property
    def dt_s(self) -> float:
        return self.dt_ms * 1e-3

    def exact_coeffs_f32(self) -> tuple[np.float32, np.float32, np.float32]:
        """Exact linear ODE coefficients over dt_ms, rounded to float32."""
        av = math.exp(-self.dt_ms / self.t_mbr_ms)
        ag = math.exp(-self.dt_ms / self.tau_ms)
        if abs(self.t_mbr_ms - self.tau_ms) < 1e-12:
            bvg = (self.dt_ms / self.t_mbr_ms) * av
        else:
            bvg = (self.tau_ms / (self.t_mbr_ms - self.tau_ms)) * (av - ag)
        return np.float32(av), np.float32(ag), np.float32(bvg)


@dataclass
class LIFState:
    """Complete mutable state of a single connectome simulation (all checkpointed)."""

    v: F32  # (N,) membrane potential (mV)
    g: F32  # (N,) synaptic input variable (mV)
    refrac: I16  # (N,) remaining refractory substeps
    delay_ring: I32  # (R, N) queued integer signed synapse sums per slot
    substep: int
    tick: int
    stim_rate_hz: F32  # (N,) external Poisson stimulation rate (Hz)
    silenced: U8  # (N,) 1 = outgoing synapses zeroed (silencing intervention)
    last_tick_spikes: I32  # (N,) spikes emitted during the most recent tick
    total_spikes: I32  # (N,) cumulative spikes since tick 0
    # Optional neuron-local ALIF biophysics (None = exact Shiu et al. 2024 uniform LIF):
    adapt: F32 | None = None  # (N,) spike-frequency adaptation current (mV)
    bias_mv: F32 | None = None  # (N,) per-neuron tonic resting-potential shift (mV)
    adapt_inc_mv: F32 | None = None  # (N,) adaptation jump added on each spike (mV)

    @property
    def num_cells(self) -> int:
        return int(self.v.shape[0])

    @property
    def has_alif(self) -> bool:
        return self.adapt is not None and self.bias_mv is not None and self.adapt_inc_mv is not None

    def enable_alif(
        self,
        bias_mv: F32 | None = None,
        adapt_inc_mv: F32 | None = None,
    ) -> None:
        n = self.num_cells
        self.adapt = np.zeros(n, dtype=np.float32)
        self.bias_mv = (
            np.ascontiguousarray(bias_mv, dtype=np.float32)
            if bias_mv is not None
            else np.zeros(n, dtype=np.float32)
        )
        self.adapt_inc_mv = (
            np.ascontiguousarray(adapt_inc_mv, dtype=np.float32)
            if adapt_inc_mv is not None
            else np.zeros(n, dtype=np.float32)
        )

    def copy(self) -> LIFState:
        return LIFState(
            v=self.v.copy(),
            g=self.g.copy(),
            refrac=self.refrac.copy(),
            delay_ring=self.delay_ring.copy(),
            substep=self.substep,
            tick=self.tick,
            stim_rate_hz=self.stim_rate_hz.copy(),
            silenced=self.silenced.copy(),
            last_tick_spikes=self.last_tick_spikes.copy(),
            total_spikes=self.total_spikes.copy(),
            adapt=self.adapt.copy() if self.adapt is not None else None,
            bias_mv=self.bias_mv.copy() if self.bias_mv is not None else None,
            adapt_inc_mv=self.adapt_inc_mv.copy() if self.adapt_inc_mv is not None else None,
        )


def make_initial_state(num_cells: int, params: LIFParams) -> LIFState:
    return LIFState(
        v=np.full(num_cells, np.float32(params.v_0), dtype=np.float32),
        g=np.zeros(num_cells, dtype=np.float32),
        refrac=np.zeros(num_cells, dtype=np.int16),
        delay_ring=np.zeros((params.ring_len, num_cells), dtype=np.int32),
        substep=0,
        tick=0,
        stim_rate_hz=np.zeros(num_cells, dtype=np.float32),
        silenced=np.zeros(num_cells, dtype=np.uint8),
        last_tick_spikes=np.zeros(num_cells, dtype=np.int32),
        total_spikes=np.zeros(num_cells, dtype=np.int32),
    )


def splitmix64_u01_f32(seed: int, substep: int, indices: I32) -> F32:
    """Stateless deterministic uniform [0, 1) float32 matching the CUDA kernel bit-for-bit."""
    with np.errstate(over="ignore"):
        z = (
            np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
            + np.uint64(substep & 0xFFFFFFFFFFFFFFFF) * _SM_SALT_STEP
            + indices.astype(np.uint64) * _SM_SALT_NEURON
            + _SM_C1
        )
        z = (z ^ (z >> np.uint64(30))) * _SM_M1
        z = (z ^ (z >> np.uint64(27))) * _SM_M2
        z = z ^ (z >> np.uint64(31))
    # Top 24 bits -> exact IEEE-754 float32 in [0, 1) with step 2^-24 (identical on CPU and GPU).
    return ((z >> np.uint64(40)).astype(np.float32)) * np.float32(1.0 / float(1 << 24))


def step_substep_cpu(
    state: LIFState,
    csr: CSRGraph,
    params: LIFParams,
    seed: int,
) -> I32:
    """Advance one substep (dt_ms) on CPU in exact float32 non-FMA arithmetic.

    Returns the sorted array of neuron indices that fired on this substep.
    """
    v_0 = np.float32(params.v_0)
    v_rst = np.float32(params.v_rst)
    v_th = np.float32(params.v_th)
    w_syn = np.float32(params.effective_w_syn_mv)
    stim_w = np.float32(params.stim_weight_mv)
    dt_s_f32 = np.float32(params.dt_s)
    alpha_v, alpha_g, beta_vg = params.exact_coeffs_f32()
    alpha_a = params.alpha_adapt_f32
    refrac_steps = np.int16(params.refrac_steps)
    delay_steps = params.delay_steps
    ring_len = params.ring_len
    use_alif = state.has_alif

    s = state.substep
    slot = s % ring_len
    dst_slot = (s + delay_steps) % ring_len

    # 1. Deliver queued integer synaptic sums arriving at this substep.
    acc = state.delay_ring[slot]
    nz = np.flatnonzero(acc)
    if nz.size:
        delta_g = acc[nz].astype(np.float32) * w_syn
        state.g[nz] = (state.g[nz] + delta_g).astype(np.float32, copy=False)
        acc[nz] = 0

    # 2. Apply counter-based Poisson stimulation (target_var='v', matching Shiu et al.).
    stim_idx = np.flatnonzero(state.stim_rate_hz > np.float32(0.0)).astype(np.int32)
    if stim_idx.size:
        u = splitmix64_u01_f32(seed, s, stim_idx)
        p = (state.stim_rate_hz[stim_idx] * dt_s_f32).astype(np.float32, copy=False)
        hits = stim_idx[u < p]
        if hits.size:
            state.v[hits] = (state.v[hits] + stim_w).astype(np.float32, copy=False)
            if params.stim_no_refrac:
                state.refrac[hits] = np.int16(0)

    # 3. Spike detection, reset, and exact ODE integration for non-spiking neurons.
    #    Order matches CUDA lif_step_neurons kernel op-for-op (no FMA).
    can_spike = state.refrac == np.int16(0)
    spiked_mask = can_spike & (state.v > v_th)
    fired = np.flatnonzero(spiked_mask).astype(np.int32)

    # Non-spiking, non-refractory neurons integrate v; refractory neurons decrement counter.
    integrate_v = can_spike & (~spiked_mask)
    if integrate_v.any():
        if use_alif:
            assert state.bias_mv is not None and state.adapt is not None
            v_eq = ((v_0 + state.bias_mv[integrate_v]) - state.adapt[integrate_v]).astype(
                np.float32, copy=False
            )
            dv = (state.v[integrate_v] - v_eq).astype(np.float32, copy=False)
            v_decay = (alpha_v * dv).astype(np.float32, copy=False)
            g_drive = (beta_vg * state.g[integrate_v]).astype(np.float32, copy=False)
            state.v[integrate_v] = ((v_eq + v_decay) + g_drive).astype(np.float32, copy=False)
        else:
            dv = state.v[integrate_v] - v_0
            v_decay = alpha_v * dv
            g_drive = beta_vg * state.g[integrate_v]
            state.v[integrate_v] = (v_0 + v_decay) + g_drive

    in_refrac = ~can_spike
    if in_refrac.any():
        state.refrac[in_refrac] = (state.refrac[in_refrac] - np.int16(1)).astype(
            np.int16, copy=False
        )

    # g and adapt decay for all non-spiking neurons (spiking neurons reset g=0 & bump adapt below).
    not_spiked = ~spiked_mask
    if not_spiked.any():
        state.g[not_spiked] = (alpha_g * state.g[not_spiked]).astype(np.float32, copy=False)
        if use_alif and state.adapt is not None:
            state.adapt[not_spiked] = (alpha_a * state.adapt[not_spiked]).astype(
                np.float32, copy=False
            )

    # Reset spiking neurons and accumulate tick/total spike counts.
    if fired.size:
        state.v[fired] = v_rst
        state.g[fired] = np.float32(0.0)
        if use_alif and state.adapt is not None and state.adapt_inc_mv is not None:
            state.adapt[fired] = (
                (alpha_a * state.adapt[fired]).astype(np.float32, copy=False)
                + state.adapt_inc_mv[fired]
            ).astype(np.float32, copy=False)
        if params.stim_no_refrac:
            refrac_vals = np.where(
                state.stim_rate_hz[fired] > np.float32(0.0), np.int16(0), refrac_steps
            )
            state.refrac[fired] = refrac_vals
        else:
            state.refrac[fired] = refrac_steps
        state.last_tick_spikes[fired] += np.int32(1)
        state.total_spikes[fired] += np.int32(1)

        # 4. Event-driven scatter of integer signed synapse weights into dst_slot.
        active_senders = fired[state.silenced[fired] == np.uint8(0)]
        if active_senders.size:
            dst = state.delay_ring[dst_slot]
            indptr = csr.indptr
            post = csr.post
            weights = csr.signed_weight
            for u_idx in active_senders:
                start = int(indptr[u_idx])
                end = int(indptr[u_idx + 1])
                if start < end:
                    np.add.at(dst, post[start:end], weights[start:end])

    state.substep = s + 1
    return fired


def step_tick_cpu(
    state: LIFState,
    csr: CSRGraph,
    params: LIFParams,
    seed: int,
) -> int:
    """Advance one tick (default 10 substeps = 1.0 ms) on CPU. Returns spikes emitted in tick."""
    state.last_tick_spikes.fill(np.int32(0))
    total = 0
    for _ in range(params.substeps_per_tick):
        fired = step_substep_cpu(state, csr, params, seed)
        total += int(fired.size)
    state.tick += 1
    return total


def make_shuffled_csr(csr: CSRGraph, seed: int) -> CSRGraph:
    """Degree-preserving edge-shuffled null connectome (comparison control brain).

    Preserves each presynaptic neuron's out-degree, synapse-count multiset and Dale sign
    while randomly permuting postsynaptic targets deterministically from ``seed``.
    Any self-loops produced by the permutation are resolved by swapping with another edge.
    """
    n = csr.num_cells
    e = int(csr.post.shape[0])
    if e == 0:
        return csr
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC04EC7]))
    pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(csr.indptr))
    post = csr.post.copy()
    rng.shuffle(post)

    # Break any self-loops deterministically if n > 1.
    if n > 1:
        self_idx = np.flatnonzero(pre == post)
        for idx in self_idx:
            swap = int((idx + 1) % e)
            while post[swap] == pre[idx] or post[idx] == pre[swap]:
                swap = (swap + 1) % e
                if swap == idx:
                    break
            post[idx], post[swap] = post[swap], post[idx]

    weights = csr.signed_weight
    count = np.abs(weights).astype(np.int32)
    sign = np.where(weights > 0, np.int8(1), np.where(weights < 0, np.int8(-1), np.int8(0)))
    # Note: shuffling can merge duplicate (pre, post) pairs; sum them so CSR stays simple.
    keys = (pre.astype(np.uint64) << np.uint64(32)) | post.astype(np.uint64)
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    pre_s = pre[order]
    post_s = post[order]
    w_s = weights[order]
    kind_s = csr.kind[order]
    if e > 1 and bool(np.any(keys_s[1:] == keys_s[:-1])):
        uniq_mask = np.empty(e, dtype=bool)
        uniq_mask[0] = True
        uniq_mask[1:] = keys_s[1:] != keys_s[:-1]
        starts = np.flatnonzero(uniq_mask)
        w_merged = np.add.reduceat(w_s, starts).astype(np.int32)
        keep = w_merged != 0
        pre_s = pre_s[starts][keep]
        post_s = post_s[starts][keep]
        w_s = w_merged[keep]
        kind_s = kind_s[starts][keep]
    else:
        _ = (count, sign)

    counts = np.bincount(pre_s, minlength=n)
    indptr = np.empty(n + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return CSRGraph(
        num_cells=n,
        indptr=indptr,
        post=np.ascontiguousarray(post_s, dtype=np.int32),
        signed_weight=np.ascontiguousarray(w_s, dtype=np.int32),
        kind=np.ascontiguousarray(kind_s, dtype=np.uint8),
    )


def make_synthetic_tiny_pack() -> ConnectomePack:
    """Built-in 4-neuron synthetic circuit for offline tests and CPU-only CI.

    Topology:
      0 (sugar_grn, sensory) --(+180)--> 1 (interneuron) --(+180)--> 2 (mn9_proboscis_extension)
      3 (bitter_grn, sensory) --(-240)--> 2 (mn9_proboscis_extension)
    """
    n = 4
    cells = CellTable(
        source_id=np.array([10, 11, 12, 13], dtype=np.int64),
        species_names=("synthetic_tiny",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=("LB3c", "IN1", "CB0701", "LB1a"),
        super_class=("sensory", "central", "motor", "sensory"),
        cell_class=("gustatory", "", "brain_motor_neuron", "gustatory"),
        cell_sub_class=("sugar", "", "ingestion_motor_neuron", "bitter"),
        region=("GNG", "GNG", "GNG", "GNG"),
        side=("left", "left", "right", "right"),
        role=np.array([ROLE_SENSORY, ROLE_INTERNEURON, ROLE_MOTOR, ROLE_SENSORY], dtype=np.uint8),
        cell_model=np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8),
        neurotransmitter=("acetylcholine", "acetylcholine", "glutamate", "gaba"),
        position_nm=np.array(
            [
                [0.0, 0.0, 0.0],
                [1000.0, 0.0, 0.0],
                [2000.0, 0.0, 0.0],
                [1000.0, 1000.0, 0.0],
            ],
            dtype=np.float32,
        ),
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    edges = EdgeTable(
        pre=np.array([0, 1, 3], dtype=np.int32),
        post=np.array([1, 2, 2], dtype=np.int32),
        count=np.array([180, 180, 240], dtype=np.int32),
        sign=np.array([1, 1, -1], dtype=np.int8),
        kind=np.full(3, SYNAPSE_CHEMICAL, dtype=np.uint8),
    )
    io_ports = {
        "sugar_grn": IOPort(
            port_id="sugar_grn",
            direction="sensory",
            label="Synthetic sugar sensory",
            description="Excitatory sensory input driving interneuron 1.",
            cell_indices=np.array([0], dtype=np.int32),
            selection_rule="synthetic index 0",
        ),
        "bitter_grn": IOPort(
            port_id="bitter_grn",
            direction="sensory",
            label="Synthetic bitter sensory",
            description="Inhibitory sensory input suppressing motor neuron 2.",
            cell_indices=np.array([3], dtype=np.int32),
            selection_rule="synthetic index 3",
        ),
        "mn9_proboscis_extension": IOPort(
            port_id="mn9_proboscis_extension",
            direction="motor",
            label="Synthetic MN9 output",
            description="Motor neuron receiving excitatory relay and bitter inhibition.",
            cell_indices=np.array([2], dtype=np.int32),
            selection_rule="synthetic index 2",
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_tiny",
        species="synthetic_tiny",
        version="v1",
        citation="Built-in 4-neuron analytic/CI test circuit.",
        licence_note="project",
        files=(
            FileProvenance(
                filename="builtin_synthetic_tiny.json",
                url="https://example.invalid/builtin",
                sha256="0" * 64,
                size_bytes=0,
                source_label="builtin",
                licence_note="project",
            ),
        ),
    )
    digest = compute_pack_digest("synthetic_tiny", "synthetic_tiny", cells, edges, io_ports, prov)
    return ConnectomePack(
        dataset_id="synthetic_tiny",
        species="synthetic_tiny",
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
        pack_digest=digest,
    )


def make_synthetic_mb_pack() -> ConnectomePack:
    """Built-in 8-neuron synthetic Mushroom Body circuit for offline F4 plasticity tests & CI.

    Topology (models canonical gamma1pedc -> gamma5 feedforward disinhibition):
      0, 1 (odor_a_kc, KCg-m) --(+140)--> 4 (MBON11, GABAergic approach)
      0, 1 (odor_a_kc, KCg-m) --(+115)--> 5 (MBON01, avoidance)
      2, 3 (odor_b_kc, KCg-m) --(+140)--> 4 (MBON11, GABAergic approach)
      2, 3 (odor_b_kc, KCg-m) --(+115)--> 5 (MBON01, avoidance)
      4 (MBON11, inhibitory)  --(-240)--> 5 (MBON01, avoidance)
      6 (PPL101, punishment DAN) --(+15)--> 4 (MBON11, defines gamma1pedc DAN compartment)
      7 (PAM01, reward DAN)      --(+15)--> 5 (MBON01, defines gamma5 DAN compartment)
    """
    n = 8
    cells = CellTable(
        source_id=np.array([20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
        species_names=("synthetic_mb",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=("KCg-m", "KCg-m", "KCg-m", "KCg-m", "MBON11", "MBON01", "PPL101", "PAM01"),
        super_class=(
            "central",
            "central",
            "central",
            "central",
            "central",
            "central",
            "central",
            "central",
        ),
        cell_class=(
            "Kenyon_Cell",
            "Kenyon_Cell",
            "Kenyon_Cell",
            "Kenyon_Cell",
            "MBON",
            "MBON",
            "DAN",
            "DAN",
        ),
        cell_sub_class=("KCg", "KCg", "KCg", "KCg", "", "", "", ""),
        region=(
            "MB_CA_L",
            "MB_CA_R",
            "MB_CA_L",
            "MB_CA_R",
            "MB_PED_L",
            "MB_ML_L",
            "MB_PED_L",
            "MB_ML_L",
        ),
        side=("left", "right", "left", "right", "left", "left", "left", "left"),
        role=np.array(
            [
                ROLE_SENSORY,
                ROLE_SENSORY,
                ROLE_SENSORY,
                ROLE_SENSORY,
                ROLE_MOTOR,
                ROLE_MOTOR,
                ROLE_SENSORY,
                ROLE_SENSORY,
            ],
            dtype=np.uint8,
        ),
        cell_model=np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8),
        neurotransmitter=(
            "acetylcholine",
            "acetylcholine",
            "acetylcholine",
            "acetylcholine",
            "gaba",
            "glutamate",
            "dopamine",
            "dopamine",
        ),
        position_nm=np.array(
            [
                [-1200.0, 900.0, 0.0],
                [-1200.0, 300.0, 0.0],
                [-1200.0, -300.0, 0.0],
                [-1200.0, -900.0, 0.0],
                [600.0, 500.0, 0.0],
                [1400.0, -500.0, 0.0],
                [600.0, 1300.0, 0.0],
                [1400.0, -1300.0, 0.0],
            ],
            dtype=np.float32,
        ),
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    edges = EdgeTable(
        pre=np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 6, 7], dtype=np.int32),
        post=np.array([4, 5, 4, 5, 4, 5, 4, 5, 5, 4, 5], dtype=np.int32),
        count=np.array([150, 125, 150, 125, 150, 125, 150, 125, 240, 15, 15], dtype=np.int32),
        sign=np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1], dtype=np.int8),
        kind=np.full(11, SYNAPSE_CHEMICAL, dtype=np.uint8),
    )
    io_ports = {
        "odor_a_kc": IOPort(
            "odor_a_kc",
            "sensory",
            "Odor A KCs (synthetic)",
            "Synthetic Odor A Kenyon cells (indices 0, 1).",
            np.array([0, 1], dtype=np.int32),
            "index in (0, 1)",
        ),
        "odor_b_kc": IOPort(
            "odor_b_kc",
            "sensory",
            "Odor B KCs (synthetic)",
            "Synthetic Odor B Kenyon cells (indices 2, 3).",
            np.array([2, 3], dtype=np.int32),
            "index in (2, 3)",
        ),
        "ppl1_punishment_dan": IOPort(
            "ppl1_punishment_dan",
            "sensory",
            "PPL101 punishment DAN (synthetic)",
            "Synthetic punishment dopaminergic neuron innervating MBON11.",
            np.array([6], dtype=np.int32),
            "cell_type == 'PPL101'",
        ),
        "pam_reward_dan": IOPort(
            "pam_reward_dan",
            "sensory",
            "PAM01 reward DAN (synthetic)",
            "Synthetic reward dopaminergic neuron innervating MBON01.",
            np.array([7], dtype=np.int32),
            "cell_type == 'PAM01'",
        ),
        "mbon11_approach": IOPort(
            "mbon11_approach",
            "motor",
            "MBON11 approach output (synthetic)",
            "Synthetic GABAergic approach MBON (inhibits MBON01).",
            np.array([4], dtype=np.int32),
            "cell_type == 'MBON11'",
        ),
        "mbon01_avoidance": IOPort(
            "mbon01_avoidance",
            "motor",
            "MBON01 avoidance output (synthetic)",
            "Synthetic avoidance MBON disinhibited when MBON11 is depressed.",
            np.array([5], dtype=np.int32),
            "cell_type == 'MBON01'",
        ),
        "mushroom_body_kc": IOPort(
            "mushroom_body_kc",
            "circuit",
            "Kenyon cells (synthetic)",
            "All synthetic Kenyon cells.",
            np.array([0, 1, 2, 3], dtype=np.int32),
            "cell_class == 'Kenyon_Cell'",
        ),
        "mushroom_body_mbon": IOPort(
            "mushroom_body_mbon",
            "circuit",
            "MBONs (synthetic)",
            "All synthetic MBONs.",
            np.array([4, 5], dtype=np.int32),
            "cell_class == 'MBON'",
        ),
        "mushroom_body_dan": IOPort(
            "mushroom_body_dan",
            "circuit",
            "DANs (synthetic)",
            "All synthetic DANs.",
            np.array([6, 7], dtype=np.int32),
            "cell_class == 'DAN'",
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_mb",
        species="synthetic_mb",
        version="v1",
        citation=(
            "Built-in 8-neuron synthetic Mushroom Body circuit for offline F4 plasticity tests."
        ),
        licence_note="Project source (offline test fixture).",
        files=(
            FileProvenance("builtin", "builtin://synthetic_mb", "0" * 64, 0, "builtin", "project"),
        ),
    )
    digest = compute_pack_digest("synthetic_mb", "synthetic_mb", cells, edges, io_ports, prov)
    return ConnectomePack(
        dataset_id="synthetic_mb",
        species="synthetic_mb",
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
        pack_digest=digest,
    )
