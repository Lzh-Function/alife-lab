"""Connectome spiking LIF substrate adapter (Shiu et al. 2024, F1).

Supports CPU oracle (``numpy``) and CUDA C++ / CUDA-graph GPU execution (``cupy``) with
exact bitwise equivalence across backends and checkpoints. Never imports CuPy unless the
``cupy`` backend is selected inside a worker process.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.connectome import (
    CELL_ROLE_NAMES,
    MODEL_GRADED,
    MODEL_SPIKING_LIF,
    ConnectomePack,
    CSRGraph,
    load_connectome_pack,
)
from alife_lab.connectome.celegans import make_synthetic_celegans_302_pack
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
    LIFState,
    make_initial_state,
    make_shuffled_csr,
    make_synthetic_mb_pack,
    make_synthetic_tiny_pack,
    step_tick_cpu,
)
from alife_lab.connectome.plasticity import (
    MBPlasticityParams,
    PlasticityState,
    PlasticSynapseMap,
    build_plastic_synapse_map,
    effective_weights,
    make_initial_plasticity_state,
    plasticity_summary,
    step_plasticity_tick,
)
from alife_lab.core.spec import ConnectomeExperiment, ExperimentSpec
from alife_lab.service.settings import load_settings
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest

SERIALIZE_VERSION = 1
_TOP_REGIONS = 16

# Process-local cache of immutable loaded packs & CSRs so worker restarts/replays stay fast.
_PACK_CACHE: dict[tuple[str, str], tuple[ConnectomePack, CSRGraph]] = {}
_SHUFFLED_CSR_CACHE: dict[tuple[str, int], CSRGraph] = {}
# Cached top-region lookup per pack_digest: (region_names, region_counts, cell_to_top_region)
_RegionRenderIndex = tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int32]],
    np.ndarray[Any, np.dtype[np.int16]],
]
_REGION_RENDER_CACHE: dict[str, _RegionRenderIndex] = {}


def _region_render_index(pack: ConnectomePack) -> _RegionRenderIndex:
    cached = _REGION_RENDER_CACHE.get(pack.pack_digest)
    if cached is not None:
        return cached
    counts = Counter(pack.cells.region)
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_TOP_REGIONS]
    names = tuple(r for r, _ in top)
    totals = np.asarray([int(c) for _, c in top], dtype=np.int32)
    name_to_idx = {r: i for i, r in enumerate(names)}
    mapping = np.full(pack.cells.num_cells, -1, dtype=np.int16)
    for i, r in enumerate(pack.cells.region):
        j = name_to_idx.get(r)
        if j is not None:
            mapping[i] = j
    cached = (names, totals, mapping)
    _REGION_RENDER_CACHE[pack.pack_digest] = cached
    return cached


def _ce(spec: ExperimentSpec) -> ConnectomeExperiment:
    exp = spec.experiment
    if not isinstance(exp, ConnectomeExperiment):
        raise TypeError("connectome adapter received a different spec")
    return exp


def _params_from_exp(exp: ConnectomeExperiment) -> LIFParams:
    d = exp.dynamics
    return LIFParams(
        v_0=d.v_0,
        v_rst=d.v_rst,
        v_th=d.v_th,
        t_mbr_ms=d.t_mbr_ms,
        tau_ms=d.tau_ms,
        t_rfc_ms=d.t_rfc_ms,
        t_dly_ms=d.t_dly_ms,
        w_syn_mv=d.w_syn_mv,
        f_poi=d.f_poi,
        dt_ms=d.dt_ms,
        substeps_per_tick=d.substeps_per_tick,
        stim_no_refrac=d.stim_no_refrac,
    )


def _plasticity_from_exp(exp: ConnectomeExperiment) -> MBPlasticityParams:
    p = exp.plasticity
    if p is None or not p.enabled:
        return MBPlasticityParams(enabled=False)
    return MBPlasticityParams(
        enabled=True,
        tau_elig_ms=p.tau_elig_ms,
        tau_dan_ms=p.tau_dan_ms,
        dan_threshold=p.dan_threshold,
        lr_ltd=p.lr_ltd,
        lr_ext=p.lr_ext,
        w_min=p.w_min,
        w_max=p.w_max,
        min_dan_synapses=p.min_dan_synapses,
    )


_PLASTIC_MAP_CACHE: dict[tuple[str, str, int, int], PlasticSynapseMap] = {}


def _get_pack_and_csr(exp: ConnectomeExperiment) -> tuple[ConnectomePack, CSRGraph]:
    ds_dir = load_settings().resolved_datasets_dir
    key = (exp.dataset_id, str(ds_dir))
    cached = _PACK_CACHE.get(key)
    if cached is None:
        if exp.dataset_id == "synthetic_tiny":
            pack = make_synthetic_tiny_pack()
        elif exp.dataset_id == "synthetic_mb":
            pack = make_synthetic_mb_pack()
        elif exp.dataset_id == "synthetic_celegans_302":
            pack = make_synthetic_celegans_302_pack()
        elif exp.dataset_id.startswith("synthetic_"):
            pack = load_connectome_pack(exp.dataset_id, ds_dir, verify_hashes=False)
        else:
            pack = load_connectome_pack(exp.dataset_id, ds_dir, verify_hashes=True)
        csr = pack.edges.to_csr(pack.cells.num_cells)
        cached = (pack, csr)
        _PACK_CACHE[key] = cached
    pack, base_csr = cached
    if exp.wiring == "shuffled_control":
        skey = (pack.pack_digest, int(exp.seed))
        shuf = _SHUFFLED_CSR_CACHE.get(skey)
        if shuf is None:
            shuf = make_shuffled_csr(base_csr, int(exp.seed))
            _SHUFFLED_CSR_CACHE[skey] = shuf
        return pack, shuf
    return pack, base_csr


@dataclass
class ConnectomeWorld:
    """Mutable state + immutable pack/CSR references + optional resident GPU runner & plasticity."""

    pack: ConnectomePack
    csr: CSRGraph
    params: LIFParams
    lif: LIFState
    backend: str
    gpu_runner: Any = None
    gpu_synced_to_host: bool = True
    plasticity_params: MBPlasticityParams = field(
        default_factory=lambda: MBPlasticityParams(enabled=False)
    )
    plastic_map: PlasticSynapseMap | None = None
    plastic_state: PlasticityState | None = None
    graded_circuit: GradedCircuit | None = None
    graded_params: GradedParams | None = None

    @property
    def tick(self) -> int:
        return self.lif.tick

    def ensure_host(self) -> LIFState:
        if self.gpu_runner is not None and not self.gpu_synced_to_host:
            self.gpu_runner.download_state(self.lif)
            self.gpu_synced_to_host = True
        return self.lif


_M = MetricDefinition
_METRICS = (
    _M("spikes_tick", "metric.spikes_tick", "spikes", "tick",
       "Total spikes emitted across all neurons during the current 1 ms tick.", "sum",
       "every tick", 1, "never null"),
    _M("active_neurons_tick", "metric.active_neurons_tick", "neurons", "tick",
       "Number of distinct neurons that fired at least once during the current tick.", "none",
       "every tick", 1, "never null"),
    _M("total_spikes", "metric.total_spikes", "spikes", "tick",
       "Cumulative spikes emitted across all neurons since tick 0.", "sum",
       "every tick", 1, "never null"),
    _M("mean_firing_rate_hz", "metric.mean_firing_rate_hz", "Hz", "tick",
       "Population mean firing rate over the elapsed simulated time.", "mean",
       "every tick", 1, "never null"),
    _M("mn9_spikes_tick", "metric.mn9_spikes_tick", "spikes", "tick",
       "Spikes emitted by MN9 proboscis extension motor neurons during the current tick.", "sum",
       "every tick", 1, "never null"),
    _M("mn9_rate_hz", "metric.mn9_rate_hz", "Hz", "tick",
       "Mean firing rate of MN9 proboscis extension motor neurons over elapsed time.", "mean",
       "every tick", 1, "never null"),
    _M("mbon11_rate_hz", "metric.mbon11_rate_hz", "Hz", "tick",
       "Mean firing rate of MBON11 approach output neurons over elapsed time.", "mean",
       "every tick", 1, "never null"),
    _M("mbon01_rate_hz", "metric.mbon01_rate_hz", "Hz", "tick",
       "Mean firing rate of MBON01 avoidance output neurons over elapsed time.", "mean",
       "every tick", 1, "never null"),
    _M("plastic_changed_edges", "metric.plastic_changed_edges", "edges", "tick",
       "Number of plastic KC->MBON edges whose scale differs from 1.0 by more than 0.01.", "none",
       "every tick", 1, "never null"),
    _M("plastic_mean_scale", "metric.plastic_mean_scale", "scale", "tick",
       "Mean synaptic multiplier across all plastic KC->MBON edges (1.0 = naive).", "mean",
       "every tick", 1, "never null"),
    _M("plastic_min_scale", "metric.plastic_min_scale", "scale", "tick",
       "Minimum synaptic multiplier across plastic KC->MBON edges.", "min",
       "every tick", 1, "never null"),
    _M("mean_v_mv", "metric.mean_v_mv", "mV", "tick",
       "Mean membrane potential across all neurons.", "mean",
       "every tick", 1, "never null"),
    _M("time_ms", "metric.time_ms", "ms", "tick",
       "Elapsed simulated time in milliseconds.", "none",
       "every tick", 1, "never null"),
)  # fmt: skip

CONNECTOME_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.connectome",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="connectome",
    display_name_key="substrate.connectome",
    scientific_modes=("reference", "exploratory"),
    capabilities=Capabilities(
        state_kind="network",
        clock_unit="tick",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=True,
        supports_lineage=False,
        supports_differentiation=False,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=("stimulate", "silence", "clear_stimuli"),
    ),
    backends=("numpy", "cupy"),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": (
            "Whole-brain leaky integrate-and-fire model derived from FlyWire v783 "
            "(Shiu et al. 2024, Nature 634:210-219). Hypothesis-testing wiring model, "
            "not a biological brain."
        ),
        "paper_urls": [
            "https://www.nature.com/articles/s41586-024-07763-9",
            "https://www.nature.com/articles/s41586-024-07558-y",
        ],
        "implementation": "https://github.com/philshiu/Drosophila_brain_model",
    },
)


def _resolve_target_indices(
    pack: ConnectomePack, payload: dict[str, Any]
) -> np.ndarray[Any, np.dtype[np.int32]]:
    n = pack.cells.num_cells
    indices: list[int] = []
    port = payload.get("port")
    if port is not None:
        if not isinstance(port, str) or port not in pack.io_ports:
            known = ", ".join(sorted(pack.io_ports))
            raise ValueError(f"unknown connectome port {port!r} (available: {known})")
        indices.extend(int(i) for i in pack.io_ports[port].cell_indices)
    raw_cells = payload.get("cell_indices")
    if raw_cells is not None:
        if not isinstance(raw_cells, list) or not all(isinstance(x, int) for x in raw_cells):
            raise ValueError("cell_indices must be a list of integers")
        for idx in raw_cells:
            if idx < 0 or idx >= n:
                raise ValueError(f"cell index {idx} out of range [0, {n})")
            indices.append(idx)
    if not indices:
        raise ValueError("intervention requires 'port' and/or non-empty 'cell_indices'")
    return np.unique(np.asarray(indices, dtype=np.int32))


class ConnectomeAdapter:
    backend: str = "numpy"

    def set_backend(self, backend: str) -> None:
        if backend not in CONNECTOME_MANIFEST.backends:
            raise ValueError(f"unsupported connectome backend {backend!r}")
        self.backend = backend

    @property
    def manifest(self) -> PluginManifest:
        return CONNECTOME_MANIFEST

    def _make_world(
        self,
        exp: ConnectomeExperiment,
        lif: LIFState,
        plastic_state: PlasticityState | None = None,
    ) -> ConnectomeWorld:
        pack, base_csr = _get_pack_and_csr(exp)
        params = _params_from_exp(exp)
        is_graded = has_graded_or_gap(pack, base_csr)
        graded_circuit: GradedCircuit | None = None
        graded_params: GradedParams | None = None
        p_params = _plasticity_from_exp(exp)
        pmap: PlasticSynapseMap | None = None
        pstate: PlasticityState | None = None
        csr = base_csr

        if is_graded:
            graded_params = GradedParams()
            graded_circuit = GradedCircuit.from_csr(csr)
            has_lif = bool(np.any(pack.cells.cell_model == MODEL_SPIKING_LIF))
            if not has_lif:
                params = LIFParams(
                    v_0=graded_params.e_leak_mv,
                    v_rst=graded_params.e_leak_mv,
                    v_th=graded_params.v_active_mv,
                    dt_ms=graded_params.dt_ms,
                    substeps_per_tick=graded_params.substeps_per_tick,
                )

        if p_params.enabled and not is_graded:
            pkey = (
                pack.pack_digest,
                exp.wiring,
                int(exp.seed) if exp.wiring == "shuffled_control" else 0,
                p_params.min_dan_synapses,
            )
            pmap = _PLASTIC_MAP_CACHE.get(pkey)
            if pmap is None:
                pmap = build_plastic_synapse_map(
                    pack, base_csr, min_dan_synapses=p_params.min_dan_synapses
                )
                _PLASTIC_MAP_CACHE[pkey] = pmap
            # Clone signed_weight so per-world plasticity never mutates the shared cached CSR.
            csr = CSRGraph(
                num_cells=base_csr.num_cells,
                indptr=base_csr.indptr,
                post=base_csr.post,
                signed_weight=base_csr.signed_weight.copy(),
                kind=base_csr.kind,
            )
            pstate = (
                plastic_state if plastic_state is not None else make_initial_plasticity_state(pmap)
            )
            if pmap.num_edges > 0:
                eff_w = effective_weights(pmap, pstate.scale)
                pstate.last_eff_weight[:] = eff_w
                csr.signed_weight[pmap.csr_edge_idx] = eff_w

        gpu_runner: Any = None
        backend = "numpy" if is_graded else self.backend
        if backend == "cupy":
            from alife_lab.connectome.lif_cupy import CupyLIFRunner

            gpu_runner = CupyLIFRunner(csr, params, int(exp.seed))
            gpu_runner.upload_state(lif)
        return ConnectomeWorld(
            pack=pack,
            csr=csr,
            params=params,
            lif=lif,
            backend=backend,
            gpu_runner=gpu_runner,
            gpu_synced_to_host=True,
            plasticity_params=p_params,
            plastic_map=pmap,
            plastic_state=pstate,
            graded_circuit=graded_circuit,
            graded_params=graded_params,
        )

    def initialize(self, spec: ExperimentSpec) -> ConnectomeWorld:
        exp = _ce(spec)
        pack, base_csr = _get_pack_and_csr(exp)
        params = _params_from_exp(exp)
        if has_graded_or_gap(pack, base_csr):
            has_lif = bool(np.any(pack.cells.cell_model == MODEL_SPIKING_LIF))
            has_grad = bool(np.any(pack.cells.cell_model == MODEL_GRADED))
            if has_lif and has_grad:
                lif = make_initial_hybrid_state(pack, params, GradedParams())
            else:
                lif = make_initial_graded_state(pack.cells.num_cells, GradedParams())
        else:
            lif = make_initial_state(pack.cells.num_cells, params)
        for stim in exp.stimuli:
            if stim.port not in pack.io_ports:
                known = ", ".join(sorted(pack.io_ports))
                raise ValueError(f"unknown stimulus port {stim.port!r} (available: {known})")
            lif.stim_rate_hz[pack.io_ports[stim.port].cell_indices] = np.float32(stim.rate_hz)
        for port_id in exp.silenced_ports:
            if port_id not in pack.io_ports:
                known = ", ".join(sorted(pack.io_ports))
                raise ValueError(f"unknown silenced port {port_id!r} (available: {known})")
            lif.silenced[pack.io_ports[port_id].cell_indices] = np.uint8(1)
        if exp.silenced_cell_types:
            wanted = set(exp.silenced_cell_types)
            matched_types: set[str] = set()
            for i, ct in enumerate(pack.cells.cell_type):
                if ct in wanted:
                    lif.silenced[i] = np.uint8(1)
                    matched_types.add(ct)
            missing = sorted(wanted - matched_types)
            if missing:
                raise ValueError(
                    f"unknown silenced cell_type(s) for {exp.dataset_id!r}: {', '.join(missing)}"
                )
        return self._make_world(exp, lif)

    def step(self, state: ConnectomeWorld, spec: ExperimentSpec) -> ConnectomeWorld:
        exp = _ce(spec)
        if state.graded_circuit is not None and state.graded_params is not None:
            has_lif = bool(np.any(state.pack.cells.cell_model == MODEL_SPIKING_LIF))
            has_grad = bool(np.any(state.pack.cells.cell_model == MODEL_GRADED))
            if has_lif and has_grad:
                step_hybrid_tick_cpu(
                    state.lif,
                    state.pack,
                    state.csr,
                    state.graded_circuit,
                    state.params,
                    state.graded_params,
                    int(exp.seed),
                )
            else:
                step_graded_tick_cpu(state.lif, state.graded_circuit, state.graded_params)
            return state
        if state.gpu_runner is not None:
            state.gpu_runner.step_tick(use_graph=True)
            state.lif.substep += state.params.substeps_per_tick
            state.lif.tick += 1
            state.gpu_synced_to_host = False
            if (
                state.plasticity_params.enabled
                and state.plastic_map is not None
                and state.plastic_state is not None
            ):
                last_sp = state.gpu_runner.download_last_tick_spikes()
                state.lif.last_tick_spikes[:] = last_sp
                dt_tick_ms = state.params.substeps_per_tick * state.params.dt_ms
                chg_idx, chg_w = step_plasticity_tick(
                    state.plastic_state,
                    state.plastic_map,
                    state.plasticity_params,
                    last_sp,
                    state.lif.silenced,
                    dt_tick_ms=dt_tick_ms,
                )
                if chg_idx.size > 0:
                    state.csr.signed_weight[chg_idx] = chg_w
                    state.gpu_runner.update_edge_weights(chg_idx, chg_w)
        else:
            step_tick_cpu(state.lif, state.csr, state.params, int(exp.seed))
            if (
                state.plasticity_params.enabled
                and state.plastic_map is not None
                and state.plastic_state is not None
            ):
                dt_tick_ms = state.params.substeps_per_tick * state.params.dt_ms
                chg_idx, chg_w = step_plasticity_tick(
                    state.plastic_state,
                    state.plastic_map,
                    state.plasticity_params,
                    state.lif.last_tick_spikes,
                    state.lif.silenced,
                    dt_tick_ms=dt_tick_ms,
                )
                if chg_idx.size > 0:
                    state.csr.signed_weight[chg_idx] = chg_w
        return state

    def metrics(self, state: ConnectomeWorld, spec: ExperimentSpec) -> dict[str, float]:
        exp = _ce(spec)
        wanted = set(exp.observations.metrics)
        lif = state.ensure_host()
        elapsed_s = max(lif.substep * state.params.dt_s, 1e-9)
        n = lif.num_cells
        spikes_tick = float(int(lif.last_tick_spikes.sum()))
        active_tick = float(int((lif.last_tick_spikes > 0).sum()))
        total_spikes = float(int(lif.total_spikes.sum()))
        mean_rate_hz = (total_spikes / (n * elapsed_s)) if (n > 0 and lif.substep > 0) else 0.0

        def _port_rate(port_id: str) -> tuple[float, float]:
            p = state.pack.io_ports.get(port_id)
            if p is None or p.cell_indices.size == 0:
                return 0.0, 0.0
            idx = p.cell_indices
            sp_tick = float(int(lif.last_tick_spikes[idx].sum()))
            sp_tot = float(int(lif.total_spikes[idx].sum()))
            rate = (sp_tot / (int(idx.size) * elapsed_s)) if lif.substep > 0 else 0.0
            return sp_tick, rate

        mn9_tick, mn9_rate = _port_rate("mn9_proboscis_extension")
        _, mbon11_rate = _port_rate("mbon11_approach")
        _, mbon01_rate = _port_rate("mbon01_avoidance")

        if state.plastic_state is not None and state.plastic_state.scale.size > 0:
            sc = state.plastic_state.scale
            p_changed = float(int((np.abs(sc - np.float32(1.0)) > np.float32(0.01)).sum()))
            p_mean = float(sc.mean())
            p_min = float(sc.min())
        else:
            p_changed = 0.0
            p_mean = 1.0
            p_min = 1.0

        out = {
            "spikes_tick": spikes_tick,
            "active_neurons_tick": active_tick,
            "total_spikes": total_spikes,
            "mean_firing_rate_hz": mean_rate_hz,
            "mn9_spikes_tick": mn9_tick,
            "mn9_rate_hz": mn9_rate,
            "mbon11_rate_hz": mbon11_rate,
            "mbon01_rate_hz": mbon01_rate,
            "plastic_changed_edges": p_changed,
            "plastic_mean_scale": p_mean,
            "plastic_min_scale": p_min,
            "mean_v_mv": float(lif.v.mean()) if n > 0 else float(state.params.v_0),
            "time_ms": float(lif.substep * state.params.dt_ms),
        }
        return {k: v for k, v in out.items() if k in wanted}

    def render(self, state: ConnectomeWorld) -> tuple[dict[str, Any], bytes]:
        """Structured connectome preview (codec ``connectome_v1``) for 2D raster/heatmap + 3D brain.

        Emits exact fired cell indices (little-endian ``int32``) as the binary payload alongside
        per-port and top-region firing summaries in the header. Pure function of ``LIFState`` so
        replay keyframe verification is byte-for-byte deterministic.
        """
        lif = state.ensure_host()
        n = lif.num_cells
        elapsed_s = max(lif.substep * state.params.dt_s, 1e-9)
        fired_idx = np.flatnonzero(lif.last_tick_spikes > 0).astype("<i4", copy=False)

        # Per-port activity summary (12 ports in FlyWire v783, 4 in synthetic_tiny).
        ports_summary: list[dict[str, Any]] = []
        for pid, port in state.pack.io_ports.items():
            idx = port.cell_indices
            cnt = int(idx.size)
            if cnt > 0:
                sp_tick = int(lif.last_tick_spikes[idx].sum())
                sp_tot = int(lif.total_spikes[idx].sum())
                rate = (sp_tot / (cnt * elapsed_s)) if lif.substep > 0 else 0.0
                stim_hz = float(lif.stim_rate_hz[idx].max())
                sil_cnt = int(lif.silenced[idx].sum())
            else:
                sp_tick = 0
                rate = 0.0
                stim_hz = 0.0
                sil_cnt = 0
            ports_summary.append(
                {
                    "port_id": pid,
                    "direction": port.direction,
                    "label": port.label,
                    "count": cnt,
                    "spikes_tick": sp_tick,
                    "rate_hz": round(rate, 2),
                    "stim_rate_hz": round(stim_hz, 2),
                    "silenced_count": sil_cnt,
                }
            )

        # Top-region firing intensity summary via vectorized bincount.
        reg_names, reg_counts, cell_to_reg = _region_render_index(state.pack)
        k = len(reg_names)
        if k > 0 and fired_idx.size > 0:
            fired_regs = cell_to_reg[fired_idx]
            valid = fired_regs >= 0
            reg_spikes = np.bincount(
                fired_regs[valid].astype(np.int32),
                weights=lif.last_tick_spikes[fired_idx[valid]].astype(np.float64),
                minlength=k,
            )
        else:
            reg_spikes = np.zeros(k, dtype=np.float64)

        regions_summary = []
        for j, rname in enumerate(reg_names):
            cnt = int(reg_counts[j])
            sp = int(reg_spikes[j])
            # Fraction of neurons firing in this tick, scaled so 5% active saturates at 1.0.
            intensity = min(1.0, sp / max(cnt * 0.05, 1.0))
            regions_summary.append(
                {
                    "region": rname,
                    "count": cnt,
                    "spikes_tick": sp,
                    "intensity": round(float(intensity), 4),
                }
            )

        header: dict[str, Any] = {
            "codec": "connectome_v1",
            "dataset_id": state.pack.dataset_id,
            "num_cells": n,
            "tick": int(lif.tick),
            "time_ms": round(lif.substep * state.params.dt_ms, 2),
            "spikes_tick": int(lif.last_tick_spikes.sum()),
            "active_neurons_tick": int(fired_idx.size),
            "fired_count": int(fired_idx.size),
            "stimulated_count": int((lif.stim_rate_hz > 0).sum()),
            "silenced_count": int((lif.silenced > 0).sum()),
            "ports": ports_summary,
            "regions": regions_summary,
        }
        if (
            state.plasticity_params.enabled
            and state.plastic_map is not None
            and state.plastic_state is not None
        ):
            header["plasticity"] = plasticity_summary(
                state.pack, state.plastic_map, state.plastic_state
            )
        return header, fired_idx.tobytes()

    def apply_intervention(
        self,
        state: ConnectomeWorld,
        spec: ExperimentSpec,
        kind: str,
        payload: dict[str, Any],
    ) -> tuple[ConnectomeWorld, dict[str, Any]]:
        lif = state.ensure_host()
        if kind == "clear_stimuli":
            prev_active = int((lif.stim_rate_hz > 0).sum())
            lif.stim_rate_hz.fill(np.float32(0.0))
            if state.gpu_runner is not None:
                state.gpu_runner.upload_controls(lif)
            return state, {"cleared_neurons": prev_active}

        if kind == "stimulate":
            try:
                rate_hz = float(payload.get("rate_hz", 150.0))
            except (TypeError, ValueError) as exc:
                raise ValueError("stimulate requires numeric rate_hz") from exc
            if not (0.0 <= rate_hz <= 1000.0):
                raise ValueError("rate_hz must be in [0, 1000]")
            idx = _resolve_target_indices(state.pack, payload)
            lif.stim_rate_hz[idx] = np.float32(rate_hz)
            if state.gpu_runner is not None:
                state.gpu_runner.upload_controls(lif)
            return state, {"affected_neurons": int(idx.size), "rate_hz": rate_hz}

        if kind == "silence":
            flag = bool(payload.get("silenced", True))
            idx = _resolve_target_indices(state.pack, payload)
            lif.silenced[idx] = np.uint8(1 if flag else 0)
            if state.gpu_runner is not None:
                state.gpu_runner.upload_controls(lif)
            return state, {"affected_neurons": int(idx.size), "silenced": flag}

        raise ValueError(f"unsupported connectome intervention kind {kind!r}")

    def inspect(
        self, state: ConnectomeWorld, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        lif = state.ensure_host()
        cells = state.pack.cells
        n = cells.num_cells
        port = query.get("port")
        if isinstance(port, str) and port in state.pack.io_ports:
            p = state.pack.io_ports[port]
            idx = p.cell_indices
            elapsed_s = max(lif.substep * state.params.dt_s, 1e-9)
            tot = int(lif.total_spikes[idx].sum()) if idx.size else 0
            return {
                "substrate": "connectome",
                "dataset_id": state.pack.dataset_id,
                "tick": lif.tick,
                "port": port,
                "count": int(idx.size),
                "spikes_tick": int(lif.last_tick_spikes[idx].sum()) if idx.size else 0,
                "total_spikes": tot,
                "mean_rate_hz": (tot / (int(idx.size) * elapsed_s))
                if (idx.size and lif.substep > 0)
                else 0.0,
                "mean_v_mv": float(lif.v[idx].mean()) if idx.size else float(state.params.v_0),
            }

        cell_idx = int(query.get("cell_index", 0))
        if cell_idx < 0 or cell_idx >= n:
            raise ValueError(f"cell_index {cell_idx} out of range [0, {n})")
        pos = cells.position_nm[cell_idx]
        sp_prov = {
            sp: round(float((cells.species_idx == i).mean()), 4)
            for i, sp in enumerate(cells.species_names)
        }
        return {
            "substrate": "connectome",
            "dataset_id": state.pack.dataset_id,
            "tick": lif.tick,
            "cell_index": cell_idx,
            "source_id": int(cells.source_id[cell_idx]),
            "species_name": cells.species_names[int(cells.species_idx[cell_idx])],
            "species_provenance": sp_prov,
            "cell_type": cells.cell_type[cell_idx],
            "super_class": cells.super_class[cell_idx],
            "cell_class": cells.cell_class[cell_idx],
            "region": cells.region[cell_idx],
            "side": cells.side[cell_idx],
            "role": CELL_ROLE_NAMES[int(cells.role[cell_idx])],
            "neurotransmitter": cells.neurotransmitter[cell_idx],
            "position_nm": [round(float(v), 1) for v in pos],
            "v_mv": float(lif.v[cell_idx]),
            "g_mv": float(lif.g[cell_idx]),
            "refrac_substeps": int(lif.refrac[cell_idx]),
            "stim_rate_hz": float(lif.stim_rate_hz[cell_idx]),
            "silenced": bool(lif.silenced[cell_idx]),
            "last_tick_spikes": int(lif.last_tick_spikes[cell_idx]),
            "total_spikes": int(lif.total_spikes[cell_idx]),
        }

    def state_distance(self, a: ConnectomeWorld, b: ConnectomeWorld) -> dict[str, Any]:
        la = a.ensure_host()
        lb = b.ensure_host()
        dv = float(np.abs(la.v - lb.v).max()) if la.num_cells else 0.0
        dg = float(np.abs(la.g - lb.g).max()) if la.num_cells else 0.0
        spike_l1 = int(np.abs(la.total_spikes - lb.total_spikes).sum())
        exact = bool(
            np.array_equal(la.v.view(np.uint32), lb.v.view(np.uint32))
            and np.array_equal(la.g.view(np.uint32), lb.g.view(np.uint32))
            and np.array_equal(la.refrac, lb.refrac)
            and np.array_equal(la.delay_ring, lb.delay_ring)
            and np.array_equal(la.total_spikes, lb.total_spikes)
            and np.array_equal(la.stim_rate_hz.view(np.uint32), lb.stim_rate_hz.view(np.uint32))
            and np.array_equal(la.silenced, lb.silenced)
        )
        max_scale_diff = 0.0
        if a.plastic_state is not None and b.plastic_state is not None:
            pa, pb = a.plastic_state, b.plastic_state
            if pa.scale.size and pa.scale.shape == pb.scale.shape:
                max_scale_diff = float(np.abs(pa.scale - pb.scale).max())
            exact = exact and bool(
                np.array_equal(pa.scale.view(np.uint32), pb.scale.view(np.uint32))
                and np.array_equal(pa.kc_elig.view(np.uint32), pb.kc_elig.view(np.uint32))
                and np.array_equal(pa.mbon_dan.view(np.uint32), pb.mbon_dan.view(np.uint32))
            )
        return {
            "metric": "max_v_diff_mv",
            "value": dv,
            "max_g_diff_mv": dg,
            "spike_count_l1": spike_l1,
            "max_plastic_scale_diff": max_scale_diff,
            "exact_equal": exact,
        }

    def serialize_state(self, state: ConnectomeWorld) -> dict[str, Any]:
        lif = state.ensure_host()
        arrays: dict[str, Any] = {
            "v": lif.v.copy(),
            "g": lif.g.copy(),
            "refrac": lif.refrac.copy(),
            "delay_ring": lif.delay_ring.copy(),
            "stim_rate_hz": lif.stim_rate_hz.copy(),
            "silenced": lif.silenced.copy(),
            "last_tick_spikes": lif.last_tick_spikes.copy(),
            "total_spikes": lif.total_spikes.copy(),
        }
        if state.plastic_state is not None:
            arrays["plastic_scale"] = state.plastic_state.scale.copy()
            arrays["plastic_kc_elig"] = state.plastic_state.kc_elig.copy()
            arrays["plastic_mbon_dan"] = state.plastic_state.mbon_dan.copy()
        return {
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "dataset_id": state.pack.dataset_id,
                "pack_digest": state.pack.pack_digest,
                "substep": int(lif.substep),
                "tick": int(lif.tick),
            },
            "arrays": arrays,
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> ConnectomeWorld:
        exp = _ce(spec)
        meta = payload["meta"]
        if meta.get("serialize_version") != SERIALIZE_VERSION:
            raise ValueError(f"unsupported connectome checkpoint version: {meta}")
        arrays = payload["arrays"]
        lif = LIFState(
            v=np.ascontiguousarray(arrays["v"], dtype=np.float32),
            g=np.ascontiguousarray(arrays["g"], dtype=np.float32),
            refrac=np.ascontiguousarray(arrays["refrac"], dtype=np.int16),
            delay_ring=np.ascontiguousarray(arrays["delay_ring"], dtype=np.int32),
            substep=int(meta["substep"]),
            tick=int(meta["tick"]),
            stim_rate_hz=np.ascontiguousarray(arrays["stim_rate_hz"], dtype=np.float32),
            silenced=np.ascontiguousarray(arrays["silenced"], dtype=np.uint8),
            last_tick_spikes=np.ascontiguousarray(arrays["last_tick_spikes"], dtype=np.int32),
            total_spikes=np.ascontiguousarray(arrays["total_spikes"], dtype=np.int32),
        )
        pstate: PlasticityState | None = None
        if "plastic_scale" in arrays:
            scale = np.ascontiguousarray(arrays["plastic_scale"], dtype=np.float32)
            pstate = PlasticityState(
                scale=scale,
                kc_elig=np.ascontiguousarray(arrays["plastic_kc_elig"], dtype=np.float32),
                mbon_dan=np.ascontiguousarray(arrays["plastic_mbon_dan"], dtype=np.float32),
                last_eff_weight=np.zeros(scale.shape[0], dtype=np.int32),
            )
        world = self._make_world(exp, lif, plastic_state=pstate)
        if meta.get("pack_digest") and meta["pack_digest"] != world.pack.pack_digest:
            raise ValueError("checkpoint pack_digest does not match loaded connectome pack")
        return world
