"""Pavlovian olfactory associative conditioning protocol & controls for F4.

Tests all four F4 hypotheses and three negative controls across independent seeds:
1. Paired acquisition: Odor A + PPL101 punishment DAN -> Odor A depresses MBON11 (and
   disinhibits avoidance MBON01 in the synthetic MB / downstream circuits).
2. Odor specificity: unpaired Odor B response remains unchanged.
3. Trial-count dose-response: 1 training trial vs N training trials strengthens memory.
4. Extinction: repeated Odor A without punishment recovers synaptic scale and MBON response.
Negative controls:
- Shifted timing (unpaired): Odor A followed by a 1500 ms gap before PPL101 -> minimal/no learning.
- DAN silenced: PPL101 silenced during pairing -> exact zero synaptic change.
- Shuffled wiring: degree-preserving shuffled brain -> compare against real MB anatomy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from alife_lab.analysis.stats import compare, describe
from alife_lab.connectome.lif import LIFParams, make_initial_state, make_shuffled_csr, step_tick_cpu
from alife_lab.connectome.plasticity import (
    MBPlasticityParams,
    PlasticSynapseMap,
    build_plastic_synapse_map,
    make_initial_plasticity_state,
    step_plasticity_tick,
)
from alife_lab.connectome.schema import ConnectomePack, CSRGraph


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def _clone_csr(csr: CSRGraph) -> CSRGraph:
    return CSRGraph(
        num_cells=csr.num_cells,
        indptr=csr.indptr,
        post=csr.post,
        signed_weight=csr.signed_weight.copy(),
        kind=csr.kind,
    )


class _PlasticSim:
    """Self-contained CPU or GPU plastic simulation helper for Pavlovian protocols."""

    def __init__(
        self,
        pack: ConnectomePack,
        base_csr: CSRGraph,
        pmap: PlasticSynapseMap,
        lif_params: LIFParams,
        p_params: MBPlasticityParams,
        seed: int,
        backend: str,
    ) -> None:
        self.pack = pack
        self.csr = _clone_csr(base_csr)
        self.pmap = pmap
        self.lif_params = lif_params
        self.p_params = p_params
        self.seed = int(seed)
        self.backend = backend
        self.lif = make_initial_state(pack.cells.num_cells, lif_params)
        self.pstate = make_initial_plasticity_state(pmap)
        self.gpu: Any = None
        if backend == "cupy":
            from alife_lab.connectome.lif_cupy import CupyLIFRunner

            self.gpu = CupyLIFRunner(self.csr, lif_params, self.seed)
            self.gpu.upload_state(self.lif)

    def set_controls(
        self, stimuli: list[tuple[str, float]], silenced_ports: tuple[str, ...] = ()
    ) -> None:
        self.lif.stim_rate_hz.fill(np.float32(0.0))
        self.lif.silenced.fill(np.uint8(0))
        for pid, hz in stimuli:
            self.lif.stim_rate_hz[self.pack.io_ports[pid].cell_indices] = np.float32(hz)
        for pid in silenced_ports:
            self.lif.silenced[self.pack.io_ports[pid].cell_indices] = np.uint8(1)
        if self.gpu is not None:
            self.gpu.upload_controls(self.lif)

    def reset_dynamics_keep_weights(self) -> None:
        """Reset membrane voltages/traces between probe tests while keeping learned weights."""
        if self.gpu is not None:
            self.gpu.download_state(self.lif)
        self.lif.v.fill(np.float32(self.lif_params.v_0))
        self.lif.g.fill(np.float32(0.0))
        self.lif.refrac.fill(np.int16(0))
        self.lif.delay_ring.fill(np.int32(0))
        self.lif.last_tick_spikes.fill(np.int32(0))
        self.pstate.kc_elig.fill(np.float32(0.0))
        self.pstate.mbon_dan.fill(np.float32(0.0))
        if self.gpu is not None:
            self.gpu.upload_state(self.lif)

    def run_ticks(self, ticks: int, *, plastic: bool = True) -> dict[str, float]:
        """Run ``ticks`` ms and return mean firing rates of MBON11 and MBON01 over this window."""
        m11_idx = self.pack.io_ports["mbon11_approach"].cell_indices
        m01_port = self.pack.io_ports.get("mbon01_avoidance")
        m01_idx = m01_port.cell_indices if m01_port is not None else np.zeros(0, dtype=np.int32)

        m11_sp = 0
        m01_sp = 0
        dt_tick_ms = self.lif_params.substeps_per_tick * self.lif_params.dt_ms
        for _ in range(ticks):
            if self.gpu is not None:
                self.gpu.step_tick(use_graph=True)
                self.lif.substep += self.lif_params.substeps_per_tick
                self.lif.tick += 1
                last_sp = self.gpu.download_last_tick_spikes()
                self.lif.last_tick_spikes[:] = last_sp
            else:
                step_tick_cpu(self.lif, self.csr, self.lif_params, self.seed)
                last_sp = self.lif.last_tick_spikes

            m11_sp += int(last_sp[m11_idx].sum()) if m11_idx.size else 0
            m01_sp += int(last_sp[m01_idx].sum()) if m01_idx.size else 0

            if plastic and self.p_params.enabled:
                chg_idx, chg_w = step_plasticity_tick(
                    self.pstate,
                    self.pmap,
                    self.p_params,
                    last_sp,
                    self.lif.silenced,
                    dt_tick_ms=dt_tick_ms,
                )
                if chg_idx.size > 0:
                    self.csr.signed_weight[chg_idx] = chg_w
                    if self.gpu is not None:
                        self.gpu.update_edge_weights(chg_idx, chg_w)

        dur_s = max(ticks * dt_tick_ms / 1000.0, 1e-9)
        return {
            "mbon11_hz": m11_sp / (max(1, int(m11_idx.size)) * dur_s),
            "mbon01_hz": (m01_sp / (max(1, int(m01_idx.size)) * dur_s)) if m01_idx.size else 0.0,
            "changed_edges": float(
                int((np.abs(self.pstate.scale - np.float32(1.0)) > np.float32(0.01)).sum())
            ),
            "min_scale": float(self.pstate.scale.min()) if self.pstate.scale.size else 1.0,
        }

    def probe_odor(self, port: str, rate_hz: float, probe_ms: int) -> dict[str, float]:
        """Read-only probe of an odor without updating synaptic weights."""
        self.reset_dynamics_keep_weights()
        self.set_controls([(port, rate_hz)])
        return self.run_ticks(probe_ms, plastic=False)


def _run_one_seed_protocol(
    pack: ConnectomePack,
    base_csr: CSRGraph,
    pmap: PlasticSynapseMap,
    lif_params: LIFParams,
    p_params: MBPlasticityParams,
    *,
    seed: int,
    backend: str,
    odor_hz: float,
    dan_hz: float,
    probe_ms: int,
    trial_ms: int,
    n_trials: int,
    ext_ms: int,
    unpaired_gap_ms: int,
) -> dict[str, Any]:
    # 1. Main paired arm: Pre -> 1 trial -> N trials -> Extinction
    sim = _PlasticSim(pack, base_csr, pmap, lif_params, p_params, seed, backend)
    pre_a = sim.probe_odor("odor_a_kc", odor_hz, probe_ms)
    pre_b = sim.probe_odor("odor_b_kc", odor_hz, probe_ms)

    # Trial 1 (dose-response checkpoint)
    sim.reset_dynamics_keep_weights()
    sim.set_controls([("odor_a_kc", odor_hz), ("ppl1_punishment_dan", dan_hz)])
    sim.run_ticks(trial_ms, plastic=True)
    after1_a = sim.probe_odor("odor_a_kc", odor_hz, probe_ms)

    # Remaining trials 2..n_trials
    for _ in range(max(0, n_trials - 1)):
        sim.reset_dynamics_keep_weights()
        sim.set_controls([("odor_a_kc", odor_hz), ("ppl1_punishment_dan", dan_hz)])
        sim.run_ticks(trial_ms, plastic=True)
    post_a = sim.probe_odor("odor_a_kc", odor_hz, probe_ms)
    post_b = sim.probe_odor("odor_b_kc", odor_hz, probe_ms)
    changed_post = int(post_a["changed_edges"])
    min_scale_post = float(post_a["min_scale"])

    # Extinction: Odor A alone without DAN
    sim.reset_dynamics_keep_weights()
    sim.set_controls([("odor_a_kc", odor_hz)])
    sim.run_ticks(ext_ms, plastic=True)
    ext_a = sim.probe_odor("odor_a_kc", odor_hz, probe_ms)

    # 2. Control: Shifted timing (unpaired: Odor A, then long gap, then PPL101)
    sim_unp = _PlasticSim(pack, base_csr, pmap, lif_params, p_params, seed, backend)
    for _ in range(n_trials):
        sim_unp.reset_dynamics_keep_weights()
        sim_unp.set_controls([("odor_a_kc", odor_hz)])
        sim_unp.run_ticks(trial_ms, plastic=True)
        sim_unp.set_controls([])
        sim_unp.run_ticks(unpaired_gap_ms, plastic=True)
        sim_unp.set_controls([("ppl1_punishment_dan", dan_hz)])
        sim_unp.run_ticks(trial_ms, plastic=True)
    unp_a = sim_unp.probe_odor("odor_a_kc", odor_hz, probe_ms)

    # 3. Control: all mushroom body DANs silenced during paired training
    sim_nodan = _PlasticSim(pack, base_csr, pmap, lif_params, p_params, seed, backend)
    for _ in range(n_trials):
        sim_nodan.reset_dynamics_keep_weights()
        sim_nodan.set_controls(
            [("odor_a_kc", odor_hz), ("ppl1_punishment_dan", dan_hz)],
            silenced_ports=("mushroom_body_dan",),
        )
        sim_nodan.run_ticks(trial_ms, plastic=True)
    nodan_a = sim_nodan.probe_odor("odor_a_kc", odor_hz, probe_ms)

    return {
        "seed": seed,
        "pre_odor_a_mbon11_hz": pre_a["mbon11_hz"],
        "pre_odor_b_mbon11_hz": pre_b["mbon11_hz"],
        "after1_odor_a_mbon11_hz": after1_a["mbon11_hz"],
        "post_odor_a_mbon11_hz": post_a["mbon11_hz"],
        "post_odor_b_mbon11_hz": post_b["mbon11_hz"],
        "ext_odor_a_mbon11_hz": ext_a["mbon11_hz"],
        "unpaired_odor_a_mbon11_hz": unp_a["mbon11_hz"],
        "no_dan_odor_a_mbon11_hz": nodan_a["mbon11_hz"],
        "pre_odor_a_mbon01_hz": pre_a["mbon01_hz"],
        "post_odor_a_mbon01_hz": post_a["mbon01_hz"],
        "post_odor_b_mbon01_hz": post_b["mbon01_hz"],
        "changed_edges_post": changed_post,
        "min_scale_post": min_scale_post,
        "changed_edges_no_dan": int(nodan_a["changed_edges"]),
        "changed_edges_unpaired": int(unp_a["changed_edges"]),
    }


def run_pavlovian_experiment(
    pack: ConnectomePack,
    *,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    p_params: MBPlasticityParams | None = None,
    lif_params: LIFParams | None = None,
    odor_hz: float = 150.0,
    dan_hz: float = 220.0,
    probe_ms: int = 300,
    trial_ms: int = 200,
    n_trials: int = 4,
    ext_ms: int = 1200,
    unpaired_gap_ms: int = 1500,
    wiring: str = "reference",
    backend: str = "auto",
) -> dict[str, Any]:
    """Run the full F4 Pavlovian conditioning + controls protocol across multiple seeds."""
    lp = lif_params or LIFParams()
    pp = p_params or MBPlasticityParams(enabled=True)
    eff_backend = ("cupy" if _has_cupy() else "numpy") if backend == "auto" else backend

    t0 = time.perf_counter()
    base_csr = pack.edges.to_csr(pack.cells.num_cells)
    if wiring == "shuffled_control":
        base_csr = make_shuffled_csr(base_csr, int(seeds[0]))
    pmap = build_plastic_synapse_map(pack, base_csr, min_dan_synapses=pp.min_dan_synapses)

    per_seed = [
        _run_one_seed_protocol(
            pack,
            base_csr,
            pmap,
            lp,
            pp,
            seed=int(s),
            backend=eff_backend,
            odor_hz=odor_hz,
            dan_hz=dan_hz,
            probe_ms=probe_ms,
            trial_ms=trial_ms,
            n_trials=n_trials,
            ext_ms=ext_ms,
            unpaired_gap_ms=unpaired_gap_ms,
        )
        for s in seeds
    ]
    wall_s = time.perf_counter() - t0

    def _col(k: str) -> list[float]:
        return [float(r[k]) for r in per_seed]

    pre_a = _col("pre_odor_a_mbon11_hz")
    post_a = _col("post_odor_a_mbon11_hz")
    after1_a = _col("after1_odor_a_mbon11_hz")
    pre_b = _col("pre_odor_b_mbon11_hz")
    post_b = _col("post_odor_b_mbon11_hz")
    ext_a = _col("ext_odor_a_mbon11_hz")
    unp_a = _col("unpaired_odor_a_mbon11_hz")
    nodan_a = _col("no_dan_odor_a_mbon11_hz")

    return {
        "dataset_id": pack.dataset_id,
        "pack_digest": pack.pack_digest,
        "wiring": wiring,
        "backend": eff_backend,
        "plastic_edges_total": pmap.num_edges,
        "seeds": [int(s) for s in seeds],
        "wall_seconds": round(wall_s, 3),
        "framing_note": (
            "Simplified three-factor KC->MBON eligibility-trace plasticity rule "
            "(Hige et al. 2015 / Handler et al. 2019 inspired); not a biochemical cAMP/PKA model."
        ),
        "summary": {
            "pre_odor_a_mbon11_hz": describe(pre_a),
            "after1_odor_a_mbon11_hz": describe(after1_a),
            "post_odor_a_mbon11_hz": describe(post_a),
            "pre_odor_b_mbon11_hz": describe(pre_b),
            "post_odor_b_mbon11_hz": describe(post_b),
            "ext_odor_a_mbon11_hz": describe(ext_a),
            "unpaired_odor_a_mbon11_hz": describe(unp_a),
            "no_dan_odor_a_mbon11_hz": describe(nodan_a),
            "pre_odor_a_mbon01_hz": describe(_col("pre_odor_a_mbon01_hz")),
            "post_odor_a_mbon01_hz": describe(_col("post_odor_a_mbon01_hz")),
            "post_odor_b_mbon01_hz": describe(_col("post_odor_b_mbon01_hz")),
            "changed_edges_post": describe(_col("changed_edges_post")),
            "changed_edges_unpaired": describe(_col("changed_edges_unpaired")),
            "changed_edges_no_dan": describe(_col("changed_edges_no_dan")),
        },
        "comparisons": {
            "paired_post_vs_pre_odor_a": compare(post_a, pre_a, seed=401),
            "paired_post_odor_a_vs_odor_b": compare(post_a, post_b, seed=402),
            "paired_post_vs_unpaired_odor_a": compare(post_a, unp_a, seed=403),
            "paired_post_vs_no_dan_odor_a": compare(post_a, nodan_a, seed=404),
            "extinction_vs_paired_post_odor_a": compare(ext_a, post_a, seed=405),
        },
        "per_seed": per_seed,
    }
