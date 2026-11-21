"""Reference & verification tests for the F1 LIF connectome engine (Shiu et al. 2024).

Covers all F1 acceptance gates:
1. Analytic solution agreement on tiny hand-built circuits (passive decay, synaptic drive,
   1.8 ms synaptic delay, 2.2 ms refractory period, excitation/inhibition, silencing).
2. Checkpoint save -> restore bitwise continuation.
3. CPU oracle vs GPU (eager & CUDA graph) bitwise equality, plus GPU repeat determinism
   and batched multi-brain runner equality.
4. Shiu et al. (2024) FlyWire v783 sensorimotor reproduction (sugar -> MN9, bitter
   suppression, antennal grooming -> descending neurons, silencing) + shuffled-brain control.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.connectome import (
    CSRGraph,
    LIFParams,
    load_flywire_783,
    make_initial_state,
    make_shuffled_csr,
    make_synthetic_tiny_pack,
    step_substep_cpu,
    step_tick_cpu,
    verify_dataset,
)
from alife_lab.connectome.lif_cupy import (
    CupyBatchedLIFRunner,
    CupyLIFRunner,
)
from alife_lab.connectome.lif_cupy import (
    available as cupy_available,
)
from alife_lab.core.spec import ExperimentSpec
from alife_lab.service.settings import load_settings
from alife_lab.substrates.connectome.adapter import ConnectomeAdapter


def _empty_csr(n: int) -> CSRGraph:
    return CSRGraph(
        num_cells=n,
        indptr=np.zeros(n + 1, dtype=np.int32),
        post=np.zeros(0, dtype=np.int32),
        signed_weight=np.zeros(0, dtype=np.int32),
        kind=np.zeros(0, dtype=np.uint8),
    )


def _assert_states_bitwise_equal(a, b) -> None:
    assert a.substep == b.substep
    assert np.array_equal(a.v.view(np.uint32), b.v.view(np.uint32))
    assert np.array_equal(a.g.view(np.uint32), b.g.view(np.uint32))
    assert np.array_equal(a.refrac, b.refrac)
    assert np.array_equal(a.delay_ring, b.delay_ring)
    assert np.array_equal(a.total_spikes, b.total_spikes)
    assert np.array_equal(a.last_tick_spikes, b.last_tick_spikes)


# --------------------------------------------------------------------------- #
# 1. Analytic checks on tiny hand-built circuits (CPU-only, always run)
# --------------------------------------------------------------------------- #


def test_single_neuron_analytic_passive_and_synaptic_response() -> None:
    params = LIFParams()
    csr = _empty_csr(2)
    st = make_initial_state(2, params)
    # Cell 0: passive decay from -48 mV with g = 0.
    st.v[0] = np.float32(-48.0)
    # Cell 1: subthreshold synaptic pulse g(0) = 5.0 mV from v_0 = -52 mV.
    st.g[1] = np.float32(5.0)

    steps = 100  # 10.0 ms
    for _ in range(steps):
        step_substep_cpu(st, csr, params, seed=0)

    t = steps * params.dt_ms
    v0_analytic = params.v_0 + (-48.0 - params.v_0) * math.exp(-t / params.t_mbr_ms)
    g1_analytic = 5.0 * math.exp(-t / params.tau_ms)
    v1_analytic = params.v_0 + 5.0 * (params.tau_ms / (params.t_mbr_ms - params.tau_ms)) * (
        math.exp(-t / params.t_mbr_ms) - math.exp(-t / params.tau_ms)
    )
    assert abs(float(st.v[0]) - v0_analytic) < 1e-4
    assert abs(float(st.g[1]) - g1_analytic) < 1e-5
    assert abs(float(st.v[1]) - v1_analytic) < 1e-4
    assert int(st.total_spikes.sum()) == 0


def test_delay_refractory_excitation_inhibition_and_silencing() -> None:
    params = LIFParams()
    # Peak delta-v from a single pulse g0 is g0 * (tau/t_mbr)^(t_mbr/(t_mbr-tau)) ~= 0.1575 * g0.
    # 180 synapses -> g0 = 49.5 mV -> peak delta-v ~= 7.8 mV > 7.0 mV threshold.
    # 0 --(+180)--> 1, 2 --(-180)--> 1
    csr = CSRGraph(
        num_cells=3,
        indptr=np.array([0, 1, 1, 2], dtype=np.int32),
        post=np.array([1, 1], dtype=np.int32),
        signed_weight=np.array([180, -180], dtype=np.int32),
        kind=np.array([0, 0], dtype=np.uint8),
    )

    # Case A: cell 0 fires at substep 0 -> arrives at cell 1 at substep 18 (1.8 ms).
    st = make_initial_state(3, params)
    st.v[0] = np.float32(-44.0)  # above v_th = -45.0
    fired0 = step_substep_cpu(st, csr, params, seed=1)
    assert list(fired0) == [0]
    assert float(st.v[0]) == pytest.approx(params.v_rst)
    assert float(st.g[0]) == 0.0
    assert int(st.refrac[0]) == params.refrac_steps  # 22 substeps = 2.2 ms

    # For substeps 1..17, cell 1 has not received the delayed input yet.
    for _ in range(1, params.delay_steps):
        step_substep_cpu(st, csr, params, seed=1)
        assert float(st.g[1]) == 0.0
    # At substep 18, +180 * 0.275 = +49.5 mV arrives in g[1] (and decays by alpha_g over the step).
    step_substep_cpu(st, csr, params, seed=1)
    assert float(st.g[1]) > 45.0
    # Refractory period on cell 0 prevents firing until 22 substeps have elapsed.
    assert int(st.refrac[0]) == params.refrac_steps - params.delay_steps
    # Within a few ms, cell 1 must spike from the +49.5 mV drive.
    for _ in range(100):
        step_substep_cpu(st, csr, params, seed=1)
    assert int(st.total_spikes[1]) >= 1

    # Case B: exact cancellation when cell 0 (+120) and cell 2 (-120) fire simultaneously.
    st_cancel = make_initial_state(3, params)
    st_cancel.v[0] = np.float32(-44.0)
    st_cancel.v[2] = np.float32(-44.0)
    for _ in range(40):
        step_substep_cpu(st_cancel, csr, params, seed=1)
    assert float(st_cancel.g[1]) == 0.0
    assert int(st_cancel.total_spikes[1]) == 0

    # Case C: silencing cell 0 zeroes its outgoing synapses even when it spikes.
    st_sil = make_initial_state(3, params)
    st_sil.silenced[0] = np.uint8(1)
    st_sil.v[0] = np.float32(-44.0)
    for _ in range(40):
        step_substep_cpu(st_sil, csr, params, seed=1)
    assert int(st_sil.total_spikes[0]) == 1
    assert float(st_sil.g[1]) == 0.0
    assert int(st_sil.total_spikes[1]) == 0


# --------------------------------------------------------------------------- #
# 2. Checkpoint save -> restore bitwise continuation (CPU-only, always run)
# --------------------------------------------------------------------------- #


def test_checkpoint_roundtrip_is_bitwise_identical() -> None:
    spec = ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "tiny_ckpt",
                "mode": "reference",
                "substrate": "connectome",
                "seed": 7,
                "dataset_id": "synthetic_tiny",
                "stimuli": [{"port": "sugar_grn", "rate_hz": 150.0}],
            }
        }
    )
    adapter = ConnectomeAdapter()
    adapter.set_backend("numpy")
    w_full = adapter.initialize(spec)
    for _ in range(35):
        adapter.step(w_full, spec)
    ser = adapter.serialize_state(w_full)
    blob = encode_checkpoint(ser["arrays"], ser["meta"])
    for _ in range(40):
        adapter.step(w_full, spec)

    w_restored = adapter.deserialize_state(decode_checkpoint(blob), spec)
    for _ in range(40):
        adapter.step(w_restored, spec)

    dist = adapter.state_distance(w_full, w_restored)
    assert dist["exact_equal"] is True
    assert dist["value"] == 0.0
    assert dist["spike_count_l1"] == 0


# --------------------------------------------------------------------------- #
# 3. CPU vs GPU bitwise equality & batched multi-brain runner (when GPU present)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not cupy_available(), reason="CuPy CUDA GPU not available")
def test_gpu_matches_cpu_bitwise_on_synthetic_and_batched() -> None:
    pack = make_synthetic_tiny_pack()
    csr = pack.edges.to_csr(pack.cells.num_cells)
    params = LIFParams()

    s_cpu = make_initial_state(csr.num_cells, params)
    s_cpu.stim_rate_hz[0] = np.float32(150.0)
    s_gpu_eager = s_cpu.copy()
    s_gpu_graph = s_cpu.copy()
    s_gpu_repeat = s_cpu.copy()

    for _ in range(100):
        step_tick_cpu(s_cpu, csr, params, seed=123)

    r_eager = CupyLIFRunner(csr, params, seed=123)
    r_eager.upload_state(s_gpu_eager)
    for _ in range(100):
        r_eager.step_tick(use_graph=False)
    r_eager.download_state(s_gpu_eager)

    r_graph = CupyLIFRunner(csr, params, seed=123)
    r_graph.upload_state(s_gpu_graph)
    for _ in range(100):
        r_graph.step_tick(use_graph=True)
    r_graph.download_state(s_gpu_graph)

    r_repeat = CupyLIFRunner(csr, params, seed=123)
    r_repeat.upload_state(s_gpu_repeat)
    for _ in range(100):
        r_repeat.step_tick(use_graph=True)
    r_repeat.download_state(s_gpu_repeat)

    _assert_states_bitwise_equal(s_cpu, s_gpu_eager)
    _assert_states_bitwise_equal(s_cpu, s_gpu_graph)
    _assert_states_bitwise_equal(s_gpu_graph, s_gpu_repeat)
    assert int(s_cpu.total_spikes[2]) > 0  # motor neuron fired via relay

    # Batched runner (B=3 with distinct seeds) matches single-brain CPU per element.
    seeds = [11, 22, 33]
    stims = np.zeros((3, csr.num_cells), dtype=np.float32)
    stims[:, 0] = 150.0
    stims[2, 3] = 150.0  # brain 2 also stimulated on bitter inhibitory channel
    batch = CupyBatchedLIFRunner(csr, params, seeds, stims)
    batch.step_substeps(1000, use_graph=True)
    batch_spikes = batch.total_spikes_cpu()

    for b, sd in enumerate(seeds):
        sb = make_initial_state(csr.num_cells, params)
        sb.stim_rate_hz[:] = stims[b]
        for _ in range(100):
            step_tick_cpu(sb, csr, params, seed=sd)
        assert np.array_equal(batch_spikes[b], sb.total_spikes)


# --------------------------------------------------------------------------- #
# 4 & 5. Real FlyWire v783 bitwise CPU/GPU + Shiu 2024 reproduction + control
# --------------------------------------------------------------------------- #


def _has_flywire_and_gpu() -> bool:
    return cupy_available() and verify_dataset(
        "flywire_783", load_settings().resolved_datasets_dir
    )["ok"]


@pytest.mark.skipif(
    not _has_flywire_and_gpu(),
    reason="requires local verified flywire_783 dataset and CuPy CUDA GPU",
)
def test_flywire_783_bitwise_and_shiu_2024_reproduction() -> None:
    pack = load_flywire_783(load_settings().resolved_datasets_dir, verify_hashes=True)
    csr = pack.edges.to_csr(pack.cells.num_cells)
    params = LIFParams()

    sugar = pack.io_ports["sugar_grn"].cell_indices
    bitter = pack.io_ports["bitter_grn"].cell_indices
    groom = pack.io_ports["antennal_grooming_mechanosensory"].cell_indices
    mn9 = pack.io_ports["mn9_proboscis_extension"].cell_indices

    # A. Bitwise CPU vs GPU (CUDA graph) on full 138,639-neuron FlyWire v783 for 30 ms.
    s_cpu = make_initial_state(csr.num_cells, params)
    s_cpu.stim_rate_hz[sugar] = np.float32(150.0)
    s_gpu = s_cpu.copy()
    for _ in range(30):
        step_tick_cpu(s_cpu, csr, params, seed=42)
    runner = CupyLIFRunner(csr, params, seed=42)
    runner.upload_state(s_gpu)
    for _ in range(30):
        runner.step_tick(use_graph=True)
    runner.download_state(s_gpu)
    _assert_states_bitwise_equal(s_cpu, s_gpu)

    # B. Shiu 2024 reproduction (1000 ms = 10,000 substeps) across conditions via batched GPU.
    # Brain 0: sugar 150 Hz
    # Brain 1: sugar 150 Hz + bitter 150 Hz (suppression)
    # Brain 2: antennal grooming mechanosensory 150 Hz (descending activation, no MN9)
    # Brain 3: sugar 150 Hz with sugar_grn silenced (intervention control)
    n = csr.num_cells
    stims = np.zeros((4, n), dtype=np.float32)
    stims[0, sugar] = 150.0
    stims[1, sugar] = 150.0
    stims[1, bitter] = 150.0
    stims[2, groom] = 150.0
    stims[3, sugar] = 150.0

    batch = CupyBatchedLIFRunner(csr, params, [1, 1, 1, 1], stims)
    sil = np.zeros((4, n), dtype=np.uint8)
    sil[3, sugar] = 1
    batch.set_silenced(sil)
    for _ in range(100):
        batch.step_substeps(100, use_graph=True)
    sp = batch.total_spikes_cpu()

    mn9_sugar = int(sp[0, mn9].sum())
    mn9_sugar_bitter = int(sp[1, mn9].sum())
    mn9_groom = int(sp[2, mn9].sum())
    mn9_silenced = int(sp[3, mn9].sum())

    # Sugar strongly drives MN9; bitter strongly suppresses MN9; silencing stops MN9.
    assert mn9_sugar >= 40
    assert mn9_sugar_bitter <= mn9_sugar // 4
    assert mn9_groom == 0
    assert mn9_silenced == 0

    # Antennal grooming activates descending neurons (e.g. DNg84 / DNg35 / DNg57).
    desc_idx = np.array(
        [i for i in range(n) if pack.cells.super_class[i] == "descending"], dtype=np.int32
    )
    assert int(sp[2, desc_idx].sum()) >= 200

    # C. Shuffled control brain: same edge count & out-degree, sugar does NOT drive MN9.
    csr_shuf = make_shuffled_csr(csr, seed=1)
    st_shuf = make_initial_state(n, params)
    st_shuf.stim_rate_hz[sugar] = np.float32(150.0)
    r_shuf = CupyLIFRunner(csr_shuf, params, seed=1)
    r_shuf.upload_state(st_shuf)
    for _ in range(100):
        r_shuf.step_substeps(100, use_graph=True)
    r_shuf.download_state(st_shuf)
    assert int(st_shuf.total_spikes[mn9].sum()) < mn9_sugar // 5
