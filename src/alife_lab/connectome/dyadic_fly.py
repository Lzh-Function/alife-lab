"""Phase F11: Two-Fly Innate Communication (Male Courtship Song <-> Female Auditory & Pursuit).

Couples TWO distinct biological adult Drosophila whole-CNS connectomes simultaneously:
- **Male (`male_cns_v1`, 165,122 cells):** P1 courtship command (`pC1_*`) -> `pIP10` descending
  song command -> nested VNC song circuit (`dPR1`, `vPR9_a/b/c`, `TN1a_*`) -> real wing song motor
  neurons (`ps1 MN` pulse song, `hg1 MN` sine song) + pursuit steering (`DNp09`, `DNa01/02`).
- **Female (`banc_888`, 156,808 cells):** Auditory Johnston's Organ (`JO-B`, 416 cells) ->
  saddle/WED/AVLP auditory relays (`SAD052`, `SAD053`, `CB1078`, `AVLP082`, `AVLP194_a`) + `pC1a-e`
  monitoring + locomotor modulation.

Implements both Stage ① (brain-to-brain acoustic dialogue + 8-cell-type Janelia song-circuit
silencing screen + channel/shuffled negative controls) and Stage ② (two-fly shared arena with
vision, short-range cuticular pheromone, near-field acoustic song, and tactile contact).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.lif import (
    LIFParams,
    LIFState,
    make_initial_state,
    make_shuffled_csr,
    step_tick_cpu,
)
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
    IOPort,
    compute_pack_digest,
)

I32 = npt.NDArray[np.int32]
F32 = npt.NDArray[np.float32]

# Janelia Lillvis et al. (2024) nested courtship song circuit ablation panel in MaleCNS v1.0.
JANELIA_SONG_PANEL: tuple[str, ...] = (
    "pIP10",
    "dPR1",
    "vPR9_a",
    "vPR9_b",
    "vPR9_c",
    "TN1a_g",
    "TN1a_i",
    "TN1a_d",
    "ps1 MN",
    "hg1 MN",
)


def make_synthetic_male_song_pack() -> ConnectomePack:
    """Offline 82-neuron synthetic male courtship-song CNS mirroring MaleCNS v1.0 anatomy."""
    core_types = (
        "pC1_14a",  # 0: male P1 courtship command L1
        "pC1_14a",  # 1: male P1 courtship command R1
        "pIP10",  # 2: descending song command L
        "pIP10",  # 3: descending song command R
        "dPR1",  # 4: pulse-song premotor L
        "dPR1",  # 5: pulse-song premotor R
        "vPR9_a",  # 6: song-switch IN
        "TN1a_g",  # 7: sine/pulse IN
        "ps1 MN",  # 8: pulse-song wing motor neuron L
        "ps1 MN",  # 9: pulse-song wing motor neuron R
        "hg1 MN",  # 10: sine-song wing motor neuron L
        "hg1 MN",  # 11: sine-song wing motor neuron R
        "DNp09",  # 12: forward pursuit DN
        "DNa01",  # 13: left pursuit steering DN
        "DNa01",  # 14: right pursuit steering DN
        "IN_inh",  # 15: local GABAergic gain control
        "pC1_14a",  # 16: male P1 courtship command L2
        "pC1_14a",  # 17: male P1 courtship command R2
    )
    bg = tuple(f"IN_bg_{k}" for k in range(64))
    types = core_types + bg
    n = len(types)
    sides = (
        "left",
        "right",
        "left",
        "right",
        "left",
        "right",
        "left",
        "right",
        "left",
        "right",
        "left",
        "right",
        "left",
        "left",
        "right",
        "center",
        "left",
        "right",
        *tuple("left" if k % 2 == 0 else "right" for k in range(64)),
    )
    roles = np.array(
        [
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_MOTOR,
            ROLE_MOTOR,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_MOTOR,
            ROLE_MOTOR,
            ROLE_MOTOR,
            ROLE_MOTOR,
            ROLE_MOTOR,
            ROLE_MOTOR,
            ROLE_MOTOR,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
        ]
        + [ROLE_INTERNEURON] * 64,
        dtype=np.uint8,
    )
    nts = tuple(
        "gaba" if (t == "IN_inh" or t.startswith("IN_bg_")) else "acetylcholine" for t in types
    )
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 0] = np.arange(n, dtype=np.float32) * 1000.0
    cells = CellTable(
        source_id=np.arange(1, n + 1, dtype=np.int64),
        species_names=("synthetic_male_song",),
        species_idx=np.zeros(n, dtype=np.int32),
        cell_type=types,
        super_class=("central",) * n,
        cell_class=("courtship",) * n,
        cell_sub_class=("song",) * n,
        region=("brain",) * 4 + ("ventral_nerve_cord",) * 8 + ("brain",) * (n - 12),
        side=sides,
        role=roles,
        cell_model=np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8),
        neurotransmitter=nts,
        position_nm=pos,
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    # Distributed 4-cell P1 -> pIP10 -> {dPR1 -> ps1 MN, vPR9_a + TN1a_g -> hg1 MN}
    erows = [
        (0, 2, 68, 1),
        (16, 2, 68, 1),
        (1, 3, 68, 1),
        (17, 3, 68, 1),
        (0, 3, 42, 1),
        (1, 2, 42, 1),
        (0, 12, 48, 1),
        (16, 12, 48, 1),
        (1, 12, 48, 1),
        (17, 12, 48, 1),
        (2, 4, 140, 1),
        (3, 5, 140, 1),
        (2, 6, 125, 1),
        (3, 7, 125, 1),
        (4, 8, 150, 1),
        (5, 9, 150, 1),
        (6, 10, 120, 1),
        (7, 11, 120, 1),
        (6, 8, 45, 1),
        (7, 9, 45, 1),
        (4, 15, 40, 1),
        (15, 6, 20, -1),
    ] + [(18 + k, 18 + ((k + 1) % 64), 10, -1) for k in range(64)]
    edges = EdgeTable(
        pre=np.array([r[0] for r in erows], dtype=np.int32),
        post=np.array([r[1] for r in erows], dtype=np.int32),
        count=np.array([r[2] for r in erows], dtype=np.int32),
        sign=np.array([r[3] for r in erows], dtype=np.int8),
        kind=np.full(len(erows), SYNAPSE_CHEMICAL, dtype=np.uint8),
    )
    io_ports = {
        "song_dn_pip10": IOPort(
            "song_dn_pip10", "motor", "pIP10", "Song DN", np.array([2, 3], dtype=np.int32), "pIP10"
        ),
        "forward_dn": IOPort(
            "forward_dn", "motor", "DNp09", "Forward DN", np.array([12], dtype=np.int32), "DNp09"
        ),
        "turn_left_dn": IOPort(
            "turn_left_dn", "motor", "DNa01 L", "Left DN", np.array([13], dtype=np.int32), "DNa01"
        ),
        "turn_right_dn": IOPort(
            "turn_right_dn",
            "motor",
            "DNa01 R",
            "Right DN",
            np.array([14], dtype=np.int32),
            "DNa01",
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_male_song",
        species="Synthetic male Drosophila courtship circuit",
        version="f11-v1",
        citation="Offline CI fixture mirroring MaleCNS v1.0 pC1->pIP10->dPR1/vPR9/TN1a->ps1/hg1",
        licence_note="Internal ALife Lab test fixture",
        files=(),
    )
    digest = compute_pack_digest(prov.dataset_id, prov.species, cells, edges, io_ports, prov)
    return ConnectomePack(prov.dataset_id, prov.species, cells, edges, io_ports, prov, digest)


def make_synthetic_female_auditory_pack() -> ConnectomePack:
    """Offline 36-neuron synthetic female auditory CNS mirroring BANC v888 JO-B->SAD/CB1078."""
    core_types = (
        "JO-B",  # 0: auditory sensory L
        "JO-B",  # 1: auditory sensory L
        "JO-B",  # 2: auditory sensory R
        "JO-B",  # 3: auditory sensory R
        "SAD052",  # 4: saddle auditory relay
        "SAD053",  # 5: saddle auditory relay
        "CB1078",  # 6: WED/AVLP auditory relay
        "AVLP082",  # 7: higher auditory relay
        "pC1a",  # 8: female courtship receptivity L
        "pC1a",  # 9: female courtship receptivity R
        "DNp09",  # 10: female forward walking DN
        "IN_aud_inh",  # 11: auditory gain control
    )
    types = core_types + tuple(f"IN_fbg_{k}" for k in range(64))
    n = len(types)
    roles = np.array(
        [
            ROLE_SENSORY,
            ROLE_SENSORY,
            ROLE_SENSORY,
            ROLE_SENSORY,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_INTERNEURON,
            ROLE_MOTOR,
            ROLE_INTERNEURON,
        ]
        + [ROLE_INTERNEURON] * 64,
        dtype=np.uint8,
    )
    nts = tuple(
        "gaba" if (t == "IN_aud_inh" or t.startswith("IN_fbg_")) else "acetylcholine" for t in types
    )
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 0] = np.arange(n, dtype=np.float32) * 1000.0
    cells = CellTable(
        source_id=np.arange(1, n + 1, dtype=np.int64),
        species_names=("synthetic_female_auditory",),
        species_idx=np.zeros(n, dtype=np.int32),
        cell_type=types,
        super_class=("sensory",) * 4 + ("central",) * (n - 4),
        cell_class=("auditory",) * 8
        + ("courtship",) * 2
        + ("descending",)
        + ("auditory",) * (n - 11),
        cell_sub_class=("jo_b",) * 4 + ("aud_relay",) * (n - 4),
        region=("brain",) * n,
        side=(
            "left",
            "left",
            "right",
            "right",
            "left",
            "right",
            "left",
            "right",
            "left",
            "right",
            "left",
            "center",
            *tuple("left" if k % 2 == 0 else "right" for k in range(64)),
        ),
        role=roles,
        cell_model=np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8),
        neurotransmitter=nts,
        position_nm=pos,
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    erows = [
        (0, 4, 105, 1),
        (1, 4, 105, 1),
        (2, 5, 110, 1),
        (3, 5, 110, 1),
        (0, 6, 95, 1),
        (2, 6, 95, 1),
        (4, 7, 115, 1),
        (5, 7, 115, 1),
        (6, 7, 85, 1),
        (7, 8, 100, 1),
        (7, 9, 100, 1),
        (4, 11, 40, 1),
        (11, 5, 20, -1),
    ] + [(12 + k, 12 + ((k + 1) % 64), 10, -1) for k in range(64)]
    edges = EdgeTable(
        pre=np.array([r[0] for r in erows], dtype=np.int32),
        post=np.array([r[1] for r in erows], dtype=np.int32),
        count=np.array([r[2] for r in erows], dtype=np.int32),
        sign=np.array([r[3] for r in erows], dtype=np.int8),
        kind=np.full(len(erows), SYNAPSE_CHEMICAL, dtype=np.uint8),
    )
    io_ports = {
        "forward_dn": IOPort(
            "forward_dn", "motor", "DNp09", "Forward DN", np.array([10], dtype=np.int32), "DNp09"
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_female_auditory",
        species="Synthetic female Drosophila auditory circuit",
        version="f11-v1",
        citation="Offline CI fixture mirroring BANC v888 JO-B->SAD052/53/CB1078/AVLP082->pC1a",
        licence_note="Internal ALife Lab test fixture",
        files=(),
    )
    digest = compute_pack_digest(prov.dataset_id, prov.species, cells, edges, io_ports, prov)
    return ConnectomePack(prov.dataset_id, prov.species, cells, edges, io_ports, prov, digest)


def _indices_by_types(pack: ConnectomePack, names: set[str]) -> I32:
    return np.asarray(
        [i for i, ct in enumerate(pack.cells.cell_type) if ct in names], dtype=np.int32
    )


def _indices_by_prefix(pack: ConnectomePack, prefix: str) -> I32:
    return np.asarray(
        [i for i, ct in enumerate(pack.cells.cell_type) if ct.startswith(prefix)], dtype=np.int32
    )


@dataclass
class _BrainNode:
    pack: ConnectomePack
    csr: CSRGraph
    params: LIFParams
    lif: LIFState
    backend: str
    seed: int
    gpu_runner: Any = None

    def step_1ms(
        self,
        stim_idx: I32,
        stim_hz: F32,
        read_idx: I32,
    ) -> I32:
        if self.gpu_runner is not None:
            if stim_idx.size > 0:
                self.gpu_runner.upload_stim_subset(stim_idx, stim_hz)
            self.gpu_runner.step_tick(use_graph=True)
            self.lif.tick += 1
            self.lif.substep += self.params.substeps_per_tick
            out: I32 = self.gpu_runner.read_spike_subset(read_idx)
            return out
        if stim_idx.size > 0:
            self.lif.stim_rate_hz[stim_idx] = stim_hz
        step_tick_cpu(self.lif, self.csr, self.params, self.seed)
        return self.lif.last_tick_spikes[read_idx].astype(np.int32, copy=False)


def _make_node(
    pack: ConnectomePack,
    *,
    wiring: str,
    backend: str,
    seed: int,
    silence_types: set[str] | None = None,
) -> _BrainNode:
    params = LIFParams()
    csr = pack.edges.to_csr(pack.cells.num_cells)
    if wiring == "shuffled_control":
        csr = make_shuffled_csr(csr, seed)
    lif = make_initial_state(pack.cells.num_cells, params)
    if silence_types:
        sidx = _indices_by_types(pack, silence_types)
        if sidx.size > 0:
            lif.silenced[sidx] = np.uint8(1)
    gpu_runner = None
    if backend == "cupy":
        from alife_lab.connectome.lif_cupy import CupyLIFRunner

        gpu_runner = CupyLIFRunner(csr, params, seed=seed)
        gpu_runner.upload_state(lif)
    return _BrainNode(pack, csr, params, lif, backend, seed, gpu_runner)


def _acoustic_attenuation(dist_cm: float, ref_cm: float = 0.55) -> float:
    """Near-field Drosophila particle-velocity song attenuation (~1 / (1 + (d/d0)^3))."""
    return 1.0 / (1.0 + (max(0.0, dist_cm) / ref_cm) ** 3)


def _resolve_f11_backend(male_cells: int, female_cells: int, backend: str) -> str:
    if backend != "auto":
        return backend
    from alife_lab.connectome.lif_cupy import available as cupy_ok

    return "cupy" if (max(male_cells, female_cells) > 500 and cupy_ok()) else "numpy"


def run_f11_song_dialogue(
    male_pack: ConnectomePack,
    female_pack: ConnectomePack,
    *,
    backend: str = "auto",
    seed: int = 1,
    ticks: int = 400,
    run_janelia_ablations: bool = True,
) -> dict[str, Any]:
    """F11 Stage ①: Coupled male song brain -> near-field acoustics -> female auditory brain.

    Evaluates:
    - ``intact_close`` (d = 0.25 cm): male P1 drive -> song circuit -> wing MNs -> female JO-B
    - ``acoustic_blocked`` (negative control, d = 5.0 cm / muted channel)
    - ``male_shuffled`` (negative control, shuffled male wiring)
    - ``female_shuffled`` (negative control, shuffled female wiring)
    - Janelia nested song-circuit single-cell-type silencing screen on ``male_pack``.
    """
    t0 = time.perf_counter()
    eff_b = _resolve_f11_backend(male_pack.cells.num_cells, female_pack.cells.num_cells, backend)

    # Male ports
    m_p1 = _indices_by_prefix(male_pack, "pC1_")
    if m_p1.size == 0:
        m_p1 = _indices_by_types(male_pack, {"pIP10"})
    m_pip10 = _indices_by_types(male_pack, {"pIP10"})
    m_dpr1 = _indices_by_types(male_pack, {"dPR1"})
    m_ps1 = _indices_by_types(male_pack, {"ps1 MN"})
    m_hg1 = _indices_by_types(male_pack, {"hg1 MN"})
    m_read = np.unique(np.concatenate([a for a in (m_pip10, m_dpr1, m_ps1, m_hg1) if a.size > 0]))
    m_pos = {int(c): i for i, c in enumerate(m_read.tolist())}

    def _mloc(arr: I32) -> I32:
        return np.asarray([m_pos[int(c)] for c in arr.tolist() if int(c) in m_pos], dtype=np.int32)

    ml_pip, ml_dpr, ml_ps1, ml_hg1 = _mloc(m_pip10), _mloc(m_dpr1), _mloc(m_ps1), _mloc(m_hg1)

    # Female ports (JO-B auditory receptors -> SAD052/53, CB1078, AVLP082/194_a, pC1a-e)
    f_job = _indices_by_prefix(female_pack, "JO-B")
    f_aud = _indices_by_types(female_pack, {"SAD052", "SAD053", "CB1078", "AVLP082", "AVLP194_a"})
    f_pc1 = _indices_by_types(female_pack, {"pC1a", "pC1b", "pC1c", "pC1d", "pC1e"})
    f_read = np.unique(np.concatenate([a for a in (f_aud, f_pc1) if a.size > 0]))
    f_pos = {int(c): i for i, c in enumerate(f_read.tolist())}
    fl_aud = np.asarray([f_pos[int(c)] for c in f_aud.tolist() if int(c) in f_pos], dtype=np.int32)
    fl_pc1 = np.asarray([f_pos[int(c)] for c in f_pc1.tolist() if int(c) in f_pos], dtype=np.int32)

    def _run_pair(
        *,
        male_wiring: str = "reference",
        female_wiring: str = "reference",
        distance_cm: float = 0.25,
        mute_acoustics: bool = False,
        silence_male_types: set[str] | None = None,
    ) -> dict[str, float]:
        m_node = _make_node(
            male_pack,
            wiring=male_wiring,
            backend=eff_b,
            seed=seed,
            silence_types=silence_male_types,
        )
        f_node = _make_node(female_pack, wiring=female_wiring, backend=eff_b, seed=seed + 100)
        # Mask silenced wing MNs so silenced output cells emit zero wing acoustic power
        m_active = (m_node.lif.silenced[m_read] == 0).astype(np.int32)
        atten = 0.0 if mute_acoustics else _acoustic_attenuation(distance_cm)

        m_stim_val = np.full(m_p1.size, 180.0, dtype=np.float32)
        f_stim_buf = np.zeros(f_job.size, dtype=np.float32)
        song_rate = 0.0
        alpha = 1.0 - math.exp(-1.0 / 18.0)

        sum_pip = sum_dpr = sum_ps1 = sum_hg1 = 0
        sum_faud = sum_fpc1 = 0
        for _ in range(ticks):
            m_spk = m_node.step_1ms(m_p1, m_stim_val, m_read) * m_active
            ps1_k = int(m_spk[ml_ps1].sum()) if ml_ps1.size else 0
            hg1_k = int(m_spk[ml_hg1].sum()) if ml_hg1.size else 0
            sum_pip += int(m_spk[ml_pip].sum()) if ml_pip.size else 0
            sum_dpr += int(m_spk[ml_dpr].sum()) if ml_dpr.size else 0
            sum_ps1 += ps1_k
            sum_hg1 += hg1_k

            wing_hz = (ps1_k / max(1, ml_ps1.size) + 0.6 * hg1_k / max(1, ml_hg1.size)) * 1000.0
            song_rate = (1.0 - alpha) * song_rate + alpha * wing_hz
            # Acoustic coupling: male wing vibration -> attenuated near-field JO-B stimulation
            job_hz = min(220.0, song_rate * 1.45 * atten)
            f_stim_buf.fill(np.float32(job_hz))
            f_spk = f_node.step_1ms(f_job, f_stim_buf, f_read)
            sum_faud += int(f_spk[fl_aud].sum()) if fl_aud.size else 0
            sum_fpc1 += int(f_spk[fl_pc1].sum()) if fl_pc1.size else 0

        dur_s = max(1e-3, ticks * 1e-3)
        return {
            "male_pIP10_hz": round(sum_pip / (max(1, ml_pip.size) * dur_s), 2),
            "male_dPR1_hz": round(sum_dpr / (max(1, ml_dpr.size) * dur_s), 2),
            "male_pulse_ps1_hz": round(sum_ps1 / (max(1, ml_ps1.size) * dur_s), 2),
            "male_sine_hg1_hz": round(sum_hg1 / (max(1, ml_hg1.size) * dur_s), 2),
            "female_auditory_relay_hz": round(sum_faud / (max(1, fl_aud.size) * dur_s), 2),
            "female_pC1_hz": round(sum_fpc1 / (max(1, fl_pc1.size) * dur_s), 2),
        }

    scenes = {
        "intact_close": _run_pair(distance_cm=0.25),
        "acoustic_blocked": _run_pair(distance_cm=0.25, mute_acoustics=True),
        "far_distance_5cm": _run_pair(distance_cm=5.0),
        "male_shuffled": _run_pair(male_wiring="shuffled_control", distance_cm=0.25),
        "female_shuffled": _run_pair(female_wiring="shuffled_control", distance_cm=0.25),
    }

    ablations: list[dict[str, Any]] = []
    if run_janelia_ablations:
        base_ps1 = scenes["intact_close"]["male_pulse_ps1_hz"]
        base_hg1 = scenes["intact_close"]["male_sine_hg1_hz"]
        base_faud = scenes["intact_close"]["female_auditory_relay_hz"]
        for ct in JANELIA_SONG_PANEL:
            idx = _indices_by_types(male_pack, {ct})
            if idx.size == 0:
                continue
            r = _run_pair(distance_cm=0.25, silence_male_types={ct})
            ablations.append(
                {
                    "silenced_cell_type": ct,
                    "cell_count": int(idx.size),
                    "male_pulse_ps1_hz": r["male_pulse_ps1_hz"],
                    "pulse_suppression_pct": (
                        round(100.0 * (base_ps1 - r["male_pulse_ps1_hz"]) / base_ps1, 1)
                        if base_ps1 > 0
                        else 0.0
                    ),
                    "male_sine_hg1_hz": r["male_sine_hg1_hz"],
                    "sine_suppression_pct": (
                        round(100.0 * (base_hg1 - r["male_sine_hg1_hz"]) / base_hg1, 1)
                        if base_hg1 > 0
                        else 0.0
                    ),
                    "female_auditory_relay_hz": r["female_auditory_relay_hz"],
                    "female_aud_suppression_pct": (
                        round(100.0 * (base_faud - r["female_auditory_relay_hz"]) / base_faud, 1)
                        if base_faud > 0
                        else 0.0
                    ),
                }
            )

    ic = scenes["intact_close"]
    checks = {
        "male_p1_recruits_song_mns": bool(
            ic["male_pulse_ps1_hz"] > 20.0 and ic["male_sine_hg1_hz"] > 5.0
        ),
        "female_auditory_responds_to_male_song": bool(ic["female_auditory_relay_hz"] > 10.0),
        "acoustic_block_silences_female": bool(
            scenes["acoustic_blocked"]["female_auditory_relay_hz"] == 0.0
        ),
        "male_shuffled_breaks_song": bool(
            scenes["male_shuffled"]["male_pulse_ps1_hz"] < ic["male_pulse_ps1_hz"] * 0.2
        ),
        "female_shuffled_breaks_hearing": bool(
            scenes["female_shuffled"]["female_auditory_relay_hz"]
            < ic["female_auditory_relay_hz"] * 0.2
        ),
    }
    return {
        "stage": "F11_stage1_brain_to_brain_song",
        "male_dataset": male_pack.dataset_id,
        "female_dataset": female_pack.dataset_id,
        "total_cells": male_pack.cells.num_cells + female_pack.cells.num_cells,
        "backend": eff_b,
        "ticks": ticks,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "scenes": scenes,
        "janelia_song_ablations": ablations,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def run_f11_two_fly_arena(
    male_pack: ConnectomePack,
    female_pack: ConnectomePack,
    *,
    backend: str = "auto",
    seed: int = 1,
    ticks: int = 1200,
) -> dict[str, Any]:
    """F11 Stage ②: Two-fly embodied pursuit & courtship arena (vision + pheromone + song + touch).

    Simulates an intact pair alongside three channel/wiring negative controls:
    - ``intact_pair``: male sees female, turns/chases, senses pheromone at close range -> sings ->
      female hears song via JO-B -> auditory relays fire and slow female walking speed.
    - ``acoustic_blocked``: song channel muted -> female never slows down despite pursuit.
    - ``vision_blocked``: male visual bearing input zeroed -> male fails to pursue female.
    - ``male_shuffled``: shuffled male connectome -> courtship song & pursuit collapse.
    """
    t0 = time.perf_counter()
    eff_b = _resolve_f11_backend(male_pack.cells.num_cells, female_pack.cells.num_cells, backend)

    m_p1 = _indices_by_prefix(male_pack, "pC1_")
    if m_p1.size == 0:
        m_p1 = _indices_by_types(male_pack, {"pIP10"})
    m_fwd = (
        male_pack.io_ports["forward_dn"].cell_indices
        if "forward_dn" in male_pack.io_ports
        else _indices_by_types(male_pack, {"DNp09"})
    )
    m_tl = (
        male_pack.io_ports["turn_left_dn"].cell_indices
        if "turn_left_dn" in male_pack.io_ports
        else np.zeros(0, dtype=np.int32)
    )
    m_tr = (
        male_pack.io_ports["turn_right_dn"].cell_indices
        if "turn_right_dn" in male_pack.io_ports
        else np.zeros(0, dtype=np.int32)
    )
    m_ps1 = _indices_by_types(male_pack, {"ps1 MN"})
    m_hg1 = _indices_by_types(male_pack, {"hg1 MN"})
    m_stim_u = np.unique(np.concatenate([a for a in (m_p1, m_fwd, m_tl, m_tr) if a.size > 0]))
    m_read_u = np.unique(
        np.concatenate([a for a in (m_fwd, m_tl, m_tr, m_ps1, m_hg1) if a.size > 0])
    )
    m_spos = {int(c): i for i, c in enumerate(m_stim_u.tolist())}
    m_rpos = {int(c): i for i, c in enumerate(m_read_u.tolist())}

    def _loc(m: dict[int, int], arr: I32) -> I32:
        return np.asarray([m[int(c)] for c in arr.tolist() if int(c) in m], dtype=np.int32)

    ms_p1, ms_fwd, ms_tl, ms_tr = (
        _loc(m_spos, m_p1),
        _loc(m_spos, m_fwd),
        _loc(m_spos, m_tl),
        _loc(m_spos, m_tr),
    )
    mr_fwd, mr_tl, mr_tr, mr_ps1, mr_hg1 = (
        _loc(m_rpos, m_fwd),
        _loc(m_rpos, m_tl),
        _loc(m_rpos, m_tr),
        _loc(m_rpos, m_ps1),
        _loc(m_rpos, m_hg1),
    )

    f_job = _indices_by_prefix(female_pack, "JO-B")
    f_aud = _indices_by_types(female_pack, {"SAD052", "SAD053", "CB1078", "AVLP082", "AVLP194_a"})

    def _run_arena(
        *,
        male_wiring: str = "reference",
        mute_acoustics: bool = False,
        blind_male: bool = False,
    ) -> dict[str, Any]:
        m_node = _make_node(male_pack, wiring=male_wiring, backend=eff_b, seed=seed)
        f_node = _make_node(female_pack, wiring="reference", backend=eff_b, seed=seed + 200)

        # Start female 1.6 cm ahead and slightly left of male (bearing ~+0.35 rad)
        mx, my, myaw = 0.0, 0.0, 0.0
        fx, fy, fyaw = 1.50, 0.55, 0.15
        m_stim = np.zeros(m_stim_u.size, dtype=np.float32)
        f_stim = np.zeros(f_job.size, dtype=np.float32)

        alpha = 1.0 - math.exp(-1.0 / 20.0)
        r_fwd = r_tl = r_tr = r_song = r_faud = 0.0
        min_dist = math.hypot(fx - mx, fy - my)
        close_ms = 0
        song_ms = 0
        f_dist = 0.0

        for _ in range(ticks):
            dx, dy = fx - mx, fy - my
            dist = math.hypot(dx, dy)
            min_dist = min(min_dist, dist)
            if dist <= 0.65:
                close_ms += 1

            # 1. Visual bearing of female in male's retina -> bilateral DNa01/02 & DNp09 pursuit
            bearing = math.atan2(dy, dx) - myaw
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            m_stim.fill(np.float32(0.0))
            if not blind_male and dist < 3.5:
                # Male visual pursuit arousal + left/right steering
                if ms_fwd.size:
                    m_stim[ms_fwd] = np.float32(150.0 if dist > 0.38 else 40.0)
                if bearing > 0.05 and ms_tl.size:
                    m_stim[ms_tl] = np.float32(min(180.0, 140.0 * abs(bearing)))
                elif bearing < -0.05 and ms_tr.size:
                    m_stim[ms_tr] = np.float32(min(180.0, 140.0 * abs(bearing)))

            # 2. Cuticular pheromone + close visual courtship arousal -> male P1 (pC1_*)
            pheromone = 1.0 if dist <= 0.85 else (0.35 if (not blind_male and dist <= 1.8) else 0.0)
            if ms_p1.size and pheromone > 0.0:
                m_stim[ms_p1] = np.float32(190.0 * pheromone)

            m_spk = m_node.step_1ms(m_stim_u, m_stim, m_read_u)
            r_fwd = (1 - alpha) * r_fwd + alpha * (
                float(m_spk[mr_fwd].mean()) * 1000.0 if mr_fwd.size else 0.0
            )
            r_tl = (1 - alpha) * r_tl + alpha * (
                float(m_spk[mr_tl].mean()) * 1000.0 if mr_tl.size else 0.0
            )
            r_tr = (1 - alpha) * r_tr + alpha * (
                float(m_spk[mr_tr].mean()) * 1000.0 if mr_tr.size else 0.0
            )
            wing_inst = (
                (float(m_spk[mr_ps1].mean()) if mr_ps1.size else 0.0)
                + 0.6 * (float(m_spk[mr_hg1].mean()) if mr_hg1.size else 0.0)
            ) * 1000.0
            r_song = (1 - alpha) * r_song + alpha * wing_inst
            if r_song >= 25.0:
                song_ms += 1

            # 3. Near-field acoustic song -> female JO-B auditory neurons
            atten = 0.0 if mute_acoustics else _acoustic_attenuation(dist)
            job_hz = min(220.0, r_song * 1.45 * atten)
            f_stim.fill(np.float32(job_hz))
            f_spk = f_node.step_1ms(f_job, f_stim, f_aud)
            faud_inst = float(f_spk.mean()) * 1000.0 if f_aud.size else 0.0
            r_faud = (1 - alpha) * r_faud + alpha * faud_inst

            # 4. Update planar 2-fly kinematics (dt = 1 ms, realistic 1.4-2.2 cm/s speeds)
            m_speed = min(2.4, 2.2 * min(1.0, r_fwd / 80.0))
            m_turn = max(-3.5, min(3.5, 2.6 * (r_tl - r_tr) / 60.0))
            myaw += m_turn * 1e-3
            mx += m_speed * math.cos(myaw) * 1e-3
            my += m_speed * math.sin(myaw) * 1e-3

            # Female walks ~1.15 cm/s and slows when hearing courtship song (receptive pause).
            slowdown = min(0.75, r_faud / 45.0)
            f_speed = 1.15 * (1.0 - slowdown)
            fx += f_speed * math.cos(fyaw) * 1e-3
            fy += f_speed * math.sin(fyaw) * 1e-3
            f_dist += f_speed * 1e-3

        return {
            "initial_dist_cm": round(math.hypot(1.50, 0.55), 3),
            "final_dist_cm": round(math.hypot(fx - mx, fy - my), 3),
            "min_dist_cm": round(min_dist, 3),
            "close_pursuit_ms": int(close_ms),
            "male_singing_ms": int(song_ms),
            "female_distance_cm": round(f_dist, 3),
            "female_final_aud_hz": round(r_faud, 2),
        }

    scenes = {
        "intact_pair": _run_arena(),
        "acoustic_blocked": _run_arena(mute_acoustics=True),
        "vision_blocked": _run_arena(blind_male=True),
        "male_shuffled": _run_arena(male_wiring="shuffled_control"),
    }
    ip = scenes["intact_pair"]
    checks = {
        "male_chases_female": bool(ip["min_dist_cm"] < 0.70),
        "male_sings_during_pursuit": bool(ip["male_singing_ms"] > 200),
        "female_slows_when_hearing_song": bool(
            ip["female_distance_cm"] < scenes["acoustic_blocked"]["female_distance_cm"] * 0.85
        ),
        "vision_block_breaks_pursuit": bool(
            scenes["vision_blocked"]["min_dist_cm"] > ip["min_dist_cm"] + 0.5
        ),
        "male_shuffled_breaks_singing": bool(
            scenes["male_shuffled"]["male_singing_ms"] < ip["male_singing_ms"] * 0.2
        ),
    }
    return {
        "stage": "F11_stage2_two_fly_arena",
        "male_dataset": male_pack.dataset_id,
        "female_dataset": female_pack.dataset_id,
        "total_cells": male_pack.cells.num_cells + female_pack.cells.num_cells,
        "backend": eff_b,
        "ticks": ticks,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "scenes": scenes,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
