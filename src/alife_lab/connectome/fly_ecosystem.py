"""Phase F11 Grand Finale: Unified Dual-Sex Whole-CNS Ecological World (``dual_sex_ecosystem``).

Runs TWO distinct real sex-specific adult *Drosophila* whole-CNS connectomes simultaneously inside
ONE shared 3D arena (GPU ``CupyBatchedLIFRunner`` for males + GPU ``CupyBatchedLIFRunner`` for
females, or exact CPU NumPy fallback):
- **Males (``male_cns_v1``, 165,122 cells/fly):** cVA male pheromone (``ORN_DA1 -> DA1_lPN/vPN``),
  visual rival aggression (``LC10a -> aIPg -> DNg33 + hg3/hg4 MN`` bilateral wing threat & lunge),
  courtship song (``pC1_* -> pIP10 -> ps1/hg1 MN``), mechanosensory arousal (``DNg84``), and
  navigation (``DNp09``, ``DNa01/02``, ``MDN``).
- **Females (``banc_888``, 156,808 cells/fly):** sugar/bitter gustatory feeding
  (``sugar_grn -> MN9``, ``bitter_grn -| MN9``), near-field auditory song reception
  (``JO-B -> SAD052/53 + CB1078 + AVLP082/194_a`` -> receptive locomotor slowing), mechanosensory
  encounter arousal (``DNg84``), and aversive odor avoidance.

All four F11 behaviors (foraging/feeding, Ramdya leg-touch odor avoidance, male-male territorial
aggression, and male-to-female acoustic courtship) run concurrently and interact in real time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.embodied import EmbodiedPorts, resolve_embodied_ports
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

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I32 = npt.NDArray[np.int32]
U8 = npt.NDArray[np.uint8]

DEFAULT_ECOSYSTEM_FOODS: tuple[dict[str, Any], ...] = (
    {"x": 1.65, "y": 1.55, "r": 0.68, "kind": "sugar"},
    {"x": 1.65, "y": -1.55, "r": 0.68, "kind": "sugar"},
    {"x": -0.35, "y": 0.0, "r": 0.55, "kind": "bitter"},
)
ECOSYSTEM_ODOR_X_MAX = -2.0


def make_synthetic_ecosystem_male_pack() -> ConnectomePack:
    """Built-in 72-neuron sparse offline male CNS pack covering aggression, song, and foraging."""
    core_types = (
        "ORN_DA1",  # 0..3: cVA male pheromone
        "ORN_DA1",
        "ORN_DA1",
        "ORN_DA1",
        "LC10a",  # 4..7: visual rival
        "LC10a",
        "LC10a",
        "LC10a",
        "DA1_lPN",  # 8..11: cVA relay
        "DA1_lPN",
        "DA1_vPN",
        "DA1_vPN",
        "aIPg1",  # 12..15: aggression command
        "aIPg1",
        "aIPg2",
        "aIPg2",
        "DNb05",  # 16..17
        "DNb05",
        "DNg33",  # 18..19: aggression DN
        "DNg33",
        "hg3 MN",  # 20..21: bilateral wing-threat MN
        "hg4 MN",
        "pC1_14a",  # 22..25: male P1 courtship command
        "pC1_14a",
        "pC1_14a",
        "pC1_14a",
        "pIP10",  # 26..27: descending song command
        "pIP10",
        "dPR1",  # 28..29: pulse song premotor
        "dPR1",
        "ps1 MN",  # 30..31: pulse song wing MN
        "ps1 MN",
        "hg1 MN",  # 32..33: sine song wing MN
        "hg1 MN",
        "sugar_GRN",  # 34..35: taste
        "bitter_GRN",
        "CB0701",  # 36: MN9 proboscis
        "JO_B",  # 37: mechanosensory
        "ORN_L",  # 38: odor L
        "ORN_R",  # 39: odor R
        "DNp09",  # 40: forward
        "DNa01",  # 41: turn L
        "DNa01",  # 42: turn R
        "MDN",  # 43: backward
        "DNg84",  # 44: grooming/arousal
        "IN_bitter",  # 45: bitter inhibitory IN
    )
    n_core = len(core_types)  # 46
    n_bg = 26
    n = n_core + n_bg  # 72
    types = core_types + tuple("BG_M_IN" for _ in range(n_bg))
    roles = np.full(n, ROLE_INTERNEURON, dtype=np.uint8)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 34, 35, 37, 38, 39):
        roles[i] = ROLE_SENSORY
    for i in (16, 17, 18, 19, 20, 21, 26, 27, 30, 31, 32, 33, 36, 40, 41, 42, 43, 44):
        roles[i] = ROLE_MOTOR
    sides = tuple("left" if i % 2 == 0 else "right" for i in range(n))
    nts = tuple("gaba" if (i == 45 or i >= n_core) else "acetylcholine" for i in range(n))
    xs = np.linspace(-120000.0, 120000.0, n, dtype=np.float32)
    pos = np.stack([xs, np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)], axis=1)
    cells = CellTable(
        source_id=np.arange(3000, 3000 + n, dtype=np.int64),
        species_names=("synthetic_ecosystem_male",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=types,
        super_class=tuple(
            "sensory" if r == ROLE_SENSORY else ("motor" if r == ROLE_MOTOR else "central")
            for r in roles.tolist()
        ),
        cell_class=("ecosystem_male",) * n,
        cell_sub_class=("",) * n,
        region=("brain",) * n,
        side=sides,
        role=roles,
        cell_model=np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8),
        neurotransmitter=nts,
        position_nm=pos,
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    rows: list[tuple[int, int, int, int]] = []
    # cVA: ORN_DA1 (0..3) -> DA1_lPN/vPN (8..11) -> DNb05 (16..17)
    for s in range(0, 4):
        for p in range(8, 12):
            rows.append((s, p, 65, 1))
    for p in range(8, 12):
        for d in (16, 17):
            rows.append((p, d, 70, 1))
    # Visual rival aggression: LC10a (4..7) -> aIPg (12..15) -> DNg33 (18..19) & hg3/hg4 MN (20..21)
    for v in range(4, 8):
        for a in range(12, 16):
            rows.append((v, a, 62, 1))
    for a in range(12, 16):
        for m in (18, 19, 20, 21):
            rows.append((a, m, 65, 1))
    # Courtship song: pC1 (22..25) -> pIP10 (26..27) -> dPR1 (28..29) -> ps1 (30..31) & hg1 (32..33)
    for p1 in range(22, 26):
        for pip in (26, 27):
            rows.append((p1, pip, 70, 1))
    for pip in (26, 27):
        for dpr in (28, 29):
            rows.append((pip, dpr, 95, 1))
        for hg1 in (32, 33):
            rows.append((pip, hg1, 75, 1))
    for dpr in (28, 29):
        for ps1 in (30, 31):
            rows.append((dpr, ps1, 110, 1))
    # Taste, mechanosensory, steering
    rows.extend(
        [
            (34, 36, 95, 1),  # sugar -> MN9
            (35, 45, 90, 1),  # bitter -> IN_bitter
            (45, 36, 150, -1),  # IN_bitter -| MN9
            (37, 44, 85, 1),  # JO_B -> DNg84
            (38, 41, 65, 1),  # ORN_L -> DNa01 L
            (39, 42, 65, 1),  # ORN_R -> DNa01 R
        ]
    )
    # Sparse inhibitory background
    for b in range(n_core, n):
        rows.append((b, n_core + ((b - n_core + 1) % n_bg), 35, -1))
        rows.append((b, n_core + ((b - n_core + 7) % n_bg), 30, -1))
    edges = EdgeTable(
        pre=np.array([r[0] for r in rows], dtype=np.int32),
        post=np.array([r[1] for r in rows], dtype=np.int32),
        count=np.array([r[2] for r in rows], dtype=np.int32),
        sign=np.array([r[3] for r in rows], dtype=np.int8),
        kind=np.full(len(rows), SYNAPSE_CHEMICAL, dtype=np.uint8),
    )
    io_ports = {
        "sugar_grn": IOPort(
            "sugar_grn", "sensory", "Sugar GRN", "Sweet taste.", np.array([34], np.int32), "sugar"
        ),
        "bitter_grn": IOPort(
            "bitter_grn",
            "sensory",
            "Bitter GRN",
            "Bitter taste.",
            np.array([35], np.int32),
            "bitter",
        ),
        "antennal_grooming_mechanosensory": IOPort(
            "antennal_grooming_mechanosensory",
            "sensory",
            "JO-B mech",
            "Leg/antennal touch.",
            np.array([37], np.int32),
            "JO_B",
        ),
        "olfactory_sensory": IOPort(
            "olfactory_sensory",
            "sensory",
            "ORN L/R",
            "Bilateral odor.",
            np.array([38, 39], np.int32),
            "ORN",
        ),
        "mn9_proboscis_extension": IOPort(
            "mn9_proboscis_extension",
            "motor",
            "MN9",
            "Proboscis feeding.",
            np.array([36], np.int32),
            "CB0701",
        ),
        "grooming_dn": IOPort(
            "grooming_dn", "motor", "DNg84", "Touch arousal.", np.array([44], np.int32), "DNg84"
        ),
        "forward_dn": IOPort(
            "forward_dn", "motor", "DNp09", "Forward.", np.array([40], np.int32), "DNp09"
        ),
        "turn_left_dn": IOPort(
            "turn_left_dn", "motor", "DNa01 L", "Left.", np.array([41], np.int32), "DNa01"
        ),
        "turn_right_dn": IOPort(
            "turn_right_dn", "motor", "DNa01 R", "Right.", np.array([42], np.int32), "DNa01"
        ),
        "backward_dn": IOPort(
            "backward_dn", "motor", "MDN", "Backward.", np.array([43], np.int32), "MDN"
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_ecosystem_male",
        species="Synthetic male Drosophila whole-CNS ecosystem circuit",
        version="f11-eco-v1",
        citation="Built-in 72-neuron offline male fixture mirroring MaleCNS v1.0",
        licence_note="Internal ALife Lab test fixture",
        files=(),
    )
    digest = compute_pack_digest(prov.dataset_id, prov.species, cells, edges, io_ports, prov)
    return ConnectomePack(prov.dataset_id, prov.species, cells, edges, io_ports, prov, digest)


def make_synthetic_ecosystem_female_pack() -> ConnectomePack:
    """Built-in 72-neuron sparse offline female CNS pack covering auditory, taste, and touch."""
    core_types = (
        "JO-B",  # 0..3: auditory Johnston's organ
        "JO-B",
        "JO-B",
        "JO-B",
        "SAD052",  # 4..7: auditory relays
        "SAD053",
        "CB1078",
        "AVLP082",
        "pC1a",  # 8..9: female receptivity
        "pC1a",
        "sugar_GRN",  # 10..11: sugar taste
        "sugar_GRN",
        "bitter_GRN",  # 12..13: bitter taste
        "bitter_GRN",
        "IN_sugar",  # 14: excitatory taste relay
        "IN_bitter",  # 15: inhibitory bitter relay
        "CB0701",  # 16..17: MN9 proboscis extension
        "CB0701",
        "JO_B_mech",  # 18: mechanosensory touch
        "ORN_L",  # 19: odor L
        "ORN_R",  # 20: odor R
        "DNp09",  # 21: forward
        "DNa01",  # 22: turn L
        "DNa01",  # 23: turn R
        "MDN",  # 24: backward
        "DNg84",  # 25: grooming/arousal
    )
    n_core = len(core_types)  # 26
    n_bg = 46
    n = n_core + n_bg  # 72
    types = core_types + tuple("BG_F_IN" for _ in range(n_bg))
    roles = np.full(n, ROLE_INTERNEURON, dtype=np.uint8)
    for i in (0, 1, 2, 3, 10, 11, 12, 13, 18, 19, 20):
        roles[i] = ROLE_SENSORY
    for i in (16, 17, 21, 22, 23, 24, 25):
        roles[i] = ROLE_MOTOR
    sides = tuple("left" if i % 2 == 0 else "right" for i in range(n))
    nts = tuple("gaba" if (i == 15 or i >= n_core) else "acetylcholine" for i in range(n))
    xs = np.linspace(-120000.0, 120000.0, n, dtype=np.float32)
    pos = np.stack([xs, np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)], axis=1)
    cells = CellTable(
        source_id=np.arange(4000, 4000 + n, dtype=np.int64),
        species_names=("synthetic_ecosystem_female",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=types,
        super_class=tuple(
            "sensory" if r == ROLE_SENSORY else ("motor" if r == ROLE_MOTOR else "central")
            for r in roles.tolist()
        ),
        cell_class=("ecosystem_female",) * n,
        cell_sub_class=("",) * n,
        region=("brain",) * n,
        side=sides,
        role=roles,
        cell_model=np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8),
        neurotransmitter=nts,
        position_nm=pos,
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    rows: list[tuple[int, int, int, int]] = []
    # Auditory: JO-B (0..3) -> SAD052/53 + CB1078 + AVLP082 (4..7) -> pC1a (8..9)
    for j in range(0, 4):
        for r in range(4, 8):
            rows.append((j, r, 85, 1))
    for r in range(4, 8):
        for p in (8, 9):
            rows.append((r, p, 70, 1))
    # Sugar/Bitter taste -> MN9 (16..17)
    for sg in (10, 11):
        rows.append((sg, 14, 70, 1))
    rows.extend(
        [
            (14, 16, 90, 1),
            (14, 17, 90, 1),
            (12, 15, 85, 1),
            (13, 15, 85, 1),
            (15, 14, 150, -1),
            (15, 16, 160, -1),
            (15, 17, 160, -1),
            (18, 25, 85, 1),
            (19, 22, 65, 1),
            (20, 23, 65, 1),
        ]
    )
    for b in range(n_core, n):
        rows.append((b, n_core + ((b - n_core + 1) % n_bg), 35, -1))
        rows.append((b, n_core + ((b - n_core + 9) % n_bg), 30, -1))
    edges = EdgeTable(
        pre=np.array([r[0] for r in rows], dtype=np.int32),
        post=np.array([r[1] for r in rows], dtype=np.int32),
        count=np.array([r[2] for r in rows], dtype=np.int32),
        sign=np.array([r[3] for r in rows], dtype=np.int8),
        kind=np.full(len(rows), SYNAPSE_CHEMICAL, dtype=np.uint8),
    )
    io_ports = {
        "sugar_grn": IOPort(
            "sugar_grn",
            "sensory",
            "Sugar GRN",
            "Sweet taste.",
            np.array([10, 11], np.int32),
            "sugar",
        ),
        "bitter_grn": IOPort(
            "bitter_grn",
            "sensory",
            "Bitter GRN",
            "Bitter taste.",
            np.array([12, 13], np.int32),
            "bitter",
        ),
        "antennal_grooming_mechanosensory": IOPort(
            "antennal_grooming_mechanosensory",
            "sensory",
            "Touch mech",
            "Leg/antennal touch.",
            np.array([18], np.int32),
            "JO_B_mech",
        ),
        "olfactory_sensory": IOPort(
            "olfactory_sensory",
            "sensory",
            "ORN L/R",
            "Bilateral odor.",
            np.array([19, 20], np.int32),
            "ORN",
        ),
        "mn9_proboscis_extension": IOPort(
            "mn9_proboscis_extension",
            "motor",
            "MN9",
            "Proboscis feeding.",
            np.array([16, 17], np.int32),
            "CB0701",
        ),
        "grooming_dn": IOPort(
            "grooming_dn", "motor", "DNg84", "Touch arousal.", np.array([25], np.int32), "DNg84"
        ),
        "forward_dn": IOPort(
            "forward_dn", "motor", "DNp09", "Forward.", np.array([21], np.int32), "DNp09"
        ),
        "turn_left_dn": IOPort(
            "turn_left_dn", "motor", "DNa01 L", "Left.", np.array([22], np.int32), "DNa01"
        ),
        "turn_right_dn": IOPort(
            "turn_right_dn", "motor", "DNa01 R", "Right.", np.array([23], np.int32), "DNa01"
        ),
        "backward_dn": IOPort(
            "backward_dn", "motor", "MDN", "Backward.", np.array([24], np.int32), "MDN"
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_ecosystem_female",
        species="Synthetic female Drosophila whole-CNS ecosystem circuit",
        version="f11-eco-v1",
        citation="Built-in 72-neuron offline female fixture mirroring BANC v888",
        licence_note="Internal ALife Lab test fixture",
        files=(),
    )
    digest = compute_pack_digest(prov.dataset_id, prov.species, cells, edges, io_ports, prov)
    return ConnectomePack(prov.dataset_id, prov.species, cells, edges, io_ports, prov, digest)


@dataclass
class _SexBatch:
    """One sex-specific whole-CNS cohort (e.g., 8 MaleCNS males OR 8 BANC females)."""

    pack: ConnectomePack
    csr: CSRGraph
    ports: EmbodiedPorts
    count: int
    seeds: list[int]
    sens_union: I32
    cmd_union: I32
    stim_buf: F32
    prev_cmd_spikes: I32
    rates: F64  # (count, 6): fwd, tl, tr, bwd, prob, groom
    # male: (count, 6)=[DA1_PN, DNb05, DNg33, hg34, ps1, hg1]; female: (count, 2)=[aud, pC1]
    extra_rates: F64
    # Local indices inside sens_union
    s_sugar: I32
    s_bitter: I32
    s_groom: I32
    s_fwd: I32
    s_tl: I32
    s_tr: I32
    s_cva: I32
    s_lc10a: I32
    s_aipg: I32
    s_pc1: I32
    s_job: I32
    # Local indices inside cmd_union
    c_extra: tuple[I32, ...]
    gpu_brain: Any = None
    cpu_brains: list[LIFState] = field(default_factory=list)


def _build_sex_batch(
    pack: ConnectomePack,
    *,
    is_male: bool,
    count: int,
    wiring: str,
    seed: int,
    backend: str,
    params: LIFParams,
) -> _SexBatch:
    csr = pack.edges.to_csr(pack.cells.num_cells)
    if wiring == "shuffled_control":
        csr = make_shuffled_csr(csr, seed)
    ports = resolve_embodied_ports(pack)
    ctypes = np.asarray(pack.cells.cell_type)
    n = pack.cells.num_cells
    seeds = [int(seed) + i * 131 for i in range(count)]

    cva = np.flatnonzero(ctypes == "ORN_DA1").astype(np.int32) if is_male else np.zeros(0, np.int32)
    lc10a = np.flatnonzero(ctypes == "LC10a").astype(np.int32) if is_male else np.zeros(0, np.int32)
    aipg = (
        np.flatnonzero([str(t).startswith("aIPg") for t in ctypes]).astype(np.int32)
        if is_male
        else np.zeros(0, np.int32)
    )
    pc1 = (
        np.flatnonzero([str(t).startswith("pC1_") for t in ctypes]).astype(np.int32)
        if is_male
        else np.zeros(0, np.int32)
    )
    job = (
        np.flatnonzero((ctypes == "JO-B") | (ctypes == "JO_B")).astype(np.int32)
        if not is_male
        else np.zeros(0, np.int32)
    )

    sens_parts = [
        a
        for a in (
            ports.sugar_grn,
            ports.bitter_grn,
            ports.groom_mech,
            ports.forward_dn,
            ports.turn_left_dn,
            ports.turn_right_dn,
            ports.backward_dn,
            cva,
            lc10a,
            aipg,
            pc1,
            job,
        )
        if a.size > 0
    ]
    sens_union = np.unique(np.concatenate(sens_parts)).astype(np.int32)
    spos = {int(c): k for k, c in enumerate(sens_union.tolist())}

    def _sloc(arr: I32) -> I32:
        return np.asarray([spos[int(c)] for c in arr.tolist() if int(c) in spos], dtype=np.int32)

    ex_arrs: tuple[I32, ...]
    if is_male:
        ex_arrs = (
            np.flatnonzero((ctypes == "DA1_lPN") | (ctypes == "DA1_vPN")).astype(np.int32),
            np.flatnonzero(ctypes == "DNb05").astype(np.int32),
            np.flatnonzero(ctypes == "DNg33").astype(np.int32),
            np.flatnonzero((ctypes == "hg3 MN") | (ctypes == "hg4 MN")).astype(np.int32),
            np.flatnonzero(ctypes == "ps1 MN").astype(np.int32),
            np.flatnonzero(ctypes == "hg1 MN").astype(np.int32),
        )
    else:
        aud_names = {"SAD052", "SAD053", "CB1078", "AVLP082", "AVLP194_a"}
        ex_arrs = (
            np.flatnonzero([str(t) in aud_names for t in ctypes]).astype(np.int32),
            np.flatnonzero(
                [str(t) in {"pC1a", "pC1b", "pC1c", "pC1d", "pC1e"} for t in ctypes]
            ).astype(np.int32),
        )

    non_empty_ex = [a for a in ex_arrs if a.size > 0]
    ex_unique = (
        np.unique(np.concatenate(non_empty_ex)).astype(np.int32)
        if non_empty_ex
        else np.zeros(0, dtype=np.int32)
    )
    base_cmd = ports.all_command_union
    cmd_union = np.concatenate([base_cmd, ex_unique]).astype(np.int32)
    cpos = {int(c): int(base_cmd.size) + k for k, c in enumerate(ex_unique.tolist())}
    c_extra = tuple(
        np.asarray([cpos[int(c)] for c in a.tolist() if int(c) in cpos], dtype=np.int32)
        for a in ex_arrs
    )

    gpu_brain: Any = None
    cpu_brains: list[LIFState] = []
    if backend == "cupy":
        from alife_lab.connectome.lif_cupy import CupyBatchedLIFRunner

        gpu_brain = CupyBatchedLIFRunner(csr, params, seeds, np.zeros((count, n), dtype=np.float32))
    else:
        cpu_brains = [make_initial_state(n, params) for _ in range(count)]

    return _SexBatch(
        pack=pack,
        csr=csr,
        ports=ports,
        count=count,
        seeds=seeds,
        sens_union=sens_union,
        cmd_union=cmd_union,
        stim_buf=np.zeros((count, sens_union.size), dtype=np.float32),
        prev_cmd_spikes=np.zeros((count, cmd_union.size), dtype=np.int32),
        rates=np.zeros((count, 6), dtype=np.float64),
        extra_rates=np.zeros((count, len(ex_arrs)), dtype=np.float64),
        s_sugar=_sloc(ports.sugar_grn),
        s_bitter=_sloc(ports.bitter_grn),
        s_groom=_sloc(ports.groom_mech),
        s_fwd=_sloc(ports.forward_dn),
        s_tl=_sloc(ports.turn_left_dn),
        s_tr=_sloc(ports.turn_right_dn),
        s_cva=_sloc(cva),
        s_lc10a=_sloc(lc10a),
        s_aipg=_sloc(aipg),
        s_pc1=_sloc(pc1),
        s_job=_sloc(job),
        c_extra=c_extra,
        gpu_brain=gpu_brain,
        cpu_brains=cpu_brains,
    )


def _step_sex_batch(batch: _SexBatch, params: LIFParams, alpha: float) -> None:
    if batch.gpu_brain is not None:
        batch.gpu_brain.upload_stim_subset_batch(batch.sens_union, batch.stim_buf)
        batch.gpu_brain.step_substeps(params.substeps_per_tick, use_graph=True)
        cur_tot = batch.gpu_brain.read_total_spikes_subset(batch.cmd_union)
    else:
        cur_tot = np.empty((batch.count, batch.cmd_union.size), dtype=np.int32)
        for i, st in enumerate(batch.cpu_brains):
            st.stim_rate_hz[batch.sens_union] = batch.stim_buf[i]
            step_tick_cpu(st, batch.csr, params, batch.seeds[i])
            cur_tot[i] = st.total_spikes[batch.cmd_union]

    delta = cur_tot - batch.prev_cmd_spikes
    batch.prev_cmd_spikes = cur_tot
    for ci, cname in enumerate(
        ("forward", "turn_left", "turn_right", "backward", "proboscis", "grooming")
    ):
        lo, hi = batch.ports.cmd_slices[cname]
        if hi > lo:
            inst_hz = delta[:, lo:hi].mean(axis=1) * 1000.0
            batch.rates[:, ci] = (1.0 - alpha) * batch.rates[:, ci] + alpha * inst_hz
    for ei, sl in enumerate(batch.c_extra):
        if sl.size > 0:
            inst_hz = delta[:, sl].mean(axis=1) * 1000.0
            batch.extra_rates[:, ei] = (1.0 - alpha) * batch.extra_rates[:, ei] + alpha * inst_hz


def _ensure_batch_cpu(batch: _SexBatch, params: LIFParams, tick: int) -> list[LIFState]:
    if batch.gpu_brain is None:
        return batch.cpu_brains
    gb = batch.gpu_brain
    gb.synchronize()
    v = gb.cp.asnumpy(gb.d_v)
    g = gb.cp.asnumpy(gb.d_g)
    refrac = gb.cp.asnumpy(gb.d_refrac)
    delay = gb.cp.asnumpy(gb.d_delay_ring)
    stim = gb.cp.asnumpy(gb.d_stim_rate_hz)
    sil = gb.cp.asnumpy(gb.d_silenced)
    tot = gb.cp.asnumpy(gb.d_total_spikes)
    substep = int(gb.cp.asnumpy(gb.d_substep)[0])
    if len(batch.cpu_brains) != batch.count:
        batch.cpu_brains = [
            make_initial_state(batch.pack.cells.num_cells, params) for _ in range(batch.count)
        ]
    for i, st in enumerate(batch.cpu_brains):
        st.v[...] = v[i]
        st.g[...] = g[i]
        st.refrac[...] = refrac[i]
        st.delay_ring[...] = delay[i]
        st.stim_rate_hz[...] = stim[i]
        st.silenced[...] = sil[i]
        st.total_spikes[...] = tot[i]
        st.tick = tick
        st.substep = substep
    return batch.cpu_brains


@dataclass
class DualSexEcosystemWorld:
    """Unified F11 ecological arena running real male + female whole-CNS batches concurrently.

    Flies ``0 .. n_males-1`` are males (``male_batch``), ``n_males .. num_flies-1`` are females
    (``female_batch``).
    """

    male_batch: _SexBatch
    female_batch: _SexBatch
    params: LIFParams
    num_flies: int
    n_males: int
    n_females: int
    seed: int
    backend: str
    wiring: str
    social_mode: str = "dual_sex_ecosystem"
    enable_touch: bool = True
    odor_enabled: bool = True
    arena_half_size: float = 4.5
    explore_drive_hz: float = 65.0
    foods: tuple[dict[str, Any], ...] = DEFAULT_ECOSYSTEM_FOODS
    # Kinematics & behavior states (length num_flies)
    x: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    y: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    yaw: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    speed: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    sex: U8 = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    active_state: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    touch_timer_ms: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    lunge_timer_ms: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    retreat_timer_ms: I32 = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    dominance_score: F32 = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    song_intensity: F32 = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    threat_intensity: F32 = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    feeding_intensity: F32 = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    listening_intensity: F32 = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    rng_state_bitgen: Any = None
    active_contacts: list[list[float]] = field(default_factory=list)
    active_clashes: list[list[float]] = field(default_factory=list)
    # Cumulative metrics
    tick: int = 0
    total_encounters: int = 0
    total_lunges: int = 0
    total_wing_threat_ms: int = 0
    total_feeding_ms: int = 0
    total_song_ms: int = 0
    clean_zone_samples: int = 0
    total_fly_samples: int = 0
    last_clean_fraction: float = 0.8
    last_mean_speed: float = 0.0
    last_active_singers: int = 0
    last_active_threats: int = 0
    last_active_feeders: int = 0
    last_female_auditory_hz: float = 0.0
    step_ms_ema: float = 0.0

    @property
    def total_neurons(self) -> int:
        return (
            self.male_batch.pack.cells.num_cells * self.n_males
            + self.female_batch.pack.cells.num_cells * self.n_females
        )

    @property
    def realtime_factor(self) -> float:
        return round(1.0 / max(self.step_ms_ema, 1e-4), 3) if self.tick > 0 else 0.0

    def ensure_cpu_brains(self) -> tuple[list[LIFState], list[LIFState]]:
        return (
            _ensure_batch_cpu(self.male_batch, self.params, self.tick),
            _ensure_batch_cpu(self.female_batch, self.params, self.tick),
        )


def make_dual_sex_ecosystem_world(
    male_pack: ConnectomePack,
    female_pack: ConnectomePack,
    *,
    num_flies: int = 16,
    wiring: str = "reference",
    seed: int = 1,
    backend: str = "auto",
    enable_touch: bool = True,
    odor_enabled: bool = True,
    arena_half_size: float = 4.5,
    explore_drive_hz: float = 65.0,
) -> DualSexEcosystemWorld:
    """Create the unified F11 ecological world with distinct male + female whole-CNS batches."""
    if num_flies < 2 or num_flies > 64:
        raise ValueError(f"dual_sex_ecosystem requires num_flies in 2..64, got {num_flies}")
    n_males = num_flies // 2
    n_females = num_flies - n_males

    eff_b = backend
    if eff_b == "auto":
        from alife_lab.connectome.lif_cupy import available as cupy_ok

        big = max(male_pack.cells.num_cells, female_pack.cells.num_cells) > 500
        eff_b = "cupy" if (big and cupy_ok()) else "numpy"

    params = LIFParams()
    male_batch = _build_sex_batch(
        male_pack,
        is_male=True,
        count=n_males,
        wiring=wiring,
        seed=seed,
        backend=eff_b,
        params=params,
    )
    female_batch = _build_sex_batch(
        female_pack,
        is_male=False,
        count=n_females,
        wiring=wiring,
        seed=seed + 7919,
        backend=eff_b,
        params=params,
    )

    rng = np.random.default_rng(seed)
    x = np.empty(num_flies, dtype=np.float64)
    y = np.empty(num_flies, dtype=np.float64)
    # Place males & females distributed across clean foraging/social zones and near the odor edge
    for i in range(num_flies):
        if i % 4 == 0:
            # 25% start near the aversive odor edge to immediately exercise Ramdya odor escape
            x[i] = float(rng.uniform(-3.4, -1.6))
            y[i] = float(rng.uniform(-2.6, 2.6))
        else:
            # 75% start around the two sugar patches & central social arena
            patch = DEFAULT_ECOSYSTEM_FOODS[i % 2]
            ang = float(rng.uniform(-math.pi, math.pi))
            rad = float(rng.uniform(0.35, 1.95))
            x[i] = float(np.clip(patch["x"] + rad * math.cos(ang), -3.5, 3.5))
            y[i] = float(np.clip(patch["y"] + rad * math.sin(ang), -3.5, 3.5))

    yaw = rng.uniform(-math.pi, math.pi, size=num_flies).astype(np.float64)
    sex = np.zeros(num_flies, dtype=np.uint8)
    sex[:n_males] = 1
    return DualSexEcosystemWorld(
        male_batch=male_batch,
        female_batch=female_batch,
        params=params,
        num_flies=num_flies,
        n_males=n_males,
        n_females=n_females,
        seed=seed,
        backend=eff_b,
        wiring=wiring,
        enable_touch=enable_touch,
        odor_enabled=odor_enabled,
        arena_half_size=arena_half_size,
        explore_drive_hz=explore_drive_hz,
        x=x,
        y=y,
        yaw=yaw,
        speed=np.zeros(num_flies, dtype=np.float64),
        sex=sex,
        active_state=np.ones(num_flies, dtype=np.float64),
        touch_timer_ms=np.zeros(num_flies, dtype=np.int32),
        lunge_timer_ms=np.zeros(num_flies, dtype=np.int32),
        retreat_timer_ms=np.zeros(num_flies, dtype=np.int32),
        dominance_score=np.zeros(num_flies, dtype=np.float32),
        song_intensity=np.zeros(num_flies, dtype=np.float32),
        threat_intensity=np.zeros(num_flies, dtype=np.float32),
        feeding_intensity=np.zeros(num_flies, dtype=np.float32),
        listening_intensity=np.zeros(num_flies, dtype=np.float32),
        rng_state_bitgen=rng,
    )


def _strip_odor(x: F64, enabled: bool) -> F64:
    if not enabled:
        return np.zeros_like(x)
    return 1.0 / (1.0 + np.exp(np.clip((x - ECOSYSTEM_ODOR_X_MAX) / 0.45, -12.0, 12.0)))


def step_dual_sex_ecosystem_tick(world: DualSexEcosystemWorld) -> None:
    """Advance all N_m real male brains + N_f real female brains by 1.0 ms in ONE shared arena."""
    t0 = time.perf_counter()
    n_all = world.num_flies
    n_m = world.n_males
    n_f = world.n_females
    rng: np.random.Generator = world.rng_state_bitgen

    # 1. Pairwise distances, steric separation & Ramdya leg-touch encounters
    dx = world.x[None, :] - world.x[:, None]
    dy = world.y[None, :] - world.y[:, None]
    dist = np.hypot(dx, dy)
    np.fill_diagonal(dist, 999.0)

    touch_radius = 0.58
    steric_radius = 0.36
    repulse_turn = np.zeros(n_all, dtype=np.float64)
    contacts: list[list[float]] = []
    encounters_now = 0
    for i in range(n_all):
        for j in range(i + 1, n_all):
            d = float(dist[i, j])
            if d < touch_radius:
                if d < steric_radius and d > 1e-4:
                    push = 0.5 * (steric_radius - d)
                    ux, uy = float(dx[i, j]) / d, float(dy[i, j]) / d
                    world.x[i] -= ux * push
                    world.y[i] -= uy * push
                    world.x[j] += ux * push
                    world.y[j] += uy * push
                if world.enable_touch:
                    if world.touch_timer_ms[i] == 0:
                        encounters_now += 1
                    if world.touch_timer_ms[j] == 0:
                        encounters_now += 1
                    world.touch_timer_ms[i] = 240
                    world.touch_timer_ms[j] = 240
                    ang_ij = math.atan2(float(dy[i, j]), float(dx[i, j]))
                    rel_i = math.atan2(
                        math.sin(ang_ij - world.yaw[i]), math.cos(ang_ij - world.yaw[i])
                    )
                    rel_j = math.atan2(
                        math.sin((ang_ij + math.pi) - world.yaw[j]),
                        math.cos((ang_ij + math.pi) - world.yaw[j]),
                    )
                    repulse_turn[i] -= 1.0 if rel_i >= 0 else -1.0
                    repulse_turn[j] -= 1.0 if rel_j >= 0 else -1.0
                    contacts.append(
                        [
                            round(float(world.x[i]), 3),
                            round(float(world.y[i]), 3),
                            round(float(world.x[j]), 3),
                            round(float(world.y[j]), 3),
                        ]
                    )

    world.active_contacts = contacts[:32]
    world.total_encounters += encounters_now
    world.touch_timer_ms = np.maximum(0, world.touch_timer_ms - 1)
    world.lunge_timer_ms = np.maximum(0, world.lunge_timer_ms - 1)
    world.retreat_timer_ms = np.maximum(0, world.retreat_timer_ms - 1)

    # 2. Food taste (sugar vs bitter) & left-strip aversive odor
    sugar_hz = np.zeros(n_all, dtype=np.float64)
    bitter_hz = np.zeros(n_all, dtype=np.float64)
    food_turn = np.zeros(n_all, dtype=np.float64)
    for i in range(n_all):
        best_sd = 999.0
        best_ang = 0.0
        for fd in world.foods:
            fx, fy, fr = float(fd["x"]), float(fd["y"]), float(fd["r"])
            df = math.hypot(fx - float(world.x[i]), fy - float(world.y[i]))
            if fd["kind"] == "sugar":
                if df < fr:
                    sugar_hz[i] = 220.0
                if df < best_sd:
                    best_sd = df
                    best_ang = math.atan2(fy - float(world.y[i]), fx - float(world.x[i]))
            elif df < fr:
                bitter_hz[i] = 240.0
                away = math.atan2(float(world.y[i]) - fy, float(world.x[i]) - fx)
                food_turn[i] += 1.6 * math.atan2(
                    math.sin(away - world.yaw[i]), math.cos(away - world.yaw[i])
                )
        if best_sd > 0.55 and best_sd < 3.6 and bitter_hz[i] == 0.0:
            bear_f = math.atan2(
                math.sin(best_ang - world.yaw[i]), math.cos(best_ang - world.yaw[i])
            )
            food_turn[i] += 0.65 * float(np.clip(bear_f, -1.0, 1.0))

    cy, sy = np.cos(world.yaw), np.sin(world.yaw)
    odor_c = _strip_odor(world.x, world.odor_enabled)
    odor_l = _strip_odor(world.x + 0.25 * cy - 0.14 * sy, world.odor_enabled)
    odor_r = _strip_odor(world.x + 0.25 * cy + 0.14 * sy, world.odor_enabled)
    turn_cmd_raw = (odor_r - odor_l) * 9.0 + food_turn + 0.30 * repulse_turn

    # 3. Social sensing:
    #    - Male <-> Male: cVA (ORN_DA1) + visual rival (LC10a) + aIPg escalation
    #    - Male -> Female: visual pursuit + female cuticular pheromone -> pC1_* song command
    #    - Female <- Male: near-field acoustic song (ps1/hg1 MN) -> female JO-B (416 cells!)
    m_cva = np.zeros(n_m, dtype=np.float64)
    m_lc10a = np.zeros(n_m, dtype=np.float64)
    m_aipg = np.zeros(n_m, dtype=np.float64)
    m_pc1 = np.zeros(n_m, dtype=np.float64)
    nearest_rival = np.full(n_m, -1, dtype=np.int32)
    nearest_rival_dist = np.full(n_m, 999.0, dtype=np.float64)

    for i in range(n_m):
        # Nearest rival male (0..n_m-1)
        if n_m >= 2:
            rm_d = dist[i, :n_m]
            j_m = int(np.argmin(rm_d))
            d_mm = float(rm_d[j_m])
            nearest_rival[i] = j_m
            nearest_rival_dist[i] = d_mm
        else:
            j_m, d_mm = -1, 999.0

        # Nearest female (n_m..n_all-1)
        rf_d = dist[i, n_m:]
        j_f = n_m + int(np.argmin(rf_d))
        d_mf = float(rf_d[j_f - n_m])

        if world.retreat_timer_ms[i] > 0 and j_m >= 0:
            ang_r = math.atan2(float(dy[i, j_m]), float(dx[i, j_m]))
            away = math.atan2(
                math.sin((ang_r + math.pi) - world.yaw[i]),
                math.cos((ang_r + math.pi) - world.yaw[i]),
            )
            turn_cmd_raw[i] = float(np.clip(2.0 * away, -1.5, 1.5))
            continue

        # Male-male aggression cues (when rival male is close)
        if j_m >= 0 and d_mm < 2.1:
            ang_r = math.atan2(float(dy[i, j_m]), float(dx[i, j_m]))
            bear_r = math.atan2(math.sin(ang_r - world.yaw[i]), math.cos(ang_r - world.yaw[i]))
            if world.enable_touch and abs(bear_r) < 1.85:
                m_lc10a[i] = min(130.0, 125.0 * (1.0 - d_mm / 2.4) + 25.0)
                if d_mm < d_mf + 0.35:
                    turn_cmd_raw[i] += float(np.clip(1.4 * bear_r, -1.2, 1.2))
            m_cva[i] = min(210.0, 200.0 * math.exp(-((d_mm / 1.15) ** 2)))
            if world.enable_touch and d_mm < 1.15:
                m_aipg[i] = 140.0

        # Male-to-female courtship cues (when female is close and male isn't fighting closer rival)
        if d_mf < 2.6 and (d_mf <= d_mm + 0.45):
            ang_f = math.atan2(float(dy[i, j_f]), float(dx[i, j_f]))
            bear_f = math.atan2(math.sin(ang_f - world.yaw[i]), math.cos(ang_f - world.yaw[i]))
            if world.enable_touch:
                turn_cmd_raw[i] += float(np.clip(1.55 * bear_f, -1.3, 1.3))
            if d_mf < 1.75:
                m_pc1[i] = min(180.0, 170.0 * math.exp(-((d_mf / 1.15) ** 2)))

    # Female near-field acoustic hearing: real male ps1/hg1 MN firing -> female JO-B!
    f_job = np.zeros(n_f, dtype=np.float64)
    male_song_hz = (
        world.male_batch.extra_rates[:, 4] + 0.6 * world.male_batch.extra_rates[:, 5]
    )  # ps1 MN + hg1 MN
    for fi in range(n_f):
        idx_f = n_m + fi
        ac_in = 0.0
        for mi in range(n_m):
            if male_song_hz[mi] > 5.0:
                d_fm = float(dist[idx_f, mi])
                if d_fm < 2.2:
                    atten = 1.0 / (1.0 + (d_fm / 0.70) ** 3)
                    ac_in += float(male_song_hz[mi]) * atten
        f_job[fi] = min(220.0, ac_in * 1.65)

    # 4. Upload sensory stimuli & step BOTH real whole-CNS batches (MaleCNS + BANC)!
    mb = world.male_batch
    fb = world.female_batch
    mb.stim_buf.fill(np.float32(0.0))
    fb.stim_buf.fill(np.float32(0.0))

    touch_act = (world.touch_timer_ms > 0).astype(np.float64)
    if mb.s_sugar.size > 0:
        mb.stim_buf[:, mb.s_sugar] = sugar_hz[:n_m, None].astype(np.float32)
    if mb.s_bitter.size > 0:
        mb.stim_buf[:, mb.s_bitter] = bitter_hz[:n_m, None].astype(np.float32)
    if mb.s_groom.size > 0:
        mb.stim_buf[:, mb.s_groom] = (140.0 * touch_act[:n_m])[:, None].astype(np.float32)
    if mb.s_cva.size > 0:
        mb.stim_buf[:, mb.s_cva] = m_cva[:, None].astype(np.float32)
    if mb.s_lc10a.size > 0:
        mb.stim_buf[:, mb.s_lc10a] = m_lc10a[:, None].astype(np.float32)
    if mb.s_aipg.size > 0:
        mb.stim_buf[:, mb.s_aipg] = m_aipg[:, None].astype(np.float32)
    if mb.s_pc1.size > 0:
        mb.stim_buf[:, mb.s_pc1] = m_pc1[:, None].astype(np.float32)

    m_arousal = np.where(
        world.lunge_timer_ms[:n_m] > 0,
        170.0,
        world.explore_drive_hz + 75.0 * odor_c[:n_m] + 55.0 * touch_act[:n_m],
    )
    if mb.s_fwd.size > 0:
        mb.stim_buf[:, mb.s_fwd] = np.clip(m_arousal, 0.0, 220.0)[:, None].astype(np.float32)
    if mb.s_tl.size > 0:
        mb.stim_buf[:, mb.s_tl] = np.clip(110.0 * turn_cmd_raw[:n_m], 0.0, 190.0)[:, None].astype(
            np.float32
        )
    if mb.s_tr.size > 0:
        mb.stim_buf[:, mb.s_tr] = np.clip(-110.0 * turn_cmd_raw[:n_m], 0.0, 190.0)[:, None].astype(
            np.float32
        )

    if fb.s_sugar.size > 0:
        fb.stim_buf[:, fb.s_sugar] = sugar_hz[n_m:, None].astype(np.float32)
    if fb.s_bitter.size > 0:
        fb.stim_buf[:, fb.s_bitter] = bitter_hz[n_m:, None].astype(np.float32)
    if fb.s_groom.size > 0:
        fb.stim_buf[:, fb.s_groom] = (140.0 * touch_act[n_m:])[:, None].astype(np.float32)
    if fb.s_job.size > 0:
        fb.stim_buf[:, fb.s_job] = f_job[:, None].astype(np.float32)

    f_aud_slow = np.clip(1.0 - fb.extra_rates[:, 0] / 65.0, 0.25, 1.0)
    f_feed_halt = np.where(fb.rates[:, 4] > 6.0, 0.15, 1.0)
    f_arousal = (
        (world.explore_drive_hz + 80.0 * odor_c[n_m:] + 60.0 * touch_act[n_m:])
        * f_aud_slow
        * f_feed_halt
    )
    if fb.s_fwd.size > 0:
        fb.stim_buf[:, fb.s_fwd] = np.clip(f_arousal, 0.0, 220.0)[:, None].astype(np.float32)
    if fb.s_tl.size > 0:
        fb.stim_buf[:, fb.s_tl] = np.clip(110.0 * turn_cmd_raw[n_m:], 0.0, 190.0)[:, None].astype(
            np.float32
        )
    if fb.s_tr.size > 0:
        fb.stim_buf[:, fb.s_tr] = np.clip(-110.0 * turn_cmd_raw[n_m:], 0.0, 190.0)[:, None].astype(
            np.float32
        )

    alpha = 1.0 - math.exp(-1.0 / 22.0)
    _step_sex_batch(mb, world.params, alpha)
    _step_sex_batch(fb, world.params, alpha)

    # 5. Decode all 4 biological behaviors from real connectome outputs!
    # (a) Feeding via real MN9 proboscis extension (CB0701)
    all_mn9 = np.concatenate([mb.rates[:, 4], fb.rates[:, 4]])
    feed_now = np.clip((all_mn9 - 4.0) / 20.0, 0.0, 1.0).astype(np.float32)
    world.feeding_intensity = (0.80 * world.feeding_intensity + 0.20 * feed_now).astype(np.float32)

    # (b) Male courtship song from real ps1 MN + hg1 MN
    song_now = np.zeros(n_all, dtype=np.float32)
    song_now[:n_m] = np.clip(
        (mb.extra_rates[:, 4] + 0.5 * mb.extra_rates[:, 5] - 8.0) / 70.0, 0.0, 1.0
    ).astype(np.float32)
    world.song_intensity = (0.82 * world.song_intensity + 0.18 * song_now).astype(np.float32)

    # (c) Female auditory reception from real BANC SAD052/53 + CB1078 + AVLP082 relays
    listen_now = np.zeros(n_all, dtype=np.float32)
    listen_now[n_m:] = np.clip((fb.extra_rates[:, 0] - 4.0) / 45.0, 0.0, 1.0).astype(np.float32)
    world.listening_intensity = (0.82 * world.listening_intensity + 0.18 * listen_now).astype(
        np.float32
    )

    # (d) Male-male bilateral wing threat & oriented lunges from real DA1_PN + DNg33 + hg3/hg4 MN
    threat_now = np.zeros(n_all, dtype=np.float32)
    cva_id = np.clip((mb.extra_rates[:, 0] - 12.0) / 85.0, 0.0, 1.0)
    dng33_sig = np.clip((mb.extra_rates[:, 2] - 15.0) / 90.0, 0.0, 1.0)
    hg34_sig = np.clip((mb.extra_rates[:, 3] - 32.0) / 70.0, 0.0, 1.0)
    m_threat = cva_id * hg34_sig * np.clip(0.35 + 0.65 * dng33_sig, 0.0, 1.0)
    threat_now[:n_m] = np.where(world.retreat_timer_ms[:n_m] == 0, m_threat, 0.0).astype(np.float32)
    world.threat_intensity = (0.82 * world.threat_intensity + 0.18 * threat_now).astype(np.float32)

    clashes: list[list[float]] = []
    for i in range(n_m):
        j = int(nearest_rival[i])
        if j < 0:
            continue
        ang_ij = math.atan2(float(dy[i, j]), float(dx[i, j]))
        bear_ij = abs(math.atan2(math.sin(ang_ij - world.yaw[i]), math.cos(ang_ij - world.yaw[i])))
        if (
            nearest_rival_dist[i] < 1.15
            and bear_ij < 0.85
            and world.threat_intensity[i] > 0.28
            and world.lunge_timer_ms[i] == 0
            and world.retreat_timer_ms[i] == 0
        ):
            world.lunge_timer_ms[i] = 140
            world.total_lunges += 1
            world.dominance_score[i] += np.float32(1.0)
            world.dominance_score[j] -= np.float32(1.0)
            d = max(1e-4, float(nearest_rival_dist[i]))
            ux, uy = float(dx[i, j]) / d, float(dy[i, j]) / d
            world.x[j] += ux * 0.28
            world.y[j] += uy * 0.28
            world.retreat_timer_ms[j] = 300
            clashes.append(
                [
                    round(float(world.x[i]), 3),
                    round(float(world.y[i]), 3),
                    round(float(world.x[j]), 3),
                    round(float(world.y[j]), 3),
                ]
            )
        elif world.lunge_timer_ms[i] > 60 and nearest_rival_dist[i] < 1.5:
            clashes.append(
                [
                    round(float(world.x[i]), 3),
                    round(float(world.y[i]), 3),
                    round(float(world.x[j]), 3),
                    round(float(world.y[j]), 3),
                ]
            )
    world.active_clashes = clashes[:32]

    # 6. Planar locomotion kinematics from connectome DNp09 + DNa01/02
    all_fwd = np.concatenate([mb.rates[:, 0], fb.rates[:, 0]])
    all_tl = np.concatenate([mb.rates[:, 1], fb.rates[:, 1]])
    all_tr = np.concatenate([mb.rates[:, 2], fb.rates[:, 2]])
    fwd_cmd = np.clip(all_fwd / 45.0, 0.0, 1.0)
    lunge_boost = np.where(world.lunge_timer_ms > 0, 1.75, 0.92)
    feed_brake = np.where(world.feeding_intensity > 0.35, 0.12, 1.0)
    eff_fwd = np.clip(fwd_cmd * lunge_boost * feed_brake, 0.0, 1.85)
    turn_cmd = np.clip((all_tl - all_tr) / 35.0, -1.0, 1.0)

    noise_yaw = rng.normal(0.0, 0.35, size=n_all) * np.where(eff_fwd > 0.05, 1.0, 0.0)
    world.yaw = (world.yaw + (2.8 * turn_cmd + noise_yaw) * 1e-3 + math.pi) % (
        2.0 * math.pi
    ) - math.pi
    world.speed = 2.1 * eff_fwd
    world.x += world.speed * np.cos(world.yaw) * 1e-3
    world.y += world.speed * np.sin(world.yaw) * 1e-3

    lim = world.arena_half_size - 0.25
    hit_x = np.abs(world.x) > lim
    hit_y = np.abs(world.y) > lim
    world.x = np.clip(world.x, -lim, lim)
    world.y = np.clip(world.y, -lim, lim)
    world.yaw = np.where(hit_x, math.pi - world.yaw, world.yaw)
    world.yaw = np.where(hit_y, -world.yaw, world.yaw)

    # 7. Scientific metrics update
    world.tick += 1
    in_clean = int(np.sum(world.x > ECOSYSTEM_ODOR_X_MAX))
    world.last_clean_fraction = in_clean / n_all
    world.clean_zone_samples += in_clean
    world.total_fly_samples += n_all
    world.last_mean_speed = float(world.speed.mean())
    world.last_active_singers = int(np.sum(world.song_intensity > 0.15))
    world.last_active_threats = int(np.sum(world.threat_intensity > 0.20))
    world.last_active_feeders = int(np.sum(world.feeding_intensity > 0.20))
    world.last_female_auditory_hz = float(fb.extra_rates[:, 0].mean())
    world.total_wing_threat_ms += world.last_active_threats
    world.total_feeding_ms += world.last_active_feeders
    world.total_song_ms += world.last_active_singers
    dt_ms = (time.perf_counter() - t0) * 1e3
    world.step_ms_ema = dt_ms if world.tick == 1 else (0.92 * world.step_ms_ema + 0.08 * dt_ms)


def run_dual_sex_ecosystem_assay(
    male_pack: ConnectomePack,
    female_pack: ConnectomePack,
    *,
    num_flies: int = 16,
    ticks: int = 1500,
    seed: int = 1,
    backend: str = "auto",
) -> dict[str, Any]:
    """Verify all 4 F11 behaviors concurrently on real MaleCNS + BANC vs shuffled wiring."""
    t0 = time.perf_counter()

    def _run(wiring: str) -> dict[str, Any]:
        w = make_dual_sex_ecosystem_world(
            male_pack,
            female_pack,
            num_flies=num_flies,
            wiring=wiring,
            seed=seed,
            backend=backend,
        )
        aud_sum = 0.0
        da1_sum = 0.0
        hg34_sum = 0.0
        ps1_sum = 0.0
        for _ in range(ticks):
            step_dual_sex_ecosystem_tick(w)
            aud_sum += w.last_female_auditory_hz
            da1_sum += float(w.male_batch.extra_rates[:, 0].mean())
            hg34_sum += float(w.male_batch.extra_rates[:, 3].mean())
            ps1_sum += float(w.male_batch.extra_rates[:, 4].mean())
        return {
            "wiring": wiring,
            "num_flies": num_flies,
            "n_males": w.n_males,
            "n_females": w.n_females,
            "total_neurons": w.total_neurons,
            "feeding_fly_ms": int(w.total_feeding_ms),
            "song_fly_ms": int(w.total_song_ms),
            "total_lunges": int(w.total_lunges),
            "wing_threat_fly_ms": int(w.total_wing_threat_ms),
            "mean_clean_fraction": round(w.clean_zone_samples / max(1, w.total_fly_samples), 3),
            "mean_da1_pn_hz": round(da1_sum / max(1, ticks), 2),
            "mean_hg34_hz": round(hg34_sum / max(1, ticks), 2),
            "mean_male_ps1_hz": round(ps1_sum / max(1, ticks), 2),
            "mean_female_aud_hz": round(aud_sum / max(1, ticks), 2),
            "realtime_factor": round(w.realtime_factor, 2),
        }

    intact = _run("reference")
    shuffled = _run("shuffled_control")
    checks = {
        "foraging_feeding_occurs": bool(intact["feeding_fly_ms"] >= 50),
        "male_aggression_occurs": bool(
            intact["total_lunges"] >= 2 and intact["wing_threat_fly_ms"] >= 100
        ),
        "male_to_female_song_dialogue_occurs": bool(
            intact["song_fly_ms"] >= 100 and intact["mean_female_aud_hz"] >= 5.0
        ),
        "collective_odor_avoidance_occurs": bool(intact["mean_clean_fraction"] >= 0.70),
        "shuffled_collapses_social_circuits": bool(
            shuffled["total_lunges"] < intact["total_lunges"] * 0.25
            and shuffled["mean_female_aud_hz"] < intact["mean_female_aud_hz"] * 0.25
        ),
    }
    return {
        "mode": "dual_sex_ecosystem",
        "male_dataset": male_pack.dataset_id,
        "female_dataset": female_pack.dataset_id,
        "total_neurons": intact["total_neurons"],
        "num_flies": num_flies,
        "ticks": ticks,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "conditions": {"intact_ecosystem": intact, "shuffled_control": shuffled},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
