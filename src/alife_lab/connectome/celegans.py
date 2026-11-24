"""Stage 2 / S1 C. elegans 302-neuron species pack & pure-CSV loader (ADR 0051).

Supports:
1. ``synthetic_celegans_302`` — built-in offline 302-neuron hermaphrodite pack using exact
   canonical somatic neuron names, ``MODEL_GRADED`` dynamics, chemical synapses, and symmetric
   bidirectional ohmic gap junctions (``SYNAPSE_GAP_JUNCTION``).
2. ``celegans_302`` — pure tabular CSV parser (ZERO pickles) for real OpenWorm / WormWiring
   edgelists stored under ``.alife/datasets/celegans_302/raw/`` after explicit user approval.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from alife_lab.connectome.registry import (
    CELEGANS_302_SPEC,
    DatasetError,
    raw_dir,
    verify_dataset,
)
from alife_lab.connectome.schema import (
    MODEL_GRADED,
    POS_SOMA,
    ROLE_INTERNEURON,
    ROLE_MOTOR,
    ROLE_SENSORY,
    SYNAPSE_CHEMICAL,
    SYNAPSE_GAP_JUNCTION,
    CellTable,
    ConnectomePack,
    DatasetProvenance,
    EdgeTable,
    IOPort,
    compute_pack_digest,
)

_BILATERAL_SENSORY_STEMS: tuple[str, ...] = (
    "ADE",
    "ADF",
    "ADL",
    "AFD",
    "ALM",
    "ALN",
    "ASE",
    "ASG",
    "ASH",
    "ASI",
    "ASJ",
    "ASK",
    "AWA",
    "AWB",
    "AWC",
    "BAG",
    "FLP",
    "OLL",
    "PDE",
    "PHA",
    "PHB",
    "PHC",
    "PLM",
    "PLN",
    "PVD",
    "SDQ",
    "URB",
    "URX",
)

_BILATERAL_INTERNEURON_STEMS: tuple[str, ...] = (
    "ADA",
    "AIA",
    "AIB",
    "AIM",
    "AIN",
    "AIY",
    "AIZ",
    "AUA",
    "AVA",
    "AVB",
    "AVD",
    "AVE",
    "AVF",
    "AVH",
    "AVJ",
    "AVK",
    "BDU",
    "CAN",
    "HSN",
    "LUA",
    "PVC",
    "PVN",
    "PVP",
    "PVQ",
    "PVW",
    "RIA",
    "RIB",
    "RIC",
    "RIF",
    "RIG",
    "RIM",
    "RIP",
    "RIV",
    "RMF",
    "RMG",
    "RMH",
)

_SENSORY_EXPLICIT: tuple[str, ...] = (
    "AVM",
    "PVM",
    "AQR",
    "PQR",
    "CEPDL",
    "CEPDR",
    "CEPVL",
    "CEPVR",
    "IL1DL",
    "IL1DR",
    "IL1L",
    "IL1R",
    "IL1VL",
    "IL1VR",
    "IL2DL",
    "IL2DR",
    "IL2L",
    "IL2R",
    "IL2VL",
    "IL2VR",
    "OLQDL",
    "OLQDR",
    "OLQVL",
    "OLQVR",
    "URYDL",
    "URYDR",
    "URYVL",
    "URYVR",
)

_INTERNEURON_EXPLICIT: tuple[str, ...] = (
    "ALA",
    "AVG",
    "AVL",
    "DVA",
    "DVB",
    "DVC",
    "PVR",
    "PVT",
    "RID",
    "RIH",
    "RIR",
    "RIS",
    "SAADL",
    "SAADR",
    "SAAVL",
    "SAAVR",
)

_HEAD_MOTOR_EXPLICIT: tuple[str, ...] = (
    "PDA",
    "PDB",
    "RMDDL",
    "RMDDR",
    "RMDL",
    "RMDR",
    "RMDVL",
    "RMDVR",
    "RMED",
    "RMEL",
    "RMER",
    "RMEV",
    "SABD",
    "SABVL",
    "SABVR",
    "SIADL",
    "SIADR",
    "SIAVL",
    "SIAVR",
    "SIBDL",
    "SIBDR",
    "SIBVL",
    "SIBVR",
    "SMBDL",
    "SMBDR",
    "SMBVL",
    "SMBVR",
    "SMDDL",
    "SMDDR",
    "SMDVL",
    "SMDVR",
    "URADL",
    "URADR",
    "URAVL",
    "URAVR",
)

_PHARYNGEAL_EXPLICIT: tuple[str, ...] = (
    "I1L",
    "I1R",
    "I2L",
    "I2R",
    "I3",
    "I4",
    "I5",
    "I6",
    "M1",
    "M2L",
    "M2R",
    "M3L",
    "M3R",
    "M4",
    "M5",
    "MCL",
    "MCR",
    "MI",
    "NSML",
    "NSMR",
)


def canonical_celegans_302_names() -> tuple[tuple[str, int], ...]:
    """Return canonical 302 somatic neuron names paired with CELL_ROLE constants."""
    rows: list[tuple[str, int]] = []
    for s in _BILATERAL_SENSORY_STEMS:
        rows.append((f"{s}L", ROLE_SENSORY))
        rows.append((f"{s}R", ROLE_SENSORY))
    for name in _SENSORY_EXPLICIT:
        rows.append((name, ROLE_SENSORY))
    for s in _BILATERAL_INTERNEURON_STEMS:
        role = ROLE_MOTOR if s in ("RIM", "RIV", "RMF", "RMH", "HSN") else ROLE_INTERNEURON
        rows.append((f"{s}L", role))
        rows.append((f"{s}R", role))
    for name in _INTERNEURON_EXPLICIT:
        rows.append((name, ROLE_INTERNEURON))
    for name in _HEAD_MOTOR_EXPLICIT:
        rows.append((name, ROLE_MOTOR))
    for cls_name, n_cls in (
        ("DA", 9),
        ("DB", 7),
        ("DD", 6),
        ("VA", 12),
        ("VB", 11),
        ("VC", 6),
        ("VD", 13),
        ("AS", 11),
    ):
        for k in range(1, n_cls + 1):
            rows.append((f"{cls_name}{k}", ROLE_MOTOR))
    for name in _PHARYNGEAL_EXPLICIT:
        role = (
            ROLE_MOTOR
            if name.startswith(("M1", "M2", "M3", "M4", "M5", "MC", "MI"))
            else ROLE_INTERNEURON
        )
        rows.append((name, role))

    if len(rows) != 302:
        raise AssertionError(f"canonical C. elegans list must contain 302 cells, got {len(rows)}")
    if len({n for n, _ in rows}) != 302:
        raise AssertionError("duplicate cell name in canonical C. elegans 302 list")
    return tuple(rows)


def _cell_class_of(name: str) -> str:
    # Numbered VNC motors: DB1 -> DB, VA12 -> VA; pharyngeal M2L -> M2; bilateral AVAL -> AVA.
    if name in ("RMEL", "RMER", "RMED", "RMEV"):
        return "RME"
    if len(name) >= 2 and name[-1] in ("L", "R"):
        base = name[:-1]
        if (
            base.endswith(("D", "V"))
            and len(base) >= 4
            and base[:3]
            in (
                "CEP",
                "IL1",
                "IL2",
                "OLQ",
                "RMD",
                "SAA",
                "SIA",
                "SIB",
                "SMB",
                "SMD",
                "URA",
                "URY",
            )
        ):
            return base[:-1]
        return base
    for prefix in ("DA", "DB", "DD", "VA", "VB", "VC", "VD", "AS"):
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            return prefix
    if name.startswith("SAB"):
        return "SAB"
    return name


def _side_of(name: str) -> str:
    if name.endswith("L"):
        return "left"
    if name.endswith("R"):
        return "right"
    return "midline"


def _neurotransmitter_of(name: str, cls_name: str, role: int) -> str:
    if cls_name in ("DD", "VD", "RME", "AVL", "DVB", "RIS"):
        return "gaba"
    if cls_name in (
        "ALM",
        "PLM",
        "AVM",
        "PVM",
        "ASH",
        "ADL",
        "ASE",
        "AFD",
        "AWC",
        "AWA",
        "AWB",
        "FLP",
        "OLQ",
        "PHA",
        "PHB",
        "PHC",
        "AVA",
        "AVD",
        "AVE",
        "RIM",
        "AIB",
        "AIA",
        "AIY",
        "AIZ",
        "RIA",
    ):
        return "glutamate" if role != ROLE_MOTOR or cls_name in ("RIA",) else "acetylcholine"
    if cls_name in ("ADE", "CEP", "PDE"):
        return "dopamine"
    if cls_name in ("NSM", "HSN", "ADF"):
        return "serotonin"
    if cls_name in ("RIC",):
        return "octopamine"
    if role == ROLE_MOTOR or cls_name in (
        "AVB",
        "PVC",
        "DA",
        "DB",
        "VA",
        "VB",
        "AS",
        "VC",
        "SMB",
        "SMD",
        "RMD",
    ):
        return "acetylcholine"
    if name.startswith(("DD", "VD")):
        return "gaba"
    return "acetylcholine"


def _worm_coords_nm(name: str, cls_name: str, idx: int) -> tuple[str, tuple[float, float, float]]:
    """Map canonical cell onto anterior-posterior C. elegans body frame in nm."""
    # Worm body coordinate convention: x in [0, 1_000_000] nm (head->tail), y = L/R, z = D/V.
    side = _side_of(name)
    y = -18_000.0 if side == "left" else (18_000.0 if side == "right" else 0.0)
    z = 8_000.0 if cls_name in ("DA", "DB", "DD", "AS") or "D" in name[-2:] else -8_000.0

    for prefix, span in (
        ("DA", 9),
        ("DB", 7),
        ("DD", 6),
        ("VA", 12),
        ("VB", 11),
        ("VC", 6),
        ("VD", 13),
        ("AS", 11),
    ):
        if cls_name == prefix and name[len(prefix) :].isdigit():
            k = int(name[len(prefix) :])
            frac = k / float(span + 1)
            return "ventral_nerve_cord", (200_000.0 + frac * 680_000.0, y, z)

    if name in _PHARYNGEAL_EXPLICIT:
        return "pharynx", (45_000.0 + (idx % 10) * 9_000.0, y * 0.7, z * 0.7)

    if cls_name in (
        "PLM",
        "PVM",
        "PVC",
        "PHA",
        "PHB",
        "PHC",
        "PLN",
        "ALN",
        "PQR",
        "PVR",
        "PVT",
        "PVW",
        "PVN",
        "PVP",
        "PVQ",
        "LUA",
        "DVA",
        "DVB",
        "DVC",
        "PDA",
        "PDB",
    ):
        return "tail_ganglia", (880_000.0 + (idx % 9) * 11_000.0, y, z)

    if cls_name in ("ALM", "AVM", "PVD", "SDQ", "HSN", "CAN", "BDU"):
        return "ventral_nerve_cord", (360_000.0 + (idx % 7) * 45_000.0, y, z)

    # Default: head nerve ring & amphid ganglia (0..180 um)
    return "head_ganglia", (70_000.0 + (idx % 12) * 9_000.0, y, z)


def _build_cell_table_and_ports(
    names_roles: tuple[tuple[str, int], ...], dataset_id: str
) -> tuple[CellTable, dict[str, int], dict[str, IOPort]]:
    n = len(names_roles)
    names = [n_ for n_, _ in names_roles]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    classes = tuple(_cell_class_of(nm) for nm in names)
    sides = tuple(_side_of(nm) for nm in names)
    roles = np.asarray([r for _, r in names_roles], dtype=np.uint8)
    regions_list: list[str] = []
    coords = np.zeros((n, 3), dtype=np.float32)
    nts: list[str] = []
    for i, (nm, r) in enumerate(names_roles):
        cls_nm = classes[i]
        reg, xyz = _worm_coords_nm(nm, cls_nm, i)
        regions_list.append(reg)
        coords[i] = np.asarray(xyz, dtype=np.float32)
        nts.append(_neurotransmitter_of(nm, cls_nm, r))

    cells = CellTable(
        source_id=np.arange(1, n + 1, dtype=np.int64),
        species_names=(dataset_id,),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(names),
        super_class=tuple(
            "sensory" if r == ROLE_SENSORY else ("motor" if r == ROLE_MOTOR else "central")
            for r in roles
        ),
        cell_class=classes,
        cell_sub_class=tuple("" for _ in range(n)),
        region=tuple(regions_list),
        side=sides,
        role=roles,
        cell_model=np.full(n, MODEL_GRADED, dtype=np.uint8),
        neurotransmitter=tuple(nts),
        position_nm=coords,
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )

    def _indices(prefixes: tuple[str, ...]) -> np.ndarray[Any, np.dtype[np.int32]]:
        matched = [
            i
            for i, (nm, cls_nm) in enumerate(zip(names, classes, strict=True))
            if nm in prefixes or cls_nm in prefixes
        ]
        return np.asarray(matched, dtype=np.int32)

    port_defs = (
        (
            "anterior_touch",
            "sensory",
            "Anterior Touch (ALM/AVM)",
            "Mechanosensors for nose/head touch reversal.",
            ("ALM", "AVM"),
        ),
        (
            "posterior_touch",
            "sensory",
            "Posterior Touch (PLM/PVM)",
            "Mechanosensors driving forward acceleration.",
            ("PLM", "PVM"),
        ),
        (
            "chemosensory_attract",
            "sensory",
            "Chemosensory Attract (ASE/AWC/AWA)",
            "Amphid odor/salt attraction receptors.",
            ("ASE", "AWC", "AWA"),
        ),
        (
            "nociceptive_avoid",
            "sensory",
            "Nociceptive Avoid (ASH/ADL)",
            "Avid avoidance polymodal sensors.",
            ("ASH", "ADL"),
        ),
        (
            "forward_command",
            "circuit",
            "Forward Command (AVB/PVC)",
            "Command interneurons gating forward B-type motor wave.",
            ("AVB", "PVC"),
        ),
        (
            "reverse_command",
            "circuit",
            "Reverse Command (AVA/AVD/AVE)",
            "Command interneurons gating reverse A-type motor wave.",
            ("AVA", "AVD", "AVE"),
        ),
        (
            "navigation_interneurons",
            "circuit",
            "Navigation Interneurons (AIY/AIA/AIB/AIZ/RIA)",
            "Sensorimotor klinotaxis & reversal-rate interneurons.",
            ("AIY", "AIA", "AIB", "AIZ", "RIA"),
        ),
        (
            "dorsal_forward_motor",
            "motor",
            "Dorsal Forward Motor (DB1..7)",
            "Excitatory cholinergic dorsal B motor neurons.",
            ("DB",),
        ),
        (
            "ventral_forward_motor",
            "motor",
            "Ventral Forward Motor (VB1..11)",
            "Excitatory cholinergic ventral B motor neurons.",
            ("VB",),
        ),
        (
            "dorsal_reverse_motor",
            "motor",
            "Dorsal Reverse Motor (DA1..9)",
            "Excitatory cholinergic dorsal A motor neurons.",
            ("DA",),
        ),
        (
            "ventral_reverse_motor",
            "motor",
            "Ventral Reverse Motor (VA1..12)",
            "Excitatory cholinergic ventral A motor neurons.",
            ("VA",),
        ),
        (
            "dorsal_inhibitory_motor",
            "motor",
            "Dorsal GABAergic Motor (DD1..6)",
            "GABAergic inhibitory cross-coupling to dorsal muscles/MNs.",
            ("DD",),
        ),
        (
            "ventral_inhibitory_motor",
            "motor",
            "Ventral GABAergic Motor (VD1..13)",
            "GABAergic inhibitory cross-coupling to ventral muscles/MNs.",
            ("VD",),
        ),
        (
            "head_steering_motor",
            "motor",
            "Head Steering Motor (SMD/SMB/RMD)",
            "Head/neck klinotaxis steering motor neurons.",
            ("SMD", "SMB", "RMD"),
        ),
    )
    io_ports = {
        pid: IOPort(
            port_id=pid,
            direction=direction,  # type: ignore[arg-type]
            label=label,
            description=desc,
            cell_indices=_indices(stems),
            selection_rule=f"canonical classes {','.join(stems)}",
        )
        for pid, direction, label, desc, stems in port_defs
    }
    return cells, name_to_idx, io_ports


def _aggregate_edges(
    triplets: list[tuple[int, int, int, int, int]],
) -> EdgeTable:
    """Merge duplicate (pre, post, sign, kind) edges into canonical EdgeTable."""
    merged: dict[tuple[int, int, int, int], int] = defaultdict(int)
    for pre, post, cnt, sign, kind in triplets:
        if pre == post or cnt <= 0:
            continue
        merged[(int(pre), int(post), int(sign), int(kind))] += int(cnt)
    if not merged:
        return EdgeTable(
            pre=np.zeros(0, dtype=np.int32),
            post=np.zeros(0, dtype=np.int32),
            count=np.zeros(0, dtype=np.int32),
            sign=np.zeros(0, dtype=np.int8),
            kind=np.zeros(0, dtype=np.uint8),
        )
    keys = sorted(merged.keys())
    return EdgeTable(
        pre=np.asarray([k[0] for k in keys], dtype=np.int32),
        post=np.asarray([k[1] for k in keys], dtype=np.int32),
        count=np.asarray([merged[k] for k in keys], dtype=np.int32),
        sign=np.asarray([k[2] for k in keys], dtype=np.int8),
        kind=np.asarray([k[3] for k in keys], dtype=np.uint8),
    )


def make_synthetic_celegans_302_pack() -> ConnectomePack:
    """Offline 302-neuron C. elegans hermaphrodite species pack with chemical + gap wiring."""
    names_roles = canonical_celegans_302_names()
    cells, idx, io_ports = _build_cell_table_and_ports(names_roles, "synthetic_celegans_302")
    edges_raw: list[tuple[int, int, int, int, int]] = []

    def add_chem(u: str, v: str, count: int, sign: int = 1) -> None:
        edges_raw.append((idx[u], idx[v], count, sign, SYNAPSE_CHEMICAL))

    def add_gap(u: str, v: str, count: int) -> None:
        iu, iv = idx[u], idx[v]
        edges_raw.append((iu, iv, count, 1, SYNAPSE_GAP_JUNCTION))
        edges_raw.append((iv, iu, count, 1, SYNAPSE_GAP_JUNCTION))

    # 1. Bilateral homologous gap junctions (left <-> right) across major amphid & command pairs
    for s in (
        "AVA",
        "AVB",
        "AVD",
        "AVE",
        "PVC",
        "ASE",
        "AWC",
        "ASH",
        "AIY",
        "AIB",
        "RIA",
        "RIM",
        "ALM",
        "PLM",
        "AFD",
    ):
        add_gap(f"{s}L", f"{s}R", 4)

    # 2. Posterior touch & forward command -> AVB <-> DB/VB gap junctions (anatomy-matched)
    for plm in ("PLML", "PLMR", "PVM"):
        for pvc in ("PVCL", "PVCR"):
            add_chem(plm, pvc, 10, 1)
            add_gap(plm, pvc, 3)
        for avb in ("AVBL", "AVBR"):
            add_chem(plm, avb, 12, 1)
        for ava in ("AVAL", "AVAR"):
            add_chem(plm, ava, 8, -1)
    for pvc in ("PVCL", "PVCR"):
        for avb in ("AVBL", "AVBR"):
            add_chem(pvc, avb, 18, 1)
            add_gap(pvc, avb, 4)

    # 3. Anterior touch & nociception -> AVD/AVE/AVA reverse circuit + AVB suppression
    for ant in ("ALML", "ALMR", "AVM", "ASHL", "ASHR", "ADLL", "ADLR", "FLPL", "FLPR"):
        for avd in ("AVDL", "AVDR", "AVEL", "AVER"):
            add_chem(ant, avd, 12, 1)
        for ava in ("AVAL", "AVAR"):
            add_chem(ant, ava, 14, 1)
            add_gap(ant, ava, 2)
        for avb in ("AVBL", "AVBR", "PVCL", "PVCR"):
            add_chem(ant, avb, 16, -1)
    for avd in ("AVDL", "AVDR", "AVEL", "AVER"):
        for ava in ("AVAL", "AVAR"):
            add_chem(avd, ava, 10, 1)
            add_gap(avd, ava, 3)
        for avb in ("AVBL", "AVBR"):
            add_chem(avd, avb, 8, -1)

    # Reciprocal command interneuron inhibition: AVA <-|-> AVB
    for ava in ("AVAL", "AVAR"):
        for avb in ("AVBL", "AVBR"):
            add_chem(ava, avb, 14, -1)
            add_chem(avb, ava, 12, -1)

    # 4. Command-to-VNC gap junctions (anatomical backbone!) + reciprocal GABA cross-inhibition
    dbs = [f"DB{k}" for k in range(1, 8)]
    vbs = [f"VB{k}" for k in range(1, 12)]
    das = [f"DA{k}" for k in range(1, 10)]
    vas = [f"VA{k}" for k in range(1, 13)]
    dds = [f"DD{k}" for k in range(1, 7)]
    vds = [f"VD{k}" for k in range(1, 14)]

    # AVB couples to DB/VB purely via gap junctions (no chemical bypass), suppressing DA/VA
    for avb in ("AVBL", "AVBR"):
        for db in dbs:
            add_gap(avb, db, 4)
        for vb in vbs:
            add_gap(avb, vb, 4)
        for da in das:
            add_chem(avb, da, 8, -1)
        for va in vas:
            add_chem(avb, va, 8, -1)

    # AVA couples strongly to DA/VA via gap junctions (+ minor chemical), while suppressing DB/VB
    for ava in ("AVAL", "AVAR"):
        for da in das:
            add_gap(ava, da, 4)
            add_chem(ava, da, 2, 1)
        for va in vas:
            add_gap(ava, va, 4)
            add_chem(ava, va, 2, 1)
        for db in dbs:
            add_chem(ava, db, 8, -1)
        for vb in vbs:
            add_chem(ava, vb, 8, -1)

    # Longitudinal VNC electrical gap-junction chains + DV contralateral GABAergic inhibition
    for k in range(len(dbs) - 1):
        add_gap(dbs[k], dbs[k + 1], 2)
    for k in range(len(vbs) - 1):
        add_gap(vbs[k], vbs[k + 1], 2)
    for k in range(len(das) - 1):
        add_gap(das[k], das[k + 1], 2)
    for k in range(len(vas) - 1):
        add_gap(vas[k], vas[k + 1], 2)

    for k, db in enumerate(dbs):
        vd = vds[min(k * 2, len(vds) - 1)]
        vb = vbs[min(k + 1, len(vbs) - 1)]
        dd = dds[min(k, len(dds) - 1)]
        add_chem(db, vd, 3, 1)
        add_chem(vd, vb, 3, -1)
        add_chem(vb, dd, 3, 1)
        add_chem(dd, db, 3, -1)

    for k, da in enumerate(das):
        vd = vds[min(k, len(vds) - 1)]
        va = vas[min(k, len(vas) - 1)]
        dd = dds[min(k, len(dds) - 1)]
        add_chem(da, vd, 3, 1)
        add_chem(vd, va, 3, -1)
        add_chem(va, dd, 3, 1)
        add_chem(dd, da, 3, -1)

    # 5. Chemosensory klinotaxis & steering circuit: ASE/AWC -> AIY/AIA/AIB -> RIA -> SMD/SMB/RMD
    for lr, side_s in (("L", "L"), ("R", "R")):
        opp = "R" if lr == "L" else "L"
        add_chem(f"ASE{lr}", f"AIY{side_s}", 10, 1)
        add_chem(f"ASE{lr}", f"AIB{side_s}", 8, -1)
        add_chem(f"AWC{lr}", f"AIY{side_s}", 10, 1)
        add_chem(f"AWC{lr}", f"AIB{side_s}", 8, 1)
        add_chem(f"AWA{lr}", f"AIA{side_s}", 8, 1)
        add_chem(f"AIA{side_s}", f"AIB{side_s}", 8, -1)
        add_chem(f"AIY{side_s}", f"RIA{side_s}", 10, 1)
        add_chem(f"AIZ{side_s}", f"RIA{opp}", 8, 1)
        add_chem(f"AIB{side_s}", f"RIM{side_s}", 8, 1)
        add_chem(f"RIM{side_s}", f"AVA{side_s}", 5, 1)
        add_chem(f"RIA{side_s}", f"SMDD{side_s}", 10, 1)
        add_chem(f"RIA{side_s}", f"SMDV{side_s}", 10, 1)
        add_chem(f"RIA{side_s}", f"SMBD{side_s}", 8, 1)
        add_chem(f"RIA{side_s}", f"SMBV{side_s}", 8, 1)
        add_chem(f"RIA{side_s}", f"RMDD{opp}", 6, -1)
        add_chem(f"RIA{side_s}", f"RMDV{opp}", 6, -1)
        add_gap(f"SMDD{lr}", f"SMBD{lr}", 5)
        add_gap(f"SMDV{lr}", f"SMBV{lr}", 5)

    # 6. Pharyngeal autonomous recurrent loop + AS/VC local couplings so all 302 cells are connected
    rng = np.random.default_rng(np.random.SeedSequence(0x302CE1))
    all_names = [nm for nm, _ in names_roles]
    core_protected = {
        "AVAL",
        "AVAR",
        "AVBL",
        "AVBR",
        "AVDL",
        "AVDR",
        "AVEL",
        "AVER",
        "PVCL",
        "PVCR",
        *dbs,
        *vbs,
        *das,
        *vas,
        *dds,
        *vds,
    }
    for i in range(len(dbs)):
        add_chem(f"AS{i + 1}", dbs[i], 2, 1)
    for k in range(len(_PHARYNGEAL_EXPLICIT) - 1):
        add_chem(_PHARYNGEAL_EXPLICIT[k], _PHARYNGEAL_EXPLICIT[k + 1], 4, 1)
        if k % 2 == 0:
            add_gap(_PHARYNGEAL_EXPLICIT[k], _PHARYNGEAL_EXPLICIT[k + 1], 3)
    # Deterministic sparse anatomical recurrent background excluding protected command/VNC pools
    for idx_u in range(0, 302, 3):
        idx_v = int(rng.integers(0, 160))
        if idx_u != idx_v and all_names[idx_v] not in core_protected:
            edges_raw.append((idx_u, idx_v, 2, 1, SYNAPSE_CHEMICAL))
        if idx_u % 6 == 0:
            idx_g = (idx_u + 1) % 302
            if all_names[idx_u] not in core_protected and all_names[idx_g] not in core_protected:
                edges_raw.append((idx_u, idx_g, 2, 1, SYNAPSE_GAP_JUNCTION))
                edges_raw.append((idx_g, idx_u, 2, 1, SYNAPSE_GAP_JUNCTION))

    edges = _aggregate_edges(edges_raw)
    edges.validate(cells.num_cells)
    prov = DatasetProvenance(
        dataset_id="synthetic_celegans_302",
        species="caenorhabditis_elegans_hermaphrodite",
        version="synthetic_s1_302_v1",
        citation=(
            "Canonical 302-neuron C. elegans wiring pattern (White 1986; Cook 2019; offline S1)."
        ),
        licence_note="Built-in offline synthetic wiring pack; not empirical EM synapse counts.",
        files=(),
    )
    digest = compute_pack_digest(
        "synthetic_celegans_302",
        "caenorhabditis_elegans_hermaphrodite",
        cells,
        edges,
        io_ports,
        prov,
    )
    return ConnectomePack(
        dataset_id="synthetic_celegans_302",
        species="caenorhabditis_elegans_hermaphrodite",
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
        pack_digest=digest,
    )


_VNC_ZERO_PAD_RE = re.compile(r"^(DA|DB|DD|VA|VB|VC|VD|AS)0([1-9])$")


def normalize_celegans_name(raw_name: str) -> str:
    """Normalize C. elegans neuron names (strip whitespace + VA01 -> VA1 zero-pad)."""
    clean = str(raw_name).strip()
    match = _VNC_ZERO_PAD_RE.match(clean)
    return f"{match.group(1)}{match.group(2)}" if match else clean


_normalize_celegans_name = normalize_celegans_name


def load_celegans_from_edgelist_csv(
    edgelist_path: Path,
    *,
    dataset_id: str = "celegans_302",
    provenance: DatasetProvenance | None = None,
) -> ConnectomePack:
    """Parse a pure CSV hermaphrodite C. elegans edgelist (zero pickles) into ConnectomePack."""
    if not edgelist_path.is_file():
        raise DatasetError(f"missing C. elegans edgelist CSV: {edgelist_path}")
    names_roles = canonical_celegans_302_names()
    cells, idx, io_ports = _build_cell_table_and_ports(names_roles, dataset_id)
    chem_merged: dict[tuple[int, int, int], int] = defaultdict(int)
    gap_pairs: dict[tuple[int, int], list[int]] = defaultdict(list)

    with edgelist_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean = {
                str(k).strip().lower(): ("" if v is None else str(v).strip())
                for k, v in row.items()
                if k is not None
            }
            src = normalize_celegans_name(clean.get("source") or clean.get("pre") or "")
            dst = normalize_celegans_name(clean.get("target") or clean.get("post") or "")
            if src not in idx or dst not in idx or src == dst:
                continue
            w_raw = clean.get("weight") or clean.get("count") or "1"
            try:
                w = max(1, round(float(w_raw)))
            except ValueError:
                continue
            kind_str = (clean.get("type") or clean.get("kind") or "chemical").lower()
            iu, iv = idx[src], idx[dst]
            if any(tok in kind_str for tok in ("ej", "gap", "elec")):
                lo, hi = (iu, iv) if iu < iv else (iv, iu)
                gap_pairs[(lo, hi)].append(w)
            else:
                sign = -1 if cells.neurotransmitter[iu] in ("gaba",) else 1
                chem_merged[(iu, iv, sign)] += w

    edges_raw: list[tuple[int, int, int, int, int]] = [
        (pre, post, cnt, sign, SYNAPSE_CHEMICAL) for (pre, post, sign), cnt in chem_merged.items()
    ]
    for (lo, hi), ws in gap_pairs.items():
        # Deduplicate bidirectional CSV gap-junction entries without doubling electrical weight.
        w_sym = max(ws)
        edges_raw.append((lo, hi, w_sym, 1, SYNAPSE_GAP_JUNCTION))
        edges_raw.append((hi, lo, w_sym, 1, SYNAPSE_GAP_JUNCTION))

    edges = _aggregate_edges(edges_raw)
    edges.validate(cells.num_cells)
    prov = provenance or CELEGANS_302_SPEC.to_provenance()
    digest = compute_pack_digest(
        dataset_id,
        "caenorhabditis_elegans_hermaphrodite",
        cells,
        edges,
        io_ports,
        prov,
    )
    return ConnectomePack(
        dataset_id=dataset_id,
        species="caenorhabditis_elegans_hermaphrodite",
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
        pack_digest=digest,
    )


def load_celegans_302(datasets_dir: Path, *, verify_hashes: bool = True) -> ConnectomePack:
    """Load real C. elegans CSV edgelist from ``.alife/datasets/celegans_302/raw/``."""
    raw = raw_dir(datasets_dir, CELEGANS_302_SPEC.dataset_id)
    csv_path = raw / CELEGANS_302_SPEC.files[0].filename
    if not csv_path.is_file():
        raise DatasetError(
            f"dataset 'celegans_302' is not installed at {raw}; "
            "download requires explicit user approval"
        )
    if verify_hashes and CELEGANS_302_SPEC.files[0].sha256:
        report = verify_dataset(CELEGANS_302_SPEC.dataset_id, datasets_dir, spec=CELEGANS_302_SPEC)
        if not report["ok"]:
            raise DatasetError(f"dataset 'celegans_302' verification failed: {report}")
    return load_celegans_from_edgelist_csv(
        csv_path,
        dataset_id="celegans_302",
        provenance=CELEGANS_302_SPEC.to_provenance(),
    )


def celegans_302_reference_checks(pack: ConnectomePack) -> list[dict[str, Any]]:
    """Verify 302-neuron C. elegans structural & electrical-synapse reference properties."""
    n = pack.cells.num_cells
    n_graded = int((pack.cells.cell_model == MODEL_GRADED).sum())
    n_gap = int((pack.edges.kind == SYNAPSE_GAP_JUNCTION).sum())
    n_chem = int((pack.edges.kind == SYNAPSE_CHEMICAL).sum())
    fwd_cnt = (
        int(pack.io_ports["forward_command"].cell_indices.size)
        if "forward_command" in pack.io_ports
        else 0
    )
    rev_cnt = (
        int(pack.io_ports["reverse_command"].cell_indices.size)
        if "reverse_command" in pack.io_ports
        else 0
    )
    return [
        {
            "check": "celegans_somatic_neuron_count",
            "expected": 302,
            "actual": n,
            "ok": n == 302,
            "note": "Exact hermaphrodite somatic connectome count (282 somatic + 20 pharyngeal).",
        },
        {
            "check": "graded_nonspiking_models",
            "expected": 302,
            "actual": n_graded,
            "ok": n_graded == 302,
            "note": "All neurons marked MODEL_GRADED.",
        },
        {
            "check": "gap_junction_and_chemical_edges",
            "expected": "> 0 chemical & > 0 gap_junction",
            "actual": {"chemical": n_chem, "gap_junction": n_gap},
            "ok": n_chem > 0 and n_gap > 0 and (n_gap % 2 == 0),
            "note": (
                "Both chemical synapses and symmetric bidirectional ohmic gap junctions present."
            ),
        },
        {
            "check": "command_interneuron_ports",
            "expected": {"forward_command": 4, "reverse_command": 6},
            "actual": {"forward_command": fwd_cnt, "reverse_command": rev_cnt},
            "ok": fwd_cnt == 4 and rev_cnt == 6,
            "note": "AVBL/R + PVCL/R (forward) and AVAL/R + AVDL/R + AVEL/R (reverse).",
        },
    ]
