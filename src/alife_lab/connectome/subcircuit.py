"""Connectome subcircuit extraction and offline synthetic olfactory circuit for Phase F5.

Extracts self-contained anatomical subcircuits (e.g. FlyWire v783 Antennal Lobe + Mushroom Body
+ Lateral Horn: 11,276 cells, 1,105,291 internal edges) while preserving cell types, classes,
neuropils, glomerular input ports and degree-preserving shuffled-wiring negative controls.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from alife_lab.connectome.lif import make_shuffled_csr
from alife_lab.connectome.schema import (
    MODEL_SPIKING_LIF,
    POS_SOMA,
    ROLE_INTERNEURON,
    ROLE_MOTOR,
    ROLE_SENSORY,
    SYNAPSE_CHEMICAL,
    CellTable,
    ConnectomePack,
    DatasetProvenance,
    EdgeTable,
    FileProvenance,
    IOPort,
    compute_pack_digest,
)

SubcircuitId = Literal["olfactory_mb_lh", "full"]

_OLF_REGIONS = frozenset(
    {
        "AL_L",
        "AL_R",
        "MB_CA_L",
        "MB_CA_R",
        "MB_ML_L",
        "MB_ML_R",
        "MB_PED_L",
        "MB_PED_R",
        "MB_VL_L",
        "MB_VL_R",
        "LH_L",
        "LH_R",
    }
)
_OLF_CLASSES = frozenset(
    {
        "olfactory",
        "ALPN",
        "ALLN",
        "ALIN",
        "ALON",
        "Kenyon_Cell",
        "MBON",
        "DAN",
        "MBIN",
        "LHLN",
        "LHCENT",
    }
)


@dataclass(frozen=True)
class Subcircuit:
    """Compact indexed connectome subcircuit ready for differentiable system identification."""

    dataset_id: str
    subcircuit_id: str
    wiring: Literal["reference", "shuffled_control"]
    pack_digest: str
    num_cells: int
    pre: npt.NDArray[np.int32]
    post: npt.NDArray[np.int32]
    signed_weight: npt.NDArray[np.int32]
    cell_types: tuple[str, ...]
    cell_classes: tuple[str, ...]
    regions: tuple[str, ...]
    roles: npt.NDArray[np.uint8]
    type_ids: npt.NDArray[np.int32]
    type_names: tuple[str, ...]
    class_ids: npt.NDArray[np.int32]
    class_names: tuple[str, ...]
    input_groups: tuple[tuple[str, npt.NDArray[np.int32]], ...]
    global_indices: npt.NDArray[np.int32]

    @property
    def num_edges(self) -> int:
        return int(self.pre.shape[0])

    @property
    def num_types(self) -> int:
        return len(self.type_names)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def make_synthetic_olfactory_pack() -> ConnectomePack:
    """Built-in 40-neuron multi-layer olfactory circuit for offline CPU-only F5 tests & UI.

    Architecture mirrors the Drosophila olfactory pathway at miniature scale:
    - 8 olfactory sensory neurons (4 glomeruli: ORN_A..D, 2 cells each) -> excitatory
    - 8 antennal lobe projection neurons (ALPN_A..D, 2 cells each) -> excitatory
    - 8 antennal lobe local interneurons (ALLN_1..2, 4 cells each) -> inhibitory (GABA)
    - 8 mushroom body Kenyon cells (KC_A..D, 2 cells each) -> excitatory
    - 4 mushroom body output neurons (MBON_1..2, 2 cells each) -> 1 inhibitory, 1 excitatory
    - 4 lateral horn local/output neurons (LHLN_1..2, 2 cells each) -> 1 inhibitory, 1 excitatory
    Total: 40 cells, 16 distinct cell types, 6 cell classes, rich feedforward + recurrent E/I loops.
    """
    specs: list[tuple[str, str, str, int, int, str]] = [
        # (cell_type, cell_class, region, count, role, nt)
        ("ORN_A", "olfactory", "AL_L", 2, ROLE_SENSORY, "ACH"),
        ("ORN_B", "olfactory", "AL_L", 2, ROLE_SENSORY, "ACH"),
        ("ORN_C", "olfactory", "AL_R", 2, ROLE_SENSORY, "ACH"),
        ("ORN_D", "olfactory", "AL_R", 2, ROLE_SENSORY, "ACH"),
        ("ALPN_A", "ALPN", "AL_L", 2, ROLE_INTERNEURON, "ACH"),
        ("ALPN_B", "ALPN", "AL_L", 2, ROLE_INTERNEURON, "ACH"),
        ("ALPN_C", "ALPN", "AL_R", 2, ROLE_INTERNEURON, "ACH"),
        ("ALPN_D", "ALPN", "AL_R", 2, ROLE_INTERNEURON, "ACH"),
        ("ALLN_1", "ALLN", "AL_L", 4, ROLE_INTERNEURON, "GABA"),
        ("ALLN_2", "ALLN", "AL_R", 4, ROLE_INTERNEURON, "GABA"),
        ("KC_A", "Kenyon_Cell", "MB_CA_L", 2, ROLE_INTERNEURON, "ACH"),
        ("KC_B", "Kenyon_Cell", "MB_CA_L", 2, ROLE_INTERNEURON, "ACH"),
        ("KC_C", "Kenyon_Cell", "MB_CA_R", 2, ROLE_INTERNEURON, "ACH"),
        ("KC_D", "Kenyon_Cell", "MB_CA_R", 2, ROLE_INTERNEURON, "ACH"),
        ("MBON_1", "MBON", "MB_ML_L", 2, ROLE_MOTOR, "GABA"),
        ("MBON_2", "MBON", "MB_ML_R", 2, ROLE_MOTOR, "ACH"),
        ("LHLN_1", "LHLN", "LH_L", 2, ROLE_INTERNEURON, "GABA"),
        ("LHLN_2", "LHLN", "LH_R", 2, ROLE_INTERNEURON, "ACH"),
    ]
    ctypes: list[str] = []
    cclasses: list[str] = []
    regions: list[str] = []
    roles: list[int] = []
    nts: list[str] = []
    pos: list[list[float]] = []
    by_type: dict[str, list[int]] = {}
    idx = 0
    layer_x = {
        "olfactory": 0.0,
        "ALPN": 120.0,
        "ALLN": 120.0,
        "Kenyon_Cell": 260.0,
        "MBON": 380.0,
        "LHLN": 380.0,
    }
    for ct, cls, reg, cnt, role, nt in specs:
        by_type[ct] = []
        for k in range(cnt):
            by_type[ct].append(idx)
            ctypes.append(ct)
            cclasses.append(cls)
            regions.append(reg)
            roles.append(role)
            nts.append(nt)
            pos.append([layer_x[cls], float(idx % 8) * 35.0, float(k) * 30.0])
            idx += 1

    n = idx
    edge_map: dict[tuple[int, int], tuple[int, int]] = {}

    def connect(src_t: str, dst_t: str, weight: int, sign: int) -> None:
        for u in by_type[src_t]:
            for v in by_type[dst_t]:
                if u != v:
                    edge_map[(u, v)] = (weight, sign)

    # 1. Glomerular ORN -> cognate ALPN + lateral ALLN recruitment.
    for g, ln in (("A", "ALLN_1"), ("B", "ALLN_1"), ("C", "ALLN_2"), ("D", "ALLN_2")):
        connect(f"ORN_{g}", f"ALPN_{g}", 80, +1)
        connect(f"ORN_{g}", ln, 35, +1)
    # Cross-talk weak ORN -> neighbour ALPN
    connect("ORN_A", "ALPN_B", 20, +1)
    connect("ORN_C", "ALPN_D", 20, +1)

    # 2. Lateral inhibitory ALLN -> ALPN & cross-ALLN.
    for g in ("A", "B", "C", "D"):
        connect("ALLN_1", f"ALPN_{g}", 25, -1)
        connect("ALLN_2", f"ALPN_{g}", 25, -1)
    connect("ALLN_1", "ALLN_2", 18, -1)
    connect("ALLN_2", "ALLN_1", 18, -1)

    # 3. ALPN -> KC (combinatorial expansion) and ALPN -> LHLN (innate pathway).
    connect("ALPN_A", "KC_A", 65, +1)
    connect("ALPN_B", "KC_A", 45, +1)
    connect("ALPN_B", "KC_B", 65, +1)
    connect("ALPN_C", "KC_B", 45, +1)
    connect("ALPN_C", "KC_C", 65, +1)
    connect("ALPN_D", "KC_C", 45, +1)
    connect("ALPN_D", "KC_D", 65, +1)
    connect("ALPN_A", "KC_D", 45, +1)

    connect("ALPN_A", "LHLN_1", 55, +1)
    connect("ALPN_B", "LHLN_1", 40, +1)
    connect("ALPN_C", "LHLN_2", 55, +1)
    connect("ALPN_D", "LHLN_2", 40, +1)

    # 4. KC -> MBON readout + recurrent MBON/LHLN cross-inhibition.
    connect("KC_A", "MBON_1", 70, +1)
    connect("KC_B", "MBON_1", 55, +1)
    connect("KC_C", "MBON_2", 70, +1)
    connect("KC_D", "MBON_2", 55, +1)
    connect("MBON_1", "MBON_2", 40, -1)
    connect("MBON_1", "LHLN_2", 30, -1)
    connect("LHLN_1", "LHLN_2", 35, -1)
    connect("LHLN_2", "MBON_2", 35, +1)
    connect("MBON_2", "LHLN_1", 25, +1)

    sorted_pairs = sorted(edge_map.items())
    pre = np.asarray([u for (u, _), _ in sorted_pairs], dtype=np.int32)
    post = np.asarray([v for (_, v), _ in sorted_pairs], dtype=np.int32)
    w = np.asarray([wt for _, (wt, _) in sorted_pairs], dtype=np.int32)
    s = np.asarray([sg for _, (_, sg) in sorted_pairs], dtype=np.int8)

    cells = CellTable(
        source_id=np.arange(3000, 3000 + n, dtype=np.int64),
        species_names=("synthetic_olfactory",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(ctypes),
        super_class=tuple("sensory" if r == ROLE_SENSORY else "central" for r in roles),
        cell_class=tuple(cclasses),
        cell_sub_class=tuple(cclasses),
        region=tuple(regions),
        side=tuple("left" if reg.endswith("_L") else "right" for reg in regions),
        role=np.asarray(roles, dtype=np.uint8),
        cell_model=np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8),
        neurotransmitter=tuple(nts),
        position_nm=np.asarray(pos, dtype=np.float32),
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    edges = EdgeTable(
        pre=pre,
        post=post,
        count=w,
        sign=s,
        kind=np.full(len(pre), SYNAPSE_CHEMICAL, dtype=np.uint8),
    )
    io_ports = {
        "olfactory_sensory": IOPort(
            port_id="olfactory_sensory",
            direction="sensory",
            label="Synthetic ORN glomeruli",
            description="All synthetic ORN glomeruli (ORN_A..D)",
            cell_indices=np.asarray(
                by_type["ORN_A"] + by_type["ORN_B"] + by_type["ORN_C"] + by_type["ORN_D"],
                dtype=np.int32,
            ),
            selection_rule="cell_class == 'olfactory'",
        ),
        "mushroom_body_kc": IOPort(
            port_id="mushroom_body_kc",
            direction="circuit",
            label="Synthetic Kenyon cells",
            description="Synthetic Kenyon cells (KC_A..D)",
            cell_indices=np.asarray(
                by_type["KC_A"] + by_type["KC_B"] + by_type["KC_C"] + by_type["KC_D"],
                dtype=np.int32,
            ),
            selection_rule="cell_class == 'Kenyon_Cell'",
        ),
        "mushroom_body_mbon": IOPort(
            port_id="mushroom_body_mbon",
            direction="motor",
            label="Synthetic MBONs",
            description="Synthetic Mushroom Body output neurons",
            cell_indices=np.asarray(by_type["MBON_1"] + by_type["MBON_2"], dtype=np.int32),
            selection_rule="cell_class == 'MBON'",
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_olfactory",
        species="synthetic_olfactory",
        version="v1",
        citation="Built-in 40-neuron F5 hidden-truth system identification circuit.",
        licence_note="project",
        files=(
            FileProvenance(
                filename="builtin_synthetic_olfactory.json",
                url="https://example.invalid/builtin",
                sha256="0" * 64,
                size_bytes=0,
                source_label="builtin",
                licence_note="project",
            ),
        ),
    )
    digest = compute_pack_digest(
        "synthetic_olfactory", "synthetic_olfactory", cells, edges, io_ports, prov
    )
    return ConnectomePack(
        dataset_id="synthetic_olfactory",
        species="synthetic_olfactory",
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
        pack_digest=digest,
    )


def extract_subcircuit(
    pack: ConnectomePack,
    *,
    subcircuit_id: str = "olfactory_mb_lh",
    wiring: Literal["reference", "shuffled_control"] = "reference",
    shuffle_seed: int = 0,
    max_input_groups: int = 12,
) -> Subcircuit:
    """Extract a self-contained subcircuit from ``pack`` with glomerular input groups."""
    n_all = pack.cells.num_cells
    if subcircuit_id == "olfactory_mb_lh" and n_all > 100:
        selected = np.asarray(
            [
                i
                for i in range(n_all)
                if (pack.cells.region[i] in _OLF_REGIONS)
                or (pack.cells.cell_class[i] in _OLF_CLASSES)
            ],
            dtype=np.int32,
        )
    else:
        selected = np.arange(n_all, dtype=np.int32)

    remap = np.full(n_all, -1, dtype=np.int32)
    remap[selected] = np.arange(selected.shape[0], dtype=np.int32)
    n_sub = int(selected.shape[0])

    sw_all = pack.edges.signed_weights()
    mask = (remap[pack.edges.pre] >= 0) & (remap[pack.edges.post] >= 0)
    pre = remap[pack.edges.pre[mask]].astype(np.int32, copy=False)
    post = remap[pack.edges.post[mask]].astype(np.int32, copy=False)
    sw = sw_all[mask].astype(np.int32, copy=False)

    if wiring == "shuffled_control":
        # Degree-preserving shuffle inside the subcircuit so in/out degree and weight
        # distributions match the biological subcircuit exactly.
        sub_edges = EdgeTable(
            pre=pre,
            post=post,
            count=np.abs(sw).astype(np.int32),
            sign=np.where(sw >= 0, 1, -1).astype(np.int8),
            kind=np.full(pre.shape[0], SYNAPSE_CHEMICAL, dtype=np.uint8),
        )
        csr = sub_edges.to_csr(n_sub)
        shuf = make_shuffled_csr(csr, int(shuffle_seed))
        # Unpack shuffled CSR back to edge lists.
        counts = np.diff(shuf.indptr)
        pre = np.repeat(np.arange(n_sub, dtype=np.int32), counts)
        post = shuf.post.astype(np.int32, copy=True)
        sw = shuf.signed_weight.astype(np.int32, copy=True)

    cell_types = tuple(pack.cells.cell_type[int(i)] or "unknown" for i in selected)
    cell_classes = tuple(pack.cells.cell_class[int(i)] or "other" for i in selected)
    regions = tuple(pack.cells.region[int(i)] or "UNASGD" for i in selected)
    roles = pack.cells.role[selected].astype(np.uint8, copy=True)

    type_names = tuple(sorted(set(cell_types)))
    type_to_id = {name: i for i, name in enumerate(type_names)}
    type_ids = np.asarray([type_to_id[t] for t in cell_types], dtype=np.int32)

    class_names = tuple(sorted(set(cell_classes)))
    class_to_id = {name: i for i, name in enumerate(class_names)}
    class_ids = np.asarray([class_to_id[c] for c in cell_classes], dtype=np.int32)

    # Build glomerular input groups from olfactory sensory neurons (ordered by cell count).
    sensory_idx = [
        i for i in range(n_sub) if cell_classes[i] == "olfactory" or int(roles[i]) == ROLE_SENSORY
    ]
    if not sensory_idx:
        sensory_idx = list(range(min(8, n_sub)))
    by_type = Counter(cell_types[i] for i in sensory_idx)
    top_types = [t for t, _ in by_type.most_common(max_input_groups)]
    groups: list[tuple[str, npt.NDArray[np.int32]]] = []
    for t in top_types:
        arr = np.asarray([i for i in sensory_idx if cell_types[i] == t], dtype=np.int32)
        if arr.size > 0:
            groups.append((t, arr))

    return Subcircuit(
        dataset_id=pack.dataset_id,
        subcircuit_id=subcircuit_id,
        wiring=wiring,
        pack_digest=pack.pack_digest,
        num_cells=n_sub,
        pre=pre,
        post=post,
        signed_weight=sw,
        cell_types=cell_types,
        cell_classes=cell_classes,
        regions=regions,
        roles=roles,
        type_ids=type_ids,
        type_names=type_names,
        class_ids=class_ids,
        class_names=class_names,
        input_groups=tuple(groups),
        global_indices=selected,
    )
