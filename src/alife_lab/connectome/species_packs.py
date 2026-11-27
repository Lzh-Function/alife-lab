"""Stage 4 / S2-S4 connectome species packs: Fly Larva L1, Platynereis 3D, and Ciona Larva.

Provides both:
1. Pure tabular parsers (ZERO pickles, ZipSlip-checked CSV archives only) for verified empirical
   datasets stored under ``.alife/datasets/<dataset_id>/raw/``:
   - ``fly_larva_l1``: Winding et al. (2023) Science, Drosophila L1 larva whole-brain (2,956 cells).
   - ``platynereis_3d``: Verasztó, Jasek, Jékely et al. (2024/2025) eLife, 3-day larva whole-body
     synaptic connectome matrix (1,720 connected neurons/muscles/ciliary cells).
   - ``ciona_larva``: Ryan, Lu & Meinertzhagen (2016) eLife, Ciona intestinalis tadpole CNS
     connectome (205 cells, 2,903 weighted edges).
2. Deterministic offline synthetic packs (``synthetic_fly_larva``, ``synthetic_platynereis``,
   ``synthetic_ciona``) sharing identical IO port IDs, cell models, and anatomical motifs so
   minimal CPU CI exercises every pipeline, comparative analysis, and F13 chimera without downloads.
"""

from __future__ import annotations

import csv
import io
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from alife_lab.connectome.registry import (
    CIONA_LARVA_SPEC,
    FLY_LARVA_L1_SPEC,
    PLATYNEREIS_3D_SPEC,
    DatasetError,
    DatasetSpec,
    raw_dir,
    verify_dataset,
)
from alife_lab.connectome.schema import (
    MODEL_GRADED,
    MODEL_SPIKING_LIF,
    POS_ANCHOR,
    POS_SOMA,
    ROLE_INTERNEURON,
    ROLE_MOTOR,
    ROLE_OTHER,
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

_POS_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _safe_read_zip_csv(zip_path: Path, member: str) -> list[list[str]]:
    """Read a CSV member safely from a zip archive without extracting to disk."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for nm in names:
            if ".." in nm or nm.startswith("/") or nm.startswith("\\"):
                raise DatasetError(f"unsafe member path {nm!r} in {zip_path.name}")
            if not nm.lower().endswith(".csv"):
                raise DatasetError(f"non-CSV member {nm!r} refused in {zip_path.name}")
        if member not in names:
            raise DatasetError(f"missing {member!r} in {zip_path.name}")
        text = zf.read(member).decode("utf-8", errors="replace")
    return list(csv.reader(io.StringIO(text)))


def _parse_pos_2d(
    raw: str, fallback_idx: int, z_nm: float = 0.0, scale_nm: float = 15_000.0
) -> tuple[float, float, float]:
    nums = [float(m) for m in _POS_RE.findall(raw)]
    if len(nums) >= 2:
        return (nums[0] * scale_nm, nums[1] * scale_nm, z_nm)
    ang = (fallback_idx * 2.3999632) % (2.0 * math.pi)
    rad = math.sqrt(fallback_idx + 1) * 5_000.0
    return (rad * math.cos(ang), rad * math.sin(ang), z_nm)


def _make_port(
    port_id: str,
    direction: Any,
    label: str,
    description: str,
    indices: list[int],
    rule: str,
) -> IOPort:
    arr = np.asarray(sorted(set(indices)), dtype=np.int32)
    return IOPort(
        port_id=port_id,
        direction=direction,
        label=label,
        description=description,
        cell_indices=arr,
        selection_rule=rule,
    )


def _build_pack(
    *,
    dataset_id: str,
    species: str,
    cells: CellTable,
    edges: EdgeTable,
    io_ports: dict[str, IOPort],
    provenance: DatasetProvenance,
) -> ConnectomePack:
    cells.validate()
    edges.validate(cells.num_cells)
    digest = compute_pack_digest(dataset_id, species, cells, edges, io_ports, provenance)
    return ConnectomePack(
        dataset_id=dataset_id,
        species=species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=provenance,
        pack_digest=digest,
    )


# =========================================================================== #
# S2: Drosophila melanogaster L1 Larva (Winding et al. 2023, Science)
# =========================================================================== #

FLY_LARVA_L1_REFERENCE: dict[str, Any] = {
    "neurons": 2956,
    "raw_multigraph_edges": 116922,
    "total_synapses": 352611,
    "kc_count": 144,
    "mbon_count": 48,
    "mbin_count": 28,
    "olfactory_sn_count": 42,
    "dn_vnc_count": 182,
}


def load_fly_larva_l1(
    datasets_dir: Path,
    *,
    verify_hashes: bool = True,
    spec: DatasetSpec | None = None,
) -> ConnectomePack:
    """Load verified Drosophila L1 larva whole-brain connectome (Winding et al. 2023)."""
    dspec = spec or FLY_LARVA_L1_SPEC
    if verify_hashes:
        rep = verify_dataset(dspec.dataset_id, datasets_dir, spec=dspec)
        if not rep["ok"]:
            raise DatasetError(f"dataset {dspec.dataset_id!r} verification failed: {rep['files']}")
    zpath = raw_dir(datasets_dir, dspec.dataset_id) / "fly_larva.csv.zip"
    if not zpath.is_file():
        raise DatasetError(f"missing {zpath}")

    node_rows = _safe_read_zip_csv(zpath, "nodes.csv")[1:]
    n = len(node_rows)
    source_ids = np.empty(n, dtype=np.int64)
    cell_types: list[str] = []
    super_classes: list[str] = []
    cell_classes: list[str] = []
    sub_classes: list[str] = []
    regions: list[str] = []
    sides: list[str] = []
    roles = np.full(n, ROLE_INTERNEURON, dtype=np.uint8)
    models = np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8)
    nts: list[str] = []
    pos = np.zeros((n, 3), dtype=np.float32)
    pos_src = np.full(n, POS_ANCHOR, dtype=np.uint8)

    p_olf: list[int] = []
    p_gust: list[int] = []
    p_vis: list[int] = []
    p_mech: list[int] = []
    p_kc: list[int] = []
    p_mbon: list[int] = []
    p_mbin: list[int] = []
    p_pn: list[int] = []
    p_dn_vnc: list[int] = []
    p_dn_sez: list[int] = []

    inhibitory_types = {"LN", "MB-FBN"}

    for row in node_rows:
        idx = int(row[0])
        vid = int(row[1]) if row[1].strip() else idx + 1
        hemi = row[2].strip().lower()
        ct = row[4].strip() or "unclassified"
        ann = row[5].strip()
        cluster = row[6].strip() or "0"
        pos_raw = row[7] if len(row) > 7 else ""

        source_ids[idx] = vid
        side = "left" if hemi == "left" else ("right" if hemi == "right" else "center")
        sides.append(side)
        cell_types.append(ct)
        sub_classes.append(ann if ann and ann != "no official annotation" else ct)

        ann_l = ann.lower()
        if ct == "sensory":
            roles[idx] = ROLE_SENSORY
            super_classes.append("sensory")
            cell_classes.append(ann if ann and ann != "no official annotation" else "sensory")
            regions.append("sensory_entry")
            if "olfactory" in ann_l:
                p_olf.append(idx)
            if "gustatory" in ann_l:
                p_gust.append(idx)
            if "visual" in ann_l:
                p_vis.append(idx)
            if "mechano" in ann_l or "proprio" in ann_l or "noci" in ann_l:
                p_mech.append(idx)
        elif ct in ("DN-VNC", "DN-SEZ", "RGN"):
            roles[idx] = ROLE_MOTOR
            super_classes.append("descending_output")
            cell_classes.append(ct)
            regions.append(ct)
            if ct == "DN-VNC":
                p_dn_vnc.append(idx)
            elif ct == "DN-SEZ":
                p_dn_sez.append(idx)
        else:
            roles[idx] = ROLE_INTERNEURON
            super_classes.append("central_brain")
            cell_classes.append(ct)
            if ct in ("KC", "MBON", "MBIN", "MB-FBN", "MB-FFN"):
                regions.append("MB")
            elif ct in ("PN", "PN-somato", "LN"):
                regions.append("AL")
            elif ct == "LHN":
                regions.append("LH")
            else:
                regions.append(f"cluster_{cluster}")
            if ct == "KC":
                p_kc.append(idx)
            elif ct == "MBON":
                p_mbon.append(idx)
            elif ct == "MBIN":
                p_mbin.append(idx)
            elif ct == "PN":
                p_pn.append(idx)

        nts.append("GABA_est" if ct in inhibitory_types else "ACH_est")
        z_nm = (-20_000.0 if side == "left" else (20_000.0 if side == "right" else 0.0)) + (
            int(cluster) % 10 - 5
        ) * 3_500.0
        pos[idx] = _parse_pos_2d(pos_raw, idx, z_nm=z_nm, scale_nm=18_000.0)

    # Collapse parallel multigraph edges (ad, aa, dd, da) into canonical (pre, post) weights
    edge_rows = _safe_read_zip_csv(zpath, "edges.csv")[1:]
    agg: dict[tuple[int, int], int] = defaultdict(int)
    for row in edge_rows:
        u, v, cnt = int(row[0]), int(row[1]), int(row[2])
        if 0 <= u < n and 0 <= v < n and cnt > 0:
            agg[(u, v)] += cnt

    m = len(agg)
    pre = np.empty(m, dtype=np.int32)
    post = np.empty(m, dtype=np.int32)
    count = np.empty(m, dtype=np.int32)
    sign = np.empty(m, dtype=np.int8)
    kind = np.full(m, SYNAPSE_CHEMICAL, dtype=np.uint8)
    for k, ((u, v), w) in enumerate(sorted(agg.items())):
        pre[k] = u
        post[k] = v
        count[k] = w
        sign[k] = -1 if nts[u] == "GABA_est" else 1

    cells = CellTable(
        source_id=source_ids,
        species_names=(dspec.species,),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(cell_types),
        super_class=tuple(super_classes),
        cell_class=tuple(cell_classes),
        cell_sub_class=tuple(sub_classes),
        region=tuple(regions),
        side=tuple(sides),
        role=roles,
        cell_model=models,
        neurotransmitter=tuple(nts),
        position_nm=pos,
        position_source=pos_src,
    )
    edges = EdgeTable(pre=pre, post=post, count=count, sign=sign, kind=kind)
    io_ports = {
        "larva_olfactory_sn": _make_port(
            "larva_olfactory_sn",
            "sensory",
            "Larval olfactory ORNs",
            "Olfactory sensory neurons projecting to larval antennal lobe.",
            p_olf,
            "cell_type == 'sensory' and 'olfactory' in annotations",
        ),
        "larva_gustatory_sn": _make_port(
            "larva_gustatory_sn",
            "sensory",
            "Larval gustatory GRNs",
            "External & pharyngeal gustatory sensory neurons.",
            p_gust,
            "cell_type == 'sensory' and 'gustatory' in annotations",
        ),
        "larva_visual_sn": _make_port(
            "larva_visual_sn",
            "sensory",
            "Larval Bolwig visual SNs",
            "Larval visual photoreceptors.",
            p_vis,
            "cell_type == 'sensory' and 'visual' in annotations",
        ),
        "larva_mechano_sn": _make_port(
            "larva_mechano_sn",
            "sensory",
            "Larval mechanosensory/nociceptive SNs",
            "Chordotonal, proprioceptive, and nociceptive sensory neurons.",
            p_mech,
            "cell_type == 'sensory' and mechano/proprio/noci",
        ),
        "larva_pn": _make_port(
            "larva_pn",
            "circuit",
            "Larval projection neurons (PN)",
            "Antennal lobe projection neurons carrying odor to MB/LH.",
            p_pn,
            "cell_type == 'PN'",
        ),
        "larva_kc": _make_port(
            "larva_kc",
            "circuit",
            "Larval Kenyon cells (KC)",
            "Mushroom body Kenyon cells encoding sparse olfactory memory.",
            p_kc,
            "cell_type == 'KC'",
        ),
        "larva_mbon": _make_port(
            "larva_mbon",
            "circuit",
            "Larval MB output neurons (MBON)",
            "Mushroom body output neurons driving approach/avoidance.",
            p_mbon,
            "cell_type == 'MBON'",
        ),
        "larva_mbin": _make_port(
            "larva_mbin",
            "circuit",
            "Larval MB input/modulatory neurons (MBIN)",
            "Dopaminergic/octopaminergic MBIN valence signals.",
            p_mbin,
            "cell_type == 'MBIN'",
        ),
        "larva_dn_vnc": _make_port(
            "larva_dn_vnc",
            "motor",
            "Larval VNC descending neurons (DN-VNC)",
            "Descending command outputs projecting to larval VNC crawling circuit.",
            p_dn_vnc,
            "cell_type == 'DN-VNC'",
        ),
        "larva_dn_sez": _make_port(
            "larva_dn_sez",
            "motor",
            "Larval SEZ descending neurons (DN-SEZ)",
            "Descending command outputs controlling feeding/head cast.",
            p_dn_sez,
            "cell_type == 'DN-SEZ'",
        ),
        # Canonical alias ports for cross-species & F13 chimera interfaces
        "olfactory_sn": _make_port(
            "olfactory_sn",
            "sensory",
            "Olfactory sensory entry",
            "Alias of larva_olfactory_sn.",
            p_olf,
            "alias: larva_olfactory_sn",
        ),
        "primary_motor_out": _make_port(
            "primary_motor_out",
            "motor",
            "Primary motor/descending output",
            "Alias of larva_dn_vnc.",
            p_dn_vnc,
            "alias: larva_dn_vnc",
        ),
    }
    return _build_pack(
        dataset_id=dspec.dataset_id,
        species=dspec.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=dspec.to_provenance(),
    )


def fly_larva_l1_reference_checks(pack: ConnectomePack) -> list[dict[str, Any]]:
    ref = FLY_LARVA_L1_REFERENCE
    syn = int(pack.edges.count.astype(np.int64).sum())
    is_real = pack.dataset_id == "fly_larva_l1"
    if not is_real:
        return [
            {
                "check": "synthetic_larva_ports_populated",
                "expected": True,
                "actual": all(
                    pack.io_ports[k].cell_indices.size > 0
                    for k in ("larva_olfactory_sn", "larva_kc", "larva_mbon", "larva_dn_vnc")
                ),
                "ok": all(
                    pack.io_ports[k].cell_indices.size > 0
                    for k in ("larva_olfactory_sn", "larva_kc", "larva_mbon", "larva_dn_vnc")
                ),
                "note": (
                    "Offline synthetic larval circuit preserves ORN->PN->KC->MBON->DN-VNC motif."
                ),
            }
        ]
    return [
        {
            "check": "winding_2023_neuron_count",
            "expected": ref["neurons"],
            "actual": pack.cells.num_cells,
            "ok": pack.cells.num_cells == ref["neurons"],
            "note": "Winding et al. (2023) Science / Netzschleuder connected brain neurons.",
        },
        {
            "check": "winding_2023_total_synapses",
            "expected": ref["total_synapses"],
            "actual": syn,
            "ok": syn == ref["total_synapses"],
            "note": (
                "Total synapses preserved exactly after collapsing ad/aa/dd/da multigraph edges."
            ),
        },
        {
            "check": "larva_mushroom_body_kc_mbon_mbin",
            "expected": [ref["kc_count"], ref["mbon_count"], ref["mbin_count"]],
            "actual": [
                int(pack.io_ports["larva_kc"].cell_indices.size),
                int(pack.io_ports["larva_mbon"].cell_indices.size),
                int(pack.io_ports["larva_mbin"].cell_indices.size),
            ],
            "ok": (
                int(pack.io_ports["larva_kc"].cell_indices.size) == ref["kc_count"]
                and int(pack.io_ports["larva_mbon"].cell_indices.size) == ref["mbon_count"]
                and int(pack.io_ports["larva_mbin"].cell_indices.size) == ref["mbin_count"]
            ),
            "note": "Complete larval mushroom body (144 KC, 48 MBON, 28 MBIN) resolved.",
        },
        {
            "check": "larva_dn_vnc_count",
            "expected": ref["dn_vnc_count"],
            "actual": int(pack.io_ports["larva_dn_vnc"].cell_indices.size),
            "ok": int(pack.io_ports["larva_dn_vnc"].cell_indices.size) == ref["dn_vnc_count"],
            "note": "Descending VNC command neurons resolved.",
        },
    ]


# =========================================================================== #
# S3: Platynereis dumerilii 3-Day Larva (Verasztó, Jasek, Jékely et al. eLife)
# =========================================================================== #

PLATYNEREIS_3D_REFERENCE: dict[str, Any] = {
    "matrix_cells": 1720,
    "directed_edges": 8451,
    "total_synapses": 19991,
    "min_prc_cells": 20,
    "min_muscle_cells": 80,
}


def _classify_platy_cell(
    name: str, ct_meta: dict[str, dict[str, str]]
) -> tuple[int, str, str, str, str, str]:
    """Return (role, cell_type, super_class, region, side, neurotransmitter)."""
    base = name.split("_")[0]
    stem = re.sub(r"\d+$", "", base)
    meta = ct_meta.get(name) or ct_meta.get(base) or ct_meta.get(stem) or {}
    raw_role = (meta.get("Sensory/inter/motor neuron") or "").lower()
    raw_reg = (meta.get("region") or "").strip()
    raw_tx = (meta.get("transmitter phenotype") or "").lower()

    lower = name.lower()
    if "_l" in lower or lower.endswith("l") or "_al" in lower or "_pl" in lower:
        side = "left"
    elif "_r" in lower or lower.endswith("r") or "_ar" in lower or "_pr" in lower:
        side = "right"
    else:
        side = "center"

    if "sensory" in raw_role or name.startswith(
        ("PRC", "SN", "chaeMech", "antPU", "ventralParaPU", "pygCirrus", "neuropodPU")
    ):
        role = ROLE_SENSORY
        scls = "sensory"
    elif (
        "motoneuron" in raw_role
        or "muscle" in raw_role
        or "multiciliated" in raw_role
        or name.startswith(("MN", "MUS", "prototroch", "paratroch", "ciliomotor"))
    ):
        role = ROLE_MOTOR
        scls = (
            "muscle"
            if name.startswith("MUS") or "muscle" in raw_role
            else ("ciliary_effector" if "troch" in lower or "ciliat" in raw_role else "motor")
        )
    elif "interneuron" in raw_role or name.startswith(("IN", "neuron", "Neuron")):
        role = ROLE_INTERNEURON
        scls = "interneuron"
    else:
        role = ROLE_OTHER
        scls = "other"

    if raw_reg:
        region = raw_reg.split("  ")[0].strip()
    elif name.startswith("PRC"):
        region = "visual system"
    elif name.startswith("INMB"):
        region = "mushroom body"
    elif name.startswith(("prototroch", "paratroch", "MNring")):
        region = "ciliomotor"
    elif name.startswith("MUS"):
        region = "trunk_musculature"
    elif name.startswith(("SNnuch", "SN-NS", "SN-DLSO", "SN-DSO")):
        region = "head_sensory"
    else:
        region = "ventral nerve cord"

    if any(tok in raw_tx for tok in ("gaba", "inhib", "fmrf", "glyc")):
        nt = "GABA_inh"
    elif any(tok in raw_tx for tok in ("cholin", "glut", "seroton")):
        nt = "ACH_exc"
    elif name.startswith(("INcomm", "INsplit")):
        nt = "GABA_inh_est"
    else:
        nt = "ACH_exc_est"

    ct = (meta.get("name of cell type") or base or name).strip()
    return role, ct, scls, region, side, nt


def load_platynereis_3d(
    datasets_dir: Path,
    *,
    verify_hashes: bool = True,
    spec: DatasetSpec | None = None,
) -> ConnectomePack:
    """Load verified Platynereis dumerilii 3-day larval connectome (Verasztó et al. eLife)."""
    dspec = spec or PLATYNEREIS_3D_SPEC
    if verify_hashes:
        rep = verify_dataset(dspec.dataset_id, datasets_dir, spec=dspec)
        if not rep["ok"]:
            raise DatasetError(f"dataset {dspec.dataset_id!r} verification failed: {rep['files']}")
    rdir = raw_dir(datasets_dir, dspec.dataset_id)
    mat_path = rdir / "full_connectome_adjacency_matrix.csv"
    ct_path = rdir / "neuronal_celltypes_table.csv"
    if not mat_path.is_file() or not ct_path.is_file():
        raise DatasetError(f"missing Platynereis CSV tables in {rdir}")

    ct_meta: dict[str, dict[str, str]] = {}
    with ct_path.open("r", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("name of cell type") or "").strip()
            if key:
                ct_meta[key] = row

    with mat_path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col_names = header[1:]
        n = len(col_names)
        row_names: list[str] = []
        pre_list: list[int] = []
        post_list: list[int] = []
        cnt_list: list[int] = []
        for i, mrow in enumerate(reader):
            row_names.append(mrow[0].strip())
            for j, val in enumerate(mrow[1:]):
                iv = int(val)
                if iv > 0:
                    pre_list.append(i)
                    post_list.append(j)
                    cnt_list.append(iv)

    if len(row_names) != n:
        raise DatasetError(f"non-square Platynereis matrix: {len(row_names)}x{n}")

    source_ids = np.arange(1, n + 1, dtype=np.int64)
    cell_types: list[str] = []
    super_classes: list[str] = []
    cell_classes: list[str] = []
    sub_classes: list[str] = []
    regions: list[str] = []
    sides: list[str] = []
    roles = np.full(n, ROLE_OTHER, dtype=np.uint8)
    # Platynereis larval circuits combine graded/ciliomotor and slow potentials
    models = np.full(n, MODEL_GRADED, dtype=np.uint8)
    nts: list[str] = []
    pos = np.zeros((n, 3), dtype=np.float32)
    pos_src = np.full(n, POS_ANCHOR, dtype=np.uint8)

    p_prc: list[int] = []
    p_mech: list[int] = []
    p_chemo: list[int] = []
    p_mb: list[int] = []
    p_cilio: list[int] = []
    p_mus: list[int] = []
    p_vis_in: list[int] = []

    reg_x = {
        "visual system": 10_000.0,
        "apical organ": 12_000.0,
        "mushroom body": 28_000.0,
        "head_sensory": 32_000.0,
        "central brain": 48_000.0,
        "ciliomotor": 64_000.0,
        "ventral nerve cord": 115_000.0,
        "trunk_musculature": 160_000.0,
    }

    for i, nm in enumerate(row_names):
        role, ct, scls, reg, side, nt = _classify_platy_cell(nm, ct_meta)
        roles[i] = role
        cell_types.append(ct)
        super_classes.append(scls)
        cell_classes.append(nm)
        sub_classes.append(ct)
        regions.append(reg)
        sides.append(side)
        nts.append(nt)

        bx = reg_x.get(reg, 90_000.0) + ((i * 37) % 19 - 9) * 1_800.0
        by = (-22_000.0 if side == "left" else (22_000.0 if side == "right" else 0.0)) + (
            (i * 17) % 13 - 6
        ) * 1_500.0
        bz = (12_000.0 if role == ROLE_SENSORY else (-12_000.0 if role == ROLE_MOTOR else 0.0)) + (
            (i * 11) % 7 - 3
        ) * 1_400.0
        pos[i] = (bx, by, bz)

        if nm.startswith("PRC"):
            p_prc.append(i)
        if nm.startswith(("chaeMech", "SN-DLSO", "SN-DSO", "antPU", "ventralParaPU", "pygCirrus")):
            p_mech.append(i)
        if nm.startswith(("SNnuch", "SN-NS")):
            p_chemo.append(i)
        if nm.startswith("INMB"):
            p_mb.append(i)
        if nm.startswith(("MNring", "prototroch", "paratroch")) or reg == "ciliomotor":
            p_cilio.append(i)
        if nm.startswith("MUS"):
            p_mus.append(i)
        if nm.startswith(("IN1", "INton", "INsn")):
            p_vis_in.append(i)

    pre = np.asarray(pre_list, dtype=np.int32)
    post = np.asarray(post_list, dtype=np.int32)
    count = np.asarray(cnt_list, dtype=np.int32)
    sign = np.asarray([-1 if "inh" in nts[int(u)] else 1 for u in pre_list], dtype=np.int8)
    kind = np.full(pre.size, SYNAPSE_CHEMICAL, dtype=np.uint8)

    cells = CellTable(
        source_id=source_ids,
        species_names=(dspec.species,),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(cell_types),
        super_class=tuple(super_classes),
        cell_class=tuple(cell_classes),
        cell_sub_class=tuple(sub_classes),
        region=tuple(regions),
        side=tuple(sides),
        role=roles,
        cell_model=models,
        neurotransmitter=tuple(nts),
        position_nm=pos,
        position_source=pos_src,
    )
    edges = EdgeTable(pre=pre, post=post, count=count, sign=sign, kind=kind)
    io_ports = {
        "platy_photoreceptor_prc": _make_port(
            "platy_photoreceptor_prc",
            "sensory",
            "Platynereis eye photoreceptors (PRC)",
            "Larval eye photoreceptor neurons driving phototaxis.",
            p_prc,
            "name startswith 'PRC'",
        ),
        "platy_mechanosensory": _make_port(
            "platy_mechanosensory",
            "sensory",
            "Platynereis mechanosensory girdle SNs",
            "Chaetal, parapodial, and dorsal mechanosensory neurons.",
            p_mech,
            "name startswith chaeMech/SN-DLSO/SN-DSO/antPU",
        ),
        "platy_chemosensory_nuchal": _make_port(
            "platy_chemosensory_nuchal",
            "sensory",
            "Platynereis nuchal/neurosecretory SNs",
            "Head chemosensory and apical neurosecretory neurons.",
            p_chemo,
            "name startswith SNnuch/SN-NS",
        ),
        "platy_mb_interneurons": _make_port(
            "platy_mb_interneurons",
            "circuit",
            "Platynereis mushroom body interneurons",
            "Annelid mushroom-body-like sensory integration interneurons (INMB).",
            p_mb,
            "name startswith 'INMB'",
        ),
        "platy_visual_in": _make_port(
            "platy_visual_in",
            "circuit",
            "Platynereis visual relay INs (IN1/INton)",
            "Primary visual interneurons downstream of PRC.",
            p_vis_in,
            "name startswith IN1/INton/INsn",
        ),
        "platy_ciliomotor_mn": _make_port(
            "platy_ciliomotor_mn",
            "motor",
            "Platynereis ciliomotor & troch cells",
            "Ciliary swimming effectors (prototroch/paratroch) and ring MNs.",
            p_cilio,
            "name startswith MNring/prototroch/paratroch",
        ),
        "platy_muscle_effectors": _make_port(
            "platy_muscle_effectors",
            "motor",
            "Platynereis trunk muscle cells",
            "Longitudinal and oblique muscle effectors directly innervated in the connectome.",
            p_mus,
            "name startswith 'MUS'",
        ),
        "primary_sensory_in": _make_port(
            "primary_sensory_in",
            "sensory",
            "Primary phototactic sensory entry",
            "Alias of platy_photoreceptor_prc.",
            p_prc,
            "alias: platy_photoreceptor_prc",
        ),
        "primary_motor_out": _make_port(
            "primary_motor_out",
            "motor",
            "Primary swimming & muscular output",
            "Union of ciliomotor and muscle effectors.",
            p_cilio + p_mus,
            "union: platy_ciliomotor_mn + platy_muscle_effectors",
        ),
    }
    return _build_pack(
        dataset_id=dspec.dataset_id,
        species=dspec.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=dspec.to_provenance(),
    )


def platynereis_3d_reference_checks(pack: ConnectomePack) -> list[dict[str, Any]]:
    ref = PLATYNEREIS_3D_REFERENCE
    syn = int(pack.edges.count.astype(np.int64).sum())
    if pack.dataset_id != "platynereis_3d":
        ok = (
            pack.io_ports["platy_photoreceptor_prc"].cell_indices.size > 0
            and pack.io_ports["platy_ciliomotor_mn"].cell_indices.size > 0
        )
        return [
            {
                "check": "synthetic_platynereis_phototaxis_circuit",
                "expected": True,
                "actual": ok,
                "ok": ok,
                "note": (
                    "Offline synthetic Platynereis preserves PRC->IN1->ciliomotor/muscle motif."
                ),
            }
        ]
    return [
        {
            "check": "veraszto_2024_connected_matrix_cells",
            "expected": ref["matrix_cells"],
            "actual": pack.cells.num_cells,
            "ok": pack.cells.num_cells == ref["matrix_cells"],
            "note": (
                "1,720 synaptically connected cells in full_connectome_adjacency_matrix.csv "
                "(subset of the 9,162 total somatic/epidermal cells in the full body EM volume)."
            ),
        },
        {
            "check": "veraszto_2024_edges_and_synapses",
            "expected": [ref["directed_edges"], ref["total_synapses"]],
            "actual": [pack.edges.num_edges, syn],
            "ok": pack.edges.num_edges == ref["directed_edges"] and syn == ref["total_synapses"],
            "note": (
                "Exact directed edge and synapse total match against JekelyLab official matrix."
            ),
        },
        {
            "check": "photoreceptor_and_muscle_ports",
            "expected": f"PRC>={ref['min_prc_cells']}, MUS>={ref['min_muscle_cells']}",
            "actual": {
                "prc": int(pack.io_ports["platy_photoreceptor_prc"].cell_indices.size),
                "mus": int(pack.io_ports["platy_muscle_effectors"].cell_indices.size),
                "cilio": int(pack.io_ports["platy_ciliomotor_mn"].cell_indices.size),
            },
            "ok": (
                pack.io_ports["platy_photoreceptor_prc"].cell_indices.size >= ref["min_prc_cells"]
                and pack.io_ports["platy_muscle_effectors"].cell_indices.size
                >= ref["min_muscle_cells"]
            ),
            "note": "Whole-body sensory-to-cilia and sensory-to-muscle effectors resolved.",
        },
    ]


# =========================================================================== #
# S4: Ciona intestinalis Tadpole Larva (Ryan, Lu & Meinertzhagen 2016, eLife)
# =========================================================================== #

CIONA_LARVA_REFERENCE: dict[str, Any] = {
    "cells": 205,
    "directed_edges": 2903,
    "photoreceptor_count": 30,
    "antenna_gravity_count": 2,
    "motor_mn_count": 10,
}


def load_ciona_larva(
    datasets_dir: Path,
    *,
    verify_hashes: bool = True,
    spec: DatasetSpec | None = None,
) -> ConnectomePack:
    """Load verified Ciona intestinalis tadpole CNS connectome (Ryan et al. 2016 eLife)."""
    dspec = spec or CIONA_LARVA_SPEC
    if verify_hashes:
        rep = verify_dataset(dspec.dataset_id, datasets_dir, spec=dspec)
        if not rep["ok"]:
            raise DatasetError(f"dataset {dspec.dataset_id!r} verification failed: {rep['files']}")
    zpath = raw_dir(datasets_dir, dspec.dataset_id) / "cintestinalis.csv.zip"
    if not zpath.is_file():
        raise DatasetError(f"missing {zpath}")

    node_rows = _safe_read_zip_csv(zpath, "nodes.csv")[1:]
    n = len(node_rows)
    source_ids = np.empty(n, dtype=np.int64)
    cell_types: list[str] = []
    super_classes: list[str] = []
    cell_classes: list[str] = []
    sub_classes: list[str] = []
    regions: list[str] = []
    sides: list[str] = []
    roles = np.full(n, ROLE_INTERNEURON, dtype=np.uint8)
    models = np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8)
    nts: list[str] = []
    pos = np.zeros((n, 3), dtype=np.float32)
    pos_src = np.full(n, POS_ANCHOR, dtype=np.uint8)

    p_pr: list[int] = []
    p_ant: list[int] = []
    p_pns: list[int] = []
    p_cor: list[int] = []
    p_mgin: list[int] = []
    p_acin: list[int] = []
    p_mn: list[int] = []
    p_mn_l: list[int] = []
    p_mn_r: list[int] = []

    for row in node_rows:
        idx = int(row[0])
        nm = row[1].strip()
        color = row[2].strip()
        pos_raw = row[3] if len(row) > 3 else ""
        source_ids[idx] = idx + 1

        if nm.endswith("L"):
            side = "left"
        elif nm.endswith("R"):
            side = "right"
        else:
            side = "center"
        sides.append(side)

        if re.fullmatch(r"pr(\d+|[a-g])", nm):
            roles[idx] = ROLE_SENSORY
            ct = "photoreceptor"
            scls = "sensory"
            reg = "sensory_vesicle"
            nt = "GLUT_exc"
            p_pr.append(idx)
        elif nm.startswith("Ant"):
            roles[idx] = ROLE_SENSORY
            ct = "antenna_gravity"
            scls = "sensory"
            reg = "sensory_vesicle"
            nt = "GLUT_exc"
            p_ant.append(idx)
        elif nm.startswith(("pns", "ATEN", "pn")):
            roles[idx] = ROLE_SENSORY
            ct = "peripheral_sensory"
            scls = "sensory"
            reg = "peripheral_epidermis"
            nt = "GLUT_exc"
            p_pns.append(idx)
        elif nm.startswith("coronet"):
            roles[idx] = ROLE_SENSORY
            ct = "coronet"
            scls = "sensory"
            reg = "sensory_vesicle"
            nt = "DA_inh_est"
            p_cor.append(idx)
        elif nm.startswith("MN"):
            roles[idx] = ROLE_MOTOR
            ct = "motor_neuron"
            scls = "motor"
            reg = "motor_ganglion"
            nt = "ACH_exc"
            p_mn.append(idx)
            if nm.endswith("L"):
                p_mn_l.append(idx)
            elif nm.endswith("R"):
                p_mn_r.append(idx)
        elif nm.startswith(("midtail", "PMGN")):
            roles[idx] = ROLE_MOTOR
            ct = "tail_motor"
            scls = "motor"
            reg = "nerve_cord"
            nt = "ACH_exc"
        elif nm.startswith("ACIN"):
            roles[idx] = ROLE_INTERNEURON
            ct = "ACIN_commissural"
            scls = "interneuron"
            reg = "motor_ganglion"
            nt = "GLY_inh"
            p_acin.append(idx)
        elif nm.startswith(("MGIN", "AMG")):
            roles[idx] = ROLE_INTERNEURON
            ct = "MGIN" if nm.startswith("MGIN") else "AMG"
            scls = "interneuron"
            reg = "motor_ganglion"
            nt = "GABA_inh" if nm.startswith("AMG") else "ACH_exc"
            p_mgin.append(idx)
        elif nm.startswith(("lens", "ukn")):
            roles[idx] = ROLE_OTHER
            ct = "accessory"
            scls = "other"
            reg = "sensory_vesicle"
            nt = "ACH_exc"
        else:
            roles[idx] = ROLE_INTERNEURON
            ct = "relay_IN"
            scls = "interneuron"
            reg = "posterior_brain_vesicle"
            nt = "GABA_inh" if nm.startswith(("aaIN", "vacIN")) else "ACH_exc"

        cell_types.append(nm)
        super_classes.append(scls)
        cell_classes.append(ct)
        sub_classes.append(color)
        regions.append(reg)
        nts.append(nt)
        z_nm = -8_000.0 if side == "left" else (8_000.0 if side == "right" else 0.0)
        pos[idx] = _parse_pos_2d(pos_raw, idx, z_nm=z_nm, scale_nm=10_000.0)

    # Edge weights in Ryan et al. 2016 are cumulative synaptic depth (um);
    # scale x10 -> integer 0.1 um units.
    edge_rows = _safe_read_zip_csv(zpath, "edges.csv")[1:]
    agg: dict[tuple[int, int], int] = defaultdict(int)
    for row in edge_rows:
        u, v = int(row[0]), int(row[1])
        depth_um = float(row[2])
        w_int = max(1, round(depth_um * 10.0))
        if 0 <= u < n and 0 <= v < n:
            agg[(u, v)] += w_int

    m = len(agg)
    pre = np.empty(m, dtype=np.int32)
    post = np.empty(m, dtype=np.int32)
    count = np.empty(m, dtype=np.int32)
    sign = np.empty(m, dtype=np.int8)
    kind = np.full(m, SYNAPSE_CHEMICAL, dtype=np.uint8)
    for k, ((u, v), w) in enumerate(sorted(agg.items())):
        pre[k] = u
        post[k] = v
        count[k] = w
        sign[k] = -1 if "inh" in nts[u] else 1

    cells = CellTable(
        source_id=source_ids,
        species_names=(dspec.species,),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(cell_types),
        super_class=tuple(super_classes),
        cell_class=tuple(cell_classes),
        cell_sub_class=tuple(sub_classes),
        region=tuple(regions),
        side=tuple(sides),
        role=roles,
        cell_model=models,
        neurotransmitter=tuple(nts),
        position_nm=pos,
        position_source=pos_src,
    )
    edges = EdgeTable(pre=pre, post=post, count=count, sign=sign, kind=kind)
    io_ports = {
        "ciona_photoreceptors": _make_port(
            "ciona_photoreceptors",
            "sensory",
            "Ciona ocellus photoreceptors (pr)",
            "30 photoreceptor cells mediating shadow/phototactic swimming.",
            p_pr,
            "name matches pr[0-9]+ or pr[a-g]",
        ),
        "ciona_gravity_antenna": _make_port(
            "ciona_gravity_antenna",
            "sensory",
            "Ciona otolith antenna cells (Ant1/Ant2)",
            "Gravity-sensing antenna cells mediating geotaxis.",
            p_ant,
            "name startswith 'Ant'",
        ),
        "ciona_peripheral_sn": _make_port(
            "ciona_peripheral_sn",
            "sensory",
            "Ciona peripheral epidermal SNs",
            "Mechanosensory and chemosensory epidermal neurons (pns/ATEN).",
            p_pns,
            "name startswith pns/ATEN/pn",
        ),
        "ciona_coronet": _make_port(
            "ciona_coronet",
            "sensory",
            "Ciona coronet cells",
            "Dopaminergic coronet sensory/modulatory cells.",
            p_cor,
            "name startswith 'coronet'",
        ),
        "ciona_relay_mgin": _make_port(
            "ciona_relay_mgin",
            "circuit",
            "Ciona motor ganglion INs (MGIN/AMG)",
            "Motor ganglion command & ascending relay interneurons.",
            p_mgin,
            "name startswith MGIN/AMG",
        ),
        "ciona_inhibitory_acin": _make_port(
            "ciona_inhibitory_acin",
            "circuit",
            "Ciona inhibitory ACIN commissural cells",
            "Glycinergic contralateral inhibitory interneurons for alternating tail beats.",
            p_acin,
            "name startswith 'ACIN'",
        ),
        "ciona_motor_mn": _make_port(
            "ciona_motor_mn",
            "motor",
            "Ciona tail motor neurons (MN1-MN5 L/R)",
            "10 primary cholinergic motor neurons driving left/right tail undulation.",
            p_mn,
            "name startswith 'MN'",
        ),
        "ciona_motor_left": _make_port(
            "ciona_motor_left",
            "motor",
            "Ciona left tail MNs",
            "Left-side motor neurons (MN1L..MN5L).",
            p_mn_l,
            "name startswith 'MN' and endswith 'L'",
        ),
        "ciona_motor_right": _make_port(
            "ciona_motor_right",
            "motor",
            "Ciona right tail MNs",
            "Right-side motor neurons (MN1R..MN5R).",
            p_mn_r,
            "name startswith 'MN' and endswith 'R'",
        ),
        "primary_sensory_in": _make_port(
            "primary_sensory_in",
            "sensory",
            "Primary photoreceptor entry",
            "Alias of ciona_photoreceptors.",
            p_pr,
            "alias: ciona_photoreceptors",
        ),
        "primary_motor_out": _make_port(
            "primary_motor_out",
            "motor",
            "Primary tail motor output",
            "Alias of ciona_motor_mn.",
            p_mn,
            "alias: ciona_motor_mn",
        ),
    }
    return _build_pack(
        dataset_id=dspec.dataset_id,
        species=dspec.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=dspec.to_provenance(),
    )


def ciona_larva_reference_checks(pack: ConnectomePack) -> list[dict[str, Any]]:
    ref = CIONA_LARVA_REFERENCE
    if pack.dataset_id != "ciona_larva":
        ok = (
            pack.io_ports["ciona_photoreceptors"].cell_indices.size > 0
            and pack.io_ports["ciona_motor_mn"].cell_indices.size > 0
        )
        return [
            {
                "check": "synthetic_ciona_visuomotor_circuit",
                "expected": True,
                "actual": ok,
                "ok": ok,
                "note": "Offline synthetic Ciona preserves pr/Ant->MGIN/ACIN->MN1-5L/R motif.",
            }
        ]
    return [
        {
            "check": "ryan_2016_cell_count",
            "expected": ref["cells"],
            "actual": pack.cells.num_cells,
            "ok": pack.cells.num_cells == ref["cells"],
            "note": "Ryan et al. (2016) eLife 205 cells in CNS + peripheral/muscle matrix.",
        },
        {
            "check": "ryan_2016_edge_count",
            "expected": ref["directed_edges"],
            "actual": pack.edges.num_edges,
            "ok": pack.edges.num_edges == ref["directed_edges"],
            "note": "Exact 2,903 directed synaptic contacts preserved (weights in 0.1 um units).",
        },
        {
            "check": "ciona_photoreceptors_antenna_and_mn",
            "expected": [
                ref["photoreceptor_count"],
                ref["antenna_gravity_count"],
                ref["motor_mn_count"],
            ],
            "actual": [
                int(pack.io_ports["ciona_photoreceptors"].cell_indices.size),
                int(pack.io_ports["ciona_gravity_antenna"].cell_indices.size),
                int(pack.io_ports["ciona_motor_mn"].cell_indices.size),
            ],
            "ok": (
                int(pack.io_ports["ciona_photoreceptors"].cell_indices.size)
                == ref["photoreceptor_count"]
                and int(pack.io_ports["ciona_gravity_antenna"].cell_indices.size)
                == ref["antenna_gravity_count"]
                and int(pack.io_ports["ciona_motor_mn"].cell_indices.size) == ref["motor_mn_count"]
            ),
            "note": "30 photoreceptors, 2 otolith antenna cells, and 10 MN1-5 L/R motor neurons.",
        },
    ]


# =========================================================================== #
# Offline CPU CI Synthetic Species Packs (S2, S3, S4)
# =========================================================================== #


def make_synthetic_fly_larva_pack() -> ConnectomePack:
    """Offline 96-neuron synthetic Drosophila L1 larval brain with ORN->PN->KC->MBON->DN-VNC."""
    rng = np.random.default_rng(202301)
    layout: list[tuple[str, str, int, str, int]] = [
        # (cell_type, region, role, nt, count)
        ("sensory", "sensory_entry", ROLE_SENSORY, "ACH_est", 16),  # 8 olf + 4 gust + 4 vis
        ("PN", "AL", ROLE_INTERNEURON, "ACH_est", 12),
        ("LN", "AL", ROLE_INTERNEURON, "GABA_est", 8),
        ("KC", "MB", ROLE_INTERNEURON, "ACH_est", 24),
        ("MBON", "MB", ROLE_INTERNEURON, "ACH_est", 8),
        ("MBIN", "MB", ROLE_INTERNEURON, "DA_mod", 6),
        ("LHN", "LH", ROLE_INTERNEURON, "ACH_est", 8),
        ("DN-VNC", "DN-VNC", ROLE_MOTOR, "ACH_est", 10),
        ("DN-SEZ", "DN-SEZ", ROLE_MOTOR, "ACH_est", 4),
    ]
    n = sum(c for *_, c in layout)
    ct_list: list[str] = []
    sc_list: list[str] = []
    cc_list: list[str] = []
    sub_list: list[str] = []
    reg_list: list[str] = []
    side_list: list[str] = []
    roles = np.empty(n, dtype=np.uint8)
    models = np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8)
    nts: list[str] = []
    pos = np.zeros((n, 3), dtype=np.float32)

    groups: dict[str, list[int]] = defaultdict(list)
    idx = 0
    for ct, reg, role, nt, count in layout:
        for k in range(count):
            side = "left" if k % 2 == 0 else "right"
            ct_list.append(ct)
            sc_list.append(
                "sensory"
                if role == ROLE_SENSORY
                else ("descending_output" if role == ROLE_MOTOR else "central_brain")
            )
            cc_list.append(ct)
            sub_list.append(ct)
            reg_list.append(reg)
            side_list.append(side)
            roles[idx] = role
            nts.append(nt)
            pos[idx] = (
                idx * 2_000.0,
                (-15_000.0 if side == "left" else 15_000.0),
                (k % 5) * 4_000.0,
            )
            groups[ct].append(idx)
            idx += 1

    olf = groups["sensory"][:8]
    gust = groups["sensory"][8:12]
    vis = groups["sensory"][12:16]
    pn = groups["PN"]
    ln = groups["LN"]
    kc = groups["KC"]
    mbon = groups["MBON"]
    mbin = groups["MBIN"]
    lhn = groups["LHN"]
    dn_vnc = groups["DN-VNC"]
    dn_sez = groups["DN-SEZ"]

    edge_map: dict[tuple[int, int], tuple[int, int]] = {}

    def connect(srcs: list[int], dsts: list[int], w: int, sgn: int, prob: float = 0.6) -> None:
        for u in srcs:
            for v in dsts:
                if u != v and float(rng.random()) < prob:
                    edge_map[(u, v)] = (w, sgn)

    # Strong deterministic olfactory pathway ORN -> PN -> KC/LHN -> MBON -> DN-VNC
    connect(olf, pn, 32, 1, 0.85)
    connect(olf, ln, 14, 1, 0.6)
    connect(ln, pn, 6, -1, 0.45)
    connect(pn, kc, 26, 1, 0.55)
    connect(pn, lhn, 24, 1, 0.7)
    connect(kc, mbon, 22, 1, 0.65)
    connect(mbin, kc, 12, 1, 0.5)
    connect(mbin, mbon, 10, 1, 0.5)
    connect(mbon, dn_vnc, 30, 1, 0.8)
    connect(lhn, dn_vnc, 28, 1, 0.75)
    connect(gust, dn_sez, 34, 1, 0.9)
    connect(vis, lhn, 24, 1, 0.75)

    m = len(edge_map)
    pre = np.empty(m, dtype=np.int32)
    post = np.empty(m, dtype=np.int32)
    cnt = np.empty(m, dtype=np.int32)
    sgn = np.empty(m, dtype=np.int8)
    for k, ((u, v), (w, sg)) in enumerate(sorted(edge_map.items())):
        pre[k], post[k], cnt[k], sgn[k] = u, v, w, sg

    cells = CellTable(
        source_id=np.arange(1, n + 1, dtype=np.int64),
        species_names=("drosophila_melanogaster_larva_l1_synthetic",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(ct_list),
        super_class=tuple(sc_list),
        cell_class=tuple(cc_list),
        cell_sub_class=tuple(sub_list),
        region=tuple(reg_list),
        side=tuple(side_list),
        role=roles,
        cell_model=models,
        neurotransmitter=tuple(nts),
        position_nm=pos,
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    edges = EdgeTable(
        pre=pre, post=post, count=cnt, sign=sgn, kind=np.full(m, SYNAPSE_CHEMICAL, dtype=np.uint8)
    )
    io_ports = {
        "larva_olfactory_sn": _make_port(
            "larva_olfactory_sn",
            "sensory",
            "Larval ORNs",
            "Synthetic larval ORNs",
            olf,
            "synthetic",
        ),
        "larva_gustatory_sn": _make_port(
            "larva_gustatory_sn",
            "sensory",
            "Larval GRNs",
            "Synthetic larval GRNs",
            gust,
            "synthetic",
        ),
        "larva_visual_sn": _make_port(
            "larva_visual_sn",
            "sensory",
            "Larval visual SNs",
            "Synthetic larval visual",
            vis,
            "synthetic",
        ),
        "larva_mechano_sn": _make_port(
            "larva_mechano_sn",
            "sensory",
            "Larval mechano SNs",
            "Synthetic mechano",
            gust[:2],
            "synthetic",
        ),
        "larva_pn": _make_port(
            "larva_pn", "circuit", "Larval PNs", "Synthetic larval PNs", pn, "synthetic"
        ),
        "larva_kc": _make_port(
            "larva_kc", "circuit", "Larval KCs", "Synthetic larval KCs", kc, "synthetic"
        ),
        "larva_mbon": _make_port(
            "larva_mbon", "circuit", "Larval MBONs", "Synthetic larval MBONs", mbon, "synthetic"
        ),
        "larva_mbin": _make_port(
            "larva_mbin", "circuit", "Larval MBINs", "Synthetic larval MBINs", mbin, "synthetic"
        ),
        "larva_dn_vnc": _make_port(
            "larva_dn_vnc", "motor", "Larval DN-VNC", "Synthetic DN-VNC", dn_vnc, "synthetic"
        ),
        "larva_dn_sez": _make_port(
            "larva_dn_sez", "motor", "Larval DN-SEZ", "Synthetic DN-SEZ", dn_sez, "synthetic"
        ),
        "olfactory_sn": _make_port(
            "olfactory_sn", "sensory", "Olfactory entry", "Alias", olf, "alias"
        ),
        "primary_motor_out": _make_port(
            "primary_motor_out", "motor", "Primary motor out", "Alias", dn_vnc, "alias"
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_fly_larva",
        species="drosophila_melanogaster_larva_l1_synthetic",
        version="synthetic_v1",
        citation="Offline synthetic larval Drosophila test pack (inspired by Winding et al. 2023).",
        licence_note="Generated offline inside ALife Lab for CPU CI verification.",
        files=(),
    )
    return _build_pack(
        dataset_id="synthetic_fly_larva",
        species=prov.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
    )


def make_synthetic_platynereis_pack() -> ConnectomePack:
    """Offline 72-cell synthetic Platynereis 3-day larva pack (MODEL_GRADED + visual-ciliomotor)."""
    rng = np.random.default_rng(202403)
    specs = [
        ("PRC", "visual system", ROLE_SENSORY, "ACH_exc", 12),
        ("chaeMech", "head_sensory", ROLE_SENSORY, "ACH_exc", 8),
        ("IN1", "visual system", ROLE_INTERNEURON, "ACH_exc", 10),
        ("INMB", "mushroom body", ROLE_INTERNEURON, "ACH_exc", 10),
        ("INcomm", "ventral nerve cord", ROLE_INTERNEURON, "GABA_inh", 8),
        ("MNring", "ciliomotor", ROLE_MOTOR, "ACH_exc", 12),
        ("MUSlong", "trunk_musculature", ROLE_MOTOR, "ACH_exc", 12),
    ]
    n = sum(c for *_, c in specs)
    ct_l, sc_l, cc_l, sub_l, reg_l, side_l, nt_l = [], [], [], [], [], [], []
    roles = np.empty(n, dtype=np.uint8)
    models = np.full(n, MODEL_GRADED, dtype=np.uint8)
    pos = np.zeros((n, 3), dtype=np.float32)
    groups: dict[str, list[int]] = defaultdict(list)
    idx = 0
    for ct, reg, role, nt, cnt in specs:
        for k in range(cnt):
            side = "left" if k % 2 == 0 else "right"
            ct_l.append(ct)
            sc_l.append(
                "sensory"
                if role == ROLE_SENSORY
                else ("motor" if role == ROLE_MOTOR else "interneuron")
            )
            cc_l.append(f"{ct}_{k}")
            sub_l.append(ct)
            reg_l.append(reg)
            side_l.append(side)
            roles[idx] = role
            nt_l.append(nt)
            pos[idx] = (idx * 2_500.0, (-18_000.0 if side == "left" else 18_000.0), 0.0)
            groups[ct].append(idx)
            idx += 1

    edge_map: dict[tuple[int, int], tuple[int, int, int]] = {}

    def conn(
        srcs: list[int],
        dsts: list[int],
        w: int,
        sgn: int,
        p: float = 0.7,
        knd: int = SYNAPSE_CHEMICAL,
    ) -> None:
        for u in srcs:
            for v in dsts:
                if u != v and float(rng.random()) < p:
                    edge_map[(u, v)] = (w, sgn, knd)

    conn(groups["PRC"], groups["IN1"], 14, 1, 0.85)
    conn(groups["PRC"], groups["INMB"], 10, 1, 0.6)
    conn(groups["chaeMech"], groups["INcomm"], 12, 1, 0.75)
    conn(groups["IN1"], groups["MNring"], 16, 1, 0.8)
    conn(groups["IN1"], groups["MUSlong"], 14, 1, 0.7)
    conn(groups["INMB"], groups["MNring"], 10, 1, 0.6)
    conn(groups["INcomm"], groups["MUSlong"], 8, -1, 0.6)
    # Local gap junctions among prototroch ciliomotor ring cells
    mn = groups["MNring"]
    for k in range(len(mn)):
        u, v = mn[k], mn[(k + 1) % len(mn)]
        edge_map[(u, v)] = (6, 1, SYNAPSE_GAP_JUNCTION)
        edge_map[(v, u)] = (6, 1, SYNAPSE_GAP_JUNCTION)

    m = len(edge_map)
    pre = np.empty(m, dtype=np.int32)
    post = np.empty(m, dtype=np.int32)
    cnt_a = np.empty(m, dtype=np.int32)
    sgn_a = np.empty(m, dtype=np.int8)
    knd_a = np.empty(m, dtype=np.uint8)
    for k, ((u, v), (w, sg, kd)) in enumerate(sorted(edge_map.items())):
        pre[k], post[k], cnt_a[k], sgn_a[k], knd_a[k] = u, v, w, sg, kd

    cells = CellTable(
        source_id=np.arange(1, n + 1, dtype=np.int64),
        species_names=("platynereis_dumerilii_3day_larva_synthetic",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(ct_l),
        super_class=tuple(sc_l),
        cell_class=tuple(cc_l),
        cell_sub_class=tuple(sub_l),
        region=tuple(reg_l),
        side=tuple(side_l),
        role=roles,
        cell_model=models,
        neurotransmitter=tuple(nt_l),
        position_nm=pos,
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    edges = EdgeTable(pre=pre, post=post, count=cnt_a, sign=sgn_a, kind=knd_a)
    io_ports = {
        "platy_photoreceptor_prc": _make_port(
            "platy_photoreceptor_prc", "sensory", "PRC", "Synthetic PRC", groups["PRC"], "synthetic"
        ),
        "platy_mechanosensory": _make_port(
            "platy_mechanosensory",
            "sensory",
            "Mechano",
            "Synthetic mechano",
            groups["chaeMech"],
            "synthetic",
        ),
        "platy_chemosensory_nuchal": _make_port(
            "platy_chemosensory_nuchal",
            "sensory",
            "Chemo",
            "Synthetic chemo",
            groups["chaeMech"][:4],
            "synthetic",
        ),
        "platy_mb_interneurons": _make_port(
            "platy_mb_interneurons",
            "circuit",
            "INMB",
            "Synthetic INMB",
            groups["INMB"],
            "synthetic",
        ),
        "platy_visual_in": _make_port(
            "platy_visual_in", "circuit", "IN1", "Synthetic IN1", groups["IN1"], "synthetic"
        ),
        "platy_ciliomotor_mn": _make_port(
            "platy_ciliomotor_mn",
            "motor",
            "Ciliomotor",
            "Synthetic MNring",
            groups["MNring"],
            "synthetic",
        ),
        "platy_muscle_effectors": _make_port(
            "platy_muscle_effectors",
            "motor",
            "Muscles",
            "Synthetic MUS",
            groups["MUSlong"],
            "synthetic",
        ),
        "primary_sensory_in": _make_port(
            "primary_sensory_in", "sensory", "Primary sensory", "Alias", groups["PRC"], "alias"
        ),
        "primary_motor_out": _make_port(
            "primary_motor_out",
            "motor",
            "Primary motor",
            "Alias",
            groups["MNring"] + groups["MUSlong"],
            "alias",
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_platynereis",
        species="platynereis_dumerilii_3day_larva_synthetic",
        version="synthetic_v1",
        citation=(
            "Offline synthetic Platynereis larva test pack (inspired by Verasztó et al. 2024)."
        ),
        licence_note="Generated offline inside ALife Lab for CPU CI verification.",
        files=(),
    )
    return _build_pack(
        dataset_id="synthetic_platynereis",
        species=prov.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
    )


def make_synthetic_ciona_pack() -> ConnectomePack:
    """Offline 48-cell synthetic Ciona tadpole CNS pack (pr/Ant -> MGIN/ACIN -> MN1-5 L/R)."""
    rng = np.random.default_rng(201604)
    specs = [
        ("pr", "sensory_vesicle", ROLE_SENSORY, "GLUT_exc", 12),
        ("Ant", "sensory_vesicle", ROLE_SENSORY, "GLUT_exc", 2),
        ("pns", "peripheral_epidermis", ROLE_SENSORY, "GLUT_exc", 6),
        ("MGIN", "motor_ganglion", ROLE_INTERNEURON, "ACH_exc", 12),
        ("ACIN", "motor_ganglion", ROLE_INTERNEURON, "GLY_inh", 6),
        ("MN", "motor_ganglion", ROLE_MOTOR, "ACH_exc", 10),
    ]
    n = sum(c for *_, c in specs)
    ct_l, sc_l, cc_l, sub_l, reg_l, side_l, nt_l = [], [], [], [], [], [], []
    roles = np.empty(n, dtype=np.uint8)
    models = np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8)
    pos = np.zeros((n, 3), dtype=np.float32)
    groups: dict[str, list[int]] = defaultdict(list)
    idx = 0
    for ct, reg, role, nt, cnt in specs:
        for k in range(cnt):
            side = "left" if k % 2 == 0 else "right"
            nm = (
                f"{ct}{k // 2 + 1}{'L' if side == 'left' else 'R'}"
                if ct in ("MN", "MGIN", "ACIN")
                else f"{ct}{k + 1}"
            )
            ct_l.append(nm)
            sc_l.append(
                "sensory"
                if role == ROLE_SENSORY
                else ("motor" if role == ROLE_MOTOR else "interneuron")
            )
            cc_l.append(ct)
            sub_l.append(ct)
            reg_l.append(reg)
            side_l.append(side)
            roles[idx] = role
            nt_l.append(nt)
            pos[idx] = (idx * 2_000.0, (-10_000.0 if side == "left" else 10_000.0), 0.0)
            groups[ct].append(idx)
            idx += 1

    mn_l = [i for i in groups["MN"] if side_l[i] == "left"]
    mn_r = [i for i in groups["MN"] if side_l[i] == "right"]
    mgin_l = [i for i in groups["MGIN"] if side_l[i] == "left"]
    mgin_r = [i for i in groups["MGIN"] if side_l[i] == "right"]
    acin_l = [i for i in groups["ACIN"] if side_l[i] == "left"]
    acin_r = [i for i in groups["ACIN"] if side_l[i] == "right"]

    edge_map: dict[tuple[int, int], tuple[int, int]] = {}

    def conn(srcs: list[int], dsts: list[int], w: int, sgn: int, p: float = 0.8) -> None:
        for u in srcs:
            for v in dsts:
                if u != v and float(rng.random()) < p:
                    edge_map[(u, v)] = (w, sgn)

    conn(groups["pr"], groups["MGIN"], 28, 1, 0.85)
    conn(groups["Ant"], groups["MGIN"], 32, 1, 0.9)
    conn(groups["pns"], groups["MGIN"], 20, 1, 0.7)
    conn(mgin_l, mn_l, 34, 1, 0.9)
    conn(mgin_r, mn_r, 34, 1, 0.9)
    conn(mgin_l, acin_l, 22, 1, 0.85)
    conn(mgin_r, acin_r, 22, 1, 0.85)
    # Contralateral ACIN inhibition drives alternating left/right tail beats
    conn(acin_l, mn_r, 20, -1, 0.9)
    conn(acin_r, mn_l, 20, -1, 0.9)

    m = len(edge_map)
    pre = np.empty(m, dtype=np.int32)
    post = np.empty(m, dtype=np.int32)
    cnt_a = np.empty(m, dtype=np.int32)
    sgn_a = np.empty(m, dtype=np.int8)
    for k, ((u, v), (w, sg)) in enumerate(sorted(edge_map.items())):
        pre[k], post[k], cnt_a[k], sgn_a[k] = u, v, w, sg

    cells = CellTable(
        source_id=np.arange(1, n + 1, dtype=np.int64),
        species_names=("ciona_intestinalis_tadpole_larva_synthetic",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(ct_l),
        super_class=tuple(sc_l),
        cell_class=tuple(cc_l),
        cell_sub_class=tuple(sub_l),
        region=tuple(reg_l),
        side=tuple(side_l),
        role=roles,
        cell_model=models,
        neurotransmitter=tuple(nt_l),
        position_nm=pos,
        position_source=np.full(n, POS_SOMA, dtype=np.uint8),
    )
    edges = EdgeTable(
        pre=pre,
        post=post,
        count=cnt_a,
        sign=sgn_a,
        kind=np.full(m, SYNAPSE_CHEMICAL, dtype=np.uint8),
    )
    io_ports = {
        "ciona_photoreceptors": _make_port(
            "ciona_photoreceptors",
            "sensory",
            "Photoreceptors",
            "Synthetic pr",
            groups["pr"],
            "synthetic",
        ),
        "ciona_gravity_antenna": _make_port(
            "ciona_gravity_antenna",
            "sensory",
            "Antenna gravity",
            "Synthetic Ant",
            groups["Ant"],
            "synthetic",
        ),
        "ciona_peripheral_sn": _make_port(
            "ciona_peripheral_sn",
            "sensory",
            "Peripheral SNs",
            "Synthetic pns",
            groups["pns"],
            "synthetic",
        ),
        "ciona_coronet": _make_port(
            "ciona_coronet",
            "sensory",
            "Coronet",
            "Synthetic coronet",
            groups["pns"][:2],
            "synthetic",
        ),
        "ciona_relay_mgin": _make_port(
            "ciona_relay_mgin", "circuit", "MGIN", "Synthetic MGIN", groups["MGIN"], "synthetic"
        ),
        "ciona_inhibitory_acin": _make_port(
            "ciona_inhibitory_acin",
            "circuit",
            "ACIN",
            "Synthetic ACIN",
            groups["ACIN"],
            "synthetic",
        ),
        "ciona_motor_mn": _make_port(
            "ciona_motor_mn", "motor", "Tail MNs", "Synthetic MN", groups["MN"], "synthetic"
        ),
        "ciona_motor_left": _make_port(
            "ciona_motor_left", "motor", "Left tail MNs", "Synthetic MN-L", mn_l, "synthetic"
        ),
        "ciona_motor_right": _make_port(
            "ciona_motor_right", "motor", "Right tail MNs", "Synthetic MN-R", mn_r, "synthetic"
        ),
        "primary_sensory_in": _make_port(
            "primary_sensory_in", "sensory", "Primary sensory", "Alias", groups["pr"], "alias"
        ),
        "primary_motor_out": _make_port(
            "primary_motor_out", "motor", "Primary motor", "Alias", groups["MN"], "alias"
        ),
    }
    prov = DatasetProvenance(
        dataset_id="synthetic_ciona",
        species="ciona_intestinalis_tadpole_larva_synthetic",
        version="synthetic_v1",
        citation="Offline synthetic Ciona tadpole test pack (inspired by Ryan et al. 2016 eLife).",
        licence_note="Generated offline inside ALife Lab for CPU CI verification.",
        files=(),
    )
    return _build_pack(
        dataset_id="synthetic_ciona",
        species=prov.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
    )
