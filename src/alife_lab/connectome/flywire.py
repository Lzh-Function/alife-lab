"""FlyWire v783 adult female Drosophila connectome species-pack loader (F0).

Reads only verified tabular files (Parquet / CSV / TSV / CSV.GZ) — never pickles:
- ``Completeness_783.csv`` and ``Connectivity_783.parquet`` (Shiu et al. 2024)
- ``Supplemental_file1_neuron_annotations.tsv`` (Schlegel et al. 2024 / flywire_annotations)
- ``classification.csv.gz``, ``coordinates.csv.gz``, ``neurons.csv.gz``,
  ``neuropil_synapse_table.csv.gz`` (FlyWire Codex v783)

Coordinates are normalized to FAFB nanometres (``float32``):
- Soma voxel coordinates from ``Supplemental_file1_neuron_annotations.tsv`` multiplied by
  ``(4, 4, 40) nm`` when available,
- falling back to anchor voxel coordinates ``pos_x/y/z * (4, 4, 40) nm``,
- falling back to Codex ``coordinates.csv.gz`` (already in nm).
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from alife_lab.connectome.registry import (
    FLYWIRE_783_SPEC,
    DatasetError,
    DatasetSpec,
    raw_dir,
    verify_dataset,
)
from alife_lab.connectome.schema import (
    MODEL_SPIKING_LIF,
    POS_ANCHOR,
    POS_CODEX,
    POS_MISSING,
    POS_SOMA,
    ROLE_INTERNEURON,
    ROLE_MOTOR,
    ROLE_OTHER,
    ROLE_SENSORY,
    SYNAPSE_CHEMICAL,
    CellTable,
    ConnectomePack,
    EdgeTable,
    IOPort,
    compute_pack_digest,
    summarize_pack,
)

# FAFB v783 voxel size in nanometres (x, y, z).
FAFB_VOXEL_NM = (4.0, 4.0, 40.0)

# Published reference values used for F0 verification checks.
# Sources: Shiu et al. (2024) Nature 634:210-219 (v783 table) and
# Dorkenwald et al. (2024) Nature (FlyWire v783 whole brain: 139,255 proofread neurons,
# ~54.5M chemical synapses).
FLYWIRE_783_REFERENCE: dict[str, Any] = {
    "shiu_v783_neurons": 138_639,
    "shiu_v783_edges": 15_091_983,
    "shiu_v783_synapses": 54_492_922,
    "dorkenwald_2024_proofread_neurons": 139_255,
    "dorkenwald_2024_synapses_approx": 54_500_000,
    "mn9_root_ids": (720575940660219265, 720575940618238523),
}

_NT_NORMALIZE = {
    "ACH": "acetylcholine",
    "GABA": "gaba",
    "GLUT": "glutamate",
    "DA": "dopamine",
    "SER": "serotonin",
    "OCT": "octopamine",
}


def _map_role(super_class: str, flow: str) -> int:
    sc = super_class.lower()
    if sc in ("sensory", "sensory_ascending"):
        return ROLE_SENSORY
    if sc == "motor":
        return ROLE_MOTOR
    if sc in (
        "central",
        "optic",
        "visual_projection",
        "visual_centrifugal",
        "ascending",
        "descending",
    ):
        return ROLE_INTERNEURON
    fl = flow.lower()
    if fl == "afferent":
        return ROLE_SENSORY
    if fl == "efferent":
        return ROLE_MOTOR
    if fl == "intrinsic":
        return ROLE_INTERNEURON
    return ROLE_OTHER


def _parse_float(raw: str) -> float | None:
    s = raw.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_codex_position(raw: str) -> tuple[float, float, float] | None:
    # Codex coordinates.csv.gz formats positions like "[352484 175164 229040]".
    parts = raw.strip().strip("[]").split()
    if len(parts) != 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def _load_completeness(path: Path) -> np.ndarray[Any, np.dtype[np.int64]]:
    ids: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header: ['', 'Completed']
        for row in reader:
            if not row or not row[0].strip():
                continue
            ids.append(int(row[0]))
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size == 0:
        raise DatasetError(f"{path.name}: no neurons found")
    if np.unique(arr).size != arr.size:
        raise DatasetError(f"{path.name}: duplicate neuron root_ids found")
    return arr


def _load_connectivity(path: Path, root_ids: np.ndarray[Any, np.dtype[np.int64]]) -> EdgeTable:
    tbl = pq.read_table(
        path,
        columns=[
            "Presynaptic_ID",
            "Postsynaptic_ID",
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Connectivity",
            "Excitatory",
            "Excitatory x Connectivity",
        ],
    )
    pre_id = tbl["Presynaptic_ID"].to_numpy(zero_copy_only=False)
    post_id = tbl["Postsynaptic_ID"].to_numpy(zero_copy_only=False)
    pre_idx = tbl["Presynaptic_Index"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    post_idx = tbl["Postsynaptic_Index"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    count = tbl["Connectivity"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    sign = tbl["Excitatory"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    exc_x_conn = tbl["Excitatory x Connectivity"].to_numpy(zero_copy_only=False)

    n = int(root_ids.shape[0])
    if pre_idx.size > 0:
        if int(pre_idx.min()) < 0 or int(pre_idx.max()) >= n:
            raise DatasetError(f"{path.name}: Presynaptic_Index out of bounds")
        if int(post_idx.min()) < 0 or int(post_idx.max()) >= n:
            raise DatasetError(f"{path.name}: Postsynaptic_Index out of bounds")
        if not np.array_equal(pre_id, root_ids[pre_idx]):
            raise DatasetError(f"{path.name}: Presynaptic_ID does not match Completeness index")
        if not np.array_equal(post_id, root_ids[post_idx]):
            raise DatasetError(f"{path.name}: Postsynaptic_ID does not match Completeness index")
        if not np.array_equal(sign.astype(np.int64) * count.astype(np.int64), exc_x_conn):
            raise DatasetError(
                f"{path.name}: Excitatory * Connectivity != Excitatory x Connectivity"
            )

    kind = np.full(pre_idx.shape[0], SYNAPSE_CHEMICAL, dtype=np.uint8)
    edges = EdgeTable(
        pre=np.ascontiguousarray(pre_idx, dtype=np.int32),
        post=np.ascontiguousarray(post_idx, dtype=np.int32),
        count=np.ascontiguousarray(count, dtype=np.int32),
        sign=np.ascontiguousarray(sign, dtype=np.int8),
        kind=kind,
    )
    edges.validate(n)
    return edges


def _load_annotations(path: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rid_str = row.get("root_id", "").strip()
            if not rid_str:
                continue
            out[int(rid_str)] = row
    return out


def _load_codex_classification(path: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rid_str = row.get("root_id", "").strip()
            if not rid_str:
                continue
            out[int(rid_str)] = row
    return out


def _load_codex_neurons(path: Path) -> dict[int, tuple[str, str]]:
    """Return ``root_id -> (group, normalized_nt)``."""
    out: dict[int, tuple[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rid_str = row.get("root_id", "").strip()
            if not rid_str:
                continue
            group = row.get("group", "").strip()
            nt_raw = row.get("nt_type", "").strip().upper()
            out[int(rid_str)] = (group, _NT_NORMALIZE.get(nt_raw, nt_raw.lower()))
    return out


def _load_codex_coordinates(
    path: Path, wanted_ids: set[int]
) -> dict[int, tuple[float, float, float]]:
    """Return first nm coordinate per wanted ``root_id`` (only for fallback cells)."""
    if not wanted_ids:
        return {}
    out: dict[int, tuple[float, float, float]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rid_str = row.get("root_id", "").strip()
            if not rid_str:
                continue
            rid = int(rid_str)
            if rid in wanted_ids and rid not in out:
                pos = _parse_codex_position(row.get("position", ""))
                if pos is not None:
                    out[rid] = pos
                    if len(out) == len(wanted_ids):
                        break
    return out


def _load_primary_regions(path: Path) -> dict[int, str]:
    """Compute primary neuropil per neuron as argmax(input + output synapses), excluding UNASGD."""
    tbl = pa_csv.read_csv(path)
    names = tbl.column_names
    in_prefix = "input synapses in "
    out_prefix = "output synapses in "
    in_regions = [c[len(in_prefix) :] for c in names if c.startswith(in_prefix)]
    out_regions = set(c[len(out_prefix) :] for c in names if c.startswith(out_prefix))
    regions = [r for r in in_regions if r in out_regions and r != "UNASGD"]
    if not regions:
        return {}

    root_ids = tbl["root_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    nrows = int(root_ids.shape[0])
    if nrows == 0:
        return {}

    totals = np.empty((nrows, len(regions)), dtype=np.int32)
    for j, region in enumerate(regions):
        in_col = (
            tbl[f"{in_prefix}{region}"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        )
        out_col = (
            tbl[f"{out_prefix}{region}"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        )
        totals[:, j] = in_col + out_col

    best_idx = np.argmax(totals, axis=1)
    best_val = totals[np.arange(nrows), best_idx]
    result: dict[int, str] = {}
    for i in range(nrows):
        if int(best_val[i]) > 0:
            result[int(root_ids[i])] = regions[int(best_idx[i])]
    return result


def _build_io_ports(cells: CellTable) -> dict[str, IOPort]:
    n = cells.num_cells

    def _select(pred: Any) -> np.ndarray[Any, np.dtype[np.int32]]:
        return np.asarray([i for i in range(n) if pred(i)], dtype=np.int32)

    ports = [
        IOPort(
            port_id="sugar_grn",
            direction="sensory",
            label="Sugar GRNs",
            description="Gustatory receptor neurons labelled cell_sub_class == 'sugar'.",
            cell_indices=_select(
                lambda i: cells.cell_class[i] == "gustatory" and cells.cell_sub_class[i] == "sugar"
            ),
            selection_rule="cell_class == 'gustatory' and cell_sub_class == 'sugar'",
        ),
        IOPort(
            port_id="sugar_and_low_salt_grn",
            direction="sensory",
            label="Sugar & sugar/low-salt GRNs",
            description="Gustatory receptor neurons labelled sugar or sugar/low_salt.",
            cell_indices=_select(
                lambda i: (
                    cells.cell_class[i] == "gustatory"
                    and cells.cell_sub_class[i] in ("sugar", "sugar/low_salt")
                )
            ),
            selection_rule=(
                "cell_class == 'gustatory' and cell_sub_class in ('sugar', 'sugar/low_salt')"
            ),
        ),
        IOPort(
            port_id="water_grn",
            direction="sensory",
            label="Water GRNs",
            description="Gustatory receptor neurons labelled cell_sub_class == 'water'.",
            cell_indices=_select(
                lambda i: cells.cell_class[i] == "gustatory" and cells.cell_sub_class[i] == "water"
            ),
            selection_rule="cell_class == 'gustatory' and cell_sub_class == 'water'",
        ),
        IOPort(
            port_id="bitter_grn",
            direction="sensory",
            label="Bitter GRNs",
            description="Gustatory receptor neurons labelled cell_sub_class == 'bitter'.",
            cell_indices=_select(
                lambda i: cells.cell_class[i] == "gustatory" and cells.cell_sub_class[i] == "bitter"
            ),
            selection_rule="cell_class == 'gustatory' and cell_sub_class == 'bitter'",
        ),
        IOPort(
            port_id="antennal_grooming_mechanosensory",
            direction="sensory",
            label="Antennal grooming mechanosensory",
            description="Mechanosensory neurons labelled cell_sub_class == 'grooming'.",
            cell_indices=_select(
                lambda i: (
                    cells.cell_class[i] == "mechanosensory"
                    and cells.cell_sub_class[i] == "grooming"
                )
            ),
            selection_rule="cell_class == 'mechanosensory' and cell_sub_class == 'grooming'",
        ),
        IOPort(
            port_id="jo_mechanosensory",
            direction="sensory",
            label="Johnston's organ mechanosensory (JO-*)",
            description="Mechanosensory neurons with cell_type starting with 'JO-'.",
            cell_indices=_select(lambda i: cells.cell_type[i].startswith("JO-")),
            selection_rule="cell_type.startswith('JO-')",
        ),
        IOPort(
            port_id="olfactory_sensory",
            direction="sensory",
            label="Olfactory receptor neurons",
            description="Sensory neurons labelled cell_class == 'olfactory'.",
            cell_indices=_select(lambda i: cells.cell_class[i] == "olfactory"),
            selection_rule="cell_class == 'olfactory'",
        ),
        IOPort(
            port_id="mn9_proboscis_extension",
            direction="motor",
            label="MN9 proboscis extension motor neurons (CB0701)",
            description=(
                "Pair of MN9 motor neurons (cell_type == 'CB0701', "
                "root_ids 720575940660219265 / 720575940618238523 in FlyWire v783)."
            ),
            cell_indices=_select(lambda i: cells.cell_type[i] == "CB0701"),
            selection_rule="cell_type == 'CB0701'",
        ),
        IOPort(
            port_id="proboscis_motor",
            direction="motor",
            label="Proboscis motor neurons",
            description="Motor neurons labelled cell_sub_class == 'proboscis_motor_neuron'.",
            cell_indices=_select(
                lambda i: (
                    cells.super_class[i] == "motor"
                    and cells.cell_sub_class[i] == "proboscis_motor_neuron"
                )
            ),
            selection_rule="super_class == 'motor' and cell_sub_class == 'proboscis_motor_neuron'",
        ),
        IOPort(
            port_id="mushroom_body_kc",
            direction="circuit",
            label="Mushroom body Kenyon cells",
            description="Kenyon cells (cell_class == 'Kenyon_Cell'), olfactory memory layer.",
            cell_indices=_select(lambda i: cells.cell_class[i] == "Kenyon_Cell"),
            selection_rule="cell_class == 'Kenyon_Cell'",
        ),
        IOPort(
            port_id="mushroom_body_mbon",
            direction="circuit",
            label="Mushroom body output neurons (MBON)",
            description="Mushroom body output neurons (cell_class == 'MBON').",
            cell_indices=_select(lambda i: cells.cell_class[i] == "MBON"),
            selection_rule="cell_class == 'MBON'",
        ),
        IOPort(
            port_id="mushroom_body_dan",
            direction="circuit",
            label="Dopaminergic neurons (DAN)",
            description="Dopaminergic reinforcement neurons (cell_class == 'DAN').",
            cell_indices=_select(lambda i: cells.cell_class[i] == "DAN"),
            selection_rule="cell_class == 'DAN'",
        ),
        IOPort(
            port_id="odor_a_kc",
            direction="sensory",
            label="Odor A sparse KC ensemble (KCg-m #0..249)",
            description=(
                "Deterministic sparse subset of 250 KCg-m Kenyon cells (~4.8% of KCs) "
                "representing Odor A for F4 associative conditioning."
            ),
            cell_indices=np.asarray(
                [i for i in range(n) if cells.cell_type[i] == "KCg-m"][:250], dtype=np.int32
            ),
            selection_rule="first 250 cells with cell_type == 'KCg-m'",
        ),
        IOPort(
            port_id="odor_b_kc",
            direction="sensory",
            label="Odor B sparse KC ensemble (KCg-m #250..499)",
            description=(
                "Disjoint sparse subset of 250 KCg-m Kenyon cells (~4.8% of KCs) "
                "representing control Odor B for F4 odor-specificity tests."
            ),
            cell_indices=np.asarray(
                [i for i in range(n) if cells.cell_type[i] == "KCg-m"][250:500], dtype=np.int32
            ),
            selection_rule="cells 250..499 with cell_type == 'KCg-m'",
        ),
        IOPort(
            port_id="ppl1_punishment_dan",
            direction="sensory",
            label="PPL101 punishment DANs (gamma1pedc)",
            description=(
                "Bilateral PPL101 aversive dopaminergic neurons innervating MBON11 "
                "(gamma1pedc compartment, punishment reinforcement)."
            ),
            cell_indices=_select(lambda i: cells.cell_type[i] == "PPL101"),
            selection_rule="cell_type == 'PPL101'",
        ),
        IOPort(
            port_id="mbon11_approach",
            direction="motor",
            label="MBON11 approach output (gamma1pedc)",
            description=(
                "Bilateral GABAergic MBON11 neurons driven by KCg-m and modulated by PPL101 "
                "(depression reduces approach and disinhibits downstream avoidance MBONs)."
            ),
            cell_indices=_select(lambda i: cells.cell_type[i] == "MBON11"),
            selection_rule="cell_type == 'MBON11'",
        ),
        IOPort(
            port_id="mbon01_avoidance",
            direction="motor",
            label="MBON01 avoidance output (gamma5beta'2a)",
            description="Bilateral glutamatergic MBON01 avoidance output neurons.",
            cell_indices=_select(lambda i: cells.cell_type[i] == "MBON01"),
            selection_rule="cell_type == 'MBON01'",
        ),
    ]
    return {p.port_id: p for p in ports}


def load_flywire_783_cells(
    datasets_dir: Path,
    *,
    verify_hashes: bool = False,
    spec: DatasetSpec | None = None,
) -> tuple[CellTable, dict[str, IOPort], Any]:
    """Load only FlyWire v783 cell metadata, 3D positions and IO ports (no 100 MiB edges).

    Used by API search and 3D geometry endpoints so the API process stays lightweight.
    """
    dspec = spec or FLYWIRE_783_SPEC
    if verify_hashes:
        report = verify_dataset(dspec.dataset_id, datasets_dir, spec=dspec)
        if not report["ok"]:
            bad = [f["filename"] for f in report["files"] if not f["ok"]]
            raise DatasetError(
                f"dataset {dspec.dataset_id!r} verification failed for: {', '.join(bad)}"
            )

    rdir = raw_dir(datasets_dir, dspec.dataset_id)
    if not (rdir / "Completeness_783.csv").is_file():
        raise DatasetError(
            f"dataset {dspec.dataset_id!r} is not installed in {datasets_dir}; "
            "run 'uv run alife connectome fetch flywire_783' first"
        )
    root_ids = _load_completeness(rdir / "Completeness_783.csv")

    annotations = _load_annotations(rdir / "Supplemental_file1_neuron_annotations.tsv")
    codex_cls = _load_codex_classification(rdir / "classification.csv.gz")
    codex_neu = _load_codex_neurons(rdir / "neurons.csv.gz")
    primary_regions = _load_primary_regions(rdir / "neuropil_synapse_table.csv.gz")

    n = int(root_ids.shape[0])
    # Identify neurons missing both soma and anchor coordinates so we only retain needed Codex rows.
    need_codex_pos: set[int] = set()
    for rid_np in root_ids:
        rid = int(rid_np)
        row = annotations.get(rid)
        if row is None or (
            (_parse_float(row.get("soma_x", "")) is None)
            and (_parse_float(row.get("pos_x", "")) is None)
        ):
            need_codex_pos.add(rid)
    codex_pos = _load_codex_coordinates(rdir / "coordinates.csv.gz", need_codex_pos)

    cell_type: list[str] = []
    super_class: list[str] = []
    cell_class: list[str] = []
    cell_sub_class: list[str] = []
    region: list[str] = []
    side: list[str] = []
    neurotransmitter: list[str] = []
    role = np.empty(n, dtype=np.uint8)
    cell_model = np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8)
    position_nm = np.full((n, 3), np.nan, dtype=np.float32)
    position_source = np.full(n, POS_MISSING, dtype=np.uint8)

    vx_x, vx_y, vx_z = FAFB_VOXEL_NM
    for i in range(n):
        rid = int(root_ids[i])
        ann = annotations.get(rid)
        cls = codex_cls.get(rid)
        neu_group, neu_nt = codex_neu.get(rid, ("", ""))

        ct = ""
        sc = ""
        cc = ""
        csc = ""
        sd = ""
        fl = ""
        nt = ""
        if ann is not None:
            ct = ann.get("cell_type", "").strip() or ann.get("hemibrain_type", "").strip()
            sc = ann.get("super_class", "").strip()
            cc = ann.get("cell_class", "").strip()
            csc = ann.get("cell_sub_class", "").strip()
            sd = ann.get("side", "").strip()
            fl = ann.get("flow", "").strip()
            nt = ann.get("top_nt", "").strip().lower()
        if cls is not None:
            sc = sc or cls.get("super_class", "").strip()
            cc = cc or cls.get("class", "").strip()
            csc = csc or cls.get("sub_class", "").strip()
            sd = sd or cls.get("side", "").strip()
            fl = fl or cls.get("flow", "").strip()
        if not nt:
            nt = neu_nt

        fallback_reg = neu_group if neu_group and neu_group != "NO_CONS" else ""
        reg = primary_regions.get(rid) or fallback_reg or "UNASGD"

        cell_type.append(ct)
        super_class.append(sc)
        cell_class.append(cc)
        cell_sub_class.append(csc)
        region.append(reg)
        side.append(sd)
        neurotransmitter.append(nt)
        role[i] = _map_role(sc, fl)

        # Position resolution: soma_x/y/z -> pos_x/y/z -> Codex nm coordinates.
        placed = False
        if ann is not None:
            sx = _parse_float(ann.get("soma_x", ""))
            sy = _parse_float(ann.get("soma_y", ""))
            sz = _parse_float(ann.get("soma_z", ""))
            if sx is not None and sy is not None and sz is not None:
                position_nm[i] = (sx * vx_x, sy * vx_y, sz * vx_z)
                position_source[i] = POS_SOMA
                placed = True
            else:
                px = _parse_float(ann.get("pos_x", ""))
                py = _parse_float(ann.get("pos_y", ""))
                pz = _parse_float(ann.get("pos_z", ""))
                if px is not None and py is not None and pz is not None:
                    position_nm[i] = (px * vx_x, py * vx_y, pz * vx_z)
                    position_source[i] = POS_ANCHOR
                    placed = True
        if not placed and rid in codex_pos:
            position_nm[i] = codex_pos[rid]
            position_source[i] = POS_CODEX

    cells = CellTable(
        source_id=root_ids,
        species_names=(dspec.species,),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(cell_type),
        super_class=tuple(super_class),
        cell_class=tuple(cell_class),
        cell_sub_class=tuple(cell_sub_class),
        region=tuple(region),
        side=tuple(side),
        role=role,
        cell_model=cell_model,
        neurotransmitter=tuple(neurotransmitter),
        position_nm=position_nm,
        position_source=position_source,
    )
    cells.validate()
    io_ports = _build_io_ports(cells)
    provenance = dspec.to_provenance()
    return cells, io_ports, provenance


def load_flywire_783(
    datasets_dir: Path,
    *,
    verify_hashes: bool = True,
    spec: DatasetSpec | None = None,
) -> ConnectomePack:
    """Load the verified FlyWire v783 tables into a canonical ``ConnectomePack``."""
    dspec = spec or FLYWIRE_783_SPEC
    cells, io_ports, provenance = load_flywire_783_cells(
        datasets_dir, verify_hashes=verify_hashes, spec=dspec
    )
    rdir = raw_dir(datasets_dir, dspec.dataset_id)
    edges = _load_connectivity(rdir / "Connectivity_783.parquet", cells.source_id)
    digest = compute_pack_digest(
        dspec.dataset_id, dspec.species, cells, edges, io_ports, provenance
    )
    return ConnectomePack(
        dataset_id=dspec.dataset_id,
        species=dspec.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=provenance,
        pack_digest=digest,
    )


def flywire_783_reference_checks(pack: ConnectomePack) -> list[dict[str, Any]]:
    """Compare loaded FlyWire v783 summary metrics against published paper values."""
    ref = FLYWIRE_783_REFERENCE
    n = pack.cells.num_cells
    e = pack.edges.num_edges
    syn = int(pack.edges.count.astype(np.int64).sum()) if e else 0
    mn9_port = pack.io_ports.get("mn9_proboscis_extension")
    mn9_ids = (
        sorted(int(pack.cells.source_id[i]) for i in mn9_port.cell_indices)
        if mn9_port is not None
        else []
    )
    expected_mn9 = sorted(int(x) for x in ref["mn9_root_ids"])
    missing_pos = int((pack.cells.position_source == POS_MISSING).sum())
    return [
        {
            "check": "shiu_v783_neuron_count",
            "expected": ref["shiu_v783_neurons"],
            "actual": n,
            "ok": n == ref["shiu_v783_neurons"],
            "note": (
                "Shiu et al. v783 completeness table has 138,639 connected neurons "
                "(616 fewer than Dorkenwald et al.'s 139,255 total proofread neurons)."
            ),
        },
        {
            "check": "shiu_v783_edge_count",
            "expected": ref["shiu_v783_edges"],
            "actual": e,
            "ok": e == ref["shiu_v783_edges"],
            "note": "Directed cell-to-cell pairs in Connectivity_783.parquet.",
        },
        {
            "check": "shiu_v783_total_synapses",
            "expected": ref["shiu_v783_synapses"],
            "actual": syn,
            "ok": syn == ref["shiu_v783_synapses"],
            "note": "Matches Dorkenwald et al. 2024 (~54.5M chemical synapses).",
        },
        {
            "check": "mn9_motor_neurons_resolved",
            "expected": expected_mn9,
            "actual": mn9_ids,
            "ok": mn9_ids == expected_mn9,
            "note": "Left and right MN9 proboscis extension motor neurons (cell_type == 'CB0701').",
        },
        {
            "check": "all_cells_have_3d_coordinates",
            "expected": 0,
            "actual": missing_pos,
            "ok": missing_pos == 0,
            "note": "Soma voxel -> anchor voxel -> Codex nm fallback covers 100% of neurons.",
        },
    ]


def load_connectome_pack(
    dataset_id: str,
    datasets_dir: Path,
    *,
    verify_hashes: bool = True,
) -> ConnectomePack:
    """Dispatch loader by dataset_id (extensible for whole-CNS and S1..S7 species packs)."""
    if dataset_id == "flywire_783":
        return load_flywire_783(datasets_dir, verify_hashes=verify_hashes)
    if dataset_id == "banc_888":
        from alife_lab.connectome.cns_packs import load_banc_888

        return load_banc_888(datasets_dir, verify_hashes=verify_hashes)
    if dataset_id == "male_cns_v1":
        from alife_lab.connectome.cns_packs import load_male_cns_v1

        return load_male_cns_v1(datasets_dir, verify_hashes=verify_hashes)
    raise DatasetError(f"no loader implemented for dataset {dataset_id!r}")


def summarize_connectome_pack(pack: ConnectomePack) -> dict[str, Any]:
    checks = flywire_783_reference_checks(pack) if pack.dataset_id == "flywire_783" else []
    return summarize_pack(pack, reference_checks=checks)
