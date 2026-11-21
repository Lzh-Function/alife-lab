"""Read-only connectome metadata, cell search and 3D point-cloud geometry index (F2).

CPU-only and lazy-loaded inside the API process: never imports GPU runtimes and skips the
100 MiB FlyWire connectivity table when serving browser metadata, neuron search and static
3D coordinates.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alife_lab.connectome.flywire import load_flywire_783_cells
from alife_lab.connectome.lif import make_synthetic_mb_pack, make_synthetic_tiny_pack
from alife_lab.connectome.registry import DatasetError
from alife_lab.connectome.schema import CELL_ROLE_NAMES, CellTable, DatasetProvenance, IOPort

_DATASET_ID_RE = re.compile(r"^[a-z0-9_]{1,64}$")
_LOCK = threading.Lock()
_INDEX_CACHE: dict[tuple[str, str], ConnectomeIndex] = {}

# Maximum number of distinct region palette indices encoded in uint8 (255 = other).
_MAX_REGION_PALETTE = 254


@dataclass(frozen=True)
class ConnectomeIndex:
    """In-memory search & 3D geometry index for a connectome dataset."""

    dataset_id: str
    species: str
    version: str
    citation: str
    licence_note: str
    cells: CellTable
    io_ports: dict[str, IOPort]
    regions: tuple[dict[str, Any], ...]
    region_names: tuple[str, ...]
    geometry_header: dict[str, Any]
    geometry_bytes: bytes
    search_blob: tuple[str, ...]


def _build_geometry(
    dataset_id: str,
    cells: CellTable,
    io_ports: dict[str, IOPort],
    region_names: tuple[str, ...],
    region_idx: np.ndarray[Any, np.dtype[np.uint8]],
) -> tuple[dict[str, Any], bytes]:
    n = cells.num_cells
    pos = np.nan_to_num(cells.position_nm.astype(np.float64, copy=False), nan=0.0)
    if n > 0:
        center = np.median(pos, axis=0)
        centered = pos - center
        # FAFB y-axis increases ventrally; invert y so dorsal faces +Y in Three.js.
        centered[:, 1] = -centered[:, 1]
        radii = np.linalg.norm(centered, axis=1)
        scale = float(np.percentile(radii, 99.5)) if n > 100 else float(np.max(radii))
        if scale <= 1e-9:
            scale = 1.0
        norm = np.clip(centered / scale, -1.0, 1.0)
        quant = np.round(norm * 32000.0).astype("<i2")
        bounds_nm = {
            "min": [round(float(v), 1) for v in pos.min(axis=0)],
            "max": [round(float(v), 1) for v in pos.max(axis=0)],
            "scale_nm": round(scale, 1),
        }
    else:
        quant = np.zeros((0, 3), dtype="<i2")
        bounds_nm = {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "scale_nm": 1.0}

    # Include compact port index lists for small named ports (<= 2500 cells) so the 3D view can
    # highlight sensory/motor populations without extra round-trips.
    port_indices: dict[str, list[int]] = {}
    for pid, port in io_ports.items():
        if port.cell_indices.size <= 2500:
            port_indices[pid] = [int(i) for i in port.cell_indices.tolist()]

    body = (
        np.ascontiguousarray(quant, dtype="<i2").tobytes()
        + np.ascontiguousarray(cells.role, dtype=np.uint8).tobytes()
        + np.ascontiguousarray(region_idx, dtype=np.uint8).tobytes()
    )
    header = {
        "codec": "connectome_geom_v1",
        "dataset_id": dataset_id,
        "num_cells": n,
        "pos_scale": 32000.0,
        "role_names": list(CELL_ROLE_NAMES),
        "region_names": list(region_names),
        "bounds_nm": bounds_nm,
        "port_indices": port_indices,
    }
    return header, body


def _build_index(
    dataset_id: str,
    cells: CellTable,
    io_ports: dict[str, IOPort],
    provenance: DatasetProvenance,
) -> ConnectomeIndex:
    counts = Counter(cells.region)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    regions = tuple({"region": r, "count": int(c)} for r, c in ordered)
    palette_names = tuple(r for r, _ in ordered[:_MAX_REGION_PALETTE])
    name_to_idx = {r: i for i, r in enumerate(palette_names)}
    n = cells.num_cells
    region_idx = np.full(n, 255, dtype=np.uint8)
    for i, r in enumerate(cells.region):
        idx = name_to_idx.get(r)
        if idx is not None:
            region_idx[i] = idx

    geom_header, geom_bytes = _build_geometry(
        dataset_id, cells, io_ports, palette_names, region_idx
    )
    search_blob = tuple(
        f"{cells.cell_type[i]} {cells.super_class[i]} {cells.cell_class[i]} "
        f"{cells.cell_sub_class[i]} {cells.region[i]} {int(cells.source_id[i])}".lower()
        for i in range(n)
    )
    return ConnectomeIndex(
        dataset_id=dataset_id,
        species=provenance.species,
        version=provenance.version,
        citation=provenance.citation,
        licence_note=provenance.licence_note,
        cells=cells,
        io_ports=io_ports,
        regions=regions,
        region_names=palette_names,
        geometry_header=geom_header,
        geometry_bytes=geom_bytes,
        search_blob=search_blob,
    )


def get_connectome_index(dataset_id: str, datasets_dir: Path) -> ConnectomeIndex:
    """Return cached ``ConnectomeIndex`` for ``dataset_id``."""
    if not _DATASET_ID_RE.match(dataset_id):
        raise DatasetError(f"invalid connectome dataset_id {dataset_id!r}")
    key = (dataset_id, str(datasets_dir.resolve()))
    with _LOCK:
        cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    if dataset_id == "synthetic_tiny":
        pack = make_synthetic_tiny_pack()
        index = _build_index(dataset_id, pack.cells, pack.io_ports, pack.provenance)
    elif dataset_id == "synthetic_mb":
        pack = make_synthetic_mb_pack()
        index = _build_index(dataset_id, pack.cells, pack.io_ports, pack.provenance)
    elif dataset_id == "flywire_783":
        cells, io_ports, prov = load_flywire_783_cells(datasets_dir, verify_hashes=False)
        index = _build_index(dataset_id, cells, io_ports, prov)
    else:
        raise DatasetError(f"unknown connectome dataset {dataset_id!r}")

    with _LOCK:
        _INDEX_CACHE[key] = index
    return index


def get_connectome_meta(dataset_id: str, datasets_dir: Path) -> dict[str, Any]:
    idx = get_connectome_index(dataset_id, datasets_dir)
    role_counts = {
        name: int((idx.cells.role == i).sum()) for i, name in enumerate(CELL_ROLE_NAMES)
    }
    return {
        "dataset_id": idx.dataset_id,
        "species": idx.species,
        "version": idx.version,
        "num_cells": idx.cells.num_cells,
        "citation": idx.citation,
        "licence_note": idx.licence_note,
        "roles": role_counts,
        "regions": list(idx.regions),
        "ports": [
            {
                **p.to_summary(),
                "description": p.description,
            }
            for p in idx.io_ports.values()
        ],
    }


def search_connectome_cells(
    dataset_id: str,
    datasets_dir: Path,
    *,
    q: str = "",
    region: str = "",
    role: str = "",
    port: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    idx = get_connectome_index(dataset_id, datasets_dir)
    cells = idx.cells
    n = cells.num_cells
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))

    role_id: int | None = None
    if role:
        if role not in CELL_ROLE_NAMES:
            raise DatasetError(f"unknown role {role!r}")
        role_id = CELL_ROLE_NAMES.index(role)

    if port:
        if port not in idx.io_ports:
            raise DatasetError(f"unknown port {port!r}")
        candidates: Any = idx.io_ports[port].cell_indices.tolist()
    else:
        candidates = range(n)

    tokens = [tok for tok in q.strip().lower().split() if tok]
    matches: list[int] = []
    blob = idx.search_blob
    reg_tuple = cells.region
    role_arr = cells.role

    for i in candidates:
        if role_id is not None and int(role_arr[i]) != role_id:
            continue
        if region and reg_tuple[i] != region:
            continue
        if tokens:
            s = blob[i]
            if not all(tok in s for tok in tokens):
                continue
        matches.append(int(i))

    total = len(matches)
    page = matches[offset : offset + limit]
    items = [
        {
            "index": i,
            "source_id": int(cells.source_id[i]),
            "cell_type": cells.cell_type[i],
            "super_class": cells.super_class[i],
            "cell_class": cells.cell_class[i],
            "region": cells.region[i],
            "side": cells.side[i],
            "role": CELL_ROLE_NAMES[int(cells.role[i])],
            "neurotransmitter": cells.neurotransmitter[i],
        }
        for i in page
    ]
    return {
        "dataset_id": idx.dataset_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
    }


def get_connectome_geometry(dataset_id: str, datasets_dir: Path) -> tuple[dict[str, Any], bytes]:
    idx = get_connectome_index(dataset_id, datasets_dir)
    return idx.geometry_header, idx.geometry_bytes
