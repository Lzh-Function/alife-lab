"""Common connectome species-pack schema (ADR 0035, connectome plan section 3.5).

Every biological wiring diagram (adult fly, C. elegans, fly larva, Platynereis,
mouse visual cortex, etc.) is normalized on load into four structures:

1. ``CellTable``   — dense 0..N-1 neuron table with taxonomy, region, role, model kind,
                     neurotransmitter and 3D position in nanometres.
2. ``EdgeTable``   — directed connectivity table with integer synapse counts, sign (+1/-1/0)
                     and synapse kind (chemical vs gap junction), convertible to canonical CSR.
3. ``IOPort``      — named sensory input / motor output / circuit cell-index sets.
4. ``DatasetProvenance`` — citation, version, licence note and verified file SHA-256 hashes.

All simulation engines, interventions, checkpoints and chimeras consume this representation.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from alife_lab.core.hashing import canonical_json

CELL_ROLE_NAMES: tuple[str, ...] = ("sensory", "interneuron", "motor", "other")
ROLE_SENSORY = 0
ROLE_INTERNEURON = 1
ROLE_MOTOR = 2
ROLE_OTHER = 3

CELL_MODEL_NAMES: tuple[str, ...] = ("spiking_lif", "graded", "other")
MODEL_SPIKING_LIF = 0
MODEL_GRADED = 1
MODEL_OTHER = 2

SYNAPSE_KIND_NAMES: tuple[str, ...] = ("chemical", "gap_junction")
SYNAPSE_CHEMICAL = 0
SYNAPSE_GAP_JUNCTION = 1

POS_SOURCE_NAMES: tuple[str, ...] = ("missing", "soma", "anchor", "codex")
POS_MISSING = 0
POS_SOMA = 1
POS_ANCHOR = 2
POS_CODEX = 3

I8 = npt.NDArray[np.int8]
U8 = npt.NDArray[np.uint8]
I32 = npt.NDArray[np.int32]
I64 = npt.NDArray[np.int64]
F32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class FileProvenance:
    filename: str
    url: str
    sha256: str
    size_bytes: int
    source_label: str
    licence_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "url": self.url,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_label": self.source_label,
            "licence_note": self.licence_note,
        }


@dataclass(frozen=True)
class DatasetProvenance:
    dataset_id: str
    species: str
    version: str
    citation: str
    licence_note: str
    files: tuple[FileProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "species": self.species,
            "version": self.version,
            "citation": self.citation,
            "licence_note": self.licence_note,
            "files": [f.to_dict() for f in self.files],
        }


@dataclass(frozen=True)
class CellTable:
    """Dense 0..N-1 table of neurons in a connectome (or chimera)."""

    source_id: I64
    species_names: tuple[str, ...]
    species_idx: U8
    cell_type: tuple[str, ...]
    super_class: tuple[str, ...]
    cell_class: tuple[str, ...]
    cell_sub_class: tuple[str, ...]
    region: tuple[str, ...]
    side: tuple[str, ...]
    role: U8
    cell_model: U8
    neurotransmitter: tuple[str, ...]
    position_nm: F32
    position_source: U8

    @property
    def num_cells(self) -> int:
        return int(self.source_id.shape[0])

    def validate(self) -> None:
        n = self.num_cells
        if self.source_id.ndim != 1:
            raise ValueError("source_id must be 1D")
        if self.species_idx.shape != (n,):
            raise ValueError("species_idx shape mismatch")
        if self.role.shape != (n,) or self.cell_model.shape != (n,):
            raise ValueError("role/cell_model shape mismatch")
        if self.position_nm.shape != (n, 3) or self.position_source.shape != (n,):
            raise ValueError("position shape mismatch")
        for field_name, seq in (
            ("cell_type", self.cell_type),
            ("super_class", self.super_class),
            ("cell_class", self.cell_class),
            ("cell_sub_class", self.cell_sub_class),
            ("region", self.region),
            ("side", self.side),
            ("neurotransmitter", self.neurotransmitter),
        ):
            if len(seq) != n:
                raise ValueError(f"{field_name} length {len(seq)} != {n}")
        if n > 0 and int(self.species_idx.max()) >= len(self.species_names):
            raise ValueError("species_idx out of bounds")


@dataclass(frozen=True)
class CSRGraph:
    """Canonical outgoing CSR adjacency (targets sorted ascending per pre neuron)."""

    num_cells: int
    indptr: I32
    post: I32
    signed_weight: I32
    kind: U8


@dataclass(frozen=True)
class EdgeTable:
    """Directed synapse edge list with exact integer counts and signs."""

    pre: I32
    post: I32
    count: I32
    sign: I8
    kind: U8

    @property
    def num_edges(self) -> int:
        return int(self.pre.shape[0])

    def signed_weights(self) -> I32:
        return (self.count * self.sign.astype(np.int32)).astype(np.int32, copy=False)

    def validate(self, num_cells: int) -> None:
        e = self.num_edges
        for name, arr in (
            ("post", self.post),
            ("count", self.count),
            ("sign", self.sign),
            ("kind", self.kind),
        ):
            if arr.shape != (e,):
                raise ValueError(f"edge array {name} shape {arr.shape} != ({e},)")
        if e == 0:
            return
        if int(self.pre.min()) < 0 or int(self.pre.max()) >= num_cells:
            raise ValueError("pre index out of bounds")
        if int(self.post.min()) < 0 or int(self.post.max()) >= num_cells:
            raise ValueError("post index out of bounds")
        if int(self.count.min()) <= 0:
            raise ValueError("synapse count must be strictly positive")
        if int(self.sign.min()) < -1 or int(self.sign.max()) > 1:
            raise ValueError("sign must be in {-1, 0, +1}")

    def to_csr(self, num_cells: int) -> CSRGraph:
        """Build deterministic CSR sorted by (pre, post) ascending."""
        self.validate(num_cells)
        e = self.num_edges
        if e == 0:
            return CSRGraph(
                num_cells=num_cells,
                indptr=np.zeros(num_cells + 1, dtype=np.int32),
                post=np.zeros(0, dtype=np.int32),
                signed_weight=np.zeros(0, dtype=np.int32),
                kind=np.zeros(0, dtype=np.uint8),
            )
        # Single 64-bit sort key guarantees deterministic ordering by (pre, post).
        keys = (self.pre.astype(np.uint64) << np.uint64(32)) | self.post.astype(np.uint64)
        order = np.argsort(keys, kind="mergesort")
        pre_sorted = self.pre[order]
        post_sorted = self.post[order].astype(np.int32, copy=False)
        weights_sorted = (
            self.count[order].astype(np.int32) * self.sign[order].astype(np.int32)
        ).astype(np.int32, copy=False)
        kind_sorted = self.kind[order].astype(np.uint8, copy=False)
        counts = np.bincount(pre_sorted, minlength=num_cells)
        indptr = np.empty(num_cells + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        return CSRGraph(
            num_cells=num_cells,
            indptr=indptr,
            post=post_sorted,
            signed_weight=weights_sorted,
            kind=kind_sorted,
        )


@dataclass(frozen=True)
class IOPort:
    """Named sensory input, motor output, or circuit population."""

    port_id: str
    direction: Literal["sensory", "motor", "circuit"]
    label: str
    description: str
    cell_indices: I32
    selection_rule: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "port_id": self.port_id,
            "direction": self.direction,
            "label": self.label,
            "count": int(self.cell_indices.shape[0]),
            "selection_rule": self.selection_rule,
        }


@dataclass(frozen=True)
class ConnectomePack:
    """Complete loaded species pack ready for simulation, inspection and provenance logging."""

    dataset_id: str
    species: str
    cells: CellTable
    edges: EdgeTable
    io_ports: dict[str, IOPort]
    provenance: DatasetProvenance
    pack_digest: str


def compute_pack_digest(
    dataset_id: str,
    species: str,
    cells: CellTable,
    edges: EdgeTable,
    io_ports: dict[str, IOPort],
    provenance: DatasetProvenance,
) -> str:
    """Deterministic SHA-256 over all normalized cell, edge, port and provenance data."""
    h = hashlib.sha256()

    def _feed_str(tag: str, text: str) -> None:
        h.update(tag.encode("ascii"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        h.update(b"\0")

    def _feed_arr(tag: str, arr: npt.NDArray[Any]) -> None:
        c = np.ascontiguousarray(arr)
        h.update(tag.encode("ascii"))
        h.update(b"\0")
        h.update(str(c.dtype).encode("ascii"))
        h.update(str(c.shape).encode("ascii"))
        h.update(c.tobytes())
        h.update(b"\0")

    _feed_str("dataset_id", dataset_id)
    _feed_str("species", species)
    _feed_str("provenance", canonical_json(provenance.to_dict()))
    _feed_arr("cells.source_id", cells.source_id)
    _feed_str("cells.species_names", canonical_json(list(cells.species_names)))
    _feed_arr("cells.species_idx", cells.species_idx)
    for name, seq in (
        ("cells.cell_type", cells.cell_type),
        ("cells.super_class", cells.super_class),
        ("cells.cell_class", cells.cell_class),
        ("cells.cell_sub_class", cells.cell_sub_class),
        ("cells.region", cells.region),
        ("cells.side", cells.side),
        ("cells.neurotransmitter", cells.neurotransmitter),
    ):
        _feed_str(name, "\n".join(seq))
    _feed_arr("cells.role", cells.role)
    _feed_arr("cells.cell_model", cells.cell_model)
    # Canonicalize NaN payload in position_nm before hashing.
    pos = np.where(np.isnan(cells.position_nm), np.float32("nan"), cells.position_nm)
    _feed_arr("cells.position_nm", pos)
    _feed_arr("cells.position_source", cells.position_source)

    _feed_arr("edges.pre", edges.pre)
    _feed_arr("edges.post", edges.post)
    _feed_arr("edges.count", edges.count)
    _feed_arr("edges.sign", edges.sign)
    _feed_arr("edges.kind", edges.kind)

    for port_id in sorted(io_ports):
        port = io_ports[port_id]
        _feed_str(
            f"port.{port_id}",
            canonical_json(
                {
                    "direction": port.direction,
                    "label": port.label,
                    "selection_rule": port.selection_rule,
                }
            ),
        )
        _feed_arr(f"port.{port_id}.indices", port.cell_indices)
    return h.hexdigest()


def summarize_pack(
    pack: ConnectomePack,
    reference_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic JSON-serializable summary of a loaded connectome pack."""
    cells = pack.cells
    edges = pack.edges
    n = cells.num_cells
    e = edges.num_edges

    exc_mask = edges.sign > 0
    inh_mask = edges.sign < 0
    unk_mask = edges.sign == 0
    total_synapses = int(edges.count.astype(np.int64).sum()) if e else 0
    exc_synapses = int(edges.count[exc_mask].astype(np.int64).sum()) if e else 0
    inh_synapses = int(edges.count[inh_mask].astype(np.int64).sum()) if e else 0
    unk_synapses = int(edges.count[unk_mask].astype(np.int64).sum()) if e else 0

    self_loops = int((edges.pre == edges.post).sum()) if e else 0
    out_deg = np.bincount(edges.pre, minlength=n) if e else np.zeros(n, dtype=np.int64)
    in_deg = np.bincount(edges.post, minlength=n) if e else np.zeros(n, dtype=np.int64)
    isolated = int(((out_deg == 0) & (in_deg == 0)).sum()) if n else 0

    # Dale's law check per presynaptic neuron for chemical synapses.
    chem_mask = edges.kind == SYNAPSE_CHEMICAL
    if e and bool(chem_mask.any()):
        chem_pre = edges.pre[chem_mask]
        chem_sign = edges.sign[chem_mask]
        max_sign = np.full(n, -2, dtype=np.int8)
        min_sign = np.full(n, 2, dtype=np.int8)
        np.maximum.at(max_sign, chem_pre, chem_sign)
        np.minimum.at(min_sign, chem_pre, chem_sign)
        has_out = max_sign > -2
        mixed_pre = int((has_out & (max_sign != min_sign)).sum())
        exc_pre_neurons = int((has_out & (max_sign == 1) & (min_sign == 1)).sum())
        inh_pre_neurons = int((has_out & (max_sign == -1) & (min_sign == -1)).sum())
    else:
        mixed_pre = 0
        exc_pre_neurons = 0
        inh_pre_neurons = 0

    role_counts = {name: int((cells.role == idx).sum()) for idx, name in enumerate(CELL_ROLE_NAMES)}
    model_counts = {
        name: int((cells.cell_model == idx).sum()) for idx, name in enumerate(CELL_MODEL_NAMES)
    }
    pos_source_counts = {
        name: int((cells.position_source == idx).sum()) for idx, name in enumerate(POS_SOURCE_NAMES)
    }
    syn_kind_counts = {
        name: int((edges.kind == idx).sum()) for idx, name in enumerate(SYNAPSE_KIND_NAMES)
    }

    super_class_counts = dict(
        sorted(Counter(cells.super_class).items(), key=lambda kv: (-kv[1], kv[0]))
    )
    nt_counts = dict(
        sorted(Counter(cells.neurotransmitter).items(), key=lambda kv: (-kv[1], kv[0]))
    )
    region_counts_all = Counter(cells.region)
    top_regions = dict(sorted(region_counts_all.items(), key=lambda kv: (-kv[1], kv[0]))[:20])
    named_types = sum(1 for t in cells.cell_type if t)

    return {
        "dataset_id": pack.dataset_id,
        "species": pack.species,
        "pack_digest": pack.pack_digest,
        "provenance": pack.provenance.to_dict(),
        "counts": {
            "cells": n,
            "edges": e,
            "total_synapses": total_synapses,
            "excitatory_edges": int(exc_mask.sum()) if e else 0,
            "inhibitory_edges": int(inh_mask.sum()) if e else 0,
            "unsigned_edges": int(unk_mask.sum()) if e else 0,
            "excitatory_edge_ratio": round(float(exc_mask.sum()) / e, 6) if e else 0.0,
            "inhibitory_edge_ratio": round(float(inh_mask.sum()) / e, 6) if e else 0.0,
            "excitatory_synapses": exc_synapses,
            "inhibitory_synapses": inh_synapses,
            "unsigned_synapses": unk_synapses,
            "excitatory_synapse_ratio": round(exc_synapses / total_synapses, 6)
            if total_synapses
            else 0.0,
            "inhibitory_synapse_ratio": round(inh_synapses / total_synapses, 6)
            if total_synapses
            else 0.0,
            "excitatory_presynaptic_neurons": exc_pre_neurons,
            "inhibitory_presynaptic_neurons": inh_pre_neurons,
            "mixed_sign_presynaptic_neurons": mixed_pre,
            "self_loops": self_loops,
            "isolated_cells": isolated,
            "cells_with_cell_type": named_types,
            "unique_regions": len(region_counts_all),
        },
        "degree": {
            "out_max": int(out_deg.max()) if n else 0,
            "out_median": float(np.median(out_deg)) if n else 0.0,
            "out_mean": round(float(out_deg.mean()), 4) if n else 0.0,
            "out_zero": int((out_deg == 0).sum()) if n else 0,
            "in_max": int(in_deg.max()) if n else 0,
            "in_median": float(np.median(in_deg)) if n else 0.0,
            "in_mean": round(float(in_deg.mean()), 4) if n else 0.0,
            "in_zero": int((in_deg == 0).sum()) if n else 0,
        },
        "synapse_kinds": syn_kind_counts,
        "roles": role_counts,
        "cell_models": model_counts,
        "position_sources": pos_source_counts,
        "super_classes": super_class_counts,
        "neurotransmitters": nt_counts,
        "top_regions": top_regions,
        "io_ports": {k: pack.io_ports[k].to_summary() for k in sorted(pack.io_ports)},
        "reference_checks": reference_checks or [],
    }
