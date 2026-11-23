"""Whole-CNS (Brain + Ventral Nerve Cord) Drosophila connectome species-pack loaders (Phase F8 L2).

Supports two real whole-CNS adult Drosophila connectomes from pure Apache Arrow ``.feather`` and
``.csv`` tables (ZERO pickles):
1. ``banc_888``    — BANC v888 adult female Brain + VNC (Lee Lab / Bates et al., CC-BY 4.0)
2. ``male_cns_v1`` — MaleCNS v1.0 adult male Brain + VNC (HHMI Janelia / Google Research, CC-BY 4.0)

Both loaders normalize into the common ``ConnectomePack`` schema with:
- Full 36-channel antagonist leg motor neuron pools across all 6 legs
  (``mn_{LF,LM,LH,RF,RM,RH}_{coxa_pro,coxa_ret,fem_flex,fem_ext,tib_ext,tib_flex}``)
- 6-leg proprioceptive & mechanosensory feedback ports (``proprio_{LF..RH}``)
- Brain sensory & descending command ports (``sugar_grn``, ``bitter_grn``,
  ``antennal_grooming_mechanosensory``, ``olfactory_sensory``, ``mn9_proboscis_extension``,
  ``forward_dn``, ``turn_left_dn``, ``turn_right_dn``, ``backward_dn``, ``grooming_dn``)
- Local non-pickle binary cache (``pack_cache_v1.npz`` + ``pack_cache_v1.json``) keyed on the
  verified dataset provenance digest so worker startup takes ~1 second.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.csv as pa_csv
import pyarrow.feather as pf

from alife_lab.connectome.registry import (
    BANC_888_SPEC,
    MALE_CNS_V1_SPEC,
    DatasetError,
    DatasetSpec,
    dataset_root,
    raw_dir,
    verify_dataset,
)
from alife_lab.connectome.schema import (
    MODEL_SPIKING_LIF,
    POS_ANCHOR,
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
)

LEG_NAMES: tuple[str, ...] = ("LF", "LM", "LH", "RF", "RM", "RH")
MUSCLE_CHANNELS: tuple[str, ...] = (
    "coxa_pro",
    "coxa_ret",
    "fem_flex",
    "fem_ext",
    "tib_ext",
    "tib_flex",
)

# Dale's-law chemical neurotransmitter signs matching Shiu et al. (2024) Drosophila convention:
# ACh excitatory (+1), GABA / Glutamate (GluCl-alpha) / Histamine inhibitory (-1).
_NT_SIGN: dict[str, int] = {
    "acetylcholine": 1,
    "ach": 1,
    "gaba": -1,
    "glutamate": -1,
    "glut": -1,
    "histamine": -1,
    "dopamine": 1,
    "da": 1,
    "serotonin": 1,
    "ser": 1,
    "octopamine": 1,
    "oct": 1,
    "tyramine": 1,
}

# BANC v888 anatomical muscle target -> 6-channel joint antagonist mapping
_BANC_MUSCLE_GROUPS: dict[str, str] = {
    "sternal_anterior_rotator_muscle": "coxa_pro",
    "tergopleural_promotor_muscle": "coxa_pro",
    "sternal_posterior_rotator_muscle": "coxa_ret",
    "pleural_remotor_and_abductor_muscle": "coxa_ret",
    "sternal_adductor_muscle": "coxa_ret",
    "trochanter_flexor_muscle": "fem_flex",
    "accessory_trochanter_flexor_muscle": "fem_flex",
    "femur_reductor_muscle": "fem_flex",
    "trochanter_extensor_muscle": "fem_ext",
    "sternotrochanter_extensor_muscle": "fem_ext",
    "tergotrochanter_extensor_muscle": "fem_ext",
    "tibia_extensor_muscle": "tib_ext",
    "tarsus_levator_muscle": "tib_ext",
    "tibia_flexor_muscle": "tib_flex",
    "accessory_tibia_flexor_muscle": "tib_flex",
    "long_tendon_muscle": "tib_flex",
    "tarsus_depressor_muscle": "tib_flex",
}

_CACHE_VERSION = 2


def _norm_nt(raw: str | None) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    if s in ("ach", "acetylcholine"):
        return "acetylcholine"
    if s in ("glut", "glutamate"):
        return "glutamate"
    if s in ("da", "dopamine"):
        return "dopamine"
    if s in ("ser", "serotonin"):
        return "serotonin"
    if s in ("oct", "octopamine"):
        return "octopamine"
    if s in ("gaba", "histamine", "tyramine"):
        return s
    return s or "unknown"


def _nt_sign(nt: str) -> int:
    return _NT_SIGN.get(nt.lower(), 0)


def _norm_side(raw: str | None) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    if s in ("l", "left", "lhs"):
        return "left"
    if s in ("r", "right", "rhs"):
        return "right"
    if s in ("m", "mid", "midline", "center"):
        return "midline"
    return s or "unknown"


def _cache_key(spec: DatasetSpec) -> str:
    h = hashlib.sha256()
    h.update(f"cns_cache_v{_CACHE_VERSION}:{spec.dataset_id}:".encode())
    for f in spec.files:
        h.update(f"{f.filename}:{f.sha256}:{f.size_bytes};".encode())
    return h.hexdigest()


def _try_load_cache(spec: DatasetSpec, datasets_dir: Path) -> ConnectomePack | None:
    droot = dataset_root(datasets_dir, spec.dataset_id)
    meta_p = droot / f"pack_cache_v{_CACHE_VERSION}.json"
    npz_p = droot / f"pack_cache_v{_CACHE_VERSION}.npz"
    if not (meta_p.is_file() and npz_p.is_file()):
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("cache_key") != _cache_key(spec):
            return None
        with np.load(npz_p, allow_pickle=False) as z:
            cells = CellTable(
                source_id=z["source_id"].astype(np.int64, copy=True),
                species_names=tuple(meta["species_names"]),
                species_idx=z["species_idx"].astype(np.uint8, copy=True),
                cell_type=tuple(meta["cell_type"]),
                super_class=tuple(meta["super_class"]),
                cell_class=tuple(meta["cell_class"]),
                cell_sub_class=tuple(meta["cell_sub_class"]),
                region=tuple(meta["region"]),
                side=tuple(meta["side"]),
                role=z["role"].astype(np.uint8, copy=True),
                cell_model=z["cell_model"].astype(np.uint8, copy=True),
                neurotransmitter=tuple(meta["neurotransmitter"]),
                position_nm=z["position_nm"].astype(np.float32, copy=True),
                position_source=z["position_source"].astype(np.uint8, copy=True),
            )
            edges = EdgeTable(
                pre=z["edge_pre"].astype(np.int32, copy=True),
                post=z["edge_post"].astype(np.int32, copy=True),
                count=z["edge_count"].astype(np.int32, copy=True),
                sign=z["edge_sign"].astype(np.int8, copy=True),
                kind=z["edge_kind"].astype(np.uint8, copy=True),
            )
            io_ports: dict[str, IOPort] = {}
            for pid, pmeta in meta["io_ports"].items():
                io_ports[pid] = IOPort(
                    port_id=pid,
                    direction=pmeta["direction"],
                    label=pmeta["label"],
                    description=pmeta["description"],
                    cell_indices=z[f"port__{pid}"].astype(np.int32, copy=True),
                    selection_rule=pmeta["selection_rule"],
                )
        return ConnectomePack(
            dataset_id=spec.dataset_id,
            species=spec.species,
            cells=cells,
            edges=edges,
            io_ports=io_ports,
            provenance=spec.to_provenance(),
            pack_digest=str(meta["pack_digest"]),
        )
    except Exception:
        return None


def _save_cache(spec: DatasetSpec, datasets_dir: Path, pack: ConnectomePack) -> None:
    droot = dataset_root(datasets_dir, spec.dataset_id)
    droot.mkdir(parents=True, exist_ok=True)
    meta_p = droot / f"pack_cache_v{_CACHE_VERSION}.json"
    npz_p = droot / f"pack_cache_v{_CACHE_VERSION}.npz"
    tmp_meta = droot / f"pack_cache_v{_CACHE_VERSION}.json.tmp"
    tmp_npz = droot / f"pack_cache_v{_CACHE_VERSION}.npz.tmp"
    c = pack.cells
    e = pack.edges
    arrays: dict[str, Any] = {
        "source_id": c.source_id,
        "species_idx": c.species_idx,
        "role": c.role,
        "cell_model": c.cell_model,
        "position_nm": c.position_nm,
        "position_source": c.position_source,
        "edge_pre": e.pre,
        "edge_post": e.post,
        "edge_count": e.count,
        "edge_sign": e.sign,
        "edge_kind": e.kind,
    }
    ports_meta: dict[str, Any] = {}
    for pid, port in sorted(pack.io_ports.items()):
        arrays[f"port__{pid}"] = port.cell_indices
        ports_meta[pid] = {
            "direction": port.direction,
            "label": port.label,
            "description": port.description,
            "selection_rule": port.selection_rule,
        }
    with tmp_npz.open("wb") as fh:
        np.savez(fh, **arrays)
    meta = {
        "cache_key": _cache_key(spec),
        "pack_digest": pack.pack_digest,
        "species_names": list(c.species_names),
        "cell_type": list(c.cell_type),
        "super_class": list(c.super_class),
        "cell_class": list(c.cell_class),
        "cell_sub_class": list(c.cell_sub_class),
        "region": list(c.region),
        "side": list(c.side),
        "neurotransmitter": list(c.neurotransmitter),
        "io_ports": ports_meta,
    }
    tmp_meta.write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp_npz, npz_p)
    os.replace(tmp_meta, meta_p)


def _fill_missing_positions(
    pos_nm: np.ndarray[Any, np.dtype[np.float32]],
    pos_src: np.ndarray[Any, np.dtype[np.uint8]],
    regions: list[str],
) -> None:
    """Fill missing 3D positions with per-region centroid so 3D CNS rendering has zero NaNs."""
    valid = pos_src != POS_MISSING
    if not bool(valid.any()):
        pos_nm.fill(0.0)
        return
    global_mean = pos_nm[valid].mean(axis=0)
    by_reg: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(regions):
        if valid[i]:
            by_reg[r].append(i)
    reg_mean = {
        r: pos_nm[np.asarray(idxs, dtype=np.int32)].mean(axis=0) for r, idxs in by_reg.items()
    }
    for i in range(pos_nm.shape[0]):
        if not valid[i]:
            pos_nm[i] = reg_mean.get(regions[i], global_mean)
            pos_src[i] = POS_ANCHOR


def _map_banc_role(super_class: str, flow: str) -> int:
    sc = super_class.lower()
    if sc in ("sensory", "sensory_ascending"):
        return ROLE_SENSORY
    if sc == "motor":
        return ROLE_MOTOR
    if sc in (
        "central_brain_intrinsic",
        "optic_lobe_intrinsic",
        "ventral_nerve_cord_intrinsic",
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


def _dedup_sort_edges(
    pre: np.ndarray[Any, np.dtype[np.int32]],
    post: np.ndarray[Any, np.dtype[np.int32]],
    count: np.ndarray[Any, np.dtype[np.int32]],
    cell_signs: np.ndarray[Any, np.dtype[np.int8]],
) -> EdgeTable:
    """Merge any duplicate (pre, post) pairs deterministically and apply Dale's-law signs."""
    if pre.size == 0:
        return EdgeTable(
            pre=np.zeros(0, dtype=np.int32),
            post=np.zeros(0, dtype=np.int32),
            count=np.zeros(0, dtype=np.int32),
            sign=np.zeros(0, dtype=np.int8),
            kind=np.zeros(0, dtype=np.uint8),
        )
    keys = (pre.astype(np.uint64) << np.uint64(32)) | post.astype(np.uint64)
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    count_s = count[order].astype(np.int64, copy=False)
    uniq_mask = np.empty(keys_s.size, dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = keys_s[1:] != keys_s[:-1]
    if bool(uniq_mask.all()):
        u_pre = pre[order].astype(np.int32, copy=False)
        u_post = post[order].astype(np.int32, copy=False)
        u_cnt = count[order].astype(np.int32, copy=False)
    else:
        starts = np.flatnonzero(uniq_mask)
        u_keys = keys_s[starts]
        u_cnt = np.add.reduceat(count_s, starts).astype(np.int32, copy=False)
        u_pre = (u_keys >> np.uint64(32)).astype(np.int32)
        u_post = (u_keys & np.uint64(0xFFFFFFFF)).astype(np.int32)
    u_sign = cell_signs[u_pre]
    u_kind = np.full(u_pre.size, SYNAPSE_CHEMICAL, dtype=np.uint8)
    return EdgeTable(
        pre=np.ascontiguousarray(u_pre, dtype=np.int32),
        post=np.ascontiguousarray(u_post, dtype=np.int32),
        count=np.ascontiguousarray(u_cnt, dtype=np.int32),
        sign=np.ascontiguousarray(u_sign, dtype=np.int8),
        kind=u_kind,
    )


def load_banc_888(datasets_dir: Path, *, verify_hashes: bool = True) -> ConnectomePack:
    """Load BANC v888 adult female Brain + VNC connectome from verified Feather/CSV tables."""
    spec = BANC_888_SPEC
    if verify_hashes:
        rep = verify_dataset(spec.dataset_id, datasets_dir, spec=spec)
        if not rep["ok"]:
            missing = [f["filename"] for f in rep["files"] if not f["ok"]]
            raise DatasetError(f"banc_888 verification failed for: {', '.join(missing)}")

    cached = _try_load_cache(spec, datasets_dir)
    if cached is not None:
        return cached

    rdir = raw_dir(datasets_dir, spec.dataset_id)
    meta_tbl = pf.read_table(
        rdir / "banc_888_meta.feather",
        columns=[
            "banc_888_id",
            "proofread",
            "roughly_proofread",
            "side",
            "root_position_nm",
            "region",
            "root_region",
            "neuromere",
            "flow",
            "super_class",
            "cell_class",
            "cell_sub_class",
            "cell_type",
            "fafb_cell_type",
            "manc_cell_type",
            "body_part_sensory",
            "body_part_effector",
            "peripheral_target_type",
            "cell_function",
            "cell_function_detailed",
            "neurotransmitter_predicted",
            "neurotransmitter_verified",
        ],
    )

    # Supplement missing neurotransmitter predictions from v2 CSV table.
    nt_extra: dict[int, str] = {}
    nt_csv = rdir / "banc_888_neurotransmitter_prediction_v2.csv"
    if nt_csv.is_file():
        nt_tbl = pa_csv.read_csv(nt_csv)
        cols = set(nt_tbl.column_names)
        id_col = (
            "root_id" if "root_id" in cols else ("banc_888_id" if "banc_888_id" in cols else None)
        )
        pred_col = (
            "top_nt" if "top_nt" in cols else ("predicted_nt" if "predicted_nt" in cols else None)
        )
        if id_col and pred_col:
            for rid_v, nt_v in zip(
                nt_tbl[id_col].to_pylist(), nt_tbl[pred_col].to_pylist(), strict=False
            ):
                if rid_v is not None and nt_v:
                    with contextlib.suppress(ValueError):
                        nt_extra[int(rid_v)] = _norm_nt(str(nt_v))

    n_raw = meta_tbl.num_rows
    col_id = meta_tbl["banc_888_id"].to_pylist()
    col_pr = meta_tbl["proofread"].to_pylist()
    col_rpr = meta_tbl["roughly_proofread"].to_pylist()
    col_side = meta_tbl["side"].to_pylist()
    col_pos = meta_tbl["root_position_nm"].to_pylist()
    col_reg = meta_tbl["region"].to_pylist()
    col_rreg = meta_tbl["root_region"].to_pylist()
    col_flow = meta_tbl["flow"].to_pylist()
    col_sc = meta_tbl["super_class"].to_pylist()
    col_cc = meta_tbl["cell_class"].to_pylist()
    col_cs = meta_tbl["cell_sub_class"].to_pylist()
    col_ct = meta_tbl["cell_type"].to_pylist()
    col_ft = meta_tbl["fafb_cell_type"].to_pylist()
    col_mt = meta_tbl["manc_cell_type"].to_pylist()
    col_bs = meta_tbl["body_part_sensory"].to_pylist()
    col_be = meta_tbl["body_part_effector"].to_pylist()
    col_pt = meta_tbl["peripheral_target_type"].to_pylist()
    col_cf = meta_tbl["cell_function"].to_pylist()
    col_cfd = meta_tbl["cell_function_detailed"].to_pylist()
    col_ntp = meta_tbl["neurotransmitter_predicted"].to_pylist()
    col_ntv = meta_tbl["neurotransmitter_verified"].to_pylist()

    keep_rows: list[int] = []
    seen_ids: set[int] = set()
    for i in range(n_raw):
        rid_raw = col_id[i]
        if not rid_raw:
            continue
        sc_v = (col_sc[i] or "").strip()
        if sc_v == "glia":
            continue
        is_kept = (
            col_pr[i] == "TRUE"
            or col_rpr[i] == "TRUE"
            or sc_v in ("motor", "descending", "ascending", "sensory", "sensory_ascending")
        )
        if not is_kept:
            continue
        rid = int(rid_raw)
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        keep_rows.append(i)

    # Sort deterministically by source_id ascending.
    keep_rows.sort(key=lambda idx: int(col_id[idx]))
    n = len(keep_rows)
    source_id = np.empty(n, dtype=np.int64)
    role = np.empty(n, dtype=np.uint8)
    pos_nm = np.full((n, 3), np.nan, dtype=np.float32)
    pos_src = np.full(n, POS_MISSING, dtype=np.uint8)
    cell_signs = np.zeros(n, dtype=np.int8)

    cell_type: list[str] = []
    super_class: list[str] = []
    cell_class: list[str] = []
    cell_sub_class: list[str] = []
    region: list[str] = []
    side: list[str] = []
    neurotransmitter: list[str] = []

    port_lists: dict[str, list[int]] = defaultdict(list)
    leg_eff_map = {
        ("front_leg", "left"): "LF",
        ("middle_leg", "left"): "LM",
        ("hind_leg", "left"): "LH",
        ("front_leg", "right"): "RF",
        ("middle_leg", "right"): "RM",
        ("hind_leg", "right"): "RH",
    }

    for out_i, i in enumerate(keep_rows):
        rid = int(col_id[i])
        source_id[out_i] = rid
        sc_v = (col_sc[i] or "unknown").strip()
        cc_v = (col_cc[i] or "").strip()
        cs_v = (col_cs[i] or "").strip()
        ct_v = (col_ct[i] or col_ft[i] or col_mt[i] or "").strip()
        ft_v = (col_ft[i] or "").strip()
        reg_v = (col_reg[i] or col_rreg[i] or "unknown").strip()
        sd_v = _norm_side(col_side[i])
        nt_v = _norm_nt(col_ntv[i] or col_ntp[i] or nt_extra.get(rid))
        role[out_i] = _map_banc_role(sc_v, col_flow[i] or "")
        cell_signs[out_i] = np.int8(_nt_sign(nt_v))

        p_raw = col_pos[i]
        if p_raw:
            parts = str(p_raw).split(",")
            if len(parts) == 3:
                try:
                    # BANC root_position_nm: (X=L/R, Y=anterior-posterior Brain->VNC, Z=D/V).
                    # Keep exact nm coordinates (Brain at lower Y, VNC at higher Y).
                    pos_nm[out_i, 0] = float(parts[0])
                    pos_nm[out_i, 1] = float(parts[1])
                    pos_nm[out_i, 2] = float(parts[2])
                    pos_src[out_i] = POS_SOMA
                except ValueError:
                    pass

        cell_type.append(ct_v)
        super_class.append(sc_v)
        cell_class.append(cc_v)
        cell_sub_class.append(cs_v)
        region.append(reg_v)
        side.append(sd_v)
        neurotransmitter.append(nt_v)

        # Resolve sensory, descending, brain motor, and 36-channel VNC leg motor pools!
        cf_v = (col_cf[i] or "").lower()
        cfd_v = (col_cfd[i] or "").lower()
        bs_v = (col_bs[i] or "").lower()
        be_v = (col_be[i] or "").lower()
        pt_v = (col_pt[i] or "").lower()

        if cf_v == "gustatory":
            if "sugar" in cfd_v or "gr5a" in cfd_v or "gr64f" in cfd_v:
                port_lists["sugar_grn"].append(out_i)
            elif "bitter" in cfd_v or "gr33a" in cfd_v or "gr66a" in cfd_v:
                port_lists["bitter_grn"].append(out_i)
        if cc_v == "olfactory_receptor_neuron":
            port_lists["olfactory_sensory"].append(out_i)
        if bs_v == "antenna" and cs_v in (
            "johnstons_organ_B_neuron",
            "johnstons_organ_C_neuron",
            "johnstons_organ_E_neuron",
            "antenna_bristle_neuron",
        ):
            port_lists["antennal_grooming_mechanosensory"].append(out_i)

        names = {ct_v, ft_v, (col_mt[i] or "").strip()}
        if "MN9" in names or "CB0701" in names:
            port_lists["mn9_proboscis_extension"].append(out_i)
        if "DNp09" in names:
            port_lists["forward_dn"].append(out_i)
        if "DNa01" in names or "DNa02" in names:
            if sd_v == "left":
                port_lists["turn_left_dn"].append(out_i)
            elif sd_v == "right":
                port_lists["turn_right_dn"].append(out_i)
        if "MDN" in names:
            port_lists["backward_dn"].append(out_i)
        if "DNg84" in names:
            port_lists["grooming_dn"].append(out_i)

        # VNC leg motor neuron pools (6 legs x 6 muscle groups = 36 channels)
        leg = leg_eff_map.get((be_v, sd_v))
        if leg is not None and sc_v == "motor":
            mgrp = _BANC_MUSCLE_GROUPS.get(pt_v)
            if mgrp is not None:
                port_lists[f"mn_{leg}_{mgrp}"].append(out_i)

        # VNC leg proprioceptive & load sensors (6 legs)
        sleg = leg_eff_map.get((bs_v, sd_v))
        if sleg is not None and (
            cf_v == "proprioception"
            or pt_v in ("chordotonal_organ", "campaniform_sensillum", "hair_plate")
        ):
            port_lists[f"proprio_{sleg}"].append(out_i)

    _fill_missing_positions(pos_nm, pos_src, region)

    # Load and map edges via vectorized np.searchsorted on sorted source_id.
    edge_tbl = pf.read_table(
        rdir / "banc_888_edgelist_simple_v3.feather", columns=["pre", "post", "count"]
    )
    pre_raw = edge_tbl["pre"].to_numpy(zero_copy_only=False)
    post_raw = edge_tbl["post"].to_numpy(zero_copy_only=False)
    cnt_raw = edge_tbl["count"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    pre_ids = pre_raw.astype(np.int64, copy=False)
    post_ids = post_raw.astype(np.int64, copy=False)

    pre_idx = np.searchsorted(source_id, pre_ids)
    post_idx = np.searchsorted(source_id, post_ids)
    valid_e = (
        (pre_idx < n)
        & (post_idx < n)
        & (cnt_raw > 0)
        & (source_id[np.minimum(pre_idx, n - 1)] == pre_ids)
        & (source_id[np.minimum(post_idx, n - 1)] == post_ids)
    )
    edges = _dedup_sort_edges(
        pre_idx[valid_e].astype(np.int32, copy=False),
        post_idx[valid_e].astype(np.int32, copy=False),
        cnt_raw[valid_e],
        cell_signs,
    )

    cells = CellTable(
        source_id=source_id,
        species_names=(spec.species,),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(cell_type),
        super_class=tuple(super_class),
        cell_class=tuple(cell_class),
        cell_sub_class=tuple(cell_sub_class),
        region=tuple(region),
        side=tuple(side),
        role=role,
        cell_model=np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8),
        neurotransmitter=tuple(neurotransmitter),
        position_nm=pos_nm,
        position_source=pos_src,
    )
    cells.validate()
    edges.validate(n)

    io_ports = _build_cns_ports(port_lists, dataset_label="BANC v888")
    prov = spec.to_provenance()
    digest = compute_pack_digest(spec.dataset_id, spec.species, cells, edges, io_ports, prov)
    pack = ConnectomePack(
        dataset_id=spec.dataset_id,
        species=spec.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
        pack_digest=digest,
    )
    _save_cache(spec, datasets_dir, pack)
    return pack


def _classify_malecns_leg_muscle(type_name: str) -> str | None:
    t = type_name.lower()
    if "anterior rotator" in t or "promotor" in t:
        return "coxa_pro"
    if "posterior rotator" in t or "remotor" in t or "adductor" in t:
        return "coxa_ret"
    if "tr flexor" in t or "fe reductor" in t:
        return "fem_flex"
    if "tr extensor" in t or "trochanter" in t or "tergotr" in t:
        return "fem_ext"
    if "ti extensor" in t or "ta levator" in t:
        return "tib_ext"
    if "ti flexor" in t or "ltm" in t or "ta depressor" in t:
        return "tib_flex"
    return None


def load_male_cns_v1(datasets_dir: Path, *, verify_hashes: bool = True) -> ConnectomePack:
    """Load FlyEM Male CNS v1.0 adult male Brain + VNC connectome from verified Feather tables."""
    spec = MALE_CNS_V1_SPEC
    if verify_hashes:
        rep = verify_dataset(spec.dataset_id, datasets_dir, spec=spec)
        if not rep["ok"]:
            missing = [f["filename"] for f in rep["files"] if not f["ok"]]
            raise DatasetError(f"male_cns_v1 verification failed for: {', '.join(missing)}")

    cached = _try_load_cache(spec, datasets_dir)
    if cached is not None:
        return cached

    rdir = raw_dir(datasets_dir, spec.dataset_id)
    meta_tbl = pf.read_table(
        rdir / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        columns=[
            "bodyId",
            "status",
            "superclass",
            "class",
            "subclass",
            "type",
            "flywireType",
            "mancType",
            "somaSide",
            "rootSide",
            "somaNeuromere",
            "entryNerve",
            "exitNerve",
            "somaLocation",
        ],
    )
    nt_tbl = pf.read_table(
        rdir / "body-neurotransmitters-male-cns-v1.0.feather",
        columns=["body", "consensus_nt", "predicted_nt", "ground_truth"],
    )
    nt_map: dict[int, str] = {}
    for bid_v, gt_v, c_v, p_v in zip(
        nt_tbl["body"].to_pylist(),
        nt_tbl["ground_truth"].to_pylist(),
        nt_tbl["consensus_nt"].to_pylist(),
        nt_tbl["predicted_nt"].to_pylist(),
        strict=False,
    ):
        if bid_v is not None:
            nt_map[int(bid_v)] = _norm_nt(gt_v or c_v or p_v)

    n_raw = meta_tbl.num_rows
    col_id = meta_tbl["bodyId"].to_pylist()
    col_stat = meta_tbl["status"].to_pylist()
    col_sc = meta_tbl["superclass"].to_pylist()
    col_cl = meta_tbl["class"].to_pylist()
    col_sub = meta_tbl["subclass"].to_pylist()
    col_tp = meta_tbl["type"].to_pylist()
    col_fw = meta_tbl["flywireType"].to_pylist()
    col_mt = meta_tbl["mancType"].to_pylist()
    col_sd = meta_tbl["somaSide"].to_pylist()
    col_rs = meta_tbl["rootSide"].to_pylist()
    col_nm = meta_tbl["somaNeuromere"].to_pylist()
    col_en = meta_tbl["entryNerve"].to_pylist()
    col_ex = meta_tbl["exitNerve"].to_pylist()
    col_pos = meta_tbl["somaLocation"].to_pylist()

    keep_rows: list[int] = []
    seen_ids: set[int] = set()
    for i in range(n_raw):
        bid = col_id[i]
        if bid is None:
            continue
        if col_stat[i] != "Traced":
            continue
        if int(bid) in seen_ids:
            continue
        seen_ids.add(int(bid))
        keep_rows.append(i)

    keep_rows.sort(key=lambda idx: int(col_id[idx]))
    n = len(keep_rows)
    source_id = np.empty(n, dtype=np.int64)
    role = np.empty(n, dtype=np.uint8)
    pos_nm = np.full((n, 3), np.nan, dtype=np.float32)
    pos_src = np.full(n, POS_MISSING, dtype=np.uint8)
    cell_signs = np.zeros(n, dtype=np.int8)

    cell_type: list[str] = []
    super_class: list[str] = []
    cell_class: list[str] = []
    cell_sub_class: list[str] = []
    region: list[str] = []
    side: list[str] = []
    neurotransmitter: list[str] = []

    port_lists: dict[str, list[int]] = defaultdict(list)
    nm_leg_map = {
        ("T1", "left"): "LF",
        ("T2", "left"): "LM",
        ("T3", "left"): "LH",
        ("T1", "right"): "RF",
        ("T2", "right"): "RM",
        ("T3", "right"): "RH",
    }
    nerve_leg_map = {
        ("ProLN", "left"): "LF",
        ("MesoLN", "left"): "LM",
        ("MetaLN", "left"): "LH",
        ("ProLN", "right"): "RF",
        ("MesoLN", "right"): "RM",
        ("MetaLN", "right"): "RH",
    }

    for out_i, i in enumerate(keep_rows):
        bid = int(col_id[i])
        source_id[out_i] = bid
        sc_v = (col_sc[i] or "unknown").strip()
        cl_v = (col_cl[i] or "").strip()
        sub_v = (col_sub[i] or "").strip()
        tp_v = (col_tp[i] or col_fw[i] or col_mt[i] or "").strip()
        fw_v = (col_fw[i] or "").strip()
        sd_v = _norm_side(col_sd[i] or col_rs[i])
        nm_v = (col_nm[i] or "").strip()
        nt_v = nt_map.get(bid, "unknown")
        cell_signs[out_i] = np.int8(_nt_sign(nt_v))

        if sc_v in ("vnc_sensory", "cb_sensory", "ol_sensory"):
            role[out_i] = ROLE_SENSORY
        elif sc_v == "vnc_motor" or "motor" in sc_v:
            role[out_i] = ROLE_MOTOR
        elif sc_v != "unknown":
            role[out_i] = ROLE_INTERNEURON
        else:
            role[out_i] = ROLE_OTHER

        if sc_v.startswith("vnc"):
            reg_v = f"ventral_nerve_cord_{nm_v}" if nm_v else "ventral_nerve_cord"
        elif sc_v.startswith("ol"):
            reg_v = "optic_lobe"
        else:
            reg_v = "central_brain"

        p_raw = col_pos[i]
        if p_raw is not None and len(p_raw) == 3:
            # MaleCNS somaLocation is in 8 nm isotropic voxels -> nanometres.
            pos_nm[out_i, 0] = float(p_raw[0]) * 8.0
            pos_nm[out_i, 1] = float(p_raw[1]) * 8.0
            pos_nm[out_i, 2] = float(p_raw[2]) * 8.0
            pos_src[out_i] = POS_SOMA

        cell_type.append(tp_v)
        super_class.append(sc_v)
        cell_class.append(cl_v)
        cell_sub_class.append(sub_v)
        region.append(reg_v)
        side.append(sd_v)
        neurotransmitter.append(nt_v)

        en_v = (col_ex[i] or "").strip()
        in_v = (col_en[i] or "").strip()
        names = {tp_v, fw_v, (col_mt[i] or "").strip()}

        # Brain sensory & motor/DN ports
        if cl_v == "olfactory":
            port_lists["olfactory_sensory"].append(out_i)
        if cl_v == "gustatory":
            if sub_v in ("labellar bristle", "taste peg") or tp_v.startswith(("LB", "claw_")):
                port_lists["sugar_grn"].append(out_i)
            elif "bitter" in tp_v.lower() or tp_v.startswith("LgAG"):
                port_lists["bitter_grn"].append(out_i)
        if cl_v == "mechanosensory" and in_v == "AN":
            port_lists["antennal_grooming_mechanosensory"].append(out_i)
        if "MN9" in names or "CB0701" in names:
            port_lists["mn9_proboscis_extension"].append(out_i)
        if "DNp09" in names:
            port_lists["forward_dn"].append(out_i)
        if "DNa01" in names or "DNa02" in names:
            if sd_v == "left":
                port_lists["turn_left_dn"].append(out_i)
            elif sd_v == "right":
                port_lists["turn_right_dn"].append(out_i)
        if "MDN" in names:
            port_lists["backward_dn"].append(out_i)
        if "DNg84" in names:
            port_lists["grooming_dn"].append(out_i)
        if "pIP10" in names:
            port_lists["song_dn_pip10"].append(out_i)

        # VNC 6-leg x 6-muscle motor pools (use neuromere + side to capture ProLN/ProAN/DProN/VProN)
        if sc_v == "vnc_motor":
            leg = nm_leg_map.get((nm_v, sd_v)) or nerve_leg_map.get((en_v, sd_v))
            if leg is not None:
                mgrp = _classify_malecns_leg_muscle(tp_v)
                if mgrp is not None:
                    port_lists[f"mn_{leg}_{mgrp}"].append(out_i)

        # VNC 6-leg proprioceptive feedback ports
        if cl_v == "mechanosensory_proprioceptive":
            sleg = nerve_leg_map.get((in_v, sd_v)) or nm_leg_map.get((nm_v, sd_v))
            if sleg is not None:
                port_lists[f"proprio_{sleg}"].append(out_i)

    _fill_missing_positions(pos_nm, pos_src, region)

    edge_tbl = pf.read_table(
        rdir / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
        columns=["body_pre", "body_post", "weight"],
    )
    pre_ids = edge_tbl["body_pre"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    post_ids = edge_tbl["body_post"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    cnt_raw = edge_tbl["weight"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)

    pre_idx = np.searchsorted(source_id, pre_ids)
    post_idx = np.searchsorted(source_id, post_ids)
    valid_e = (
        (pre_idx < n)
        & (post_idx < n)
        & (cnt_raw > 0)
        & (source_id[np.minimum(pre_idx, n - 1)] == pre_ids)
        & (source_id[np.minimum(post_idx, n - 1)] == post_ids)
    )
    edges = _dedup_sort_edges(
        pre_idx[valid_e].astype(np.int32, copy=False),
        post_idx[valid_e].astype(np.int32, copy=False),
        cnt_raw[valid_e],
        cell_signs,
    )

    cells = CellTable(
        source_id=source_id,
        species_names=(spec.species,),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(cell_type),
        super_class=tuple(super_class),
        cell_class=tuple(cell_class),
        cell_sub_class=tuple(cell_sub_class),
        region=tuple(region),
        side=tuple(side),
        role=role,
        cell_model=np.full(n, MODEL_SPIKING_LIF, dtype=np.uint8),
        neurotransmitter=tuple(neurotransmitter),
        position_nm=pos_nm,
        position_source=pos_src,
    )
    cells.validate()
    edges.validate(n)

    io_ports = _build_cns_ports(port_lists, dataset_label="MaleCNS v1.0")
    prov = spec.to_provenance()
    digest = compute_pack_digest(spec.dataset_id, spec.species, cells, edges, io_ports, prov)
    pack = ConnectomePack(
        dataset_id=spec.dataset_id,
        species=spec.species,
        cells=cells,
        edges=edges,
        io_ports=io_ports,
        provenance=prov,
        pack_digest=digest,
    )
    _save_cache(spec, datasets_dir, pack)
    return pack


def _build_cns_ports(
    port_lists: dict[str, list[int]],
    *,
    dataset_label: str,
) -> dict[str, IOPort]:
    specs: list[tuple[str, Any, str, str]] = [
        ("sugar_grn", "sensory", "Sugar gustatory receptor neurons", f"{dataset_label} sweet GRNs"),
        (
            "bitter_grn",
            "sensory",
            "Bitter gustatory receptor neurons",
            f"{dataset_label} bitter GRNs",
        ),
        (
            "olfactory_sensory",
            "sensory",
            "Antennal olfactory receptor neurons",
            f"{dataset_label} ORNs",
        ),
        (
            "antennal_grooming_mechanosensory",
            "sensory",
            "Antennal mechanosensory neurons (JO/bristles)",
            f"{dataset_label} antennal mechanosensors",
        ),
        (
            "mn9_proboscis_extension",
            "motor",
            "MN9 proboscis extension motor neurons",
            f"{dataset_label} MN9",
        ),
        (
            "forward_dn",
            "circuit",
            "DNp09 forward walking descending neurons",
            f"{dataset_label} DNp09",
        ),
        (
            "turn_left_dn",
            "circuit",
            "DNa01/DNa02 left steering descending neurons",
            f"{dataset_label} DNa01/02 L",
        ),
        (
            "turn_right_dn",
            "circuit",
            "DNa01/DNa02 right steering descending neurons",
            f"{dataset_label} DNa01/02 R",
        ),
        (
            "backward_dn",
            "circuit",
            "MDN backward walking descending neurons",
            f"{dataset_label} MDN",
        ),
        (
            "grooming_dn",
            "circuit",
            "DNg84 antennal grooming descending neurons",
            f"{dataset_label} DNg84",
        ),
        (
            "song_dn_pip10",
            "circuit",
            "pIP10 male courtship song descending neurons",
            f"{dataset_label} pIP10",
        ),
    ]
    for leg in LEG_NAMES:
        specs.append(
            (
                f"proprio_{leg}",
                "sensory",
                f"{leg} leg proprioceptors (chordotonal/campaniform/hair plate)",
                f"{dataset_label} {leg} leg proprioceptors",
            )
        )
        for mgrp in MUSCLE_CHANNELS:
            specs.append(
                (
                    f"mn_{leg}_{mgrp}",
                    "motor",
                    f"{leg} leg {mgrp} motor neurons",
                    f"{dataset_label} {leg} {mgrp} MNs",
                )
            )
    out: dict[str, IOPort] = {}
    for pid, direction, label, rule in specs:
        idxs = port_lists.get(pid, [])
        if not idxs:
            continue
        out[pid] = IOPort(
            port_id=pid,
            direction=direction,
            label=label,
            description=f"{label} resolved from {dataset_label}.",
            cell_indices=np.asarray(sorted(set(idxs)), dtype=np.int32),
            selection_rule=rule,
        )
    return out
