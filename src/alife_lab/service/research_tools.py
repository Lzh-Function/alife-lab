"""GUI-CUI parity backend services: Connectome dataset manager, research job runner, and backups.

All heavy scientific assays, training jobs, and dataset fetches run in isolated subprocess workers
via strictly allowlisted ``alife_lab.cli`` arguments (``shell=False``, zero arbitrary commands,
GPU runtimes never imported into the FastAPI controller process).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from alife_lab.connectome.celegans import (
    celegans_302_reference_checks,
    make_synthetic_celegans_302_pack,
)
from alife_lab.connectome.chimera import make_chimera_worm_fly_mb_pack
from alife_lab.connectome.embodied import make_synthetic_fly_embodied_pack
from alife_lab.connectome.flywire import (
    flywire_783_reference_checks,
    load_connectome_pack,
    summarize_connectome_pack,
)
from alife_lab.connectome.lif import make_synthetic_mb_pack, make_synthetic_tiny_pack
from alife_lab.connectome.registry import (
    DATASETS,
    DatasetError,
    list_dataset_specs,
    manifest_path,
    raw_dir,
    verify_dataset,
)
from alife_lab.connectome.species_packs import (
    ciona_larva_reference_checks,
    fly_larva_l1_reference_checks,
    make_synthetic_ciona_pack,
    make_synthetic_fly_larva_pack,
    make_synthetic_platynereis_pack,
    platynereis_3d_reference_checks,
)
from alife_lab.service.settings import Settings
from alife_lab.storage.backup import (
    MANIFEST as BACKUP_MANIFEST,
)
from alife_lab.storage.backup import (
    BackupError,
    create_backup,
    restore_backup,
    verify_backup,
)

_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.,:\-\s]{0,256}$")
_BACKUP_ID_RE = re.compile(r"^bk_[0-9TZ]+_[0-9a-f]{4,16}$")
_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12,24}$")

SYNTHETIC_DATASETS: tuple[dict[str, str], ...] = (
    {
        "dataset_id": "synthetic_tiny",
        "species": "synthetic_spiking_lif",
        "version": "synthetic_v1",
        "description": "Offline minimal spiking LIF connectome (64 cells) for CPU tests.",
    },
    {
        "dataset_id": "synthetic_mb",
        "species": "synthetic_mushroom_body",
        "version": "synthetic_v1",
        "description": "Offline synthetic Mushroom Body (KC/MBON/DAN) circuit for F4 plasticity.",
    },
    {
        "dataset_id": "synthetic_celegans_302",
        "species": "caenorhabditis_elegans_synthetic",
        "version": "synthetic_v1",
        "description": "Offline synthetic 302-cell graded + gap-junction C. elegans pack.",
    },
    {
        "dataset_id": "synthetic_fly_larva",
        "species": "drosophila_melanogaster_larva_l1_synthetic",
        "version": "synthetic_v1",
        "description": "Offline synthetic Drosophila L1 larva whole-brain + MB pack.",
    },
    {
        "dataset_id": "synthetic_platynereis",
        "species": "platynereis_dumerilii_3day_larva_synthetic",
        "version": "synthetic_v1",
        "description": "Offline synthetic Platynereis 3-day larva whole-body graded pack.",
    },
    {
        "dataset_id": "synthetic_ciona",
        "species": "ciona_intestinalis_tadpole_larva_synthetic",
        "version": "synthetic_v1",
        "description": "Offline synthetic Ciona intestinalis tadpole larva CNS pack.",
    },
    {
        "dataset_id": "synthetic_chimera_worm_fly_mb",
        "species": "chimera_synthetic_worm_plus_fly_mb",
        "version": "synthetic_v1",
        "description": "Offline exploratory F13c hybrid (synthetic C. elegans + synthetic fly MB).",
    },
    {
        "dataset_id": "synthetic_fly_embodied",
        "species": "drosophila_melanogaster_embodied_synthetic",
        "version": "synthetic_v1",
        "description": "Offline synthetic embodied fly pack for F8/F9/F11/F12 CPU tests.",
    },
)


# --------------------------------------------------------------------------- #
# Connectome dataset management (list / verify / summary)
# --------------------------------------------------------------------------- #


def list_connectome_datasets(datasets_dir: Path) -> list[dict[str, Any]]:
    """Return status and metadata for every empirical and synthetic connectome pack."""
    out: list[dict[str, Any]] = []
    for spec in list_dataset_specs():
        rdir = raw_dir(datasets_dir, spec.dataset_id)
        mpath = manifest_path(datasets_dir, spec.dataset_id)
        expected_bytes = sum(f.size_bytes for f in spec.files)
        local_bytes = 0
        present = 0
        for f in spec.files:
            fp = rdir / f.filename
            if fp.is_file():
                present += 1
                local_bytes += fp.stat().st_size
        out.append(
            {
                "dataset_id": spec.dataset_id,
                "kind": "empirical",
                "species": spec.species,
                "version": spec.version,
                "citation": spec.citation,
                "licence_note": spec.licence_note,
                "num_files": len(spec.files),
                "files_present": present,
                "installed": present == len(spec.files) and len(spec.files) > 0,
                "manifest_present": mpath.is_file(),
                "local_bytes": local_bytes,
                "expected_bytes": expected_bytes,
            }
        )
    for syn in SYNTHETIC_DATASETS:
        out.append(
            {
                "dataset_id": syn["dataset_id"],
                "kind": "synthetic",
                "species": syn["species"],
                "version": syn["version"],
                "citation": syn["description"],
                "licence_note": "Generated deterministically offline inside ALife Lab.",
                "num_files": 0,
                "files_present": 0,
                "installed": True,
                "manifest_present": True,
                "local_bytes": 0,
                "expected_bytes": 0,
            }
        )
    return out


def verify_connectome_dataset(dataset_id: str, datasets_dir: Path) -> dict[str, Any]:
    """Re-hash local files of a registered empirical connectome dataset."""
    if dataset_id not in DATASETS:
        raise DatasetError(f"dataset {dataset_id!r} is not a registered empirical dataset")
    return verify_dataset(dataset_id, datasets_dir)


def _load_any_pack(dataset_id: str, datasets_dir: Path) -> Any:
    if dataset_id == "synthetic_tiny":
        return make_synthetic_tiny_pack()
    if dataset_id == "synthetic_mb":
        return make_synthetic_mb_pack()
    if dataset_id == "synthetic_celegans_302":
        return make_synthetic_celegans_302_pack()
    if dataset_id == "synthetic_fly_larva":
        return make_synthetic_fly_larva_pack()
    if dataset_id == "synthetic_platynereis":
        return make_synthetic_platynereis_pack()
    if dataset_id == "synthetic_ciona":
        return make_synthetic_ciona_pack()
    if dataset_id == "synthetic_chimera_worm_fly_mb":
        return make_chimera_worm_fly_mb_pack(None, None)
    if dataset_id == "synthetic_fly_embodied":
        return make_synthetic_fly_embodied_pack()
    return load_connectome_pack(dataset_id, datasets_dir, verify_hashes=True)


def summarize_connectome_dataset(dataset_id: str, datasets_dir: Path) -> dict[str, Any]:
    """Load any empirical or synthetic connectome pack and return summary + reference checks."""
    pack = _load_any_pack(dataset_id, datasets_dir)
    summary = summarize_connectome_pack(pack)
    checks: list[dict[str, Any]] = []
    if dataset_id == "flywire_783":
        checks = flywire_783_reference_checks(pack)
    elif dataset_id in ("celegans_302", "synthetic_celegans_302"):
        checks = celegans_302_reference_checks(pack)
    elif dataset_id in ("fly_larva_l1", "synthetic_fly_larva"):
        checks = fly_larva_l1_reference_checks(pack)
    elif dataset_id in ("platynereis_3d", "synthetic_platynereis"):
        checks = platynereis_3d_reference_checks(pack)
    elif dataset_id in ("ciona_larva", "synthetic_ciona"):
        checks = ciona_larva_reference_checks(pack)
    return {
        "dataset_id": dataset_id,
        "summary": summary,
        "reference_checks": checks,
        "all_checks_ok": all(bool(c.get("ok")) for c in checks) if checks else None,
    }


# --------------------------------------------------------------------------- #
# Declarative catalog of ALL CLI research tools (GUI-CUI parity)
# --------------------------------------------------------------------------- #

ParamKind = Literal["str", "int", "float", "bool", "select"]


@dataclass(frozen=True)
class ToolParamSpec:
    name: str
    kind: ParamKind
    default: Any
    label_key: str
    cli_flag: str | None = None  # None => positional arg
    choices: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    required: bool = False
    is_repeatable_csv: bool = False  # e.g. --stim sugar_grn:150


@dataclass(frozen=True)
class ResearchToolSpec:
    id: str
    category: Literal[
        "species_chimera", "circuits_plasticity", "learning_fit", "embodied_social", "cultural"
    ]
    title_key: str
    description_key: str
    cli_command: tuple[str, ...]  # e.g. ("connectome", "compare-species")
    supports_json: bool
    params: tuple[ToolParamSpec, ...]
    gpu_optional: bool = False
    timeout_seconds: int = 900


_P = ToolParamSpec
_T = ResearchToolSpec

RESEARCH_TOOLS: tuple[ResearchToolSpec, ...] = (
    # ---- Category 1: Cross-Species & Chimeras (Stage 4 / S2-S4 / F10-B / F13) ----
    _T(
        id="connectome_compare_species",
        category="species_chimera",
        title_key="research.tool.compare_species.title",
        description_key="research.tool.compare_species.desc",
        cli_command=("connectome", "compare-species"),
        supports_json=True,
        params=(
            _P(
                "datasets",
                "str",
                "ciona_larva,celegans_302,platynereis_3d,fly_larva_l1",
                "research.param.datasets",
                "--datasets",
            ),
            _P(
                "ticks", "int", 120, "research.param.ticks", "--ticks", min_value=10, max_value=1000
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
            _P("no_physiology", "bool", False, "research.param.no_physiology", "--no-physiology"),
        ),
    ),
    _T(
        id="connectome_chimera",
        category="species_chimera",
        title_key="research.tool.chimera.title",
        description_key="research.tool.chimera.desc",
        cli_command=("connectome", "chimera"),
        supports_json=True,
        params=(
            _P(
                "kind",
                "select",
                "f13c",
                "research.param.chimera_kind",
                "--kind",
                choices=("f13c", "f13b", "f13a"),
            ),
            _P("host", "str", "celegans_302", "research.param.host", "--host"),
            _P("donor", "str", "fly_larva_l1", "research.param.donor", "--donor"),
            _P(
                "blend",
                "float",
                0.5,
                "research.param.blend",
                "--blend",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
            ),
            _P("trials", "int", 4, "research.param.trials", "--trials", min_value=1, max_value=24),
            _P("ticks", "int", 80, "research.param.ticks", "--ticks", min_value=10, max_value=500),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_compare_sex",
        category="species_chimera",
        title_key="research.tool.compare_sex.title",
        description_key="research.tool.compare_sex.desc",
        cli_command=("connectome", "compare-sex"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "datasets",
                "str",
                "flywire_783,banc_888,male_cns_v1",
                "research.param.datasets",
                "--datasets",
            ),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
            _P(
                "ticks", "int", 300, "research.param.ticks", "--ticks", min_value=20, max_value=2000
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    # ---- Category 2: Connectome Circuits & Plasticity (F3, F4) ----
    _T(
        id="connectome_screen",
        category="circuits_plasticity",
        title_key="research.tool.screen.title",
        description_key="research.tool.screen.desc",
        cli_command=("connectome", "screen"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "str",
                "synthetic_tiny",
                "research.param.dataset_id",
                None,
                required=True,
            ),
            _P(
                "stim",
                "str",
                "sugar_grn:150",
                "research.param.stim",
                "--stim",
                required=True,
                is_repeatable_csv=True,
            ),
            _P(
                "target", "str", "mn9_proboscis_extension", "research.param.target_port", "--target"
            ),
            _P("top_k", "int", 8, "research.param.top_k", "--top-k", min_value=1, max_value=64),
            _P("seeds", "str", "1,2", "research.param.seeds", "--seeds"),
            _P(
                "ticks", "int", 200, "research.param.ticks", "--ticks", min_value=20, max_value=2000
            ),
            _P(
                "wiring",
                "select",
                "reference",
                "research.param.wiring",
                "--wiring",
                choices=("reference", "shuffled_control"),
            ),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
        ),
    ),
    _T(
        id="connectome_pavlov",
        category="circuits_plasticity",
        title_key="research.tool.pavlov.title",
        description_key="research.tool.pavlov.desc",
        cli_command=("connectome", "pavlov"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "str",
                "synthetic_mb",
                "research.param.dataset_id",
                None,
                required=True,
            ),
            _P("seeds", "str", "1,2,3", "research.param.seeds", "--seeds"),
            _P("trials", "int", 4, "research.param.trials", "--trials", min_value=1, max_value=12),
            _P(
                "odor_hz",
                "float",
                150.0,
                "research.param.odor_hz",
                "--odor-hz",
                min_value=10.0,
                max_value=600.0,
                step=10.0,
            ),
            _P(
                "dan_hz",
                "float",
                220.0,
                "research.param.dan_hz",
                "--dan-hz",
                min_value=10.0,
                max_value=600.0,
                step=10.0,
            ),
            _P(
                "wiring",
                "select",
                "reference",
                "research.param.wiring",
                "--wiring",
                choices=("reference", "shuffled_control"),
            ),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
        ),
    ),
    # ---- Category 3: System ID, Task Learning & Real Activity Fitting (F5, F6, F7, S1) ----
    _T(
        id="connectome_sysid",
        category="learning_fit",
        title_key="research.tool.sysid.title",
        description_key="research.tool.sysid.desc",
        cli_command=("connectome", "sysid"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "str",
                "synthetic_olfactory",
                "research.param.dataset_id",
                None,
                required=True,
            ),
            _P(
                "subcircuit",
                "select",
                "olfactory_mb_lh",
                "research.param.subcircuit",
                "--subcircuit",
                choices=("olfactory_mb_lh", "full"),
            ),
            _P(
                "obs_fraction",
                "float",
                0.25,
                "research.param.obs_fraction",
                "--obs-fraction",
                min_value=0.02,
                max_value=1.0,
                step=0.05,
            ),
            _P(
                "epochs", "int", 60, "research.param.epochs", "--epochs", min_value=5, max_value=400
            ),
            _P(
                "restarts",
                "int",
                2,
                "research.param.restarts",
                "--restarts",
                min_value=1,
                max_value=8,
            ),
            _P(
                "device",
                "select",
                "auto",
                "research.param.device",
                "--device",
                choices=("auto", "cpu", "cuda"),
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_sysid_map",
        category="learning_fit",
        title_key="research.tool.sysid_map.title",
        description_key="research.tool.sysid_map.desc",
        cli_command=("connectome", "sysid-map"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "str",
                "synthetic_olfactory",
                "research.param.dataset_id",
                None,
                required=True,
            ),
            _P("fractions", "str", "0.10,0.30,0.60,1.0", "research.param.fractions", "--fractions"),
            _P(
                "epochs", "int", 40, "research.param.epochs", "--epochs", min_value=5, max_value=300
            ),
            _P(
                "restarts",
                "int",
                2,
                "research.param.restarts",
                "--restarts",
                min_value=1,
                max_value=5,
            ),
            _P(
                "device",
                "select",
                "auto",
                "research.param.device",
                "--device",
                choices=("auto", "cpu", "cuda"),
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_task",
        category="learning_fit",
        title_key="research.tool.task.title",
        description_key="research.tool.task.desc",
        cli_command=("connectome", "task"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "str",
                "synthetic_olfactory",
                "research.param.dataset_id",
                None,
                required=True,
            ),
            _P(
                "task",
                "select",
                "odor_classification",
                "research.param.task_kind",
                "--task",
                choices=("odor_classification", "short_memory", "nonlinear_xor"),
            ),
            _P(
                "scope",
                "select",
                "readout_only",
                "research.param.scope",
                "--scope",
                choices=("readout_only", "io_only", "partial_weights"),
            ),
            _P("seeds", "str", "1,2,3", "research.param.seeds", "--seeds"),
            _P(
                "epochs", "int", 50, "research.param.epochs", "--epochs", min_value=5, max_value=300
            ),
            _P(
                "device",
                "select",
                "auto",
                "research.param.device",
                "--device",
                choices=("auto", "cpu", "cuda"),
            ),
        ),
    ),
    _T(
        id="connectome_fit_real",
        category="learning_fit",
        title_key="research.tool.fit_real.title",
        description_key="research.tool.fit_real.desc",
        cli_command=("connectome", "fit-real"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "source",
                "select",
                "synthetic",
                "research.param.data_source",
                None,
                choices=("synthetic", "aimon2023", "door2"),
                required=True,
            ),
            _P("dataset_id", "str", "flywire_783", "research.param.dataset_id", "--dataset-id"),
            _P(
                "max_flies",
                "int",
                5,
                "research.param.max_flies",
                "--max-flies",
                min_value=1,
                max_value=30,
            ),
            _P(
                "epochs", "int", 60, "research.param.epochs", "--epochs", min_value=5, max_value=400
            ),
            _P("seeds", "str", "1,2,3", "research.param.seeds", "--seeds"),
            _P(
                "device",
                "select",
                "auto",
                "research.param.device",
                "--device",
                choices=("auto", "cpu", "cuda"),
            ),
        ),
    ),
    _T(
        id="connectome_fit_worm",
        category="learning_fit",
        title_key="research.tool.fit_worm.title",
        description_key="research.tool.fit_worm.desc",
        cli_command=("connectome", "fit-worm"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "source",
                "select",
                "synthetic",
                "research.param.data_source",
                "--source",
                choices=("synthetic", "real"),
            ),
            _P("dataset_id", "str", "celegans_302", "research.param.dataset_id", "--dataset-id"),
            _P(
                "max_worms",
                "int",
                5,
                "research.param.max_worms",
                "--max-worms",
                min_value=1,
                max_value=30,
            ),
            _P(
                "epochs", "int", 60, "research.param.epochs", "--epochs", min_value=5, max_value=400
            ),
            _P("seeds", "str", "1,2,3", "research.param.seeds", "--seeds"),
            _P(
                "device",
                "select",
                "auto",
                "research.param.device",
                "--device",
                choices=("auto", "cpu", "cuda"),
            ),
        ),
    ),
    # ---- Category 4: Embodied Fly, VNC, Vision, Social, Swarm & Evolution (F8-F12) ----
    _T(
        id="connectome_vision",
        category="embodied_social",
        title_key="research.tool.vision.title",
        description_key="research.tool.vision.desc",
        cli_command=("connectome", "vision"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "frames", "int", 30, "research.param.frames", "--frames", min_value=8, max_value=120
            ),
            _P("synthetic", "bool", True, "research.param.synthetic_mode", "--synthetic"),
        ),
    ),
    _T(
        id="connectome_embody",
        category="embodied_social",
        title_key="research.tool.embody.title",
        description_key="research.tool.embody.desc",
        cli_command=("connectome", "embody"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "select",
                "synthetic_fly_embodied",
                "research.param.dataset_id",
                None,
                choices=("synthetic_fly_embodied", "flywire_783"),
                required=True,
            ),
            _P(
                "ticks", "int", 120, "research.param.ticks", "--ticks", min_value=20, max_value=2000
            ),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_embody_l2",
        category="embodied_social",
        title_key="research.tool.embody_l2.title",
        description_key="research.tool.embody_l2.desc",
        cli_command=("connectome", "embody-l2"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "select",
                "synthetic_cns_embodied",
                "research.param.dataset_id",
                None,
                choices=("synthetic_cns_embodied", "banc_888", "male_cns_v1"),
                required=True,
            ),
            _P(
                "mode",
                "select",
                "level2_vnc",
                "research.param.vnc_mode",
                "--mode",
                choices=("level2_vnc", "level2_vnc_tuned"),
            ),
            _P(
                "ticks", "int", 200, "research.param.ticks", "--ticks", min_value=20, max_value=2000
            ),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_tune_vnc",
        category="embodied_social",
        title_key="research.tool.tune_vnc.title",
        description_key="research.tool.tune_vnc.desc",
        cli_command=("connectome", "tune-vnc"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "select",
                "synthetic_cns_embodied",
                "research.param.dataset_id",
                None,
                choices=("synthetic_cns_embodied", "banc_888", "male_cns_v1"),
                required=True,
            ),
            _P(
                "ticks", "int", 250, "research.param.ticks", "--ticks", min_value=50, max_value=2000
            ),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_learn_vnc",
        category="embodied_social",
        title_key="research.tool.learn_vnc.title",
        description_key="research.tool.learn_vnc.desc",
        cli_command=("connectome", "learn-vnc"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "select",
                "synthetic_cns_embodied",
                "research.param.dataset_id",
                None,
                choices=("synthetic_cns_embodied", "banc_888", "male_cns_v1"),
                required=True,
            ),
            _P(
                "generations",
                "int",
                8,
                "research.param.generations",
                "--generations",
                min_value=2,
                max_value=100,
            ),
            _P("pop", "int", 8, "research.param.population", "--pop", min_value=4, max_value=64),
            _P(
                "ticks", "int", 200, "research.param.ticks", "--ticks", min_value=50, max_value=2000
            ),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_embody_batch",
        category="embodied_social",
        title_key="research.tool.embody_batch.title",
        description_key="research.tool.embody_batch.desc",
        cli_command=("connectome", "embody-batch"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "select",
                "synthetic_fly_embodied",
                "research.param.dataset_id",
                None,
                choices=("synthetic_fly_embodied", "flywire_783"),
                required=True,
            ),
            _P(
                "mode",
                "select",
                "benchmark",
                "research.param.batch_mode",
                "--mode",
                choices=("benchmark", "screen", "search"),
            ),
            _P(
                "ticks", "int", 120, "research.param.ticks", "--ticks", min_value=20, max_value=1000
            ),
            _P(
                "batch_size",
                "int",
                8,
                "research.param.batch_size",
                "--batch-size",
                min_value=1,
                max_value=64,
            ),
            _P(
                "physics",
                "select",
                "auto",
                "research.param.physics",
                "--physics",
                choices=("auto", "warp", "cpu"),
            ),
            _P(
                "brain",
                "select",
                "auto",
                "research.param.backend",
                "--brain",
                choices=("auto", "cupy", "numpy"),
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_social",
        category="embodied_social",
        title_key="research.tool.social.title",
        description_key="research.tool.social.desc",
        cli_command=("connectome", "social"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "stage",
                "select",
                "dialogue",
                "research.param.social_stage",
                "--stage",
                choices=("dialogue", "arena"),
            ),
            _P("male", "str", "synthetic_social_male", "research.param.male_dataset", "--male"),
            _P(
                "female",
                "str",
                "synthetic_social_female",
                "research.param.female_dataset",
                "--female",
            ),
            _P(
                "ticks", "int", 250, "research.param.ticks", "--ticks", min_value=40, max_value=2000
            ),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_swarm",
        category="embodied_social",
        title_key="research.tool.swarm.title",
        description_key="research.tool.swarm.desc",
        cli_command=("connectome", "swarm"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "str",
                "synthetic_fly_embodied",
                "research.param.dataset_id",
                None,
                required=True,
            ),
            _P(
                "mode",
                "select",
                "ramdya",
                "research.param.swarm_mode",
                "--mode",
                choices=("ramdya", "aggression", "ecosystem"),
            ),
            _P("sizes", "str", "1,4,8", "research.param.sizes", "--sizes"),
            _P(
                "ticks", "int", 400, "research.param.ticks", "--ticks", min_value=50, max_value=4000
            ),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
        ),
    ),
    _T(
        id="connectome_evolve",
        category="embodied_social",
        title_key="research.tool.evolve.title",
        description_key="research.tool.evolve.desc",
        cli_command=("connectome", "evolve"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P(
                "dataset_id",
                "str",
                "synthetic_fly_embodied",
                "research.param.dataset_id",
                None,
                required=True,
            ),
            _P("seeds", "str", "1,2", "research.param.seeds", "--seeds"),
            _P(
                "evo_ticks",
                "int",
                400,
                "research.param.evo_ticks",
                "--evo-ticks",
                min_value=50,
                max_value=5000,
            ),
            _P(
                "eval_ticks",
                "int",
                200,
                "research.param.eval_ticks",
                "--eval-ticks",
                min_value=50,
                max_value=2000,
            ),
            _P("replay_fine", "bool", False, "research.param.replay_fine", "--replay-fine"),
            _P(
                "backend",
                "select",
                "auto",
                "research.param.backend",
                "--backend",
                choices=("auto", "numpy", "cupy"),
            ),
        ),
    ),
    # ---- Category 5: Cultural Evolution Assays, Phase Diagrams & Benchmarks (P4-P7) ----
    _T(
        id="cultural_assay",
        category="cultural",
        title_key="research.tool.cultural_assay.title",
        description_key="research.tool.cultural_assay.desc",
        cli_command=("cultural", "assay"),
        supports_json=True,
        params=(
            _P(
                "preset",
                "select",
                "cultural_world_compositional",
                "research.param.preset",
                None,
                choices=(
                    "cultural_world_compositional",
                    "cultural_world_no_composition",
                    "cultural_world_externalized",
                    "cultural_world_baseline",
                ),
                required=True,
            ),
            _P(
                "seeds", "int", 3, "research.param.seed_count", "--seeds", min_value=1, max_value=20
            ),
            _P(
                "steps", "int", 400, "research.param.steps", "--steps", min_value=60, max_value=2000
            ),
        ),
    ),
    _T(
        id="cultural_phase_diagram",
        category="cultural",
        title_key="research.tool.cultural_phase.title",
        description_key="research.tool.cultural_phase.desc",
        cli_command=("cultural", "phase-diagram"),
        supports_json=True,
        params=(
            _P(
                "preset",
                "select",
                "cultural_world_compositional",
                "research.param.preset",
                None,
                choices=(
                    "cultural_world_compositional",
                    "cultural_world_externalized",
                    "cultural_world_baseline",
                ),
                required=True,
            ),
            _P("periods", "str", "80,200,500", "research.param.periods", "--periods"),
            _P("costs", "str", "0.0,0.015,0.04", "research.param.costs", "--costs"),
            _P("seeds", "str", "0,1", "research.param.seeds", "--seeds"),
            _P(
                "steps", "int", 300, "research.param.steps", "--steps", min_value=60, max_value=2000
            ),
        ),
    ),
    _T(
        id="cultural_social_assay",
        category="cultural",
        title_key="research.tool.cultural_social.title",
        description_key="research.tool.cultural_social.desc",
        cli_command=("cultural", "social-assay"),
        supports_json=True,
        params=(
            _P(
                "preset",
                "select",
                "cultural_world_exchange",
                "research.param.preset",
                None,
                choices=("cultural_world_exchange", "cultural_world_compositional"),
                required=True,
            ),
            _P(
                "seeds", "int", 3, "research.param.seed_count", "--seeds", min_value=1, max_value=10
            ),
            _P(
                "steps", "int", 400, "research.param.steps", "--steps", min_value=80, max_value=1500
            ),
        ),
    ),
    _T(
        id="cultural_benchmark",
        category="cultural",
        title_key="research.tool.cultural_benchmark.title",
        description_key="research.tool.cultural_benchmark.desc",
        cli_command=("cultural", "benchmark"),
        supports_json=True,
        gpu_optional=True,
        params=(
            _P("populations", "str", "80,200,400", "research.param.populations", "--populations"),
            _P("skills", "str", "2,4,8", "research.param.skills", "--skills"),
            _P("steps", "int", 100, "research.param.steps", "--steps", min_value=20, max_value=500),
            _P("seed", "int", 1, "research.param.seed", "--seed", min_value=0, max_value=100000),
            _P("include_gpu", "bool", False, "research.param.include_gpu", "--include-gpu"),
        ),
    ),
)

RESEARCH_TOOLS_BY_ID: dict[str, ResearchToolSpec] = {t.id: t for t in RESEARCH_TOOLS}


def list_research_tools() -> list[dict[str, Any]]:
    """Serialize the allowlisted research tool catalog for the React Web GUI."""
    return [asdict(t) for t in RESEARCH_TOOLS]


def build_tool_argv(
    tool_id: str, params: dict[str, Any]
) -> tuple[ResearchToolSpec, list[str], dict[str, Any]]:
    """Validate ``params`` against ``tool_id`` and return a strict ``shell=False`` CLI argv."""
    spec = RESEARCH_TOOLS_BY_ID.get(tool_id)
    if spec is None:
        raise ValueError(f"unknown research tool {tool_id!r}")

    extra_keys = set(params) - {p.name for p in spec.params}
    if extra_keys:
        raise ValueError(f"unexpected parameter(s) for {tool_id}: {sorted(extra_keys)}")

    argv: list[str] = list(spec.cli_command)
    normalized: dict[str, Any] = {}

    for p in spec.params:
        raw_val = params.get(p.name, p.default)
        if p.required and (raw_val is None or raw_val == ""):
            raise ValueError(f"parameter {p.name!r} is required")

        if p.kind == "bool":
            val = bool(raw_val)
            normalized[p.name] = val
            if val and p.cli_flag:
                argv.append(p.cli_flag)
            continue

        if p.kind == "int":
            if isinstance(raw_val, bool):
                raise ValueError(f"parameter {p.name!r} must be an integer")
            iv = int(raw_val)
            if p.min_value is not None and iv < p.min_value:
                raise ValueError(f"parameter {p.name!r} must be >= {int(p.min_value)}")
            if p.max_value is not None and iv > p.max_value:
                raise ValueError(f"parameter {p.name!r} must be <= {int(p.max_value)}")
            normalized[p.name] = iv
            if p.cli_flag:
                argv.extend([p.cli_flag, str(iv)])
            else:
                argv.append(str(iv))
            continue

        if p.kind == "float":
            if isinstance(raw_val, bool):
                raise ValueError(f"parameter {p.name!r} must be a number")
            fv = float(raw_val)
            if p.min_value is not None and fv < p.min_value:
                raise ValueError(f"parameter {p.name!r} must be >= {p.min_value}")
            if p.max_value is not None and fv > p.max_value:
                raise ValueError(f"parameter {p.name!r} must be <= {p.max_value}")
            normalized[p.name] = fv
            if p.cli_flag:
                argv.extend([p.cli_flag, str(fv)])
            else:
                argv.append(str(fv))
            continue

        sv = str(raw_val).strip()
        if p.choices and sv not in p.choices:
            raise ValueError(f"parameter {p.name!r} must be one of {list(p.choices)}")
        if not _SAFE_TOKEN_RE.fullmatch(sv):
            raise ValueError(f"parameter {p.name!r} contains disallowed characters")
        normalized[p.name] = sv
        if p.is_repeatable_csv and p.cli_flag:
            parts = [part.strip() for part in sv.split(",") if part.strip()]
            if p.required and not parts:
                raise ValueError(f"parameter {p.name!r} cannot be empty")
            for part in parts:
                argv.extend([p.cli_flag, part])
        elif p.cli_flag:
            if sv:
                argv.extend([p.cli_flag, sv])
        else:
            argv.append(sv)

    if spec.supports_json:
        argv.append("--json")

    return spec, argv, normalized


# --------------------------------------------------------------------------- #
# Isolated subprocess ResearchJobManager
# --------------------------------------------------------------------------- #


class ResearchJobManager:
    """Runs allowlisted research tools and dataset fetches in isolated worker subprocesses."""

    def __init__(self, settings: Settings, max_concurrent: int = 2) -> None:
        self._settings = settings
        self._max_concurrent = max_concurrent
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._dir = settings.runs_root / "_research_jobs"
        self._load_existing()

    def _load_existing(self) -> None:
        if not self._dir.is_dir():
            return
        for path in sorted(self._dir.glob("job_*.json"))[-40:]:
            with contextlib.suppress(OSError, json.JSONDecodeError, ValueError):
                data = json.loads(path.read_text(encoding="utf-8"))
                jid = str(data.get("id", ""))
                if not _JOB_ID_RE.fullmatch(jid):
                    continue
                if data.get("status") == "running":
                    data["status"] = "failed"
                    data["error"] = "Interrupted by server restart"
                self._jobs[jid] = data

    def _save_locked(self, job: dict[str, Any]) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            target = self._dir / f"{job['id']}.json"
            tmp = self._dir / f".{job['id']}.tmp"
            tmp.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, target)
        except OSError:
            pass

    def _active_count_locked(self) -> int:
        return sum(1 for j in self._jobs.values() if j["status"] == "running")

    def submit_tool(self, tool_id: str, params: dict[str, Any]) -> dict[str, Any]:
        spec, argv, normalized = build_tool_argv(tool_id, params)
        return self._spawn(
            tool_id=spec.id,
            title_key=spec.title_key,
            category=spec.category,
            params=normalized,
            argv=argv,
            timeout_seconds=spec.timeout_seconds,
        )

    def submit_dataset_fetch(self, dataset_id: str) -> dict[str, Any]:
        if dataset_id not in DATASETS:
            raise DatasetError(f"unknown empirical connectome dataset {dataset_id!r}")
        return self._spawn(
            tool_id="connectome_fetch",
            title_key="research.tool.fetch.title",
            category="species_chimera",
            params={"dataset_id": dataset_id},
            argv=["connectome", "fetch", dataset_id],
            timeout_seconds=600,
        )

    def _spawn(
        self,
        *,
        tool_id: str,
        title_key: str,
        category: str,
        params: dict[str, Any],
        argv: list[str],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        with self._lock:
            if self._active_count_locked() >= self._max_concurrent:
                raise RuntimeError(
                    f"too many research jobs running (limit={self._max_concurrent}); "
                    "wait for one to finish or cancel it first"
                )
            job_id = f"job_{secrets.token_hex(8)}"
            now = time.time()
            job: dict[str, Any] = {
                "id": job_id,
                "tool_id": tool_id,
                "title_key": title_key,
                "category": category,
                "params": params,
                "status": "running",
                "created_at": now,
                "finished_at": None,
                "elapsed_seconds": 0.0,
                "exit_code": None,
                "result": None,
                "stdout_tail": "",
                "stderr_tail": "",
                "error": None,
            }
            self._jobs[job_id] = job
            self._save_locked(job)

        env = os.environ.copy()
        env["ALIFE_HOME"] = str(self._settings.home)
        env["ALIFE_RUNS_DIR"] = str(self._settings.runs_root)
        env["ALIFE_DATASETS_DIR"] = str(self._settings.resolved_datasets_dir)
        env["ALIFE_CONFIGS_DIR"] = str(self._settings.configs_dir)
        env["ALIFE_BACKUPS_DIR"] = str(self._settings.resolved_backups_dir)

        cmd = [sys.executable, "-m", "alife_lab.cli", *argv]
        thread = threading.Thread(
            target=self._run_worker,
            args=(job_id, cmd, env, timeout_seconds),
            daemon=True,
            name=f"research-{job_id}",
        )
        thread.start()
        return self.get_job(job_id)

    def _run_worker(
        self, job_id: str, cmd: list[str], env: dict[str, str], timeout_seconds: int
    ) -> None:
        t0 = time.perf_counter()
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self._settings.project_root),
                env=env,
            )
            with self._lock:
                self._procs[job_id] = proc
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            rc = proc.returncode
            parsed: Any = None
            stripped = stdout.strip()
            if stripped:
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    # Try finding the last JSON object if logs preceded JSON
                    brace = stripped.find("{")
                    if brace >= 0:
                        try:
                            parsed = json.loads(stripped[brace:])
                        except json.JSONDecodeError:
                            parsed = None
            with self._lock:
                self._procs.pop(job_id, None)
                job = self._jobs[job_id]
                elapsed = round(time.perf_counter() - t0, 3)
                job["elapsed_seconds"] = elapsed
                job["finished_at"] = time.time()
                job["exit_code"] = rc
                job["stdout_tail"] = stripped[-4000:]
                job["stderr_tail"] = stderr.strip()[-4000:]
                job["result"] = parsed
                if job["status"] == "cancelled":
                    pass
                elif rc == 0 or parsed is not None:
                    # Some assay CLIs exit 1 when an assay check fails even though JSON succeeded
                    job["status"] = "completed"
                    if rc != 0 and parsed is None:
                        job["status"] = "failed"
                        job["error"] = stderr.strip()[-600:] or f"process exited with code {rc}"
                else:
                    job["status"] = "failed"
                    job["error"] = stderr.strip()[-600:] or stripped[-600:] or f"exit code {rc}"
                self._save_locked(job)
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
                proc.communicate()
            with self._lock:
                self._procs.pop(job_id, None)
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["finished_at"] = time.time()
                job["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
                job["error"] = f"Research job timed out after {timeout_seconds}s"
                self._save_locked(job)
        except Exception as exc:
            with self._lock:
                self._procs.pop(job_id, None)
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["finished_at"] = time.time()
                job["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
                job["error"] = str(exc)
                self._save_locked(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            items = []
            for j in sorted(self._jobs.values(), key=lambda x: x["created_at"], reverse=True)[:40]:
                copy = {k: v for k, v in j.items() if k != "result"}
                copy["has_result"] = j["result"] is not None
                if copy["status"] == "running":
                    copy["elapsed_seconds"] = round(now - float(copy["created_at"]), 1)
                items.append(copy)
            return items

    def get_job(self, job_id: str) -> dict[str, Any]:
        if not _JOB_ID_RE.fullmatch(job_id):
            raise KeyError(f"invalid job id {job_id!r}")
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"research job {job_id!r} not found")
            copy = dict(self._jobs[job_id])
            if copy["status"] == "running":
                copy["elapsed_seconds"] = round(time.time() - float(copy["created_at"]), 1)
            return copy

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        if not _JOB_ID_RE.fullmatch(job_id):
            raise KeyError(f"invalid job id {job_id!r}")
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"research job {job_id!r} not found")
            job = self._jobs[job_id]
            proc = self._procs.get(job_id)
            if job["status"] == "running":
                job["status"] = "cancelled"
                job["finished_at"] = time.time()
                job["error"] = "Cancelled by user"
                if proc is not None:
                    proc.terminate()
                self._save_locked(job)
            return dict(job)

    def shutdown(self) -> None:
        with self._lock:
            procs = list(self._procs.values())
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                with contextlib.suppress(OSError):
                    p.kill()


# --------------------------------------------------------------------------- #
# Managed Backup & Restore API (closes `alife backup create/verify/restore`)
# --------------------------------------------------------------------------- #


def _resolve_backup_dir(backup_id: str, settings: Settings) -> Path:
    if not _BACKUP_ID_RE.fullmatch(backup_id):
        raise BackupError(f"invalid backup_id {backup_id!r}")
    base = settings.resolved_backups_dir.resolve()
    target = (base / backup_id).resolve()
    if target.parent != base or not target.is_dir():
        raise BackupError(f"backup {backup_id!r} not found")
    return target


def list_backups(settings: Settings) -> dict[str, Any]:
    base = settings.resolved_backups_dir
    items: list[dict[str, Any]] = []
    if base.is_dir():
        for d in sorted(base.iterdir(), reverse=True):
            if not d.is_dir() or not _BACKUP_ID_RE.fullmatch(d.name):
                continue
            mpath = d / BACKUP_MANIFEST
            if not mpath.is_file():
                continue
            with contextlib.suppress(OSError, json.JSONDecodeError, ValueError):
                m = json.loads(mpath.read_text(encoding="utf-8"))
                ts = m.get("created_at_unix")
                created_iso = (
                    time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(float(ts)))
                    if ts is not None
                    else m.get("created_at")
                )
                items.append(
                    {
                        "backup_id": m.get("backup_id", d.name),
                        "created_at": created_iso,
                        "file_count": len(m.get("files", [])),
                        "total_bytes": int(
                            m.get(
                                "total_bytes",
                                sum(int(f.get("size", 0)) for f in m.get("files", [])),
                            )
                        ),
                        "run_count": len(m.get("runs", [])),
                        "path": str(d),
                    }
                )
    return {"backups_dir": str(base), "items": items}


def run_create_backup(settings: Settings) -> dict[str, Any]:
    return create_backup(settings.db_path, settings.runs_root, settings.resolved_backups_dir)


def run_verify_backup(backup_id: str, settings: Settings) -> dict[str, Any]:
    target = _resolve_backup_dir(backup_id, settings)
    return verify_backup(target)


def run_test_restore_backup(backup_id: str, settings: Settings) -> dict[str, Any]:
    """Verify and restore a backup into a clean sibling directory without touching live state."""
    target = _resolve_backup_dir(backup_id, settings)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest_root = (
        settings.resolved_backups_dir / "_restored" / f"{backup_id}_{stamp}_{secrets.token_hex(2)}"
    )
    home_dest = dest_root / "home"
    runs_dest = dest_root / "runs"
    return restore_backup(target, home_dest, runs_dest)
