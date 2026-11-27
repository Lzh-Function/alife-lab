"""Connectome dataset registry, safe fetcher and SHA-256 verifier (ADR 0035).

Rules enforced here:
- Tabular formats only (parquet, csv, tsv, csv.gz, tsv.gz, json). Pickle files are refused.
- HTTPS URLs only, pinned to immutable Git commits or versioned release paths.
- Downloads stage into a temporary path, enforce size and total disk budgets, verify
  SHA-256, and only publish via ``os.replace`` on exact match.
- Datasets live outside Git (default: ``.alife/datasets/<dataset_id>/``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from alife_lab.connectome.schema import DatasetProvenance, FileProvenance
from alife_lab.core.hashing import canonical_json, sha256_file

ALLOWED_SUFFIXES: tuple[str, ...] = (
    ".parquet",
    ".feather",
    ".csv",
    ".tsv",
    ".txt",
    ".csv.gz",
    ".tsv.gz",
    ".csv.zip",
    ".json",
)
FORBIDDEN_SUFFIXES: tuple[str, ...] = (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt")


class DatasetError(RuntimeError):
    """Raised when a connectome dataset fails validation, budget, or integrity checks."""


@dataclass(frozen=True)
class DatasetFileSpec:
    filename: str
    url: str
    sha256: str
    size_bytes: int
    source_label: str
    licence_note: str

    def to_provenance(self) -> FileProvenance:
        return FileProvenance(
            filename=self.filename,
            url=self.url,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            source_label=self.source_label,
            licence_note=self.licence_note,
        )


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    species: str
    version: str
    citation: str
    licence_note: str
    files: tuple[DatasetFileSpec, ...]

    @property
    def total_size_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    def to_provenance(self) -> DatasetProvenance:
        return DatasetProvenance(
            dataset_id=self.dataset_id,
            species=self.species,
            version=self.version,
            citation=self.citation,
            licence_note=self.licence_note,
            files=tuple(f.to_provenance() for f in self.files),
        )


_SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
_ANNO_COMMIT = "a83b2776d60d5764cef36b927f5f9679c16c47a2"
_CODEX_BASE = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"

FLYWIRE_783_SPEC = DatasetSpec(
    dataset_id="flywire_783",
    species="drosophila_melanogaster_adult_female",
    version="v783",
    citation=(
        "Dorkenwald et al. (2024) Nature 'Neuronal wiring diagram of an adult brain'; "
        "Schlegel et al. (2024) Nature 'Whole-brain annotation and multi-connectome cell "
        "typing of Drosophila'; Shiu et al. (2024) Nature 634:210-219 "
        "'A Drosophila computational brain model reveals sensorimotor processing'."
    ),
    licence_note=(
        "Personal non-commercial research use (data not redistributed in Git). "
        "Shiu et al. code/repo: MIT; FlyWire v783 data: CC-BY 4.0 (Zenodo) / "
        "CC BY-NC 4.0 (FlyWire principles); flywire_annotations: see upstream repo."
    ),
    files=(
        DatasetFileSpec(
            filename="Completeness_783.csv",
            url=f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{_SHIU_COMMIT}/Completeness_783.csv",
            sha256="bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
            size_bytes=3_327_347,
            source_label="philshiu/Drosophila_brain_model",
            licence_note="MIT repo; FlyWire v783 derived table",
        ),
        DatasetFileSpec(
            filename="Connectivity_783.parquet",
            url=f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{_SHIU_COMMIT}/Connectivity_783.parquet",
            sha256="efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
            size_bytes=100_804_642,
            source_label="philshiu/Drosophila_brain_model",
            licence_note="MIT repo; FlyWire v783 derived connectivity table",
        ),
        DatasetFileSpec(
            filename="Supplemental_file1_neuron_annotations.tsv",
            url=(
                f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                f"{_ANNO_COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
            ),
            sha256="b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd",
            size_bytes=31_720_298,
            source_label="flyconnectome/flywire_annotations",
            licence_note="FlyWire consortium annotations (Schlegel et al. 2024)",
        ),
        DatasetFileSpec(
            filename="classification.csv.gz",
            url=f"{_CODEX_BASE}/classification.csv.gz",
            sha256="e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663",
            size_bytes=934_402,
            source_label="FlyWire Codex v783 (GCS)",
            licence_note="FlyWire v783 Codex export",
        ),
        DatasetFileSpec(
            filename="coordinates.csv.gz",
            url=f"{_CODEX_BASE}/coordinates.csv.gz",
            sha256="14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05",
            size_bytes=5_314_546,
            source_label="FlyWire Codex v783 (GCS)",
            licence_note="FlyWire v783 Codex export",
        ),
        DatasetFileSpec(
            filename="neurons.csv.gz",
            url=f"{_CODEX_BASE}/neurons.csv.gz",
            sha256="6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719",
            size_bytes=1_679_884,
            source_label="FlyWire Codex v783 (GCS)",
            licence_note="FlyWire v783 Codex export",
        ),
        DatasetFileSpec(
            filename="neuropil_synapse_table.csv.gz",
            url=f"{_CODEX_BASE}/neuropil_synapse_table.csv.gz",
            sha256="e525bdea7bc2fe585cf8ab8f9fc76bea5e29f2c3e5240588b6add518bcda5ab1",
            size_bytes=4_674_663,
            source_label="FlyWire Codex v783 (GCS)",
            licence_note="FlyWire v783 Codex export",
        ),
    ),
)

_BANC_BASE = (
    "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome"
    "/compiled_data/banc_888"
)
_MALECNS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

BANC_888_SPEC = DatasetSpec(
    dataset_id="banc_888",
    species="drosophila_melanogaster_adult_female_cns",
    version="v888",
    citation=(
        "Bates, Phelps, Kim, Yang, Tuthill, Lee et al. (2025/2026) "
        "'Distributed control circuits across a brain-and-nerve-cord connectome' "
        "(Harvard Dataverse DOI: 10.7910/DVN/7WTH1N)."
    ),
    licence_note=(
        "Personal non-commercial research use (data not redistributed in Git). "
        "BANC v888 public release: CC-BY 4.0."
    ),
    files=(
        DatasetFileSpec(
            filename="banc_888_meta.feather",
            url=f"{_BANC_BASE}/banc_888_meta.feather",
            sha256="60a7c2252841c9f05aef0803c84bc3be7802b211f6bd2a40cef123ee3ad6265f",
            size_bytes=57_550_730,
            source_label="BANC v888 GCS (Lee Lab)",
            licence_note="CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="banc_888_neurotransmitter_prediction_v2.csv",
            url=f"{_BANC_BASE}/banc_888_neurotransmitter_prediction_v2.csv",
            sha256="d941b62059f3bbd491619df72ce96651dba0a4b008d8f4ce62abd78e472ab29e",
            size_bytes=21_037_063,
            source_label="BANC v888 GCS (Lee Lab)",
            licence_note="CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="banc_888_edgelist_simple_v3.feather",
            url=f"{_BANC_BASE}/banc_888_edgelist_simple_v3.feather",
            sha256="8c296e946f3c69a8c7222f30ad75fa8a98eeb189124fec6df829c9125f4be64b",
            size_bytes=359_161_658,
            source_label="BANC v888 GCS (Lee Lab)",
            licence_note="CC-BY 4.0",
        ),
    ),
)

MALE_CNS_V1_SPEC = DatasetSpec(
    dataset_id="male_cns_v1",
    species="drosophila_melanogaster_adult_male_cns",
    version="v1.0",
    citation=(
        "Berg, Schlegel, Jefferis, Rubin, Jain et al. (2025) "
        "'Connectome of the adult male Drosophila central nervous system' "
        "(HHMI Janelia / University of Cambridge / Google Research, male-cns:v1.0)."
    ),
    licence_note=(
        "Personal non-commercial research use (data not redistributed in Git). "
        "FlyEM Male CNS v1.0 public release: CC-BY 4.0."
    ),
    files=(
        DatasetFileSpec(
            filename="body-annotations-male-cns-v1.0-minconf-0.5.feather",
            url=f"{_MALECNS_BASE}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
            sha256="2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
            size_bytes=14_483_314,
            source_label="FlyEM Male CNS v1.0 GCS (Janelia)",
            licence_note="CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="body-neurotransmitters-male-cns-v1.0.feather",
            url=f"{_MALECNS_BASE}/body-neurotransmitters-male-cns-v1.0.feather",
            sha256="95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
            size_bytes=43_282_834,
            source_label="FlyEM Male CNS v1.0 GCS (Janelia)",
            licence_note="CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
            url=f"{_MALECNS_BASE}/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
            sha256="9b3beab17bad5f618be3f2c02d3139a8d07b822565919c013f1e5506d93e604b",
            size_bytes=508_025_642,
            source_label="FlyEM Male CNS v1.0 GCS (Janelia)",
            licence_note="CC-BY 4.0",
        ),
    ),
)

_CELEGANS_COMMIT = "1ca08b14f4e795a973ce447565e0c97b897ee4dc"

CELEGANS_302_SPEC = DatasetSpec(
    dataset_id="celegans_302",
    species="caenorhabditis_elegans_hermaphrodite",
    version="openworm_edgelist_v1",
    citation=(
        "White et al. (1986) Phil. Trans. R. Soc. B 314:1-340; "
        "Varshney et al. (2011) PLoS Comput. Biol. 7:e1001066; "
        "Cook et al. (2019) Nature 571:63-71; OpenWorm CElegansNeuroML."
    ),
    licence_note=(
        "Personal/academic research use (data not redistributed in Git; "
        "stored under .alife/datasets/celegans_302/). OpenWorm CElegansNeuroML (MIT)."
    ),
    files=(
        DatasetFileSpec(
            filename="herm_full_edgelist.csv",
            url=(
                f"https://raw.githubusercontent.com/openworm/CElegansNeuroML/"
                f"{_CELEGANS_COMMIT}/herm_full_edgelist.csv"
            ),
            sha256="142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca",
            size_bytes=245_463,
            source_label="openworm/CElegansNeuroML",
            licence_note="OpenWorm MIT / WormWiring",
        ),
    ),
)

_PLATYNEREIS_COMMIT = "ab75029040b8dba768d4c2cbdb26423aec59ae4a"
_PLATY_BASE = (
    f"https://raw.githubusercontent.com/JekelyLab/Platynereis_3D_connectome_2024/"
    f"{_PLATYNEREIS_COMMIT}/data"
)

FLY_LARVA_L1_SPEC = DatasetSpec(
    dataset_id="fly_larva_l1",
    species="drosophila_melanogaster_larva_l1",
    version="winding_2023_v1",
    citation=(
        "Winding, Pedigo, Barnes, Patsolic, Park, Kazimiers, Fushiki, Andrade, Khandelwal, "
        "Valdes-Aleman, Li, Randel, Barsotti, Correia, Fetter, Hartenstein, Priebe, "
        "Vogelstein, Cardona & Zlatic (2023) Science 379(6636):eadd9330 "
        "'The connectome of an insect brain' (via Netzschleuder fly_larva CSV archive)."
    ),
    licence_note=(
        "Personal/academic research use (data not redistributed in Git; "
        "stored under .alife/datasets/fly_larva_l1/). Winding et al. (2023) Science."
    ),
    files=(
        DatasetFileSpec(
            filename="fly_larva.csv.zip",
            url="https://networks.skewed.de/net/fly_larva/files/fly_larva.csv.zip",
            sha256="49f87128185feb1fa41131050668eecc57064dbca36fc1908419fa5f6b341463",
            size_bytes=508_166,
            source_label="Netzschleuder fly_larva (Winding et al. 2023)",
            licence_note="Open connectome CSV archive (Science 2023)",
        ),
    ),
)

PLATYNEREIS_3D_SPEC = DatasetSpec(
    dataset_id="platynereis_3d",
    species="platynereis_dumerilii_3day_larva",
    version="veraszto_2024_ab75029",
    citation=(
        "Verasztó, Jasek, Gühmann, Shahidi, Ueda, Beard, Mendes, Heinz, Bezares-Calderón, "
        "Williams & Jékely (2024/2025) eLife 97964 "
        "'Whole-animal connectome and cell-type complement of the three-segmented "
        "Platynereis dumerilii larva'."
    ),
    licence_note=(
        "Personal/academic research use (data not redistributed in Git; "
        "stored under .alife/datasets/platynereis_3d/). JekelyLab GitHub / eLife CC-BY 4.0."
    ),
    files=(
        DatasetFileSpec(
            filename="full_connectome_adjacency_matrix.csv",
            url=f"{_PLATY_BASE}/full_connectome_adjacency_matrix.csv",
            sha256="889422d55c009ccc515c2828adb69ea4b377123950d830e98e7fd5fa146ddc6c",
            size_bytes=5_978_359,
            source_label="JekelyLab/Platynereis_3D_connectome_2024",
            licence_note="GPL-3.0 / eLife CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="with_soma_all_annot.txt",
            url=f"{_PLATY_BASE}/with_soma_all_annot.txt",
            sha256="9c157e2e16ea89f9387bb0958ff79d171f25225c9efe26f03bb8d5cc9636e130",
            size_bytes=7_072_816,
            source_label="JekelyLab/Platynereis_3D_connectome_2024",
            licence_note="GPL-3.0 / eLife CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="neuronal_celltypes_table.csv",
            url=f"{_PLATY_BASE}/neuronal_celltypes_table.csv",
            sha256="25a46a8b1887675df8c955e90ee44b088fab76d4b3cf977beaf6b4da647f7870",
            size_bytes=66_033,
            source_label="JekelyLab/Platynereis_3D_connectome_2024",
            licence_note="GPL-3.0 / eLife CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="connectome_SN_skids.csv",
            url=f"{_PLATY_BASE}/connectome_SN_skids.csv",
            sha256="7a3baebb5e250664976d3948fef1219734f7d9cd8d07db84a56e409d8458637e",
            size_bytes=3_642,
            source_label="JekelyLab/Platynereis_3D_connectome_2024",
            licence_note="GPL-3.0 / eLife CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="connectome_IN_skids.csv",
            url=f"{_PLATY_BASE}/connectome_IN_skids.csv",
            sha256="9a2fc83a4c5a67e8a6ada971321d02fb1b5b1a70415be60a934bfcebf2a9e658",
            size_bytes=7_267,
            source_label="JekelyLab/Platynereis_3D_connectome_2024",
            licence_note="GPL-3.0 / eLife CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="connectome_MN_skids.csv",
            url=f"{_PLATY_BASE}/connectome_MN_skids.csv",
            sha256="f5daa7c25cea66a916756c7b09e130000d2d8f64987a94d513155b7df07b4f0f",
            size_bytes=1_871,
            source_label="JekelyLab/Platynereis_3D_connectome_2024",
            licence_note="GPL-3.0 / eLife CC-BY 4.0",
        ),
        DatasetFileSpec(
            filename="connectome_neuron_skids.csv",
            url=f"{_PLATY_BASE}/connectome_neuron_skids.csv",
            sha256="c6b423e13ceff2b18e29d8a9d9b381d5b1d931a144ee2930a601f7bb4645e5d5",
            size_bytes=12_838,
            source_label="JekelyLab/Platynereis_3D_connectome_2024",
            licence_note="GPL-3.0 / eLife CC-BY 4.0",
        ),
    ),
)

CIONA_LARVA_SPEC = DatasetSpec(
    dataset_id="ciona_larva",
    species="ciona_intestinalis_tadpole_larva",
    version="ryan_2016_v1",
    citation=(
        "Ryan, Lu & Meinertzhagen (2016) eLife 5:e16962 "
        "'The CNS connectome of a tadpole larva of Ciona intestinalis (L.) highlights "
        "sidedness in the brain of a chordate sibling' (Netzschleuder cintestinalis archive)."
    ),
    licence_note=(
        "Personal/academic research use (data not redistributed in Git; "
        "stored under .alife/datasets/ciona_larva/). eLife CC-BY 4.0 / Netzschleuder."
    ),
    files=(
        DatasetFileSpec(
            filename="cintestinalis.csv.zip",
            url="https://networks.skewed.de/net/cintestinalis/files/cintestinalis.csv.zip",
            sha256="d036aeba31e94dee6b01ac9f883872177f4ec82bd720da405f3e0ac272416210",
            size_bytes=14_664,
            source_label="Netzschleuder cintestinalis (Ryan et al. 2016)",
            licence_note="eLife CC-BY 4.0",
        ),
    ),
)

DATASETS: dict[str, DatasetSpec] = {
    FLYWIRE_783_SPEC.dataset_id: FLYWIRE_783_SPEC,
    BANC_888_SPEC.dataset_id: BANC_888_SPEC,
    MALE_CNS_V1_SPEC.dataset_id: MALE_CNS_V1_SPEC,
    CELEGANS_302_SPEC.dataset_id: CELEGANS_302_SPEC,
    FLY_LARVA_L1_SPEC.dataset_id: FLY_LARVA_L1_SPEC,
    PLATYNEREIS_3D_SPEC.dataset_id: PLATYNEREIS_3D_SPEC,
    CIONA_LARVA_SPEC.dataset_id: CIONA_LARVA_SPEC,
}


def list_dataset_specs() -> list[DatasetSpec]:
    return [DATASETS[k] for k in sorted(DATASETS)]


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise DatasetError(f"unknown connectome dataset {dataset_id!r} (available: {known})")
    return DATASETS[dataset_id]


def validate_filename_and_url(filename: str, url: str) -> None:
    if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
        raise DatasetError(f"unsafe dataset filename: {filename!r}")
    lower = filename.lower()
    if any(lower.endswith(s) for s in FORBIDDEN_SUFFIXES):
        raise DatasetError(f"forbidden binary/pickle dataset file: {filename!r}")
    if not any(lower.endswith(s) for s in ALLOWED_SUFFIXES):
        raise DatasetError(f"unsupported dataset file format: {filename!r}")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise DatasetError(f"dataset URL must use https: {url!r}")


def dataset_root(datasets_dir: Path, dataset_id: str) -> Path:
    return datasets_dir / dataset_id


def raw_dir(datasets_dir: Path, dataset_id: str) -> Path:
    return dataset_root(datasets_dir, dataset_id) / "raw"


def manifest_path(datasets_dir: Path, dataset_id: str) -> Path:
    return dataset_root(datasets_dir, dataset_id) / "manifest.json"


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def verify_dataset(
    dataset_id: str,
    datasets_dir: Path,
    *,
    spec: DatasetSpec | None = None,
) -> dict[str, Any]:
    """Re-hash every file in the dataset and compare against pinned size and SHA-256."""
    dspec = spec or get_dataset_spec(dataset_id)
    rdir = raw_dir(datasets_dir, dspec.dataset_id)
    results: list[dict[str, Any]] = []
    all_ok = True
    for fspec in dspec.files:
        validate_filename_and_url(fspec.filename, fspec.url)
        path = rdir / fspec.filename
        if not path.is_file():
            all_ok = False
            results.append(
                {
                    "filename": fspec.filename,
                    "ok": False,
                    "status": "missing",
                    "expected_size": fspec.size_bytes,
                    "actual_size": None,
                    "expected_sha256": fspec.sha256,
                    "actual_sha256": None,
                }
            )
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        ok = actual_size == fspec.size_bytes and actual_sha == fspec.sha256
        if not ok:
            all_ok = False
        results.append(
            {
                "filename": fspec.filename,
                "ok": ok,
                "status": "ok" if ok else "mismatch",
                "expected_size": fspec.size_bytes,
                "actual_size": actual_size,
                "expected_sha256": fspec.sha256,
                "actual_sha256": actual_sha,
            }
        )
    return {
        "dataset_id": dspec.dataset_id,
        "species": dspec.species,
        "version": dspec.version,
        "path": str(dataset_root(datasets_dir, dspec.dataset_id)),
        "ok": all_ok,
        "files": results,
        "total_expected_bytes": dspec.total_size_bytes,
    }


def write_dataset_manifest(
    dspec: DatasetSpec,
    datasets_dir: Path,
) -> Path:
    """Atomically write the verified dataset manifest.json."""
    root = dataset_root(datasets_dir, dspec.dataset_id)
    root.mkdir(parents=True, exist_ok=True)
    mpath = manifest_path(datasets_dir, dspec.dataset_id)
    tmp = root / "manifest.json.tmp"
    payload = {
        "schema_version": 1,
        "provenance": dspec.to_provenance().to_dict(),
        "total_size_bytes": dspec.total_size_bytes,
    }
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, mpath)
    return mpath


def fetch_dataset(
    dataset_id: str,
    datasets_dir: Path,
    *,
    max_datasets_mb: int = 4096,
    timeout_seconds: float = 60.0,
    spec: DatasetSpec | None = None,
) -> dict[str, Any]:
    """Download any missing/mismatched files with staging, budget check and SHA-256 verification."""
    dspec = spec or get_dataset_spec(dataset_id)
    for fspec in dspec.files:
        validate_filename_and_url(fspec.filename, fspec.url)

    budget_bytes = max_datasets_mb * 1024 * 1024
    existing_bytes = _dir_size_bytes(datasets_dir)
    rdir = raw_dir(datasets_dir, dspec.dataset_id)
    sdir = dataset_root(datasets_dir, dspec.dataset_id) / "staging"
    rdir.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []
    reused: list[str] = []

    for fspec in dspec.files:
        target = rdir / fspec.filename
        if (
            target.is_file()
            and target.stat().st_size == fspec.size_bytes
            and sha256_file(target) == fspec.sha256
        ):
            reused.append(fspec.filename)
            continue

        existing_target = target.stat().st_size if target.is_file() else 0
        projected = existing_bytes - existing_target + fspec.size_bytes
        if projected > budget_bytes:
            raise DatasetError(
                f"downloading {fspec.filename} ({fspec.size_bytes} B) would exceed "
                f"dataset disk budget ({max_datasets_mb} MB)"
            )

        sdir.mkdir(parents=True, exist_ok=True)
        staged = sdir / f"{fspec.filename}.part"
        staged.unlink(missing_ok=True)
        req = urllib.request.Request(  # noqa: S310 — scheme verified as https above
            fspec.url,
            headers={"User-Agent": "alife-lab/0.1.0 connectome-fetcher"},
        )
        hasher = hashlib.sha256()
        written = 0
        try:
            with (
                urllib.request.urlopen(req, timeout=timeout_seconds) as resp,  # noqa: S310
                staged.open("wb") as out,
            ):
                while chunk := resp.read(1 << 20):
                    written += len(chunk)
                    if written > fspec.size_bytes:
                        raise DatasetError(
                            f"{fspec.filename}: exceeded pinned size {fspec.size_bytes} B"
                        )
                    hasher.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
        except Exception:
            staged.unlink(missing_ok=True)
            raise

        digest = hasher.hexdigest()
        if written != fspec.size_bytes or digest != fspec.sha256:
            staged.unlink(missing_ok=True)
            raise DatasetError(
                f"{fspec.filename}: integrity check failed "
                f"(size {written} vs {fspec.size_bytes}, sha256 {digest} vs {fspec.sha256})"
            )
        os.replace(staged, target)
        existing_bytes = projected
        downloaded.append(fspec.filename)

    if sdir.exists():
        with contextlib.suppress(OSError):
            sdir.rmdir()

    report = verify_dataset(dspec.dataset_id, datasets_dir, spec=dspec)
    if not report["ok"]:
        raise DatasetError(f"post-fetch verification failed for {dspec.dataset_id}")
    mpath = write_dataset_manifest(dspec, datasets_dir)
    report["manifest_path"] = str(mpath)
    report["downloaded"] = downloaded
    report["reused"] = reused
    report["manifest_sha256"] = hashlib.sha256(
        canonical_json(dspec.to_provenance().to_dict()).encode("utf-8")
    ).hexdigest()
    return report
