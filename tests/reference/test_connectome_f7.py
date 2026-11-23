"""Phase F7 reference tests: fitting connectome models to real public Drosophila activity.

Pure-NumPy region mapping, neuropil aggregation, shuffling, and SHA-256 verification tests run
unconditionally in CPU-only CI. PyTorch synthetic fitting tests are guarded with ``@needs_torch``.
Real-data smoke checks run only when ``.alife/datasets/f7_real_activity/`` is present locally.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.real_fit import (
    F7_DIR_NAME,
    build_neuropil_connectome,
    map_aimon_region_to_flywire,
    run_synthetic_f7_check,
    sha256_file,
    shuffle_neuropil_connectome,
    verify_f7_file,
)
from alife_lab.connectome.schema import (
    CellTable,
    ConnectomePack,
    DatasetProvenance,
    EdgeTable,
)

ROOT = Path(__file__).resolve().parents[2]
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch not installed (uv sync --extra asal)"
)


def _make_mini_neuropil_pack() -> ConnectomePack:
    regions = ("AL_L", "AL_R", "MB_PED_L", "MB_VL_L", "WED_L", "LH_L")
    n = len(regions)
    cells = CellTable(
        source_id=np.arange(n, dtype=np.int64),
        species_names=("synthetic",),
        species_idx=np.zeros(n, dtype=np.uint8),
        cell_type=tuple(f"T{i}" for i in range(n)),
        super_class=tuple("central" for _ in range(n)),
        cell_class=tuple("Interneuron" for _ in range(n)),
        cell_sub_class=tuple("" for _ in range(n)),
        region=regions,
        side=("left", "right", "left", "left", "left", "left"),
        role=np.ones(n, dtype=np.uint8),
        cell_model=np.zeros(n, dtype=np.uint8),
        neurotransmitter=tuple("ACH" for _ in range(n)),
        position_nm=np.zeros((n, 3), dtype=np.float32),
        position_source=np.ones(n, dtype=np.uint8),
    )
    edges = EdgeTable(
        pre=np.array([0, 0, 1, 2, 3, 4], dtype=np.int32),
        post=np.array([2, 5, 3, 3, 4, 5], dtype=np.int32),
        count=np.array([20, 15, 10, 30, 25, 12], dtype=np.int32),
        sign=np.array([1, 1, 1, -1, 1, -1], dtype=np.int8),
        kind=np.zeros(6, dtype=np.uint8),
    )
    prov = DatasetProvenance(
        dataset_id="synthetic_neuropil",
        species="synthetic",
        version="test",
        citation="synthetic test fixture",
        licence_note="internal test",
        files=(),
    )
    return ConnectomePack(
        dataset_id="synthetic_neuropil",
        species="synthetic",
        cells=cells,
        edges=edges,
        io_ports={},
        provenance=prov,
        pack_digest="0" * 64,
    )


def test_aimon_region_mapping_and_neuropil_connectome_cpu_only() -> None:
    pack = _make_mini_neuropil_pack()
    fw_regions = set(pack.cells.region)
    assert map_aimon_region_to_flywire("AL_L", fw_regions) == "AL_L"
    assert map_aimon_region_to_flywire("MBPED_L", fw_regions) == "MB_PED_L"
    assert map_aimon_region_to_flywire("MBVL_L", fw_regions) == "MB_VL_L"
    assert map_aimon_region_to_flywire("IVLP_L", fw_regions) == "WED_L"
    assert map_aimon_region_to_flywire("NONEXISTENT_X", fw_regions) is None

    aimon_names = ("AL_L", "AL_R", "MBPED_L", "MBVL_L", "IVLP_L", "LH_L", "MISSING_R")
    nc = build_neuropil_connectome(pack, aimon_names)
    assert nc.matched_count == 6
    assert nc.synapse_total == 20 + 15 + 10 + 30 + 25 + 12
    assert nc.w_exc.shape == (7, 7)
    assert nc.w_inh.shape == (7, 7)
    # Zero self-loops on diagonal.
    assert float(np.diag(nc.w_exc).sum()) == 0.0
    assert float(np.diag(nc.w_inh).sum()) == 0.0
    # Row-preserving shuffle keeps exact row sums and zero diagonal.
    shuf = shuffle_neuropil_connectome(nc, seed=42)
    assert np.allclose(nc.w_exc.sum(axis=1), shuf.w_exc.sum(axis=1), atol=1e-6)
    assert np.allclose(nc.w_inh.sum(axis=1), shuf.w_inh.sum(axis=1), atol=1e-6)
    assert float(np.diag(shuf.w_exc).sum()) == 0.0


def test_verify_f7_file_hash_and_missing(tmp_path: Path) -> None:
    sample = tmp_path / "custom.csv"
    sample.write_text("a,b\n1,2\n", encoding="utf-8")
    digest = sha256_file(sample)
    assert len(digest) == 64
    assert verify_f7_file(tmp_path, "custom.csv") == sample
    with pytest.raises(FileNotFoundError):
        verify_f7_file(tmp_path, "door_or_merged.csv")
    # Corrupted known file fails SHA-256 verification.
    bad = tmp_path / "door_or_merged.csv"
    bad.write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_f7_file(tmp_path, "door_or_merged.csv")


@needs_torch
def test_synthetic_f7_check_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    res = run_synthetic_f7_check(seed=1, epochs=60, device="cpu")
    # Spatial unobserved prediction must favor true regional wiring over shuffled/unconnected.
    assert res["reference"]["spatial_unobs_r2"] > res["shuffled"]["spatial_unobs_r2"] + 0.05
    assert res["reference"]["spatial_unobs_r2"] > res["unconnected"]["spatial_unobs_r2"] + 0.30

    rc = cli_main(
        [
            "connectome",
            "fit-real",
            "synthetic",
            "--epochs",
            "40",
            "--seeds",
            "1",
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["reference"]["spatial_unobs_r2"] > out["unconnected"]["spatial_unobs_r2"]


def test_local_f7_dataset_hashes_when_present() -> None:
    raw_dir = ROOT / ".alife" / "datasets" / F7_DIR_NAME / "raw"
    if not raw_dir.exists():
        pytest.skip("Local F7 real datasets not downloaded")
    for fname in (
        "door_or_merged.csv",
        "door_to_flywire_mapping.csv",
        "Walk_anatomical_regions.zip",
    ):
        assert verify_f7_file(raw_dir, fname).exists()
