"""Unit tests for connectome species packs, registry safety gates, and FlyWire loader (F0).

All tests run offline on synthetic mini-datasets so CPU-only CI stays fast and self-contained.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome import (
    POS_ANCHOR,
    POS_CODEX,
    POS_SOMA,
    ROLE_INTERNEURON,
    ROLE_MOTOR,
    ROLE_SENSORY,
    DatasetError,
    DatasetFileSpec,
    DatasetSpec,
    EdgeTable,
    fetch_dataset,
    load_flywire_783,
    raw_dir,
    summarize_pack,
    verify_dataset,
)
from alife_lab.connectome.registry import validate_filename_and_url
from alife_lab.core.hashing import sha256_file


def _make_mini_flywire(datasets_dir: Path) -> DatasetSpec:
    """Create a 5-neuron synthetic FlyWire v783 dataset covering all fallback paths."""
    rdir = raw_dir(datasets_dir, "mini_flywire")
    rdir.mkdir(parents=True, exist_ok=True)

    # 5 neurons:
    # 0: sugar GRN (has soma_x/y/z)
    # 1: bitter GRN (no soma, has pos_x/y/z -> anchor fallback)
    # 2: interneuron KC (has soma, primary neuropil MB_CA_L)
    # 3: MN9 motor neuron (CB0701, has soma)
    # 4: missing from Supplemental_file1_neuron_annotations.tsv -> Codex fallback (cls/neu/coords)
    root_ids = [1001, 1002, 1003, 1004, 1005]

    with (rdir / "Completeness_783.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "Completed"])
        for rid in root_ids:
            w.writerow([rid, "True"])

    # Directed edges (intentional unsorted order to verify canonical CSR sorting).
    pre_idx = np.array([0, 0, 1, 2, 4], dtype=np.int64)
    post_idx = np.array([3, 2, 2, 3, 2], dtype=np.int64)
    conn = np.array([5, 10, 4, 12, 3], dtype=np.int64)
    exc = np.array([1, 1, -1, 1, -1], dtype=np.int64)
    pq.write_table(
        pa.table(
            {
                "Presynaptic_ID": [root_ids[i] for i in pre_idx],
                "Postsynaptic_ID": [root_ids[i] for i in post_idx],
                "Presynaptic_Index": pre_idx,
                "Postsynaptic_Index": post_idx,
                "Connectivity": conn,
                "Excitatory": exc,
                "Excitatory x Connectivity": exc * conn,
            }
        ),
        rdir / "Connectivity_783.parquet",
    )

    ann_cols = [
        "root_id",
        "pos_x",
        "pos_y",
        "pos_z",
        "soma_x",
        "soma_y",
        "soma_z",
        "flow",
        "super_class",
        "cell_class",
        "cell_sub_class",
        "cell_type",
        "hemibrain_type",
        "top_nt",
        "side",
    ]
    ann_rows = [
        [
            1001,
            "10",
            "20",
            "3",
            "11",
            "21",
            "4",
            "afferent",
            "sensory",
            "gustatory",
            "sugar",
            "LB3c",
            "",
            "acetylcholine",
            "left",
        ],
        [
            1002,
            "30",
            "40",
            "5",
            "",
            "",
            "",
            "afferent",
            "sensory",
            "gustatory",
            "bitter",
            "LB1a",
            "",
            "gaba",
            "right",
        ],
        [
            1003,
            "50",
            "60",
            "7",
            "51",
            "61",
            "8",
            "intrinsic",
            "central",
            "Kenyon_Cell",
            "",
            "KCg-m",
            "",
            "acetylcholine",
            "left",
        ],
        [
            1004,
            "70",
            "80",
            "9",
            "71",
            "81",
            "10",
            "efferent",
            "motor",
            "brain_motor_neuron",
            "ingestion_motor_neuron",
            "CB0701",
            "",
            "glutamate",
            "right",
        ],
    ]
    with (rdir / "Supplemental_file1_neuron_annotations.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(ann_cols)
        w.writerows(ann_rows)

    with gzip.open(rdir / "classification.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["root_id", "flow", "super_class", "class", "sub_class", "hemilineage", "side", "nerve"]
        )
        w.writerow(
            [1005, "intrinsic", "optic", "optic_lobe_intrinsic", "transmedullary", "", "left", ""]
        )

    with gzip.open(rdir / "neurons.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root_id", "group", "nt_type", "nt_type_score"])
        w.writerow([1005, "LO", "GLUT", "0.75"])

    with gzip.open(rdir / "coordinates.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root_id", "position", "supervoxel_id"])
        w.writerow([1005, "[900 800 700]", "123"])

    with gzip.open(rdir / "neuropil_synapse_table.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "root_id",
                "input synapses in GNG",
                "input synapses in MB_CA_L",
                "input synapses in UNASGD",
                "output synapses in GNG",
                "output synapses in MB_CA_L",
                "output synapses in UNASGD",
            ]
        )
        w.writerow([1001, 0, 0, 5, 15, 0, 0])
        w.writerow([1002, 2, 0, 0, 4, 0, 0])
        w.writerow([1003, 1, 20, 0, 0, 12, 0])
        w.writerow([1004, 17, 0, 0, 0, 0, 0])

    file_specs = []
    for fname in (
        "Completeness_783.csv",
        "Connectivity_783.parquet",
        "Supplemental_file1_neuron_annotations.tsv",
        "classification.csv.gz",
        "coordinates.csv.gz",
        "neurons.csv.gz",
        "neuropil_synapse_table.csv.gz",
    ):
        p = rdir / fname
        file_specs.append(
            DatasetFileSpec(
                filename=fname,
                url=f"https://example.invalid/mini/{fname}",
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                source_label="synthetic-test",
                licence_note="test",
            )
        )
    return DatasetSpec(
        dataset_id="mini_flywire",
        species="drosophila_test",
        version="v_test",
        citation="synthetic fixture",
        licence_note="test",
        files=tuple(file_specs),
    )


def test_security_gates_reject_pickle_traversal_and_http() -> None:
    with pytest.raises(DatasetError, match="pickle"):
        validate_filename_and_url("sez_neurons.pickle", "https://example.org/a.pickle")
    with pytest.raises(DatasetError, match="pickle"):
        validate_filename_and_url("weights.pkl", "https://example.org/a.pkl")
    with pytest.raises(DatasetError, match="unsafe"):
        validate_filename_and_url("../escape.csv", "https://example.org/a.csv")
    with pytest.raises(DatasetError, match="https"):
        validate_filename_and_url("ok.csv", "http://example.org/ok.csv")


def test_mini_flywire_load_verify_csr_and_reproducibility(tmp_path: Path) -> None:
    spec = _make_mini_flywire(tmp_path)

    ver = verify_dataset(spec.dataset_id, tmp_path, spec=spec)
    assert ver["ok"] is True

    # fetch_dataset reuses already-verified files without network access and writes manifest.json.
    fetched = fetch_dataset(spec.dataset_id, tmp_path, spec=spec)
    assert fetched["ok"] is True
    assert fetched["downloaded"] == []
    assert len(fetched["reused"]) == 7
    assert Path(fetched["manifest_path"]).is_file()

    pack1 = load_flywire_783(tmp_path, verify_hashes=True, spec=spec)
    pack2 = load_flywire_783(tmp_path, verify_hashes=True, spec=spec)

    # Identical files loaded twice yield identical digest and summary (F0 acceptance gate).
    assert pack1.pack_digest == pack2.pack_digest
    s1 = summarize_pack(pack1)
    s2 = summarize_pack(pack2)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)

    # Verify normalized cell properties and all 3 coordinate fallback paths.
    cells = pack1.cells
    assert cells.num_cells == 5
    assert list(cells.role) == [
        ROLE_SENSORY,
        ROLE_SENSORY,
        ROLE_INTERNEURON,
        ROLE_MOTOR,
        ROLE_INTERNEURON,
    ]
    assert list(cells.position_source) == [
        POS_SOMA,
        POS_ANCHOR,
        POS_SOMA,
        POS_SOMA,
        POS_CODEX,
    ]
    np.testing.assert_allclose(cells.position_nm[0], [44.0, 84.0, 160.0])  # soma * (4,4,40)
    np.testing.assert_allclose(cells.position_nm[1], [120.0, 160.0, 200.0])  # pos * (4,4,40)
    np.testing.assert_allclose(cells.position_nm[4], [900.0, 800.0, 700.0])  # codex nm directly
    assert cells.region == ("GNG", "GNG", "MB_CA_L", "GNG", "LO")
    assert cells.neurotransmitter[4] == "glutamate"

    # IO ports resolve cleanly.
    assert list(pack1.io_ports["sugar_grn"].cell_indices) == [0]
    assert list(pack1.io_ports["bitter_grn"].cell_indices) == [1]
    assert list(pack1.io_ports["mushroom_body_kc"].cell_indices) == [2]
    assert list(pack1.io_ports["mn9_proboscis_extension"].cell_indices) == [3]

    # Summary statistics match hand calculation.
    c = s1["counts"]
    assert c["cells"] == 5
    assert c["edges"] == 5
    assert c["total_synapses"] == 34
    assert c["excitatory_synapses"] == 27
    assert c["inhibitory_synapses"] == 7
    assert c["mixed_sign_presynaptic_neurons"] == 0
    assert c["self_loops"] == 0
    assert c["isolated_cells"] == 0

    # Canonical CSR sorts targets ascending per presynaptic cell (cell 0: [2, 3], weights [10, 5]).
    csr = pack1.edges.to_csr(cells.num_cells)
    assert list(csr.indptr) == [0, 2, 3, 4, 4, 5]
    assert list(csr.post[csr.indptr[0] : csr.indptr[1]]) == [2, 3]
    assert list(csr.signed_weight[csr.indptr[0] : csr.indptr[1]]) == [10, 5]
    assert list(csr.signed_weight[csr.indptr[1] : csr.indptr[2]]) == [-4]


def test_tampered_file_is_rejected(tmp_path: Path) -> None:
    spec = _make_mini_flywire(tmp_path)
    target = raw_dir(tmp_path, spec.dataset_id) / "Completeness_783.csv"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    ver = verify_dataset(spec.dataset_id, tmp_path, spec=spec)
    assert ver["ok"] is False
    with pytest.raises(DatasetError, match="verification failed"):
        load_flywire_783(tmp_path, verify_hashes=True, spec=spec)


def test_edge_table_validation_rejects_bad_weights() -> None:
    bad = EdgeTable(
        pre=np.array([0], dtype=np.int32),
        post=np.array([1], dtype=np.int32),
        count=np.array([0], dtype=np.int32),
        sign=np.array([1], dtype=np.int8),
        kind=np.array([0], dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="strictly positive"):
        bad.validate(2)


def test_cli_connectome_list_runs_without_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ALIFE_DATASETS_DIR", str(tmp_path))
    rc = cli_main(["connectome", "list", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    ids = [d["dataset_id"] for d in out["datasets"]]
    assert "flywire_783" in ids
    fly = next(d for d in out["datasets"] if d["dataset_id"] == "flywire_783")
    assert fly["verified_local"] is False
