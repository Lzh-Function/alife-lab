"""Phase F10 reference tests: flyvis visual system + cross-sex connectome comparison.

Offline CPU tests run unconditionally in CI (synthetic HR motion detector + synthetic cross-sex
pack comparison + CLI JSON checks). Real pretrained ``flyvis`` test runs when ``flyvis`` and its
pretrained weights are present under ``.alife/datasets/flyvis``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.embodied import (
    make_synthetic_cns_embodied_pack,
    make_synthetic_fly_embodied_pack,
)
from alife_lab.connectome.sex_compare import compare_sex_connectomes
from alife_lab.connectome.vision_flyvis import (
    flyvis_available,
    run_flyvis_visual_assay,
    run_synthetic_visual_assay,
    visual_to_descending_hz,
)

ROOT = Path(__file__).resolve().parents[2]


def test_f10_synthetic_vision_and_bridge_cpu_only(capsys: pytest.CaptureFixture[str]) -> None:
    rep = run_synthetic_visual_assay(n_frames=20, size=32)
    assert rep["all_checks_passed"] is True
    assert rep["stimuli"]["grating_right"]["opponent_index"] > 0.2
    assert rep["stimuli"]["grating_left"]["opponent_index"] < -0.2
    assert abs(rep["stimuli"]["static_gray"]["opponent_index"]) < 0.05

    br = visual_to_descending_hz(0.5, 0.8)
    assert br["turn_left_dn_hz"] > 0.0
    assert br["turn_right_dn_hz"] == 0.0
    assert br["backward_dn_hz"] > 0.0

    rc = cli_main(["connectome", "vision", "--synthetic", "--frames", "16", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["all_checks_passed"] is True


def test_f10_cross_sex_compare_cpu_only(capsys: pytest.CaptureFixture[str]) -> None:
    packs = {
        "synthetic_fly_embodied": make_synthetic_fly_embodied_pack(),
        "synthetic_cns_embodied": make_synthetic_cns_embodied_pack(),
    }
    rep = compare_sex_connectomes(packs, backend="numpy", seed=1, ticks=150)
    assert rep["shared_cell_types_all"] >= 10
    assert rep["physiology"]["synthetic_fly_embodied"]["bitter_suppression_pct"] > 80.0
    assert (
        rep["physiology"]["synthetic_fly_embodied"]["reference"]["sugar_150hz"]["mn9_hz"]
        > rep["physiology"]["synthetic_fly_embodied"]["shuffled_control"]["sugar_150hz"]["mn9_hz"]
    )

    rc = cli_main(
        [
            "connectome",
            "compare-sex",
            "--datasets",
            "synthetic_fly_embodied,synthetic_cns_embodied",
            "--backend",
            "numpy",
            "--ticks",
            "120",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["shared_cell_types_all"] >= 10


def test_f10_real_flyvis_pretrained_when_available() -> None:
    ds_dir = ROOT / ".alife" / "datasets"
    if not flyvis_available(ds_dir):
        pytest.skip("flyvis or pretrained models not present in .alife/datasets/flyvis")
    rep = run_flyvis_visual_assay(ds_dir, model_id="000", n_frames=24, size=64)
    assert rep["num_cells"] == 45669
    assert rep["unique_cell_types"] == 65
    assert rep["all_checks_passed"] is True, rep
