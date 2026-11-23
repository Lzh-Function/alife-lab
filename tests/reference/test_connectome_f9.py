"""Phase F9 reference tests: parallel multi-fly simulation (MuJoCo Warp + Batched LIF).

CPU-only tests validate backend resolution and error handling. MuJoCo tests verify batched
behavioral ablation screening, parallel foraging search, CLI JSON output, and (when MuJoCo Warp
is installed on CUDA) GPU Warp vs CPU float64 MuJoCo behavioral equivalence.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.embodied import make_synthetic_fly_embodied_pack
from alife_lab.connectome.embodied_batch import (
    _resolve_backends,
    benchmark_batched_embodied,
    run_batched_embodied_screen,
    run_batched_embodied_search,
    warp_available,
)

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
needs_mujoco = pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not installed")


def test_f9_backend_resolution_cpu_only() -> None:
    p, b = _resolve_backends(24, "cpu", "numpy")
    assert (p, b) == ("cpu", "numpy")
    with pytest.raises(ValueError, match="physics_backend"):
        _resolve_backends(24, "invalid", "numpy")
    with pytest.raises(ValueError, match="brain_backend"):
        _resolve_backends(24, "cpu", "invalid")


@needs_mujoco
def test_f9_batched_screen_search_and_cli_cpu(capsys: pytest.CaptureFixture[str]) -> None:
    pack = make_synthetic_fly_embodied_pack()
    rep = run_batched_embodied_screen(
        pack,
        cell_types=["CB0701", "IN_sugar", "IN_groom"],
        seeds=(1, 2),
        ticks=180,
        scene="sugar_feeding",
        batch_size=4,
        physics_backend="cpu",
        brain_backend="numpy",
        verify_top_cpu=True,
    )
    assert rep["baseline"]["mean_ms"] > 80.0
    # Silencing MN9 (CB0701) or sugar interneuron (IN_sugar) eliminates feeding completely!
    by_ct = {r["cell_type"]: r for r in rep["results"]}
    assert by_ct["CB0701"]["mean_ms"] == 0.0
    assert by_ct["IN_sugar"]["mean_ms"] == 0.0
    assert by_ct["IN_groom"]["mean_ms"] > 80.0
    assert rep["cpu_confirmation"] is not None
    assert rep["cpu_confirmation"]["cpu_silenced_ms"] == 0

    srep = run_batched_embodied_search(
        pack,
        generations=2,
        pop_size=4,
        ticks=150,
        seed=1,
        physics_backend="cpu",
        brain_backend="numpy",
    )
    assert srep["best_score"] > 0.0
    assert len(srep["history"]) == 2

    rc = cli_main(
        [
            "connectome",
            "embody-batch",
            "synthetic_fly_embodied",
            "--mode",
            "benchmark",
            "--batch-sizes",
            "1,2",
            "--ticks",
            "120",
            "--physics",
            "cpu",
            "--brain",
            "numpy",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["scaling"]) == 2


@needs_mujoco
def test_f9_warp_matches_cpu_behavior_when_gpu_available() -> None:
    if not warp_available():
        pytest.skip("MuJoCo Warp CUDA not available")
    pack = make_synthetic_fly_embodied_pack()
    rep = benchmark_batched_embodied(
        pack,
        batch_sizes=(1, 4),
        ticks=180,
        seed=1,
        physics_backend="warp",
        brain_backend="numpy",
    )
    chk = rep["cpu_vs_warp_check"]
    assert chk["cpu_feeding_ms"] > 80
    assert chk["warp_feeding_ms"] > 80
    assert chk["feeding_ms_diff"] <= 2
    assert chk["distance_cm_diff"] < 0.02
