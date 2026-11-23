"""Phase F6 reference tests: fixed-topology connectome task learning (biological vs shuffled).

Pure-NumPy pre-registration, decision-rule, and dataset-generation tests run unconditionally in
CPU-only CI. PyTorch training / adapter / CLI tests are guarded with ``@needs_torch``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.subcircuit import extract_subcircuit, make_synthetic_olfactory_pack
from alife_lab.connectome.tasks import (
    F6_FRAMING_NOTE,
    F6TaskConfig,
    build_task_dataset,
    evaluate_preregistration,
    make_preregistration,
    run_topology_benchmark,
)
from alife_lab.core.spec import ExperimentSpec, load_experiment_spec
from alife_lab.substrates.connectome.task_adapter import ConnectomeTaskAdapter

ROOT = Path(__file__).resolve().parents[2]
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch not installed (uv sync --extra asal)"
)


def test_task_spec_rejects_reference_mode() -> None:
    raw = {
        "experiment": {
            "name": "task_ref_reject",
            "mode": "reference",
            "substrate": "connectome_task",
            "seed": 1,
            "dataset_id": "synthetic_olfactory",
            "subcircuit_id": "full",
        }
    }
    with pytest.raises(Exception, match="exploratory"):
        ExperimentSpec.model_validate(raw)


def test_preregistration_hash_and_decision_rules_cpu_only() -> None:
    plan1 = make_preregistration(
        "flywire_783", "olfactory_mb_lh", "odor_classification", "readout_only"
    )
    plan2 = make_preregistration(
        "flywire_783", "olfactory_mb_lh", "odor_classification", "readout_only"
    )
    assert plan1.to_dict()["plan_digest"] == plan2.to_dict()["plan_digest"]
    assert "borrowing" in F6_FRAMING_NOTE.lower()

    # Validated when reference clearly exceeds shuffled across seeds.
    res_val = evaluate_preregistration(
        plan1, [0.92, 0.90, 0.91, 0.93, 0.89], [0.55, 0.58, 0.52, 0.54, 0.56]
    )
    assert res_val["verdict"] == "validated"

    # Rejected when shuffled significantly beats biological wiring.
    res_rej = evaluate_preregistration(
        plan1, [0.50, 0.52, 0.49, 0.51, 0.48], [0.85, 0.88, 0.86, 0.84, 0.87]
    )
    assert res_rej["verdict"] == "rejected"

    # Inconclusive when difference is small / noisy.
    res_inc = evaluate_preregistration(
        plan1, [0.75, 0.76, 0.74, 0.75, 0.76], [0.74, 0.75, 0.75, 0.76, 0.74]
    )
    assert res_inc["verdict"] == "inconclusive"


def test_task_datasets_deterministic_and_balanced_cpu_only() -> None:
    pack = make_synthetic_olfactory_pack()
    sub = extract_subcircuit(pack, subcircuit_id="full", wiring="reference")
    for task, expected_c in (
        ("odor_classification", 4),
        ("short_memory", 4),
        ("nonlinear_xor", 2),
    ):
        cfg = F6TaskConfig(task=task, n_train=32, n_test=32)  # type: ignore[arg-type]
        ds1 = build_task_dataset(sub, cfg, seed=7)
        ds2 = build_task_dataset(sub, cfg, seed=7)
        assert ds1.num_classes == expected_c
        assert (ds1.x_train_groups == ds2.x_train_groups).all()
        assert (ds1.y_test == ds2.y_test).all()
        assert len(set(ds1.y_test.tolist())) == expected_c


@needs_torch
def test_topology_benchmark_all_scopes_and_reproducible() -> None:
    pack = make_synthetic_olfactory_pack()
    sub_ref = extract_subcircuit(pack, subcircuit_id="full", wiring="reference")
    for scope in ("readout_only", "io_only", "partial_weights"):
        rep = run_topology_benchmark(
            sub_ref,
            lambda s: extract_subcircuit(
                pack, subcircuit_id="full", wiring="shuffled_control", shuffle_seed=s
            ),
            F6TaskConfig(
                task="odor_classification",
                scope=scope,  # type: ignore[arg-type]
                n_train=32,
                n_test=32,
                epochs=40,
                device="cpu",
            ),
            seeds=(1, 2, 3),
        )
        assert rep["evaluation"]["reference_stats"]["mean"] > 0.50
        assert rep["evaluation"]["verdict"] in ("validated", "rejected", "inconclusive")
        assert len(rep["preregistration"]["plan_digest"]) == 64

    # Deterministic repeat with same seed.
    cfg = F6TaskConfig(task="odor_classification", scope="readout_only", epochs=30, device="cpu")
    r1 = run_topology_benchmark(
        sub_ref,
        lambda s: extract_subcircuit(
            pack, subcircuit_id="full", wiring="shuffled_control", shuffle_seed=s
        ),
        cfg,
        seeds=(1,),
    )
    r2 = run_topology_benchmark(
        sub_ref,
        lambda s: extract_subcircuit(
            pack, subcircuit_id="full", wiring="shuffled_control", shuffle_seed=s
        ),
        cfg,
        seeds=(1,),
    )
    assert r1["reference_runs"][0]["test_acc"] == r2["reference_runs"][0]["test_acc"]


@needs_torch
def test_task_adapter_render_checkpoint_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    spec = load_experiment_spec(ROOT / "configs/scenarios/connectome_task_olfactory_tiny.yaml")
    ad = ConnectomeTaskAdapter()
    ad.set_backend("torch")
    w = ad.initialize(spec)
    for _ in range(25):
        ad.step(w, spec)
    m = ad.metrics(w, spec)
    assert m["test_acc"] > m["chance_acc"]

    hdr, raw = ad.render(w)
    assert hdr["codec"] == "field_u8"
    assert hdr["width"] == 160 and hdr["height"] == 120 and hdr["channels"] == 3
    assert len(raw) == 160 * 120 * 3

    info = ad.inspect(w, spec, {"x": 0, "y": 0})
    assert info["substrate"] == "connectome_task"
    assert "borrowing" in info["framing_note"].lower()

    # Checkpoint serialize -> deserialize exact match.
    ckpt = ad.serialize_state(w)
    w2 = ad.deserialize_state(ckpt, spec)
    assert ad.state_distance(w, w2)["exact_equal"] is True

    # CLI JSON smoke test.
    rc = cli_main(
        [
            "connectome",
            "task",
            "synthetic_olfactory",
            "--task",
            "odor_classification",
            "--scope",
            "readout_only",
            "--seeds",
            "1,2",
            "--epochs",
            "25",
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dataset_id"] == "synthetic_olfactory"
    assert out["evaluation"]["reference_stats"]["mean"] > 0.50
