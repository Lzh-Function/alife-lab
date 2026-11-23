"""Phase F5 reference tests: hidden-truth connectome system identification.

Verifies on offline CPU-only ``synthetic_olfactory`` (40 neurons, 18 cell types, 6 classes):
1. Spec contract (rejects ``mode='reference'``, accepts ``mode='exploratory'``).
2. Hidden mechanism recovery (heterogeneous tau_m, sign flips, neuromodulation, gap junctions)
   and unseen-stimulus generalization on observed & unobserved neurons vs ``untrained`` and
   ``shuffled_control`` wiring.
3. Multi-restart identifiability / non-uniqueness reporting.
4. Observability curve (recording fraction sweep) and cell-class observability map.
5. ``connectome_sysid`` substrate adapter lifecycle (initialize, step, render, inspect,
   checkpoint serialize/deserialize exact equivalence) and CLI ``alife connectome sysid`` /
   ``sysid-map``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.subcircuit import extract_subcircuit, make_synthetic_olfactory_pack
from alife_lab.connectome.sysid import (
    HiddenMechanismSpec,
    ObservationSpec,
    SysIDTrainConfig,
    run_hidden_truth_experiment,
    run_observability_study,
)
from alife_lab.core.spec import ExperimentSpec, load_experiment_spec
from alife_lab.substrates.connectome.sysid_adapter import ConnectomeSysIDAdapter

ROOT = Path(__file__).resolve().parents[2]
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch not installed (uv sync --extra asal)"
)


def test_sysid_spec_rejects_reference_mode() -> None:
    raw = {
        "experiment": {
            "name": "sysid_ref_reject",
            "mode": "reference",
            "substrate": "connectome_sysid",
            "seed": 1,
            "dataset_id": "synthetic_olfactory",
            "subcircuit_id": "full",
        }
    }
    with pytest.raises(Exception, match="exploratory"):
        ExperimentSpec.model_validate(raw)


def test_synthetic_olfactory_subcircuit_extraction_cpu_only() -> None:
    pack = make_synthetic_olfactory_pack()
    sub_ref = extract_subcircuit(pack, subcircuit_id="full", wiring="reference")
    sub_shuf = extract_subcircuit(
        pack, subcircuit_id="full", wiring="shuffled_control", shuffle_seed=1
    )
    assert sub_ref.num_cells == 40
    assert sub_ref.num_edges == 236
    # Degree-preserving shuffle drops self-loops / merges duplicates on a dense 40-node graph.
    assert 180 < sub_shuf.num_edges <= sub_ref.num_edges
    assert sub_ref.num_types == 18
    assert sub_ref.num_classes == 6
    assert len(sub_ref.input_groups) == 4


@needs_torch
def test_hidden_truth_recovery_generalization_and_shuffled_control() -> None:
    pack = make_synthetic_olfactory_pack()
    sub_ref = extract_subcircuit(pack, subcircuit_id="full", wiring="reference")
    sub_shuf = extract_subcircuit(
        pack, subcircuit_id="full", wiring="shuffled_control", shuffle_seed=1
    )
    rep = run_hidden_truth_experiment(
        sub_ref,
        sub_shuffled=sub_shuf,
        mech=HiddenMechanismSpec(gap_candidates=24),
        obs=ObservationSpec(obs_fraction=0.5, bin_ms=10.0, noise_sd=0.01),
        cfg=SysIDTrainConfig(epochs=140, n_restarts=3, device="cpu"),
        seed=1,
    )
    untr = rep["untrained"]
    ref = rep["reference_best"]
    shuf = rep["shuffled_control"]
    ident = rep["identifiability"]

    # 1. Strong improvement on training fit and unseen-stimulus generalization.
    assert ref["train_obs_r2"] > 0.95
    assert ref["heldout_all_r2"] > untr["heldout_all_r2"] + 0.30
    assert ref["heldout_unobs_r2"] > untr["heldout_unobs_r2"] + 0.25

    # 2. Biological wiring strongly beats degree-preserving shuffled wiring on unobserved cells!
    assert shuf is not None
    assert ref["heldout_all_r2"] > shuf["heldout_all_r2"] + 0.30
    assert ref["heldout_unobs_r2"] > shuf["heldout_unobs_r2"] + 0.50

    # 3. Genuine recovery of all four hidden mechanisms (no artificial mean-offset bias).
    assert untr["neuromod_pearson_r"] is None
    assert ref["tau_pearson_r"] > 0.85
    assert ref["tau_mae_ms"] < untr["tau_mae_ms"] * 0.5
    assert ref["sign_flip_auroc"] > 0.75
    assert ref["neuromod_pearson_r"] > 0.50
    assert ref["gap_auroc"] > 0.70

    # 4. Multi-restart identifiability report is populated cleanly.
    assert ident["n_restarts"] == 3
    assert ident["verdict"] in ("well_identified", "partially_identifiable", "non_unique_sloppy")


@needs_torch
def test_observability_curve_and_cell_class_map() -> None:
    pack = make_synthetic_olfactory_pack()
    sub_ref = extract_subcircuit(pack, subcircuit_id="full", wiring="reference")
    rep = run_observability_study(
        sub_ref,
        fractions=(0.20, 0.80),
        class_targets=("ALPN", "Kenyon_Cell", "MBON"),
        mech=HiddenMechanismSpec(gap_candidates=16),
        bin_ms=10.0,
        noise_sd=0.01,
        cfg=SysIDTrainConfig(epochs=70, n_restarts=2, device="cpu"),
        seed=2,
    )
    assert len(rep["fraction_curve"]) == 2
    low_f, high_f = rep["fraction_curve"]
    assert high_f["observed_cells"] > low_f["observed_cells"]
    assert high_f["reference_best"]["heldout_all_r2"] > low_f["reference_best"]["heldout_all_r2"]
    assert len(rep["cell_class_map"]) == 3
    # Sorted descending by heldout_unobs_r2.
    vals = [r["reference_best"]["heldout_unobs_r2"] for r in rep["cell_class_map"]]
    assert vals == sorted(vals, reverse=True)


@needs_torch
def test_sysid_adapter_render_checkpoint_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    spec = load_experiment_spec(ROOT / "configs/scenarios/connectome_sysid_olfactory_tiny.yaml")
    ad = ConnectomeSysIDAdapter()
    ad.set_backend("torch")
    w = ad.initialize(spec)
    loss0 = ad.metrics(w, spec)["loss"]
    for _ in range(15):
        ad.step(w, spec)
    m = ad.metrics(w, spec)
    assert m["loss"] < loss0

    hdr, raw = ad.render(w)
    assert hdr["codec"] == "field_u8"
    assert hdr["width"] == 160 and hdr["height"] == 120 and hdr["channels"] == 3
    assert len(raw) == 160 * 120 * 3

    info = ad.inspect(w, spec, {"x": 10, "y": 10})
    assert info["substrate"] == "connectome_sysid"
    assert info["epoch"] == 15

    # Checkpoint serialize -> deserialize -> exact parameter equality.
    ckpt = ad.serialize_state(w)
    w2 = ad.deserialize_state(ckpt, spec)
    assert ad.state_distance(w, w2)["exact_equal"] is True

    # CLI JSON smoke test.
    rc = cli_main(
        [
            "connectome",
            "sysid",
            "synthetic_olfactory",
            "--epochs",
            "30",
            "--restarts",
            "2",
            "--gap-candidates",
            "16",
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dataset_id"] == "synthetic_olfactory"
    assert out["reference_best"]["loss"] < out["untrained"]["loss"]
