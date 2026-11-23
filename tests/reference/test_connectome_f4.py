"""Phase F4 reference tests: Mushroom Body KC -> MBON three-factor plasticity and
Pavlovian olfactory conditioning (specificity, dose-response, extinction, unpaired timing
control, DAN-silenced control, checkpoint roundtrip, CPU == GPU bitwise equivalence).

Runs offline in CPU-only CI using built-in ``synthetic_mb``; also asserts GPU bitwise match
when CuPy is installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.learning import run_pavlovian_experiment
from alife_lab.connectome.lif import make_synthetic_mb_pack
from alife_lab.connectome.plasticity import MBPlasticityParams
from alife_lab.core.spec import ExperimentSpec, load_experiment_spec
from alife_lab.substrates.connectome import ConnectomeAdapter

ROOT = Path(__file__).resolve().parents[2]


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def test_plasticity_spec_rejects_reference_mode_and_preserves_hash() -> None:
    raw = {
        "experiment": {
            "name": "mb_ref_reject",
            "mode": "reference",
            "substrate": "connectome",
            "seed": 1,
            "dataset_id": "synthetic_mb",
            "wiring": "reference",
            "plasticity": {"enabled": True},
        }
    }
    with pytest.raises(Exception, match="plasticity disabled"):
        ExperimentSpec.model_validate(raw)

    # Disabled/omitted plasticity does not pollute serialized JSON.
    base_spec = load_experiment_spec(ROOT / "configs/scenarios/connectome_synthetic_tiny.yaml")
    assert "plasticity" not in base_spec.model_dump(mode="json")["experiment"]


def test_synthetic_mb_pavlovian_specificity_dose_extinction_and_controls() -> None:
    pack = make_synthetic_mb_pack()
    pp = MBPlasticityParams(
        enabled=True,
        tau_elig_ms=200.0,
        tau_dan_ms=150.0,
        dan_threshold=18.0,
        lr_ltd=0.00022,
        lr_ext=0.03,
        w_min=0.05,
        w_max=2.0,
        min_dan_synapses=5,
    )
    rep = run_pavlovian_experiment(
        pack,
        seeds=(1, 2, 3, 4, 5),
        p_params=pp,
        odor_hz=150.0,
        dan_hz=220.0,
        probe_ms=300,
        trial_ms=200,
        n_trials=4,
        ext_ms=1200,
        unpaired_gap_ms=1500,
        wiring="reference",
        backend="numpy",
    )
    sm = rep["summary"]
    pre_a = sm["pre_odor_a_mbon11_hz"]["mean"]
    after1_a = sm["after1_odor_a_mbon11_hz"]["mean"]
    post_a = sm["post_odor_a_mbon11_hz"]["mean"]
    pre_b = sm["pre_odor_b_mbon11_hz"]["mean"]
    post_b = sm["post_odor_b_mbon11_hz"]["mean"]
    ext_a = sm["ext_odor_a_mbon11_hz"]["mean"]
    unp_a = sm["unpaired_odor_a_mbon11_hz"]["mean"]
    nodan_a = sm["no_dan_odor_a_mbon11_hz"]["mean"]

    # 1. Paired Odor A + PPL101 strongly depresses MBON11 and disinhibits avoidance MBON01.
    assert pre_a > 60.0
    assert post_a < pre_a * 0.65
    assert rep["comparisons"]["paired_post_vs_pre_odor_a"]["permutation"]["p_value"] < 0.05
    assert sm["post_odor_a_mbon01_hz"]["mean"] > sm["pre_odor_a_mbon01_hz"]["mean"] + 15.0

    # 2. Odor specificity: unpaired Odor B stays intact (only Poisson sampling variance)!
    assert post_b > pre_b * 0.85
    assert abs(post_b - pre_b) < 20.0
    assert post_a < post_b * 0.65
    assert abs(sm["post_odor_b_mbon01_hz"]["mean"] - sm["pre_odor_a_mbon01_hz"]["mean"]) < 15.0

    # 3. Dose-response: 4 trials produce strictly stronger depression than 1 trial.
    assert post_a < after1_a < pre_a

    # 4. Extinction: repeated Odor A without punishment recovers MBON11 substantially.
    assert ext_a > post_a + 15.0

    # 5. Negative controls: DAN silenced -> exact 0 changed edges & no learning;
    #    Unpaired (1500 ms shifted timing) -> no learning.
    assert sm["changed_edges_no_dan"]["mean"] == 0.0
    assert abs(nodan_a - pre_a) < 12.0
    assert unp_a > pre_a * 0.85

    # 6. GPU matches CPU bitwise when CuPy is available.
    if _has_cupy():
        rep_gpu = run_pavlovian_experiment(
            pack,
            seeds=(1, 2),
            p_params=pp,
            odor_hz=150.0,
            dan_hz=220.0,
            probe_ms=300,
            trial_ms=200,
            n_trials=4,
            ext_ms=800,
            unpaired_gap_ms=1500,
            wiring="reference",
            backend="cupy",
        )
        assert (
            rep_gpu["per_seed"][0]["post_odor_a_mbon11_hz"]
            == rep["per_seed"][0]["post_odor_a_mbon11_hz"]
        )
        assert (
            rep_gpu["per_seed"][0]["changed_edges_post"] == rep["per_seed"][0]["changed_edges_post"]
        )


def test_plastic_adapter_checkpoint_roundtrip_and_gpu_bitwise() -> None:
    spec = load_experiment_spec(ROOT / "configs/scenarios/connectome_mb_tiny_pavlov.yaml")
    ad_cpu = ConnectomeAdapter()
    ad_cpu.set_backend("numpy")
    w = ad_cpu.initialize(spec)
    for _ in range(250):
        ad_cpu.step(w, spec)

    m = ad_cpu.metrics(w, spec)
    assert m["plastic_changed_edges"] >= 2.0
    assert m["plastic_min_scale"] < 0.9

    # Checkpoint serialize -> deserialize -> step must stay bitwise identical.
    ckpt = ad_cpu.serialize_state(w)
    w_restored = ad_cpu.deserialize_state(ckpt, spec)
    assert ad_cpu.state_distance(w, w_restored)["exact_equal"] is True
    for _ in range(50):
        ad_cpu.step(w, spec)
        ad_cpu.step(w_restored, spec)
    assert ad_cpu.state_distance(w, w_restored)["exact_equal"] is True

    # Preview frame includes plasticity summary.
    hdr, _ = ad_cpu.render(w)
    assert hdr["plasticity"]["enabled"] is True
    assert hdr["plasticity"]["changed_edges"] >= 2
    assert len(hdr["plasticity"]["top_changed"]) >= 2

    if _has_cupy():
        ad_gpu = ConnectomeAdapter()
        ad_gpu.set_backend("cupy")
        wg = ad_gpu.initialize(spec)
        for _ in range(300):
            ad_gpu.step(wg, spec)
        dist = ad_cpu.state_distance(w, wg)
        assert dist["exact_equal"] is True
        assert dist["max_plastic_scale_diff"] == 0.0


def test_cli_connectome_pavlov_synthetic_mb(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(
        [
            "connectome",
            "pavlov",
            "synthetic_mb",
            "--seeds",
            "1,2",
            "--probe-ms",
            "200",
            "--trial-ms",
            "200",
            "--trials",
            "3",
            "--ext-ms",
            "600",
            "--backend",
            "numpy",
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["dataset_id"] == "synthetic_mb"
    assert (
        data["summary"]["post_odor_a_mbon11_hz"]["mean"]
        < data["summary"]["pre_odor_a_mbon11_hz"]["mean"]
    )
    assert data["summary"]["changed_edges_no_dan"]["mean"] == 0.0
