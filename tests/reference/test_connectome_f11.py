"""Phase F11 reference tests: two-fly innate courtship communication (CPU-only CI safe)."""

from __future__ import annotations

import json

import pytest

from alife_lab.cli import main as cli_main
from alife_lab.connectome.dyadic_fly import (
    make_synthetic_female_auditory_pack,
    make_synthetic_male_song_pack,
    run_f11_song_dialogue,
    run_f11_two_fly_arena,
)


def test_f11_stage1_song_dialogue_and_ablations_cpu_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    male = make_synthetic_male_song_pack()
    female = make_synthetic_female_auditory_pack()
    rep = run_f11_song_dialogue(male, female, backend="numpy", seed=1, ticks=250)
    assert rep["all_checks_passed"] is True, rep["checks"]
    assert rep["scenes"]["intact_close"]["female_auditory_relay_hz"] > 20.0
    assert rep["scenes"]["acoustic_blocked"]["female_auditory_relay_hz"] == 0.0
    assert rep["scenes"]["male_shuffled"]["male_pulse_ps1_hz"] == 0.0

    # Silencing pIP10 must abolish both pulse and sine song + female hearing!
    ab_map = {a["silenced_cell_type"]: a for a in rep["janelia_song_ablations"]}
    assert ab_map["pIP10"]["pulse_suppression_pct"] == 100.0
    assert ab_map["pIP10"]["female_aud_suppression_pct"] == 100.0
    assert ab_map["dPR1"]["pulse_suppression_pct"] > 50.0

    rc = cli_main(
        [
            "connectome",
            "social",
            "--stage",
            "dialogue",
            "--male",
            "synthetic_male_song",
            "--female",
            "synthetic_female_auditory",
            "--backend",
            "numpy",
            "--ticks",
            "200",
            "--no-ablations",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["all_checks_passed"] is True


def test_f11_stage2_two_fly_arena_and_controls_cpu_only() -> None:
    male = make_synthetic_male_song_pack()
    female = make_synthetic_female_auditory_pack()
    rep = run_f11_two_fly_arena(male, female, backend="numpy", seed=1, ticks=1000)
    assert rep["all_checks_passed"] is True, rep["checks"]
    assert rep["scenes"]["intact_pair"]["min_dist_cm"] < 0.70
    assert rep["scenes"]["intact_pair"]["male_singing_ms"] > 200
    assert (
        rep["scenes"]["intact_pair"]["female_distance_cm"]
        < rep["scenes"]["acoustic_blocked"]["female_distance_cm"] * 0.85
    )
