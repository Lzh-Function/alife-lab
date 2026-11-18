"""BFF replicator assay: positive control (an evolved program), negative controls, verdict
rule, determinism and budget validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alife_lab.substrates.bff.assay import BFFAssayParams, run_assay

# Most frequent program in the soup of a reference run (bff_noheads, 2^17 programs, seed 0,
# CuBFF mutation rate) at epoch 4795, after the entropy transition at epochs 2800-2900.
EVOLVED = (
    "5b5b732e5b5b3c2c242ee03d71bccc62bf242effffb4b4cfff7cffffa2ff2c79"
    "f1491068bcd27a657dc95a105d7cc77d2434bc3b531723e06f242c3c705b71ac"
)
COPIER_BFF = "80c05b2e3e7d5d" + "00" * 57  # hand-written 7-byte copier (bff variant)
SMALL = {"soup_size": 256, "copies": 8, "epochs": 48, "replicates": 3, "partners": 64}


def test_positive_control_evolved_program_replicates_in_soup() -> None:
    r = run_assay(EVOLVED, {**SMALL, "seed": 0})
    assert r["verdict"]["outcome"] == "replicates_in_assay"
    for rep in r["soup_assay"]["replicates"]:
        assert rep["conditions"]["scrambled"]["final_exact"] <= 8  # control does not grow
        assert rep["conditions"]["background"]["exact"][0] == 0
    # Observed: no copying within one interaction, even with itself (recorded, not assumed).
    assert r["pair_assay"]["self_pair"]["second_half_intact"] is False


def test_partial_copier_is_not_a_replicator() -> None:
    r = run_assay(COPIER_BFF, {**SMALL, "variant": "bff"})
    assert r["verdict"]["outcome"] == "no_evidence"
    # It copies its 7 bytes into an all-NUL partner, but not into random partners.
    first = r["pair_assay"]["program_first"]
    assert first["nul_partner"]["partner_became_copy"] is True
    assert first["random_partners"]["copy_rate"] == 0.0


def test_uniform_program_is_inconclusive() -> None:
    r = run_assay("00" * 64, SMALL)
    assert r["verdict"]["outcome"] == "inconclusive"
    assert r["scrambled_hex"] == r["program_hex"]


def test_assay_is_deterministic() -> None:
    a = run_assay(EVOLVED, {**SMALL, "replicates": 1, "epochs": 16})
    b = run_assay(EVOLVED, {**SMALL, "replicates": 1, "epochs": 16})
    a.pop("elapsed_seconds")
    b.pop("elapsed_seconds")
    assert a == b


def test_params_are_bounded() -> None:
    with pytest.raises(ValidationError):
        BFFAssayParams(soup_size=4096, epochs=512, replicates=7)  # budget exceeded
    with pytest.raises(ValidationError):
        BFFAssayParams(soup_size=64, copies=40)
    with pytest.raises(ValidationError):
        BFFAssayParams.model_validate({"bogus": 1})
