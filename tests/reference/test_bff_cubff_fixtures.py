"""External validation: soups after k epochs are byte-identical to a CuBFF binary.

Fixture digests were produced by the CuBFF CPU build (see tests/fixtures/cubff_soups.json and
scripts/regenerate_cubff_fixtures.sh)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from alife_lab.substrates.bff.soup import init_soup, run_epoch

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cubff_soups.json"
CASES: list[dict[str, Any]] = json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[f"{c['lang']}-{c['num']}-s{c['seed']}" for c in CASES])
def test_soup_matches_cubff_binary(case: dict[str, Any]) -> None:
    soup = init_soup(case["seed"], case["num"])
    done = 0
    for snap in case["snapshots"]:
        while done < snap["epochs_completed"]:
            run_epoch(soup, case["seed"], done, case["mutation_prob_int"], variant=case["lang"])
            done += 1
        assert hashlib.sha256(soup.tobytes()).hexdigest() == snap["soup_sha256"], (
            f"diverged from CuBFF after {done} epochs"
        )
