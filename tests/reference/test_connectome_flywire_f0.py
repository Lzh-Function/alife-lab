"""Reference F0 verification tests against local FlyWire v783 tables.

Skipped cleanly in CPU-only CI environments where the ~142 MiB dataset has not been fetched.
"""

from __future__ import annotations

import json

import pytest

from alife_lab.connectome import (
    FLYWIRE_783_REFERENCE,
    load_flywire_783,
    summarize_connectome_pack,
    verify_dataset,
)
from alife_lab.service.settings import load_settings


def _datasets_dir():
    return load_settings().resolved_datasets_dir


@pytest.mark.skipif(
    not verify_dataset("flywire_783", _datasets_dir())["ok"],
    reason=(
        "local flywire_783 dataset not present/verified (run `alife connectome fetch flywire_783`)"
    ),
)
def test_flywire_783_f0_reference_and_repeatability() -> None:
    ds_dir = _datasets_dir()
    pack1 = load_flywire_783(ds_dir, verify_hashes=True)
    pack2 = load_flywire_783(ds_dir, verify_hashes=True)

    # F0 gate: loading the same files twice yields identical digest and summary.
    assert pack1.pack_digest == pack2.pack_digest
    s1 = summarize_connectome_pack(pack1)
    s2 = summarize_connectome_pack(pack2)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)

    # F0 gate: all published reference checks pass.
    assert s1["reference_checks"]
    for chk in s1["reference_checks"]:
        assert chk["ok"], f"reference check failed: {chk}"

    counts = s1["counts"]
    assert counts["cells"] == FLYWIRE_783_REFERENCE["shiu_v783_neurons"]
    assert counts["edges"] == FLYWIRE_783_REFERENCE["shiu_v783_edges"]
    assert counts["total_synapses"] == FLYWIRE_783_REFERENCE["shiu_v783_synapses"]
    assert counts["self_loops"] == 0
    assert counts["isolated_cells"] == 0
    assert counts["mixed_sign_presynaptic_neurons"] == 0

    # Key sensory / motor / memory circuit ports are populated.
    assert s1["io_ports"]["sugar_grn"]["count"] == 32
    assert s1["io_ports"]["bitter_grn"]["count"] == 42
    assert s1["io_ports"]["mn9_proboscis_extension"]["count"] == 2
    assert s1["io_ports"]["mushroom_body_kc"]["count"] == 5177
    assert s1["io_ports"]["mushroom_body_mbon"]["count"] == 96
    assert s1["io_ports"]["mushroom_body_dan"]["count"] == 331
