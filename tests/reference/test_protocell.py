"""Protocells: exact matter ledger, conservation of hopping, division bookkeeping and lineage,
controls (no membrane / no metabolism / growth only / random partition / fixed traits),
interventions and checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.protocell import model
from alife_lab.substrates.protocell.adapter import ProtocellAdapter

A = ProtocellAdapter()


def make(seed: int = 0, size: int = 32, **dyn: Any) -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "p",
                "substrate": "protocell",
                "seed": seed,
                "world": {"width": size, "height": size},
                "initialization": {"count": 8},
                "dynamics": dyn,
            }
        }
    )


def run(sp: ExperimentSpec, n: int) -> Any:
    s = A.initialize(sp)
    for _ in range(n):
        s = A.step(s, sp)
        assert s.world.residual() == 0  # exact ledger every tick
        assert (s.world.food >= 0).all() and (s.world.cells.contents >= 0).all()
    return s


def test_hopping_conserves_molecules_exactly() -> None:
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 50, (13, 17)).astype(np.int64)
    out = model._hop(grid.copy(), 0.7, rng)
    assert out.sum() == grid.sum() and (out >= 0).all()


def test_ledger_closes_with_growth_division_and_interventions() -> None:
    sp = make()
    s = run(sp, 400)
    lg = s.world.ledger
    assert lg.divisions > 0 and s.world.cells.n > 8
    s, info = A.apply_intervention(s, sp, "add_food", {"x": 3, "y": 3, "count": 500})
    assert info == {"added": 500} and s.world.residual() == 0
    s, _ = A.apply_intervention(s, sp, "add_cell", {"x": 10, "y": 10})
    assert s.world.residual() == 0
    s, info = A.apply_intervention(s, sp, "kill", {"x": 10, "y": 10, "radius": 2})
    assert info["killed"] >= 1 and s.world.residual() == 0


def test_division_lineage_and_partition_records() -> None:
    s = run(make(), 400)
    wd = s.world
    alive = {int(i) for i in wd.cells.ids}
    for div in wd.recent_divisions:
        parent = div["parent"]
        assert wd.lineage[str(parent)][2] == div["tick"]  # the parent ends at division
        for child in div["children"]:
            assert wd.lineage[str(child)][:2] == [parent, div["tick"]]
    for cid in alive:  # every living cell traces back to an initial cell
        cur, hops = cid, 0
        while wd.lineage[str(cur)][0] >= 0 and hops < 10_000:
            cur, hops = wd.lineage[str(cur)][0], hops + 1
        assert cur < 8


@pytest.mark.parametrize("dyn", [{"membrane": False}, {"metabolism": False}])
def test_controls_without_compartment_or_metabolism_die_out(dyn: dict[str, Any]) -> None:
    s = run(make(**dyn), 300)
    assert s.world.cells.n == 0 and s.world.ledger.divisions == 0


def test_growth_only_control_never_divides() -> None:
    s = run(make(division=False), 300)
    assert s.world.ledger.divisions == 0 and s.world.cells.n == 8
    assert model.radius(s.world.cells.contents[:, 2], 20.0).mean() > 2.0


def test_random_partition_is_more_asymmetric() -> None:
    def asym(**dyn: Any) -> float:
        s = run(make(**dyn), 400)
        return float(np.mean([d["asymmetry"] for d in s.world.recent_divisions]))

    assert asym(partition="random_fraction") > 3 * asym()


def test_fixed_traits_do_not_mutate() -> None:
    s = run(make(heritable=[]), 400)
    assert s.world.ledger.divisions > 0
    assert np.all(s.world.cells.genome[:, 1] == 0.02)


def test_checkpoint_continues_bitwise_and_validates_ledger() -> None:
    sp = make()
    s = run(sp, 150)
    ser = A.serialize_state(s)
    r = A.deserialize_state(decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp)
    for _ in range(100):
        s, r = A.step(s, sp), A.step(r, sp)
    assert A.state_distance(s, r)["exact_equal"]
    assert A.metrics(s, sp) == A.metrics(r, sp) or all(
        (a == b) or (a != a and b != b)
        for a, b in zip(A.metrics(s, sp).values(), A.metrics(r, sp).values(), strict=True)
    )
    bad = A.serialize_state(s)
    bad["arrays"]["food"] = bad["arrays"]["food"] + 1
    with pytest.raises(ValueError, match="ledger"):
        A.deserialize_state(bad, sp)
    q = A.inspect(s, sp, {"x": 16, "y": 16})
    assert q["ledger_residual"] == 0 and q["protocell"]["contents"]["membrane"] > 0


def test_spec_is_exploratory_only() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(
            {"experiment": {"name": "p", "substrate": "protocell", "seed": 0, "mode": "reference"}}
        )
