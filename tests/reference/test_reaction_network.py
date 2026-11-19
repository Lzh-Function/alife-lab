"""Reaction network (Gillespie direct method): propensities, exact conservation laws,
analytic statistics (decay mean, birth-death Poisson stationary law), buffered species, event
budget, determinism, catalytic-cycle candidates and the knock-out control."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from alife_lab.artifacts.store import decode_checkpoint, encode_checkpoint
from alife_lab.core.spec import ExperimentSpec
from alife_lab.substrates.reaction_network import gillespie as g
from alife_lab.substrates.reaction_network.adapter import ReactionNetworkAdapter

A = ReactionNetworkAdapter()


def make(species: list[dict[str, Any]], reactions: list[dict[str, Any]], seed: int = 0,
         dt: float = 0.1, **dyn: Any) -> ExperimentSpec:  # fmt: skip
    return ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "rn",
                "substrate": "reaction_network",
                "seed": seed,
                "dynamics": {
                    "species": species,
                    "reactions": reactions,
                    "time_per_tick": dt,
                    **dyn,
                },
            }
        }
    )


def test_propensity_uses_binomial_combinatorics() -> None:
    sp = make([{"name": "A", "initial": 5}, {"name": "B", "initial": 3}],
              [{"id": "dimer", "reactants": {"A": 2}, "products": {"B": 1}, "rate": 0.5},
               {"id": "ab", "reactants": {"A": 1, "B": 1}, "products": {}, "rate": 2.0},
               {"id": "a3", "reactants": {"A": 3}, "products": {}, "rate": 1.0}])  # fmt: skip
    s = A.initialize(sp)
    assert g.all_propensities(s.net, s.counts, s.enabled) == [0.5 * 10, 2.0 * 15, 1.0 * 10]
    s.counts[0] = 1
    assert g.all_propensities(s.net, s.counts, s.enabled) == [0.0, 6.0, 0.0]


def test_reversible_binding_conserves_totals_exactly() -> None:
    sp = make([{"name": "A", "initial": 300}, {"name": "B", "initial": 200},
               {"name": "C", "initial": 50}],
              [{"id": "bind", "reactants": {"A": 1, "B": 1}, "products": {"C": 1}, "rate": 0.01},
               {"id": "unbind", "reactants": {"C": 1}, "products": {"A": 1, "B": 1},
                "rate": 1.0}])  # fmt: skip
    s = A.initialize(sp)
    for _ in range(300):
        s = A.step(s, sp)
        a, b, c = (int(v) for v in s.counts)
        assert (a + c, b + c) == (350, 250)
    assert s.fired.sum() > 1000


def test_decay_mean_matches_exponential_law() -> None:
    finals = []
    for seed in range(200):
        sp = make([{"name": "A", "initial": 100}],
                  [{"id": "d", "reactants": {"A": 1}, "products": {}, "rate": 1.0}],
                  seed=seed, dt=1.0)  # fmt: skip
        s = A.step(A.initialize(sp), sp)
        finals.append(int(s.counts[0]))
    p = math.exp(-1.0)  # each molecule survives to t = 1 with probability e^-1: Binomial(100, p)
    se = math.sqrt(100 * p * (1 - p) / 200)
    assert abs(np.mean(finals) - 100 * p) < 4 * se


def test_birth_death_stationary_law_is_poisson() -> None:
    sp = make([{"name": "A", "initial": 100}],
              [{"id": "birth", "reactants": {}, "products": {"A": 1}, "rate": 100.0},
               {"id": "death", "reactants": {"A": 1}, "products": {}, "rate": 1.0}],
              seed=7, dt=0.5)  # fmt: skip
    s = A.initialize(sp)
    xs = []
    for _ in range(4000):  # samples every 0.5 time units (autocorrelation time 1)
        s = A.step(s, sp)
        xs.append(int(s.counts[0]))
    x = np.array(xs[100:], dtype=float)
    assert abs(x.mean() - 100) < 1.5  # SE ~ sqrt(100 * 3 / 3900) ~ 0.28
    assert 0.85 < x.var() / x.mean() < 1.15  # Poisson: variance == mean


def test_buffered_species_stay_constant() -> None:
    sp = make([{"name": "X", "initial": 1000, "constant": True}, {"name": "Y", "initial": 10}],
              [{"id": "eat", "reactants": {"X": 1, "Y": 1}, "products": {"Y": 2}, "rate": 1e-3},
               {"id": "die", "reactants": {"Y": 1}, "products": {}, "rate": 0.5}])  # fmt: skip
    s = A.initialize(sp)
    for _ in range(100):
        s = A.step(s, sp)
        assert s.counts[0] == 1000
    with pytest.raises(ValueError):
        A.apply_intervention(s, sp, "add", {"species": "X", "count": 5})


def test_event_budget_is_an_error_not_truncation() -> None:
    sp = make([{"name": "A", "initial": 0}],
              [{"id": "birth", "reactants": {}, "products": {"A": 1}, "rate": 1e6}],
              max_events_per_tick=1000)  # fmt: skip
    with pytest.raises(g.EventBudgetExceeded):
        A.step(A.initialize(sp), sp)


def _scenario(name: str) -> ExperimentSpec:
    import yaml

    raw = yaml.safe_load(open(f"configs/scenarios/{name}.yaml", encoding="utf-8"))  # noqa: SIM115
    return ExperimentSpec.model_validate(raw)


def test_catalytic_cycle_candidate_and_knockout_control() -> None:
    sp = _scenario("reaction_network_cycle")
    s = A.initialize(sp)
    for _ in range(400):
        s = A.step(s, sp)
    cyc = A.inspect(s, sp, {})["cycle_candidates"]
    assert {"species": ["I1", "I2", "I3", "I4"], "reactions": ["rep1", "rep2", "rep3", "rep4"]} \
        in cyc["cycles"]  # fmt: skip
    assert min(int(c) for c in s.counts[1:]) > 50

    raw = sp.model_dump(mode="json")
    raw["experiment"]["dynamics"]["disabled_reactions"] = ["rep1", "rep2", "rep3", "rep4"]
    ko = ExperimentSpec.model_validate(raw)
    k = A.initialize(ko)
    for _ in range(1000):  # t = 50: expected survivors 200 * exp(-0.2 * 50) ~ 0.009
        k = A.step(k, ko)
    assert int(k.counts[1:].sum()) == 0
    assert A.inspect(k, ko, {})["cycle_candidates"]["cycles"] == []


def test_lotka_reference_has_buffered_prey_source_and_autocatalysis() -> None:
    sp = _scenario("reaction_network_lotka")
    s = A.initialize(sp)
    y1 = []
    for _ in range(300):
        s = A.step(s, sp)
        y1.append(int(s.counts[1]))
    assert s.counts[0] == 100000 and max(y1) - min(y1) > 200
    auto = {a["species"] for a in A.inspect(s, sp, {})["cycle_candidates"]["autocatalysis"]}
    assert auto == {"Y1", "Y2"}


def test_spec_validation() -> None:
    with pytest.raises(ValidationError):
        make([{"name": "A"}], [{"id": "r", "reactants": {"B": 1}, "rate": 1}])
    with pytest.raises(ValidationError):
        make([{"name": "A"}, {"name": "A"}], [{"id": "r", "rate": 1}])
    with pytest.raises(ValidationError):
        make([{"name": "A"}], [{"id": "r", "rate": 1}], disabled_reactions=["nope"])


def test_interventions_checkpoint_and_determinism() -> None:
    sp = _scenario("reaction_network_cycle")
    s = A.initialize(sp)
    for _ in range(30):
        s = A.step(s, sp)
    s, info = A.apply_intervention(s, sp, "add", {"y": 1, "count": 25})
    assert info["after"] - info["before"] == 25
    s, info = A.apply_intervention(s, sp, "disable_reaction", {"reaction": "rep2"})
    assert info == {"reaction": "rep2", "enabled": False}
    ser = A.serialize_state(s)
    r = A.deserialize_state(decode_checkpoint(encode_checkpoint(ser["arrays"], ser["meta"])), sp)
    for _ in range(50):
        s, r = A.step(s, sp), A.step(r, sp)
    assert A.state_distance(s, r)["exact_equal"]
    assert A.metrics(s, sp) == A.metrics(r, sp)
    assert A.metrics(s, sp)["count.F"] == float(s.counts[0])
    header, data = A.render(s)
    assert header["rows"] == ["F", "I1", "I2", "I3", "I4"] and len(data) == 5 * 256
