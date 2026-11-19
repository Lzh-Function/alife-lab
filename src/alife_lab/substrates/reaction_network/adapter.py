"""Abstract reaction network adapter (Gillespie direct method). See ``gillespie.py``.

One tick advances simulated time by ``time_per_tick``. The display is a species x time heatmap
of log counts (a display-only ring buffer, rebuilt from the current counts after a restore).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alife_lab import API_CONTRACT_VERSION
from alife_lab.core.spec import ExperimentSpec, ReactionNetworkExperiment
from alife_lab.substrates.base import Capabilities, MetricDefinition, PluginManifest
from alife_lab.substrates.reaction_network import gillespie as g
from alife_lab.substrates.reaction_network.gillespie import I64, Network

SERIALIZE_VERSION = 1
HISTORY = 256
MAX_ADD = 10**7


def _rn(spec: ExperimentSpec) -> ReactionNetworkExperiment:
    exp = spec.experiment
    if not isinstance(exp, ReactionNetworkExperiment):
        raise TypeError("reaction_network adapter received a different spec")
    return exp


def _network(exp: ReactionNetworkExperiment) -> Network:
    return Network.from_spec(exp.dynamics.species, exp.dynamics.reactions)


@dataclass
class ReactionState:
    counts: I64
    fired: I64
    enabled: Any  # bool array per reaction
    t: float
    tick: int
    rng: np.random.Generator = field(repr=False)
    net: Network = field(repr=False)
    last_events: int = 0
    history: Any = field(default=None, repr=False)  # display only

    def push_history(self) -> None:
        if self.history is None:
            self.history = np.zeros((self.counts.size, HISTORY), dtype=np.float64)
        self.history = np.roll(self.history, -1, axis=1)
        self.history[:, -1] = np.log1p(self.counts.astype(np.float64))


_M = MetricDefinition
_METRICS = (
    _M("total_molecules", "metric.total_molecules", "molecules", "tick",
       "Sum of all non-buffered species counts.", "none", "every tick", 1, "never null"),
    _M("species_present", "metric.species_present", "species", "tick",
       "Number of non-buffered species with count > 0.", "none", "every tick", 1, "never null"),
    _M("events", "metric.events", "events", "tick", "Reaction events during the last tick.",
       "none", "every tick", 1, "never null"),
    _M("diversity", "metric.diversity", "effective species", "tick",
       "exp(Shannon entropy) of the non-buffered composition.", "none", "every tick", 1,
       "null when no molecules are present"),
    _M("time", "metric.time", "time units", "tick", "Simulated time.", "none", "every tick",
       1, "never null"),
)  # fmt: skip

REACTION_NETWORK_MANIFEST = PluginManifest(
    plugin_id="alife_lab.substrates.reaction_network",
    plugin_version="0.1.0",
    api_contract_version=API_CONTRACT_VERSION,
    kind="substrate",
    slug="reaction_network",
    display_name_key="substrate.reaction_network",
    scientific_modes=("reference", "exploratory"),
    capabilities=Capabilities(
        state_kind="network",
        clock_unit="tick",
        supports_pause=True,
        supports_interventions=True,
        supports_inspection=True,
        supports_lineage=False,
        supports_differentiation=False,
        supports_checkpoint=True,
        supports_deterministic_cpu=True,
        intervention_types=("add", "remove", "disable_reaction", "enable_reaction"),
    ),
    backends=("numpy",),
    gpu_required=False,
    serialize_version=SERIALIZE_VERSION,
    metrics=_METRICS,
    reference={
        "description": "Gillespie (1977) exact stochastic simulation (direct method) of an "
        "abstract reaction network; no physico-chemical realism implied.",
        "paper_urls": ["https://doi.org/10.1021/j100540a008"],
        "implementation": "this repository",
    },
)


def _species_index(state: ReactionState, payload: dict[str, Any]) -> int:
    if "species" in payload:
        name = str(payload["species"])
        if name not in state.net.names:
            raise ValueError(f"unknown species {name!r}")
        return state.net.names.index(name)
    try:
        row = int(payload["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("payload needs species (name) or y (row)") from exc
    if not 0 <= row < len(state.net.names):
        raise ValueError("no species at this row")
    return row


class ReactionNetworkAdapter:
    @property
    def manifest(self) -> PluginManifest:
        return REACTION_NETWORK_MANIFEST

    def initialize(self, spec: ExperimentSpec) -> ReactionState:
        exp = _rn(spec)
        net = _network(exp)
        disabled = set(exp.dynamics.disabled_reactions)
        state = ReactionState(
            counts=np.array([sp.initial for sp in exp.dynamics.species], dtype=np.int64),
            fired=np.zeros(len(net.rates), dtype=np.int64),
            enabled=np.array([rid not in disabled for rid in net.reaction_ids], dtype=np.bool_),
            t=0.0,
            tick=0,
            rng=np.random.Generator(np.random.PCG64(exp.seed)),
            net=net,
        )
        state.push_history()
        return state

    def step(self, state: ReactionState, spec: ExperimentSpec) -> ReactionState:
        d = _rn(spec).dynamics
        t_end = (state.tick + 1) * d.time_per_tick
        state.t, state.last_events = g.simulate_until(
            state.net, state.counts, state.enabled, state.fired, state.t, t_end, state.rng,
            d.max_events_per_tick,
        )  # fmt: skip
        state.tick += 1
        state.push_history()
        return state

    def metrics(self, state: ReactionState, spec: ExperimentSpec) -> dict[str, float]:
        exp = _rn(spec)
        wanted = set(exp.observations.metrics)
        free = np.array([not c for c in state.net.constant], dtype=np.bool_)
        x = state.counts[free].astype(np.float64)
        total = float(x.sum())
        out: dict[str, float] = {}
        if "total_molecules" in wanted:
            out["total_molecules"] = total
        if "species_present" in wanted:
            out["species_present"] = float((x > 0).sum())
        if "events" in wanted:
            out["events"] = float(state.last_events)
        if "diversity" in wanted:
            if total > 0:
                p = x[x > 0] / total
                out["diversity"] = float(math.exp(-(p * np.log(p)).sum()))
            else:
                out["diversity"] = math.nan
        if "time" in wanted:
            out["time"] = state.t
        if exp.observations.record_species_counts:
            for name, c in zip(state.net.names, state.counts, strict=True):
                out[f"count.{name}"] = float(c)
        return out

    def render(self, state: ReactionState) -> tuple[dict[str, Any], bytes]:
        hist = state.history
        top = float(hist.max()) or 1.0
        img = np.rint(np.clip(hist / top, 0.0, 1.0) * 255).astype(np.uint8)
        header = {
            "codec": "field_u8",
            "width": HISTORY,
            "height": int(hist.shape[0]),
            "channels": 1,
            "scale": "rows = species, columns = last 256 ticks, log1p(count) / max",
            "rows": list(state.net.names),
        }
        return header, img.tobytes()

    def apply_intervention(
        self, state: ReactionState, spec: ExperimentSpec, kind: str, payload: dict[str, Any]
    ) -> tuple[ReactionState, dict[str, Any]]:
        if kind in ("disable_reaction", "enable_reaction"):
            rid = str(payload.get("reaction", ""))
            if rid not in state.net.reaction_ids:
                raise ValueError(f"unknown reaction {rid!r}")
            mu = state.net.reaction_ids.index(rid)
            state.enabled[mu] = kind == "enable_reaction"
            return state, {"reaction": rid, "enabled": bool(state.enabled[mu])}
        i = _species_index(state, payload)
        if state.net.constant[i]:
            raise ValueError("buffered (constant) species cannot be changed by interventions")
        try:
            count = int(payload.get("count", 100))
        except (TypeError, ValueError) as exc:
            raise ValueError("count must be an integer") from exc
        if not 0 <= count <= MAX_ADD:
            raise ValueError(f"count must be within 0..{MAX_ADD}")
        before = int(state.counts[i])
        if kind == "add":
            state.counts[i] = before + count
        elif kind == "remove":
            state.counts[i] = max(0, before - count) if payload.get("count") is not None else 0
        else:
            raise ValueError(f"unsupported intervention kind {kind!r}")
        return state, {"species": state.net.names[i], "before": before,
                       "after": int(state.counts[i])}  # fmt: skip

    def inspect(
        self, state: ReactionState, spec: ExperimentSpec, query: dict[str, Any]
    ) -> dict[str, Any]:
        net = state.net
        a = g.all_propensities(net, state.counts, state.enabled)
        reactions = [
            {
                "id": rid,
                "label": net.labels[mu],
                "equation": " + ".join(f"{s if s > 1 else ''}{net.names[i]}"
                                       for i, s in net.reactants[mu]) or "∅",
                "products": " + ".join(f"{s if s > 1 else ''}{net.names[i]}"
                                       for i, s in net.products[mu]) or "∅",
                "rate": net.rates[mu],
                "enabled": bool(state.enabled[mu]),
                "propensity": a[mu],
                "fired": int(state.fired[mu]),
            }
            for mu, rid in enumerate(net.reaction_ids)
        ]  # fmt: skip
        selected: dict[str, Any] | None = None
        if "y" in query or "species" in query:
            i = _species_index(state, query)
            selected = {
                "name": net.names[i],
                "count": int(state.counts[i]),
                "constant": net.constant[i],
                "consumed_by": [r["id"] for mu, r in enumerate(reactions)
                                if any(j == i for j, _ in net.reactants[mu])],
                "produced_by": [r["id"] for mu, r in enumerate(reactions)
                                if any(j == i for j, _ in net.products[mu])],
            }  # fmt: skip
        return {
            "substrate": "reaction_network",
            "tick": state.tick,
            "time": state.t,
            "species": [
                {"name": n, "count": int(c), "constant": k}
                for n, c, k in zip(net.names, state.counts, net.constant, strict=True)
            ],
            "selected_species": selected,
            "reactions": reactions,
            "cycle_candidates": g.cycle_candidates(net, state.fired),
        }

    def state_distance(self, a: ReactionState, b: ReactionState) -> dict[str, Any]:
        if a.counts.shape != b.counts.shape:
            return {"metric": "count_l1_difference", "value": None, "exact_equal": False}
        l1 = int(np.abs(a.counts - b.counts).sum())
        return {
            "metric": "count_l1_difference",
            "value": l1,
            "exact_equal": l1 == 0 and a.t == b.t and np.array_equal(a.fired, b.fired),
        }

    def serialize_state(self, state: ReactionState) -> dict[str, Any]:
        return {
            "arrays": {"counts": state.counts, "fired": state.fired, "enabled": state.enabled},
            "meta": {
                "serialize_version": SERIALIZE_VERSION,
                "tick": state.tick,
                "time": state.t,
                "rng_state": state.rng.bit_generator.state,
                "last_events": state.last_events,
                "species": list(state.net.names),
                "reactions": list(state.net.reaction_ids),
            },
        }

    def deserialize_state(self, payload: dict[str, Any], spec: ExperimentSpec) -> ReactionState:
        exp = _rn(spec)
        meta = payload["meta"]
        if meta["serialize_version"] != SERIALIZE_VERSION:
            raise ValueError("incompatible reaction_network checkpoint serialize_version")
        net = _network(exp)
        if meta.get("species") != list(net.names) or meta.get("reactions") != list(
            net.reaction_ids
        ):
            raise ValueError("checkpoint network does not match the spec")
        arr = payload["arrays"]
        counts = np.asarray(arr["counts"], dtype=np.int64).copy()
        fired = np.asarray(arr["fired"], dtype=np.int64).copy()
        enabled = np.asarray(arr["enabled"], dtype=np.bool_).copy()
        if counts.shape != (len(net.names),) or counts.min(initial=0) < 0:
            raise ValueError("invalid species counts in checkpoint")
        if fired.shape != (len(net.rates),) or enabled.shape != (len(net.rates),):
            raise ValueError("invalid reaction arrays in checkpoint")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = meta["rng_state"]
        state = ReactionState(
            counts=counts, fired=fired, enabled=enabled, t=float(meta["time"]),
            tick=int(meta["tick"]), rng=rng, net=net, last_events=int(meta.get("last_events", 0)),
        )  # fmt: skip
        state.push_history()
        return state
