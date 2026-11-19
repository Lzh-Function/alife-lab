"""Stochastic simulation of an abstract reaction network (Gillespie 1977, direct method).

D. T. Gillespie, "Exact stochastic simulation of coupled chemical reactions", J. Phys. Chem.
81(25):2340-2361 (1977). Propensity of reaction mu: ``a_mu = c_mu * prod_i C(x_i, s_i)`` with
binomial coefficients for the reactant stoichiometries (``h_mu`` in the paper). Each event:
``tau = -ln(r1) / a0`` with ``r1`` in (0, 1], then the reaction with cumulative propensity just
above ``r2 * a0``. When the next event would pass the end of the tick the drawn ``tau`` is
discarded (valid by memorylessness) and the clock is set to the tick end.

This is an *abstract* chemistry: no physico-chemical realism is implied. Buffered ("constant")
species keep their count. Only reactions that are enabled can fire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

I64 = npt.NDArray[np.int64]


class EventBudgetExceeded(RuntimeError):
    """More events in one tick than ``max_events_per_tick`` (raised, never truncated)."""


@dataclass(frozen=True)
class Network:
    names: tuple[str, ...]
    constant: tuple[bool, ...]
    reaction_ids: tuple[str, ...]
    labels: tuple[str, ...]
    rates: tuple[float, ...]
    reactants: tuple[tuple[tuple[int, int], ...], ...]  # per reaction: (species, stoich)
    products: tuple[tuple[tuple[int, int], ...], ...]
    change: tuple[tuple[tuple[int, int], ...], ...]  # net change on non-constant species
    dependents: tuple[tuple[int, ...], ...]  # reactions whose propensity depends on species i

    @staticmethod
    def from_spec(species: list[Any], reactions: list[Any]) -> Network:
        index = {sp.name: i for i, sp in enumerate(species)}
        constant = tuple(bool(sp.constant) for sp in species)
        reac = tuple(
            tuple((index[n], s) for n, s in sorted(r.reactants.items())) for r in reactions
        )
        prod = tuple(tuple((index[n], s) for n, s in sorted(r.products.items())) for r in reactions)
        change = []
        for rr, pp in zip(reac, prod, strict=True):
            net: dict[int, int] = {}
            for i, s in rr:
                net[i] = net.get(i, 0) - s
            for i, s in pp:
                net[i] = net.get(i, 0) + s
            change.append(tuple(sorted((i, d) for i, d in net.items() if d and not constant[i])))
        deps: list[list[int]] = [[] for _ in species]
        for mu, rr in enumerate(reac):
            for i, _ in rr:
                deps[i].append(mu)
        return Network(
            names=tuple(sp.name for sp in species),
            constant=constant,
            reaction_ids=tuple(r.id for r in reactions),
            labels=tuple(r.label for r in reactions),
            rates=tuple(float(r.rate) for r in reactions),
            reactants=reac,
            products=prod,
            change=tuple(change),
            dependents=tuple(tuple(sorted(set(d))) for d in deps),
        )


def propensity(net: Network, mu: int, counts: I64, enabled: Any) -> float:
    if not enabled[mu]:
        return 0.0
    a = net.rates[mu]
    for i, s in net.reactants[mu]:
        x = int(counts[i])
        if x < s:
            return 0.0
        a *= math.comb(x, s)
    return float(a)


def simulate_until(
    net: Network, counts: I64, enabled: Any, fired: I64, t: float, t_end: float,
    rng: np.random.Generator, max_events: int,
) -> tuple[float, int]:  # fmt: skip
    """Advance ``counts`` (in place) from time ``t`` to ``t_end``. Returns (t_end, events)."""
    a = [propensity(net, mu, counts, enabled) for mu in range(len(net.rates))]
    events = 0
    while True:
        a0 = sum(a)
        if a0 <= 0.0:
            return t_end, events
        r1 = 1.0 - float(rng.random())  # (0, 1]
        r2 = float(rng.random())
        tau = -math.log(r1) / a0
        if t + tau > t_end:
            return t_end, events
        t += tau
        target = r2 * a0
        acc = 0.0
        mu = len(a) - 1
        for k, ak in enumerate(a):
            acc += ak
            if target < acc:
                mu = k
                break
        while a[mu] == 0.0:  # guard against rounding at the upper end
            mu -= 1
        touched: set[int] = set()
        for i, d in net.change[mu]:
            counts[i] += d
            touched.add(i)
        fired[mu] += 1
        events += 1
        if events > max_events:
            raise EventBudgetExceeded(
                f"more than {max_events} reaction events in one tick; reduce time_per_tick "
                "or rates (results are never truncated silently)"
            )
        for i in touched:
            for nu in net.dependents[i]:
                a[nu] = propensity(net, nu, counts, enabled)


def all_propensities(net: Network, counts: I64, enabled: Any) -> list[float]:
    return [propensity(net, mu, counts, enabled) for mu in range(len(net.rates))]


def cycle_candidates(
    net: Network, fired: I64, max_length: int = 6, limit: int = 50
) -> dict[str, Any]:
    """Directed cycles in the species graph of reactions that have fired.

    Edge a -> b for every fired reaction with reactant a and a product b whose count the
    reaction increases (a != b; returned catalysts are not products), excluding buffered
    species. A species appearing on both sides with a net gain is reported as direct
    autocatalysis. These are *candidates* from the observed reaction history, not proofs of
    self-maintaining cycles.
    """
    edges: dict[int, dict[int, set[str]]] = {}
    auto: list[dict[str, Any]] = []
    for mu, rid in enumerate(net.reaction_ids):
        if fired[mu] <= 0:
            continue
        reac = {i for i, _ in net.reactants[mu] if not net.constant[i]}
        prod = {i for i, _ in net.products[mu] if not net.constant[i]}
        gain = dict(net.change[mu])
        made = {i for i in prod if gain.get(i, 0) > 0}  # net producers only (catalysts return)
        for i in reac & made:
            auto.append({"species": net.names[i], "reaction": rid})
        for a in reac:
            for b in made:
                if a != b:
                    edges.setdefault(a, {}).setdefault(b, set()).add(rid)
    cycles: list[dict[str, Any]] = []
    nodes = sorted(edges)
    for start in nodes:  # canonical form: the cycle's smallest node is its start
        stack: list[tuple[int, list[int]]] = [(start, [start])]
        while stack and len(cycles) < limit:
            node, path = stack.pop()
            for nxt in sorted(edges.get(node, {}), reverse=True):
                if nxt == start and len(path) > 1:
                    reactions = sorted(
                        {r for k in range(len(path)) for r in
                         edges[path[k]][path[(k + 1) % len(path)]]}
                    )  # fmt: skip
                    cycles.append({"species": [net.names[i] for i in path], "reactions": reactions})
                elif nxt > start and nxt not in path and len(path) < max_length:
                    stack.append((nxt, [*path, nxt]))
    return {
        "autocatalysis": auto,
        "cycles": cycles,
        "truncated": len(cycles) >= limit,
        "rule": "directed cycles (length <= 6) among non-buffered species over reactions that "
        "fired at least once; candidates only",
    }
