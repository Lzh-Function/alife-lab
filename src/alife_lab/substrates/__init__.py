"""Substrate registry. Each substrate is an independent adapter."""

from __future__ import annotations

from alife_lab.substrates.base import PluginManifest, SubstrateAdapter
from alife_lab.substrates.bff.adapter import BFF_MANIFEST, BFFAdapter
from alife_lab.substrates.boids.boids import BOIDS_MANIFEST, BoidsAdapter
from alife_lab.substrates.ca.conway import CONWAY_MANIFEST, ConwayAdapter
from alife_lab.substrates.lenia.lenia import LENIA_MANIFEST, LeniaAdapter
from alife_lab.substrates.particle_life.particle_life import (
    PARTICLE_LIFE_MANIFEST,
    ParticleLifeAdapter,
)

_REGISTRY: dict[str, tuple[PluginManifest, type[SubstrateAdapter]]] = {
    "conway": (CONWAY_MANIFEST, ConwayAdapter),
    "boids": (BOIDS_MANIFEST, BoidsAdapter),
    "particle_life": (PARTICLE_LIFE_MANIFEST, ParticleLifeAdapter),
    "bff": (BFF_MANIFEST, BFFAdapter),
    "lenia": (LENIA_MANIFEST, LeniaAdapter),
}


def list_substrates() -> list[PluginManifest]:
    return [manifest for manifest, _ in _REGISTRY.values()]


def get_manifest(slug: str) -> PluginManifest:
    return _REGISTRY[slug][0]


def get_adapter(slug: str) -> SubstrateAdapter:
    return _REGISTRY[slug][1]()


def has_substrate(slug: str) -> bool:
    return slug in _REGISTRY
