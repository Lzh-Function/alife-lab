"""Substrate registry. Each substrate is an independent adapter."""

from __future__ import annotations

from alife_lab.substrates.base import PluginManifest, SubstrateAdapter
from alife_lab.substrates.ca.conway import CONWAY_MANIFEST, ConwayAdapter

_REGISTRY: dict[str, tuple[PluginManifest, type[SubstrateAdapter]]] = {
    "conway": (CONWAY_MANIFEST, ConwayAdapter),
}


def list_substrates() -> list[PluginManifest]:
    return [manifest for manifest, _ in _REGISTRY.values()]


def get_manifest(slug: str) -> PluginManifest:
    return _REGISTRY[slug][0]


def get_adapter(slug: str) -> SubstrateAdapter:
    return _REGISTRY[slug][1]()


def has_substrate(slug: str) -> bool:
    return slug in _REGISTRY
