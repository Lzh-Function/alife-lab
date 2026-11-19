"""Substrate registry. Each substrate is an independent adapter."""

from __future__ import annotations

from alife_lab.substrates.agent_world.adapter import AGENT_WORLD_MANIFEST, AgentWorldAdapter
from alife_lab.substrates.base import PluginManifest, SubstrateAdapter
from alife_lab.substrates.bff.adapter import BFF_MANIFEST, BFFAdapter
from alife_lab.substrates.boids.boids import BOIDS_MANIFEST, BoidsAdapter
from alife_lab.substrates.ca.conway import CONWAY_MANIFEST, ConwayAdapter
from alife_lab.substrates.flow_lenia.flow_lenia import FLOW_LENIA_MANIFEST, FlowLeniaAdapter
from alife_lab.substrates.gray_scott.gray_scott import GRAY_SCOTT_MANIFEST, GrayScottAdapter
from alife_lab.substrates.lenia.lenia import LENIA_MANIFEST, LeniaAdapter
from alife_lab.substrates.nca.nca_adapter import NCA_MANIFEST, NCAAdapter
from alife_lab.substrates.nca.train_adapter import NCA_TRAIN_MANIFEST, NCATrainAdapter
from alife_lab.substrates.particle_life.particle_life import (
    PARTICLE_LIFE_MANIFEST,
    ParticleLifeAdapter,
)
from alife_lab.substrates.pd_nca.adapter import PD_NCA_MANIFEST, PDNCAAdapter
from alife_lab.substrates.physarum.physarum import PHYSARUM_MANIFEST, PhysarumAdapter
from alife_lab.substrates.protocell.adapter import PROTOCELL_MANIFEST, ProtocellAdapter
from alife_lab.substrates.reaction_network.adapter import (
    REACTION_NETWORK_MANIFEST,
    ReactionNetworkAdapter,
)

_REGISTRY: dict[str, tuple[PluginManifest, type[SubstrateAdapter]]] = {
    "conway": (CONWAY_MANIFEST, ConwayAdapter),
    "boids": (BOIDS_MANIFEST, BoidsAdapter),
    "particle_life": (PARTICLE_LIFE_MANIFEST, ParticleLifeAdapter),
    "bff": (BFF_MANIFEST, BFFAdapter),
    "lenia": (LENIA_MANIFEST, LeniaAdapter),
    "flow_lenia": (FLOW_LENIA_MANIFEST, FlowLeniaAdapter),
    "gray_scott": (GRAY_SCOTT_MANIFEST, GrayScottAdapter),
    "physarum": (PHYSARUM_MANIFEST, PhysarumAdapter),
    "reaction_network": (REACTION_NETWORK_MANIFEST, ReactionNetworkAdapter),
    "nca_train": (NCA_TRAIN_MANIFEST, NCATrainAdapter),
    "nca": (NCA_MANIFEST, NCAAdapter),
    "pd_nca": (PD_NCA_MANIFEST, PDNCAAdapter),
    "protocell": (PROTOCELL_MANIFEST, ProtocellAdapter),
    "agent_world": (AGENT_WORLD_MANIFEST, AgentWorldAdapter),
}


def list_substrates() -> list[PluginManifest]:
    return [manifest for manifest, _ in _REGISTRY.values()]


def get_manifest(slug: str) -> PluginManifest:
    return _REGISTRY[slug][0]


def get_adapter(slug: str) -> SubstrateAdapter:
    return _REGISTRY[slug][1]()


def has_substrate(slug: str) -> bool:
    return slug in _REGISTRY
