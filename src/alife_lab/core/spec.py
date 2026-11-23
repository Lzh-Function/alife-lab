"""Immutable scientific specification (ExperimentSpec) and ExecutionProfile.

ExperimentSpec holds everything that changes the scientific meaning of a run
(rules, world size, seed, numeric policy, stopping rules, observations).
ExecutionProfile holds compute placement, preview and resource budgets only.
The two are hashed independently; changing the profile never changes the
scientific hash, and the profile must never alter the dynamics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from alife_lab.core.hashing import sha256_json

MAX_WORLD_SIDE = 1024
MAX_STEPS = 10_000_000

_RULE_RE = re.compile(r"^B([0-8]*)/S([0-8]*)$")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# ExperimentSpec
# --------------------------------------------------------------------------- #


class WorldSpec(_Frozen):
    width: Annotated[int, Field(ge=8, le=MAX_WORLD_SIDE)]
    height: Annotated[int, Field(ge=8, le=MAX_WORLD_SIDE)]
    boundary: Literal["periodic", "dead"] = "periodic"


PatternPreset = Literal["glider", "blinker", "block", "r_pentomino", "random", "empty"]


class InitializationSpec(_Frozen):
    preset: PatternPreset = "glider"
    # Density is only meaningful for preset="random".
    density: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    # Top-left offset of a pattern preset; None centers the pattern.
    offset_x: Annotated[int, Field(ge=0, lt=MAX_WORLD_SIDE)] | None = None
    offset_y: Annotated[int, Field(ge=0, lt=MAX_WORLD_SIDE)] | None = None


class DynamicsSpec(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["exact_integer"] = "exact_integer"
    rule: str = "B3/S23"

    @field_validator("rule")
    @classmethod
    def _check_rule(cls, value: str) -> str:
        match = _RULE_RE.match(value)
        if not match:
            raise ValueError("rule must use B<digits>/S<digits> notation, e.g. B3/S23")
        for group in match.groups():
            if len(set(group)) != len(group):
                raise ValueError("rule digits must not repeat")
        return value


class RunnerSpec(_Frozen):
    max_steps: Annotated[int, Field(ge=1, le=MAX_STEPS)] = 1000
    checkpoint_interval: Annotated[int, Field(ge=1, le=MAX_STEPS)] = 100


KnownMetric = Literal["alive_cell_count", "glider_displacement", "centroid_displacement"]


class ObservationSpec(_Frozen):
    metrics: tuple[KnownMetric, ...] = ("alive_cell_count",)


_NAME = Annotated[str, Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.\- ]+$")]
_SEED = Annotated[int, Field(ge=0, lt=2**63)]


class ConwayExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["conway"]
    seed: _SEED
    world: WorldSpec
    initialization: InitializationSpec = InitializationSpec()
    dynamics: DynamicsSpec = DynamicsSpec()
    runner: RunnerSpec = RunnerSpec()
    observations: ObservationSpec = ObservationSpec()

    @model_validator(mode="after")
    def _check_reference_rule(self) -> ConwayExperiment:
        if self.mode == "reference" and self.dynamics.rule != "B3/S23":
            raise ValueError("reference mode for substrate 'conway' requires rule B3/S23")
        return self


# Backwards-compatible name used by Phase 1 code.
ExperimentBody = ConwayExperiment


# --------------------------------------------------------------------------- #
# Boids (continuous 2D particles)
# --------------------------------------------------------------------------- #

MAX_BOIDS = 10_000


class ContinuousWorldSpec(_Frozen):
    width: Annotated[float, Field(ge=10.0, le=100_000.0)]
    height: Annotated[float, Field(ge=10.0, le=100_000.0)]
    boundary: Literal["periodic", "reflect"] = "periodic"


class BoidsInitialization(_Frozen):
    count: Annotated[int, Field(ge=0, le=MAX_BOIDS)] = 300
    # "random": uniform positions, uniformly random headings at the mean speed.
    preset: Literal["random"] = "random"


class BoidsDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float64_cpu_deterministic"] = "float64_cpu_deterministic"
    dt: Annotated[float, Field(gt=0.0, le=10.0)] = 1.0
    perception_radius: Annotated[float, Field(gt=0.0, le=1_000.0)] = 10.0
    separation_radius: Annotated[float, Field(gt=0.0, le=1_000.0)] = 3.0
    separation_weight: Annotated[float, Field(ge=0.0, le=100.0)] = 0.05
    alignment_weight: Annotated[float, Field(ge=0.0, le=100.0)] = 0.05
    cohesion_weight: Annotated[float, Field(ge=0.0, le=100.0)] = 0.005
    min_speed: Annotated[float, Field(ge=0.0, le=1_000.0)] = 0.5
    max_speed: Annotated[float, Field(gt=0.0, le=1_000.0)] = 2.0
    # Std-dev of Gaussian velocity noise per tick (0 = deterministic rules only).
    noise: Annotated[float, Field(ge=0.0, le=100.0)] = 0.0

    @model_validator(mode="after")
    def _check(self) -> BoidsDynamics:
        if self.min_speed > self.max_speed:
            raise ValueError("min_speed must not exceed max_speed")
        if self.separation_radius > self.perception_radius:
            raise ValueError("separation_radius must not exceed perception_radius")
        return self


BoidsMetric = Literal["population", "polarization", "mean_speed", "mean_neighbor_count"]


class BoidsObservation(_Frozen):
    metrics: tuple[BoidsMetric, ...] = ("population", "polarization", "mean_speed")


class BoidsExperiment(_Frozen):
    name: _NAME
    # No canonical numerical reference implementation exists: exploratory only.
    mode: Literal["exploratory"] = "exploratory"
    substrate: Literal["boids"]
    seed: _SEED
    world: ContinuousWorldSpec
    initialization: BoidsInitialization = BoidsInitialization()
    dynamics: BoidsDynamics = BoidsDynamics()
    runner: RunnerSpec = RunnerSpec()
    observations: BoidsObservation = BoidsObservation()

    @model_validator(mode="after")
    def _check_world(self) -> BoidsExperiment:
        if self.dynamics.perception_radius * 3 > min(self.world.width, self.world.height):
            raise ValueError("world must be at least 3 perception radii wide and high")
        return self


# --------------------------------------------------------------------------- #
# Particle Life (typed particles with an asymmetric interaction matrix)
# --------------------------------------------------------------------------- #

MAX_TYPES = 8


class ParticleLifeInitialization(_Frozen):
    count: Annotated[int, Field(ge=0, le=MAX_BOIDS)] = 600
    types: Annotated[int, Field(ge=1, le=MAX_TYPES)] = 4
    # "random": uniform positions, zero velocity, uniformly random types.
    preset: Literal["random"] = "random"


class ParticleLifeDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float64_cpu_deterministic"] = "float64_cpu_deterministic"
    dt: Annotated[float, Field(gt=0.0, le=1.0)] = 0.05
    r_max: Annotated[float, Field(gt=0.0, le=1_000.0)] = 15.0
    beta: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.3
    force_factor: Annotated[float, Field(ge=0.0, le=1_000.0)] = 1.0
    # Velocity is multiplied by 0.5 ** (dt / friction_half_life) each tick.
    friction_half_life: Annotated[float, Field(gt=0.0, le=100.0)] = 0.1
    # matrix[a][b] = attraction of type a towards type b in [-1, 1]; None = drawn from the
    # run's RNG (uniform [-1, 1]) at initialization and stored in the state/checkpoints.
    matrix: list[list[Annotated[float, Field(ge=-1.0, le=1.0)]]] | None = None


ParticleLifeMetric = Literal["population", "mean_speed", "mean_neighbor_count", "segregation_index"]


class ParticleLifeObservation(_Frozen):
    metrics: tuple[ParticleLifeMetric, ...] = (
        "population",
        "mean_speed",
        "mean_neighbor_count",
        "segregation_index",
    )


class ParticleLifeExperiment(_Frozen):
    name: _NAME
    # Community model without a canonical reference implementation: exploratory only.
    mode: Literal["exploratory"] = "exploratory"
    substrate: Literal["particle_life"]
    seed: _SEED
    world: ContinuousWorldSpec
    initialization: ParticleLifeInitialization = ParticleLifeInitialization()
    dynamics: ParticleLifeDynamics = ParticleLifeDynamics()
    runner: RunnerSpec = RunnerSpec()
    observations: ParticleLifeObservation = ParticleLifeObservation()

    @model_validator(mode="after")
    def _check(self) -> ParticleLifeExperiment:
        k = self.initialization.types
        m = self.dynamics.matrix
        if m is not None and (len(m) != k or any(len(row) != k for row in m)):
            raise ValueError(f"matrix must be {k}x{k} (initialization.types)")
        if self.dynamics.r_max * 3 > min(self.world.width, self.world.height):
            raise ValueError("world must be at least 3 r_max wide and high")
        return self


# --------------------------------------------------------------------------- #
# BFF primordial soup (CuBFF semantics)
# --------------------------------------------------------------------------- #

BFF_DEFAULT_MUTATION = 1 << 18  # out of 2^30 (= 1/4096 ≈ 0.024 %), CuBFF default


class BFFSoup(_Frozen):
    num_programs: Annotated[int, Field(ge=2, le=1 << 20, multiple_of=2)] = 4096
    init: Literal["random", "zero"] = "random"


class BFFDynamics(_Frozen):
    clock_unit: Literal["epoch"] = "epoch"
    numeric_policy: Literal["exact_integer"] = "exact_integer"
    variant: Literal["bff_noheads", "bff"] = "bff_noheads"
    # Per-byte replacement probability numerator over 2^30 (CuBFF ``mutation_prob``).
    mutation_prob: Annotated[int, Field(ge=0, le=1 << 30)] = BFF_DEFAULT_MUTATION
    # 0 = non-execution control (pairs are shuffled and mutated but never executed).
    step_limit: Annotated[int, Field(ge=0, le=1 << 16)] = 8192


BFFMetric = Literal[
    "h0_bits_per_byte",
    "brotli_bits_per_byte",
    "higher_order_entropy",
    "ops_per_pair",
    "unique_programs",
    "top_program_fraction",
    "mutated_bytes",
]


class BFFObservation(_Frozen):
    metrics: tuple[BFFMetric, ...] = (
        "h0_bits_per_byte",
        "brotli_bits_per_byte",
        "higher_order_entropy",
        "ops_per_pair",
        "unique_programs",
        "top_program_fraction",
    )


class BFFExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["bff"]
    seed: Annotated[int, Field(ge=0, lt=2**64)]
    soup: BFFSoup = BFFSoup()
    dynamics: BFFDynamics = BFFDynamics()
    runner: RunnerSpec = RunnerSpec()
    observations: BFFObservation = BFFObservation()

    @model_validator(mode="after")
    def _check_reference(self) -> BFFExperiment:
        if self.mode == "reference" and (
            self.dynamics.step_limit != 8192 or self.soup.init != "random"
        ):
            raise ValueError(
                "reference mode for 'bff' requires CuBFF defaults: step_limit 8192 and "
                "random initialization (num_programs, seed and mutation_prob are CuBFF flags)"
            )
        return self


# --------------------------------------------------------------------------- #
# Lenia (Chan 2019; LeniaND.py semantics)
# --------------------------------------------------------------------------- #


class LeniaWorld(_Frozen):
    width: Annotated[int, Field(ge=16, le=MAX_WORLD_SIDE)] = 128
    height: Annotated[int, Field(ge=16, le=MAX_WORLD_SIDE)] = 128
    boundary: Literal["periodic"] = "periodic"  # FFT convolution is periodic


class LeniaKernel(_Frozen):
    R: Annotated[int, Field(ge=2, le=128)] = 13
    # Ring weights ("b"), e.g. [1] or [1, 0.5]; at most 4 rings.
    b: Annotated[
        list[Annotated[float, Field(ge=0.0, le=1.0)]], Field(min_length=1, max_length=4)
    ] = [1.0]
    core: Literal["polynomial", "exponential", "step", "staircase"] = "polynomial"  # kn 1..4


class LeniaGrowth(_Frozen):
    m: Annotated[float, Field(gt=0.0, le=1.0)] = 0.15
    s: Annotated[float, Field(gt=0.0, le=1.0)] = 0.015
    function: Literal["polynomial", "gaussian", "step"] = "polynomial"  # gn 1..3


class LeniaDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float64_fft_cpu"] = "float64_fft_cpu"
    T: Annotated[int, Field(ge=1, le=10_000)] = 10  # dt = 1 / T
    kernel: LeniaKernel = LeniaKernel()
    growth: LeniaGrowth = LeniaGrowth()


class LeniaInitialization(_Frozen):
    # "pattern": a named creature from Chan's animals.json (embedded, MIT);
    # "random": uniform noise in a centred square of side `size`.
    preset: Literal["pattern", "random", "empty"] = "pattern"
    pattern: Literal["O2u"] = "O2u"
    size: Annotated[int, Field(ge=1, le=MAX_WORLD_SIDE)] = 64
    density: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


LeniaMetric = Literal["mass", "alive_fraction", "max_value", "centroid_displacement"]


class LeniaObservation(_Frozen):
    metrics: tuple[LeniaMetric, ...] = (
        "mass",
        "alive_fraction",
        "max_value",
        "centroid_displacement",
    )


class LeniaExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["lenia"]
    seed: _SEED
    world: LeniaWorld = LeniaWorld()
    initialization: LeniaInitialization = LeniaInitialization()
    dynamics: LeniaDynamics = LeniaDynamics()
    runner: RunnerSpec = RunnerSpec()
    observations: LeniaObservation = LeniaObservation()

    @model_validator(mode="after")
    def _check(self) -> LeniaExperiment:
        diameter = 2 * self.dynamics.kernel.R + 1
        if diameter > min(self.world.width, self.world.height):
            raise ValueError("kernel diameter 2R+1 must fit in the world")
        return self


# --------------------------------------------------------------------------- #
# Flow-Lenia (Plantec et al. 2023; semantics of erwanplantec/FlowLenia @ dce428c)
# --------------------------------------------------------------------------- #

MAX_FLOW_LENIA_CHANNELS = 4
MAX_FLOW_LENIA_KERNELS = 32


class FlowLeniaWorld(_Frozen):
    width: Annotated[int, Field(ge=16, le=MAX_WORLD_SIDE)] = 128
    height: Annotated[int, Field(ge=16, le=MAX_WORLD_SIDE)] = 128
    # "wall": target positions are clipped inside the world; "torus": periodic transport.
    # Kernel convolution is periodic (FFT) and the Sobel gradient zero-padded in both cases.
    boundary: Literal["wall", "torus"] = "wall"


_UNIT = Annotated[float, Field(ge=0.0, le=1.0)]
_TRIPLE = Annotated[list[_UNIT], Field(min_length=3, max_length=3)]


class FlowLeniaKernel(_Frozen):
    source: Annotated[int, Field(ge=0, lt=MAX_FLOW_LENIA_CHANNELS)] = 0  # c0
    target: Annotated[int, Field(ge=0, lt=MAX_FLOW_LENIA_CHANNELS)] = 0  # channel of c1
    r: Annotated[float, Field(gt=0.0, le=1.0)]  # relative radius
    m: Annotated[float, Field(ge=0.0, le=1.0)]  # growth centre
    s: Annotated[float, Field(gt=0.0, le=1.0)]  # growth width
    h: Annotated[float, Field(ge=0.0, le=1.0)]  # growth weight
    a: _TRIPLE  # ring centres
    b: _TRIPLE  # ring weights
    w: Annotated[list[Annotated[float, Field(gt=0.0, le=1.0)]], Field(min_length=3, max_length=3)]


class FlowLeniaRule(_Frozen):
    # null => sampled from the experiment seed with the official parameter ranges.
    R: Annotated[float, Field(ge=0.0, le=100.0)] | None = None
    kernels: (
        Annotated[list[FlowLeniaKernel], Field(min_length=1, max_length=MAX_FLOW_LENIA_KERNELS)]
        | None
    ) = None
    # Used only when kernels are sampled: connectivity[s][t] = number of kernels from
    # channel s to channel t (official conn_from_matrix). Default: [[10]].
    connectivity: list[list[Annotated[int, Field(ge=0, le=MAX_FLOW_LENIA_KERNELS)]]] | None = None


class FlowLeniaDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float64_numpy_cpu"] = "float64_numpy_cpu"
    channels: Annotated[int, Field(ge=1, le=MAX_FLOW_LENIA_CHANNELS)] = 1
    dt: Annotated[float, Field(gt=0.0, le=10.0)] = 0.2
    dd: Annotated[int, Field(ge=1, le=10)] = 5  # max transport distance (cells)
    sigma: Annotated[float, Field(gt=0.0, le=5.0)] = 0.65  # half-size of the spread square
    # Mass-dependent flow mixing alpha = clip((A / theta_A)^n, 0, 1); null => channels.
    theta_A: Annotated[float, Field(gt=0.0, le=100.0)] | None = None
    n: Annotated[float, Field(gt=0.0, le=10.0)] = 2.0
    rule: FlowLeniaRule = FlowLeniaRule()


class FlowLeniaInitialization(_Frozen):
    # "random": uniform values in a centred square of side `size` in every channel.
    preset: Literal["random", "empty"] = "random"
    size: Annotated[int, Field(ge=1, le=MAX_WORLD_SIDE)] = 40
    density: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


FlowLeniaMetric = Literal[
    "mass", "mass_drift", "alive_fraction", "max_value", "centroid_displacement"
]


class FlowLeniaObservation(_Frozen):
    metrics: tuple[FlowLeniaMetric, ...] = (
        "mass",
        "mass_drift",
        "alive_fraction",
        "max_value",
        "centroid_displacement",
    )


class FlowLeniaExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["flow_lenia"]
    seed: _SEED
    world: FlowLeniaWorld = FlowLeniaWorld()
    initialization: FlowLeniaInitialization = FlowLeniaInitialization()
    dynamics: FlowLeniaDynamics = FlowLeniaDynamics()
    runner: RunnerSpec = RunnerSpec()
    observations: FlowLeniaObservation = FlowLeniaObservation()

    @model_validator(mode="after")
    def _check(self) -> FlowLeniaExperiment:
        d = self.dynamics
        c = d.channels
        if 2 * d.dd + 1 > min(self.world.width, self.world.height):
            raise ValueError("transport window 2*dd+1 must fit in the world")
        if d.sigma >= d.dd:
            raise ValueError("sigma must be smaller than dd")
        rule = d.rule
        if rule.kernels is not None:
            for i, k in enumerate(rule.kernels):
                if k.source >= c or k.target >= c:
                    raise ValueError(f"kernel {i} uses a channel >= channels ({c})")
            if rule.connectivity is not None:
                raise ValueError("give either explicit kernels or a connectivity matrix")
        if rule.connectivity is not None:
            m = rule.connectivity
            if len(m) != c or any(len(row) != c for row in m):
                raise ValueError("connectivity must be a channels x channels matrix")
            total = sum(map(sum, m))
            if not 1 <= total <= MAX_FLOW_LENIA_KERNELS:
                raise ValueError(f"connectivity must define 1..{MAX_FLOW_LENIA_KERNELS} kernels")
        elif rule.kernels is None and c != 1:
            raise ValueError("sampled rules with channels > 1 need a connectivity matrix")
        return self


# --------------------------------------------------------------------------- #
# Gray-Scott reaction-diffusion (Pearson 1993 parameterisation)
# --------------------------------------------------------------------------- #


class GrayScottWorld(_Frozen):
    width: Annotated[int, Field(ge=16, le=MAX_WORLD_SIDE)] = 256
    height: Annotated[int, Field(ge=16, le=MAX_WORLD_SIDE)] = 256
    boundary: Literal["periodic"] = "periodic"
    # Physical side length of the square domain (Pearson: 2.5 for 256 x 256 cells).
    length: Annotated[float, Field(gt=0.0, le=1_000.0)] = 2.5


class GrayScottDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float64_explicit_euler"] = "float64_explicit_euler"
    Du: Annotated[float, Field(gt=0.0, le=10.0)] = 2e-5
    Dv: Annotated[float, Field(gt=0.0, le=10.0)] = 1e-5
    F: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0367
    k: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0649
    dt: Annotated[float, Field(gt=0.0, le=100.0)] = 1.0
    reaction: bool = True  # False: diffusion-only control (no u v^2 term, no feed/kill)


class GrayScottInitialization(_Frozen):
    # "pearson": u = 1, v = 0 with a centred square (side `size`) of u = 0.5, v = 0.25,
    # both perturbed by +-`noise` relative uniform noise (Pearson used 1 %).
    preset: Literal["pearson", "uniform"] = "pearson"
    size: Annotated[int, Field(ge=1, le=MAX_WORLD_SIDE)] = 20
    noise: Annotated[float, Field(ge=0.0, le=0.5)] = 0.01


GrayScottMetric = Literal["mean_u", "mean_v", "v_area_fraction", "v_std", "total_u_plus_v"]


class GrayScottObservation(_Frozen):
    metrics: tuple[GrayScottMetric, ...] = (
        "mean_u",
        "mean_v",
        "v_area_fraction",
        "v_std",
        "total_u_plus_v",
    )


class GrayScottExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["gray_scott"]
    seed: _SEED
    world: GrayScottWorld = GrayScottWorld()
    initialization: GrayScottInitialization = GrayScottInitialization()
    dynamics: GrayScottDynamics = GrayScottDynamics()
    runner: RunnerSpec = RunnerSpec()
    observations: GrayScottObservation = GrayScottObservation()

    @model_validator(mode="after")
    def _check(self) -> GrayScottExperiment:
        w = self.world
        if w.width != w.height:
            raise ValueError("gray_scott uses a square domain (width == height)")
        dx = w.length / w.width
        limit = dx * dx / (4 * max(self.dynamics.Du, self.dynamics.Dv))
        if self.dynamics.dt > limit:
            raise ValueError(
                f"explicit Euler is unstable: dt={self.dynamics.dt} > dx^2/(4 max D)={limit:.4g}"
            )
        return self


# --------------------------------------------------------------------------- #
# Physarum-inspired stigmergy (Jones 2010 agent model)
# --------------------------------------------------------------------------- #

MAX_PHYSARUM_AGENTS = 200_000
MAX_FOOD_SOURCES = 32


class PhysarumWorld(_Frozen):
    width: Annotated[int, Field(ge=16, le=MAX_WORLD_SIDE)] = 200
    height: Annotated[int, Field(ge=16, le=MAX_WORLD_SIDE)] = 200
    boundary: Literal["periodic"] = "periodic"


class FoodSource(_Frozen):
    x: Annotated[int, Field(ge=0, lt=MAX_WORLD_SIDE)]
    y: Annotated[int, Field(ge=0, lt=MAX_WORLD_SIDE)]
    radius: Annotated[int, Field(ge=0, le=32)] = 2
    amount: Annotated[float, Field(ge=0.0, le=1_000.0)] = 10.0  # added to the trail per tick


class PhysarumDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float64_sequential_agents"] = "float64_sequential_agents"
    sensor_angle_deg: Annotated[float, Field(ge=0.0, le=180.0)] = 45.0  # SA
    rotation_angle_deg: Annotated[float, Field(ge=0.0, le=180.0)] = 45.0  # RA
    sensor_offset: Annotated[float, Field(gt=0.0, le=64.0)] = 9.0  # SO (cells)
    step_size: Annotated[float, Field(gt=0.0, le=8.0)] = 1.0  # SS (cells per tick)
    deposit: Annotated[float, Field(ge=0.0, le=1_000.0)] = 5.0  # depT
    decay: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1  # fraction removed per tick
    p_random_turn: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0  # pCD
    # "exclusion": one agent per cell, blocked agents pick a random heading (Jones);
    # "none": agents never block each other (common fast variant, exploratory only).
    collisions: Literal["exclusion", "none"] = "exclusion"
    sensing: bool = True  # False: control, agents ignore the trail (straight walks)
    food: Annotated[list[FoodSource], Field(max_length=MAX_FOOD_SOURCES)] = []


class PhysarumInitialization(_Frozen):
    population_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.15  # %p of cells


PhysarumMetric = Literal[
    "trail_mean", "trail_coverage", "trail_contrast", "heading_order", "blocked_fraction"
]


class PhysarumObservation(_Frozen):
    metrics: tuple[PhysarumMetric, ...] = (
        "trail_mean",
        "trail_coverage",
        "trail_contrast",
        "heading_order",
        "blocked_fraction",
    )


class PhysarumExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["physarum"]
    seed: _SEED
    world: PhysarumWorld = PhysarumWorld()
    initialization: PhysarumInitialization = PhysarumInitialization()
    dynamics: PhysarumDynamics = PhysarumDynamics()
    runner: RunnerSpec = RunnerSpec()
    observations: PhysarumObservation = PhysarumObservation()

    @model_validator(mode="after")
    def _check(self) -> PhysarumExperiment:
        cells = self.world.width * self.world.height
        if round(cells * self.initialization.population_fraction) > MAX_PHYSARUM_AGENTS:
            raise ValueError(f"at most {MAX_PHYSARUM_AGENTS} agents")
        for f in self.dynamics.food:
            if f.x >= self.world.width or f.y >= self.world.height:
                raise ValueError("food source outside the world")
        if self.mode == "reference" and self.dynamics.collisions != "exclusion":
            raise ValueError("reference mode uses Jones' exclusion rule (collisions: exclusion)")
        return self


# --------------------------------------------------------------------------- #
# Abstract reaction network (stochastic simulation, Gillespie 1977 direct method)
# --------------------------------------------------------------------------- #

MAX_SPECIES = 64
MAX_REACTIONS = 256
MAX_COUNT = 10**9
_SPECIES_NAME = Annotated[
    str, Field(min_length=1, max_length=24, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
]
_STOICH = Annotated[int, Field(ge=1, le=8)]


class Species(_Frozen):
    name: _SPECIES_NAME
    initial: Annotated[int, Field(ge=0, le=MAX_COUNT)] = 0
    # Buffered species (Gillespie's X-bar): count held fixed, consumption/production ignored.
    constant: bool = False


class Reaction(_Frozen):
    id: Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_\-]+$")]
    reactants: dict[_SPECIES_NAME, _STOICH] = {}
    products: dict[_SPECIES_NAME, _STOICH] = {}
    rate: Annotated[float, Field(ge=0.0, le=1e9)]  # stochastic rate constant c (per time)
    # Declared role (informational; never changes the dynamics): e.g. "catalysed", "inflow".
    label: Annotated[str, Field(max_length=40)] = ""


class ReactionNetworkDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["gillespie_direct_float64"] = "gillespie_direct_float64"
    time_per_tick: Annotated[float, Field(gt=0.0, le=1e6)] = 0.01
    max_events_per_tick: Annotated[int, Field(ge=1, le=10_000_000)] = 200_000
    species: Annotated[list[Species], Field(min_length=1, max_length=MAX_SPECIES)]
    reactions: Annotated[list[Reaction], Field(min_length=1, max_length=MAX_REACTIONS)]
    disabled_reactions: list[str] = []  # knock-out control: these reactions never fire


ReactionNetworkMetric = Literal["total_molecules", "species_present", "events", "diversity", "time"]


class ReactionNetworkObservation(_Frozen):
    metrics: tuple[ReactionNetworkMetric, ...] = (
        "total_molecules",
        "species_present",
        "events",
        "diversity",
        "time",
    )
    record_species_counts: bool = True  # adds `count.<species>` to every metrics row


class ReactionNetworkExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["reaction_network"]
    seed: _SEED
    dynamics: ReactionNetworkDynamics
    runner: RunnerSpec = RunnerSpec()
    observations: ReactionNetworkObservation = ReactionNetworkObservation()

    @model_validator(mode="after")
    def _check(self) -> ReactionNetworkExperiment:
        names = [sp.name for sp in self.dynamics.species]
        if len(set(names)) != len(names):
            raise ValueError("species names must be unique")
        ids = [r.id for r in self.dynamics.reactions]
        if len(set(ids)) != len(ids):
            raise ValueError("reaction ids must be unique")
        known = set(names)
        for r in self.dynamics.reactions:
            unknown = (set(r.reactants) | set(r.products)) - known
            if unknown:
                raise ValueError(f"reaction {r.id!r} uses unknown species {sorted(unknown)}")
        missing = set(self.dynamics.disabled_reactions) - set(ids)
        if missing:
            raise ValueError(f"disabled_reactions not found: {sorted(missing)}")
        return self


# --------------------------------------------------------------------------- #
# Growing Neural Cellular Automata (Mordvintsev et al. 2020)
# --------------------------------------------------------------------------- #

MAX_NCA_TARGET = 64
MAX_NCA_PAD = 32
_B64 = Annotated[str, Field(max_length=200_000, pattern=r"^[A-Za-z0-9+/=]*$")]


class NCAArchitecture(_Frozen):
    channels: Annotated[int, Field(ge=4, le=32)] = 16  # RGBA + hidden
    hidden: Annotated[int, Field(ge=8, le=256)] = 128
    fire_rate: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5  # stochastic cell update
    alive_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1


class NCATarget(_Frozen):
    # "preset": procedurally drawn built-in shape; "custom": RGBA bytes (base64, row-major,
    # premultiplied-free straight alpha) of size x size, e.g. drawn in the browser.
    kind: Literal["preset", "custom"] = "preset"
    preset: Literal["disc", "ring", "heart", "letter_a"] = "heart"
    size: Annotated[int, Field(ge=8, le=MAX_NCA_TARGET)] = 40
    rgba_b64: _B64 | None = None

    @model_validator(mode="after")
    def _check(self) -> NCATarget:
        if self.kind == "custom":
            import base64

            if self.rgba_b64 is None:
                raise ValueError("custom targets need rgba_b64")
            if len(base64.b64decode(self.rgba_b64)) != self.size * self.size * 4:
                raise ValueError("rgba_b64 must decode to size * size * 4 bytes")
        return self


class NCATrainDynamics(_Frozen):
    clock_unit: Literal["optimizer_step"] = "optimizer_step"
    # float32_jax_cpu: CPU only (bitwise reproducible across runs on this platform);
    # float32_jax: any JAX device (CPU or CUDA); results depend on the recorded backend.
    numeric_policy: Literal["float32_jax_cpu", "float32_jax"] = "float32_jax_cpu"
    # Mordvintsev et al. experiments: 1 = growing, 2 = persistent (pool), 3 = regenerating.
    experiment: Literal["growing", "persistent", "regenerating"] = "persistent"
    architecture: NCAArchitecture = NCAArchitecture()
    pad: Annotated[int, Field(ge=0, le=MAX_NCA_PAD)] = 8
    batch_size: Annotated[int, Field(ge=1, le=64)] = 8
    pool_size: Annotated[int, Field(ge=1, le=1024)] = 256
    damaged_per_batch: Annotated[int, Field(ge=0, le=64)] = 3
    steps_min: Annotated[int, Field(ge=1, le=512)] = 64
    steps_max: Annotated[int, Field(ge=1, le=512)] = 96
    learning_rate: Annotated[float, Field(gt=0.0, le=1.0)] = 2e-3
    lr_decay_step: Annotated[int, Field(ge=1, le=10**6)] = 2000
    lr_decay_factor: Annotated[float, Field(gt=0.0, le=1.0)] = 0.1


NCATrainMetric = Literal["loss", "log10_loss", "learning_rate", "grad_norm"]


class NCATrainObservation(_Frozen):
    metrics: tuple[NCATrainMetric, ...] = ("loss", "log10_loss", "learning_rate", "grad_norm")


class NCATrainExperiment(_Frozen):
    """Optimisation run: explicit external objective (MSE to a target image)."""

    name: _NAME
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["nca_train"]
    seed: _SEED
    target: NCATarget = NCATarget()
    dynamics: NCATrainDynamics = NCATrainDynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=2000, checkpoint_interval=250)
    observations: NCATrainObservation = NCATrainObservation()

    @model_validator(mode="after")
    def _check(self) -> NCATrainExperiment:
        d = self.dynamics
        if d.steps_min > d.steps_max:
            raise ValueError("steps_min must be <= steps_max")
        if d.experiment != "growing" and d.pool_size < d.batch_size:
            raise ValueError("pool_size must be >= batch_size")
        if d.damaged_per_batch > d.batch_size:
            raise ValueError("damaged_per_batch must be <= batch_size")
        return self


class NCAWeights(_Frozen):
    # float32 little-endian w1 (3C x hidden), b1 (hidden), w2 (hidden x C), concatenated.
    data_b64: _B64
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source: dict[str, Any] = {}  # informational provenance, e.g. training run id and step


class NCADynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float32_jax_cpu", "float32_jax"] = "float32_jax_cpu"
    architecture: NCAArchitecture = NCAArchitecture()
    # "trained": the given weights; "untrained": control with freshly initialised weights
    # (zero output layer => nothing grows), "noise": control with random output weights.
    weights_kind: Literal["trained", "untrained", "noise"] = "trained"
    weights: NCAWeights | None = None


NCAMetric = Literal["alive_fraction", "alpha_mass", "target_mse"]


class NCAObservation(_Frozen):
    metrics: tuple[NCAMetric, ...] = ("alive_fraction", "alpha_mass", "target_mse")


class NCAExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["nca"]
    seed: _SEED
    target: NCATarget = NCATarget()
    pad: Annotated[int, Field(ge=0, le=MAX_NCA_PAD)] = 8
    dynamics: NCADynamics = NCADynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=100_000, checkpoint_interval=200)
    observations: NCAObservation = NCAObservation()

    @model_validator(mode="after")
    def _check(self) -> NCAExperiment:
        d = self.dynamics
        if d.weights_kind == "trained" and d.weights is None:
            raise ValueError("weights_kind 'trained' needs weights")
        if d.weights is not None:
            import base64
            import hashlib

            raw = base64.b64decode(d.weights.data_b64)
            a = d.architecture
            expected = 4 * (3 * a.channels * a.hidden + a.hidden + a.hidden * a.channels)
            if len(raw) != expected:
                raise ValueError(f"weights must be {expected} bytes for this architecture")
            if hashlib.sha256(raw).hexdigest() != d.weights.sha256:
                raise ValueError("weights sha256 does not match the data")
        return self


# --------------------------------------------------------------------------- #
# Petri Dish NCA (Sakana AI): competing NCAs with online learning
# --------------------------------------------------------------------------- #


class PDNCAWorld(_Frozen):
    width: Annotated[int, Field(ge=8, le=256)] = 48
    height: Annotated[int, Field(ge=8, le=256)] = 48
    boundary: Literal["zero_pad"] = "zero_pad"  # convolutions use zero padding


class PDNCAModel(_Frozen):
    hidden_dim: Annotated[int, Field(ge=4, le=256)] = 32
    n_hidden_layers: Annotated[int, Field(ge=0, le=4)] = 0
    kernel_size: Annotated[int, Field(ge=1, le=9)] = 3
    # Fraction of the hidden channels each NCA may write (offset by its index).
    per_hid_upd: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    alive_visible: bool = True  # NCAs see the aliveness channels


class PDNCADynamics(_Frozen):
    clock_unit: Literal["epoch"] = "epoch"
    numeric_policy: Literal["float32_jax_cpu", "float32_jax"] = "float32_jax_cpu"
    n_ncas: Annotated[int, Field(ge=1, le=16)] = 4
    n_seeds: Annotated[int, Field(ge=1, le=16)] = 1
    cell_state_dim: Annotated[int, Field(ge=2, le=128, multiple_of=2)] = 16  # attack | defense
    cell_hidden_dim: Annotated[int, Field(ge=0, le=32)] = 4
    model: PDNCAModel = PDNCAModel()
    alive_threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.4
    softmax_temp: Annotated[float, Field(gt=0.0, le=100.0)] = 1.0
    # Controls: "uniform" removes the attack/defence contest (equal strengths among the
    # alive entities of a cell); "frozen" never updates the weights.
    competition: Literal["cosine", "uniform"] = "cosine"
    learning: Literal["online", "frozen"] = "online"
    optimizer: Literal["sgd", "rmsprop", "adam"] = "sgd"
    learning_rate: Annotated[float, Field(gt=0.0, le=10.0)] = 0.03
    grad_clip_norm: Annotated[float, Field(gt=0.0, le=1e6)] = 1.0
    steps_before_update: Annotated[int, Field(ge=0, le=256)] = 0
    steps_per_update: Annotated[int, Field(ge=1, le=64)] = 1
    # The reference training loop keeps the previous world if any NCA went extinct in the epoch
    # (off by default: observed to freeze the world once an NCA dies every epoch).
    revert_on_extinction: bool = False


PDNCAMetric = Literal[
    "territory_loss", "grad_norm", "alive_ncas", "territory_entropy", "sun_fraction", "reverted"
]


class PDNCAObservation(_Frozen):
    metrics: tuple[PDNCAMetric, ...] = (
        "territory_loss",
        "grad_norm",
        "alive_ncas",
        "territory_entropy",
        "sun_fraction",
        "reverted",
    )
    record_territories: bool = True  # adds `territory.sun` and `territory.nca_<k>`


class PDNCAExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["pd_nca"]
    seed: _SEED
    world: PDNCAWorld = PDNCAWorld()
    dynamics: PDNCADynamics = PDNCADynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=100_000, checkpoint_interval=200)
    observations: PDNCAObservation = PDNCAObservation()

    @model_validator(mode="after")
    def _check(self) -> PDNCAExperiment:
        d = self.dynamics
        if d.n_ncas * d.n_seeds > self.world.width * self.world.height:
            raise ValueError("n_ncas * n_seeds exceeds the number of cells")
        if d.model.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return self


# --------------------------------------------------------------------------- #
# Protocells: compartments, metabolism, growth and division (abstract 2D model)
# --------------------------------------------------------------------------- #

MAX_PROTOCELLS = 4000


class ProtocellWorld(_Frozen):
    width: Annotated[int, Field(ge=8, le=512)] = 64
    height: Annotated[int, Field(ge=8, le=512)] = 64
    boundary: Literal["periodic"] = "periodic"


class ProtocellEnvironment(_Frozen):
    inflow_per_cell: Annotated[float, Field(ge=0.0, le=100.0)] = 0.2  # mean food added / tick
    food_decay: Annotated[float, Field(ge=0.0, le=1.0)] = 0.002  # per molecule per tick
    waste_decay: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    hop_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5  # field diffusion
    initial_food_per_cell: Annotated[int, Field(ge=0, le=10_000)] = 20


class ProtocellGenome(_Frozen):
    """Heritable traits (initial values; mutated multiplicatively at division)."""

    k_lipid: Annotated[float, Field(ge=0.0, le=100.0)] = 0.05  # F + C -> C + M
    k_replicate: Annotated[float, Field(ge=0.0, le=100.0)] = 0.02  # F + C -> 2 C
    permeability: Annotated[float, Field(ge=0.0, le=10.0)] = 0.05  # food uptake per tick


class ProtocellDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["integer_counts_binomial"] = "integer_counts_binomial"
    environment: ProtocellEnvironment = ProtocellEnvironment()
    genome: ProtocellGenome = ProtocellGenome()
    mutation_sigma: Annotated[float, Field(ge=0.0, le=2.0)] = 0.05  # log-normal per trait
    heritable: tuple[Literal["k_lipid", "k_replicate", "permeability"], ...] = (
        "k_lipid",
        "k_replicate",
        "permeability",
    )
    membrane_decay: Annotated[float, Field(ge=0.0, le=1.0)] = 0.01  # M -> W
    catalyst_decay: Annotated[float, Field(ge=0.0, le=1.0)] = 0.005  # C -> W
    maintenance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.002  # F -> W per membrane unit
    membrane_density: Annotated[float, Field(gt=0.0, le=1000.0)] = 20.0  # M per area unit
    division_membrane: Annotated[int, Field(ge=4, le=100_000)] = 400
    min_membrane: Annotated[int, Field(ge=1, le=100_000)] = 60
    rupture_pressure: Annotated[float, Field(gt=0.0, le=1e6)] = 60.0  # contents per area
    # Controls
    membrane: bool = True  # False: contents leak freely (no compartment)
    metabolism: bool = True  # False: no lipid synthesis / replication
    division: bool = True  # False: cells only grow (and eventually rupture)
    partition: Literal["binomial", "random_fraction"] = "binomial"
    repulsion: Annotated[float, Field(ge=0.0, le=10.0)] = 0.5
    diffusion: Annotated[float, Field(ge=0.0, le=10.0)] = 0.1  # Brownian step (world units)


class ProtocellInitialization(_Frozen):
    count: Annotated[int, Field(ge=0, le=MAX_PROTOCELLS)] = 20
    membrane: Annotated[int, Field(ge=1, le=100_000)] = 200
    catalyst: Annotated[int, Field(ge=0, le=100_000)] = 20
    food: Annotated[int, Field(ge=0, le=100_000)] = 50


ProtocellMetric = Literal[
    "population", "divisions", "ruptures", "deaths", "mean_radius", "total_matter",
    "field_food", "ledger_residual", "mean_k_replicate", "mean_permeability",
    "partition_asymmetry",
]  # fmt: skip


class ProtocellObservation(_Frozen):
    metrics: tuple[ProtocellMetric, ...] = (
        "population",
        "divisions",
        "ruptures",
        "deaths",
        "mean_radius",
        "total_matter",
        "field_food",
        "ledger_residual",
        "mean_k_replicate",
        "mean_permeability",
        "partition_asymmetry",
    )


class ProtocellExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["protocell"]
    seed: _SEED
    world: ProtocellWorld = ProtocellWorld()
    initialization: ProtocellInitialization = ProtocellInitialization()
    dynamics: ProtocellDynamics = ProtocellDynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=100_000, checkpoint_interval=500)
    observations: ProtocellObservation = ProtocellObservation()

    @model_validator(mode="after")
    def _check(self) -> ProtocellExperiment:
        d = self.dynamics
        if d.min_membrane * 2 > d.division_membrane:
            raise ValueError("division_membrane must be at least 2 * min_membrane")
        if self.mode == "reference":
            raise ValueError("protocell is an abstract model without a reference implementation")
        return self


# --------------------------------------------------------------------------- #
# AgentWorld: lightweight sensing agents with fixed, evolved or learning controllers
# --------------------------------------------------------------------------- #

MAX_AGENTS = 4000


class AgentWorldWorld(_Frozen):
    width: Annotated[float, Field(ge=16.0, le=2048.0)] = 100.0
    height: Annotated[float, Field(ge=16.0, le=2048.0)] = 100.0
    boundary: Literal["periodic"] = "periodic"
    resolution: Annotated[int, Field(ge=8, le=512)] = 50  # food/signal grid cells per axis


class AgentWorldFood(_Frozen):
    initial: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5  # fraction of capacity
    capacity: Annotated[float, Field(gt=0.0, le=100.0)] = 1.0
    regrowth: Annotated[float, Field(ge=0.0, le=1.0)] = 0.01  # logistic rate per tick
    bite: Annotated[float, Field(ge=0.0, le=100.0)] = 0.4


class AgentBody(_Frozen):
    speed: Annotated[float, Field(gt=0.0, le=16.0)] = 1.0
    turn_deg: Annotated[float, Field(gt=0.0, le=180.0)] = 30.0
    sensor_distance: Annotated[float, Field(gt=0.0, le=64.0)] = 4.0
    sensor_angle_deg: Annotated[float, Field(ge=0.0, le=180.0)] = 45.0
    initial_energy: Annotated[float, Field(gt=0.0, le=1e4)] = 5.0
    metabolism: Annotated[float, Field(ge=0.0, le=100.0)] = 0.05  # per tick
    move_cost: Annotated[float, Field(ge=0.0, le=100.0)] = 0.02  # per forward move
    emit_cost: Annotated[float, Field(ge=0.0, le=100.0)] = 0.01
    reproduce_threshold: Annotated[float, Field(gt=0.0, le=1e5)] = 10.0  # child gets half


class AgentController(_Frozen):
    # baseline: hand-coded food-gradient follower; random: uniform actions (control);
    # mlp: small policy network (inputs: food L/F/R, signal L/F/R, energy, bias).
    kind: Literal["baseline", "random", "mlp"] = "mlp"
    # fixed: no change; evolution: offspring inherit mutated genetic weights;
    # lifetime: REINFORCE on the output layer with the agent's own energy change as reward
    # (learned weights are NOT inherited — offspring get the genetic weights).
    mode: Literal["fixed", "evolution", "lifetime"] = "evolution"
    hidden: Annotated[int, Field(ge=2, le=64)] = 8
    mutation_sigma: Annotated[float, Field(ge=0.0, le=5.0)] = 0.1
    learning_rate: Annotated[float, Field(ge=0.0, le=10.0)] = 0.05
    temperature: Annotated[float, Field(gt=0.0, le=100.0)] = 1.0


class AgentWorldDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float64_numpy"] = "float64_numpy"
    food: AgentWorldFood = AgentWorldFood()
    body: AgentBody = AgentBody()
    controller: AgentController = AgentController()
    communication: bool = True  # False: signal sensors read zero and emission does nothing
    signal_decay: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    signal_diffusion: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    emit_amount: Annotated[float, Field(ge=0.0, le=100.0)] = 1.0
    shuffle_policies: bool = False  # control: controllers permuted among agents every tick
    max_population: Annotated[int, Field(ge=1, le=MAX_AGENTS)] = 2000


class AgentWorldInitialization(_Frozen):
    count: Annotated[int, Field(ge=0, le=MAX_AGENTS)] = 100


AgentWorldMetric = Literal[
    "population", "mean_energy", "births", "deaths", "food_total", "mean_age", "signal_rate",
    "forward_rate", "mean_reward", "genome_spread",
]  # fmt: skip


class AgentWorldObservation(_Frozen):
    metrics: tuple[AgentWorldMetric, ...] = (
        "population",
        "mean_energy",
        "births",
        "deaths",
        "food_total",
        "mean_age",
        "signal_rate",
        "forward_rate",
        "mean_reward",
        "genome_spread",
    )
    history: Annotated[int, Field(ge=1, le=256)] = 32  # per-agent ticks kept for the inspector


class AgentWorldExperiment(_Frozen):
    """Autonomous run: no external fitness; the only reward (lifetime mode) is internal."""

    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["agent_world"]
    seed: _SEED
    world: AgentWorldWorld = AgentWorldWorld()
    initialization: AgentWorldInitialization = AgentWorldInitialization()
    dynamics: AgentWorldDynamics = AgentWorldDynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=100_000, checkpoint_interval=500)
    observations: AgentWorldObservation = AgentWorldObservation()

    @model_validator(mode="after")
    def _check(self) -> AgentWorldExperiment:
        if self.mode == "reference":
            raise ValueError("agent_world is a project model without a reference implementation")
        if self.initialization.count > self.dynamics.max_population:
            raise ValueError("initial count exceeds max_population")
        return self


# --------------------------------------------------------------------------- #
# Multichannel Lenia (Chan 2020 "expanded universe"; semantics of SakanaAI/asal lenia_impl.py,
# itself from maxencefaldor/Leniabreeder): c channels, k kernels from channel c0 to c1.
# --------------------------------------------------------------------------- #

MAX_MC_CHANNELS = 4
MAX_MC_KERNELS = 32
LENIA_MC_PATTERNS = ("5N7KKM", "5N7KKM_2", "5N7KKM_g", "VT049W")


class LeniaMCWorld(_Frozen):
    size: Annotated[int, Field(ge=32, le=MAX_WORLD_SIDE)] = 128  # square, periodic (FFT)
    boundary: Literal["periodic"] = "periodic"

    @property
    def width(self) -> int:
        return self.size

    @property
    def height(self) -> int:
        return self.size


class LeniaMCKernel(_Frozen):
    b: Annotated[list[Annotated[float, Field(ge=0.0, le=1.0)]], Field(min_length=1, max_length=4)]
    m: Annotated[float, Field(gt=0.0, le=1.0)]  # growth centre
    s: Annotated[float, Field(gt=0.0, le=1.0)]  # growth width
    h: Annotated[float, Field(ge=0.0, le=1.0)]  # growth weight
    r: Annotated[float, Field(gt=0.0, le=1.0)]  # relative radius
    c0: Annotated[int, Field(ge=0, lt=MAX_MC_CHANNELS)]  # source channel
    c1: Annotated[int, Field(ge=0, lt=MAX_MC_CHANNELS)]  # target channel


class LeniaMCDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float64_fft_cpu"] = "float64_fft_cpu"
    channels: Annotated[int, Field(ge=1, le=MAX_MC_CHANNELS)] = 3
    R: Annotated[float, Field(gt=1.0, le=200.0)] = 12.0
    T: Annotated[float, Field(gt=0.0, le=1000.0)] = 2.0  # dt = 1 / T
    kernels: Annotated[list[LeniaMCKernel], Field(min_length=1, max_length=MAX_MC_KERNELS)]


class LeniaMCInitialization(_Frozen):
    # "pattern": a catalogued creature (cells padded to 32 x 32 and centred, as Leniabreeder);
    # "random": uniform noise in a centred square of side `size` in every channel.
    preset: Literal["pattern", "random", "empty"] = "pattern"
    pattern: Literal["5N7KKM", "5N7KKM_2", "5N7KKM_g", "VT049W"] = "5N7KKM"
    size: Annotated[int, Field(ge=1, le=MAX_WORLD_SIDE)] = 32
    density: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


LeniaMCMetric = Literal["mass", "alive_fraction", "max_value", "centroid_displacement"]


class LeniaMCObservation(_Frozen):
    metrics: tuple[LeniaMCMetric, ...] = (
        "mass",
        "alive_fraction",
        "max_value",
        "centroid_displacement",
    )


def _lenia_mc_defaults(data: Any) -> Any:
    """Fill R, T, channels and kernels from the pattern when they are omitted, so that the
    canonical spec (and its hash) always lists every value used."""
    if not isinstance(data, dict):
        return data
    init = data.get("initialization") or {}
    if hasattr(init, "model_dump"):
        init = init.model_dump()
    dyn = data.get("dynamics") or {}
    if hasattr(dyn, "model_dump"):
        return data
    if dyn.get("kernels") is not None:
        return data
    from alife_lab.substrates.lenia_mc.patterns import load_pattern

    pat = load_pattern(init.get("pattern", "5N7KKM"))
    filled = {
        "channels": len(pat["cells"]),
        "R": float(pat["R"]),
        "T": float(pat["T"]),
        **{k: v for k, v in dyn.items() if v is not None},
        "kernels": [dict(kern) for kern in pat["kernels"]],
    }
    return {**data, "dynamics": filled}


class LeniaMCExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["lenia_mc"]
    seed: _SEED
    world: LeniaMCWorld = LeniaMCWorld()
    initialization: LeniaMCInitialization = LeniaMCInitialization()
    dynamics: LeniaMCDynamics
    runner: RunnerSpec = RunnerSpec()
    observations: LeniaMCObservation = LeniaMCObservation()

    @model_validator(mode="before")
    @classmethod
    def _defaults(cls, data: Any) -> Any:
        return _lenia_mc_defaults(data)

    @model_validator(mode="after")
    def _check(self) -> LeniaMCExperiment:
        d = self.dynamics
        for i, k in enumerate(d.kernels):
            if k.c0 >= d.channels or k.c1 >= d.channels:
                raise ValueError(f"kernel {i} uses a channel >= channels ({d.channels})")
        if self.world.size < 2 * d.R + 1:
            raise ValueError("kernel diameter 2R+1 must fit in the world")
        if self.initialization.preset == "pattern":
            from alife_lab.substrates.lenia_mc.patterns import load_pattern

            if len(load_pattern(self.initialization.pattern)["cells"]) != d.channels:
                raise ValueError("the pattern's number of channels differs from dynamics.channels")
        return self


# --------------------------------------------------------------------------- #
# Embodied creature (MuJoCo, CPU float64): a torso with legs driven by a CPG controller
# --------------------------------------------------------------------------- #

MAX_CREATURE_LEGS = 8


class CreatureLeg(_Frozen):
    """A leg attached on the torso's equator at ``angle_deg``: a hip joint swings it around the
    vertical axis, a knee joint lifts the lower segment around the leg's tangent axis."""

    angle_deg: Annotated[float, Field(ge=0.0, lt=360.0)]
    upper: Annotated[float, Field(ge=0.05, le=1.0)] = 0.3  # thigh length (m)
    lower: Annotated[float, Field(ge=0.05, le=1.0)] = 0.4  # shin length (m)
    radius: Annotated[float, Field(ge=0.02, le=0.15)] = 0.05


class CreatureBody(_Frozen):
    torso_radius: Annotated[float, Field(ge=0.05, le=0.6)] = 0.25
    legs: Annotated[list[CreatureLeg], Field(min_length=1, max_length=MAX_CREATURE_LEGS)]


class CreatureController(_Frozen):
    """Central pattern generator: ctrl_j = amplitude_j sin(2 pi (frequency t + phase_j)).
    Joints are ordered hip_0, knee_0, hip_1, knee_1, ... ``off`` applies no torque."""

    kind: Literal["cpg", "off"] = "cpg"
    frequency_hz: Annotated[float, Field(gt=0.0, le=5.0)] = 1.0
    amplitudes: list[Annotated[float, Field(ge=0.0, le=1.0)]]
    phases: list[Annotated[float, Field(ge=0.0, le=1.0)]]  # in turns


class CreatureReflex(_Frozen):
    """Sensory feedback added to the CPG (cf. Taga 1991; Kimura et al. Tekken; Ijspeert 2008):
    * tilt: each knee command gets ``tilt_gain * lean`` where lean is how far the torso leans
      towards that leg (the leg on the lower side pushes, levelling the torso);
    * contact: a knee whose foot is off the ground gets ``contact_gain`` more extension (the leg
      reaches down to find the ground).
    The sum is clipped to [-1, 1]."""

    tilt_gain: Annotated[float, Field(ge=0.0, le=3.0)] = 1.0
    contact_gain: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2


class CreatureDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float64_mujoco_cpu"] = "float64_mujoco_cpu"
    timestep: Annotated[float, Field(ge=0.001, le=0.02)] = 0.01  # physics step (s)
    substeps: Annotated[int, Field(ge=1, le=50)] = 5  # physics steps per tick
    gear: Annotated[float, Field(gt=0.0, le=500.0)] = 60.0  # motor torque scale (N m)
    friction: Annotated[float, Field(gt=0.0, le=5.0)] = 1.0
    density: Annotated[float, Field(gt=0.0, le=5000.0)] = 300.0  # kg / m^3 of all parts
    # Passive joint springs towards the rest posture (N m / rad), like muscle tone.
    joint_stiffness: Annotated[float, Field(ge=0.0, le=1000.0)] = 20.0
    body: CreatureBody
    controller: CreatureController
    # Optional, additive (omitted from the canonical dump when absent: digests unchanged).
    reflex: CreatureReflex | None = None

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if out.get("reflex") is None:
            out.pop("reflex", None)
        return out


class CreatureTerrain(_Frozen):
    """Rough ground as a MuJoCo height field: smooth random bumps (bilinear interpolation of
    uniform noise on a grid of spacing ``wavelength``), heights in [0, amplitude], flattened to
    0 within ``flat_radius`` of the start. Generated from the experiment seed."""

    kind: Literal["rough"] = "rough"
    amplitude: Annotated[float, Field(ge=0.0, le=0.5)] = 0.12  # m
    wavelength: Annotated[float, Field(ge=0.2, le=10.0)] = 1.0  # m
    half_size: Annotated[float, Field(ge=5.0, le=60.0)] = 30.0  # m, square from -s to s
    resolution: Annotated[float, Field(ge=0.05, le=0.5)] = 0.1  # m per height sample
    flat_radius: Annotated[float, Field(ge=0.0, le=5.0)] = 1.0  # m


class CreatureInitialization(_Frozen):
    height: Annotated[float, Field(ge=0.1, le=3.0)] = 0.75  # torso centre above the floor (m)


CreatureMetric = Literal["displacement", "forward", "speed", "height", "upright", "energy", "time"]


class CreatureObservation(_Frozen):
    metrics: tuple[CreatureMetric, ...] = (
        "displacement", "forward", "speed", "height", "upright", "energy", "time",
    )  # fmt: skip


class CreatureExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["creature"]
    seed: _SEED
    initialization: CreatureInitialization = CreatureInitialization()
    dynamics: CreatureDynamics
    terrain: CreatureTerrain | None = None  # None: flat floor
    runner: RunnerSpec = RunnerSpec(max_steps=2000, checkpoint_interval=200)
    observations: CreatureObservation = CreatureObservation()

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if out.get("terrain") is None:
            out.pop("terrain", None)
        return out

    @model_validator(mode="after")
    def _check(self) -> CreatureExperiment:
        joints = 2 * len(self.dynamics.body.legs)
        c = self.dynamics.controller
        if len(c.amplitudes) != joints or len(c.phases) != joints:
            raise ValueError(f"the controller needs {joints} amplitudes and phases (2 per leg)")
        if self.mode == "reference":
            raise ValueError("creature is a project model without a reference implementation")
        return self


# --------------------------------------------------------------------------- #
# Search: an external optimisation run over the parameters of a target world
# --------------------------------------------------------------------------- #

MAX_SEARCH_PARAMS = 64
# Parameters a search may vary, per target substrate: path -> integer-valued?
SEARCH_PATHS: dict[str, dict[str, bool]] = {
    "lenia": {
        "dynamics.growth.m": False,
        "dynamics.growth.s": False,
        "dynamics.T": True,
        "dynamics.kernel.R": True,
        "initialization.density": False,
        "initialization.size": True,
    },
}


_MC_KERNEL_PATH = re.compile(r"^dynamics\.kernels\.(\d{1,2})\.(m|s|h|r)$")
_MC_PATHS = {"dynamics.T": False, "dynamics.R": False, "initialization.density": False,
             "initialization.size": True}  # fmt: skip


_CREATURE_PATHS = (
    re.compile(r"^dynamics\.controller\.(amplitudes|phases)\.(\d{1,2})$"),
    re.compile(r"^dynamics\.body\.legs\.(\d)\.(angle_deg|upper|lower|radius)$"),
)
_CREATURE_SCALARS = {"dynamics.controller.frequency_hz", "dynamics.body.torso_radius",
                     "dynamics.gear", "dynamics.joint_stiffness"}  # fmt: skip
_CREATURE_REFLEX = {"dynamics.reflex.tilt_gain", "dynamics.reflex.contact_gain"}


_CONNECTOME_STIM_PATH = re.compile(r"^stimuli\.(\d+)\.rate_hz$")


def search_path_kind(target: Any, path: str) -> bool | None:
    """None if ``path`` cannot be searched for this target, else whether it is integer-valued.
    Multichannel Lenia exposes every kernel's m, s, h, r (``dynamics.kernels.<i>.<name>``).
    Connectome exposes stimulus firing rates only (``stimuli.<i>.rate_hz``; wiring is fixed)."""
    if target.substrate == "connectome":
        m = _CONNECTOME_STIM_PATH.match(path)
        if m:
            return False if int(m.group(1)) < len(target.stimuli) else None
        return None
    if target.substrate == "creature":
        if path in _CREATURE_SCALARS:
            return False
        if path in _CREATURE_REFLEX:  # only when the base target has a reflex section
            return False if target.dynamics.reflex is not None else None
        legs = len(target.dynamics.body.legs)
        m = _CREATURE_PATHS[0].match(path)
        if m:
            return False if int(m.group(2)) < 2 * legs else None
        m = _CREATURE_PATHS[1].match(path)
        if m:
            return False if int(m.group(1)) < legs else None
        return None
    if target.substrate == "lenia_mc":
        m = _MC_KERNEL_PATH.match(path)
        if m:
            return False if int(m.group(1)) < len(target.dynamics.kernels) else None
        return _MC_PATHS.get(path)
    return SEARCH_PATHS.get(target.substrate, {}).get(path)


class SearchParam(_Frozen):
    path: Annotated[str, Field(pattern=r"^[a-z_]{1,32}(\.[a-zA-Z0-9_]{1,32}){1,4}$")]
    low: float
    high: float
    # linear / log, or logit: uniform in logit(value) between logit(low) and logit(high)
    # (ASAL perturbs Lenia parameters as sigmoid(logit(base) + delta)).
    scale: Literal["linear", "log", "logit"] = "linear"

    @model_validator(mode="after")
    def _check(self) -> SearchParam:
        if not self.low < self.high:
            raise ValueError(f"{self.path}: low must be < high")
        if self.scale == "log" and self.low <= 0:
            raise ValueError(f"{self.path}: log scale needs low > 0")
        if self.scale == "logit" and not (self.low > 0 and self.high < 1):
            raise ValueError(f"{self.path}: logit scale needs 0 < low < high < 1")
        return self


CLIP_OBJECTIVES = ("lenia_clip_target", "lenia_clip_open_endedness")
OBJECTIVE_TARGETS = {
    "lenia_moving_localized": ("lenia",),
    "lenia_clip_target": ("lenia", "lenia_mc"),
    "lenia_clip_open_endedness": ("lenia", "lenia_mc"),
    "creature_distance": ("creature",),
    "connectome_mn9_rate": ("connectome",),
}


class SearchDynamics(_Frozen):
    clock_unit: Literal["generation"] = "generation"
    # float64_cpu: NumPy objectives; float32_clip_torch: objectives that embed frames with CLIP
    # in PyTorch (backends torch / torch_cuda); float32_connectome: bit-identical CPU/CUDA LIF
    # connectome evaluations (backends numpy / cupy).
    numeric_policy: Literal["float64_cpu", "float32_clip_torch", "float32_connectome"] = (
        "float64_cpu"
    )
    # random: uniform sampling of the box (baseline); cma_es: CMA-ES (Hansen's purecma
    # equations, positive weights) in the normalised box with a quadratic bound penalty;
    # map_elites: MAP-Elites over the ``qd`` descriptor grid (requires ``qd``).
    # novelty: novelty search (Lehman & Stanley 2011) with a kNN archive; imgep: goal
    # exploration (Forestier et al. 2017; Reinke et al. 2020 for Lenia). Both need a behaviour
    # space: the qd descriptors, or (novelty only) the CLIP embeddings of an asal search.
    algorithm: Literal["random", "cma_es", "map_elites", "novelty", "imgep"] = "cma_es"
    population: Annotated[int, Field(ge=2, le=256)] = 12
    sigma0: Annotated[float, Field(gt=0.0, le=1.0)] = 0.3  # in normalised [0, 1] units
    # Pre-declared objective (versioned in the adapter); never changed during a run.
    objective: Literal[
        "lenia_moving_localized",
        "lenia_clip_target",
        "lenia_clip_open_endedness",
        "creature_distance",
        "connectome_mn9_rate",
    ] = "lenia_moving_localized"
    max_alive_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.3
    eval_steps: Annotated[int, Field(ge=10, le=20_000)] = 300
    # Common random numbers: every candidate is scored on the same search seeds; the best
    # candidate is re-scored on disjoint holdout seeds that never drive the search.
    eval_seeds: Annotated[int, Field(ge=1, le=16)] = 2
    holdout_seeds: Annotated[int, Field(ge=0, le=16)] = 4


SearchMetric = Literal[
    "best_score",
    "generation_best",
    "generation_mean",
    "viable_fraction",
    "evaluations",
    "holdout_best",
    "sigma",
    "coverage",
    "qd_score",
    "archive_size",
    "blank_score",
    "mean_novelty",
    "illumination",
]


class SearchExplore(_Frozen):
    """Settings of the novelty and IMGEP algorithms (behaviour space: qd descriptors scaled to
    [0, 1] by their bounds, or CLIP embeddings)."""

    k: Annotated[int, Field(ge=1, le=64)] = 10  # nearest neighbours for novelty
    archive_add: Annotated[int, Field(ge=0, le=64)] = 2  # most novel added per generation
    mutation_sigma: Annotated[float, Field(gt=0.0, le=1.0)] = 0.1  # normalised units


class QDDescriptor(_Frozen):
    """One behaviour descriptor axis with pre-declared, fixed bounds (spec §7.2)."""

    name: Literal["speed", "mass", "alive_fraction"]
    low: float
    high: float
    bins: Annotated[int, Field(ge=2, le=64)] = 10

    @model_validator(mode="after")
    def _check(self) -> QDDescriptor:
        if not self.low < self.high:
            raise ValueError(f"descriptor {self.name}: low must be < high")
        return self


class SearchQD(_Frozen):
    """Quality-diversity archive: a fixed grid over two descriptors. Every algorithm reports
    coverage and QD-score on it; ``map_elites`` also selects parents from it. Only candidates
    viable on every search seed enter; quality = the objective score."""

    descriptors: Annotated[list[QDDescriptor], Field(min_length=2, max_length=2)]
    mutation_sigma: Annotated[float, Field(gt=0.0, le=1.0)] = 0.1  # normalised units
    archive_version: Annotated[int, Field(ge=1)] = 1  # bump if the grid definition changes

    @model_validator(mode="after")
    def _check(self) -> SearchQD:
        if self.descriptors[0].name == self.descriptors[1].name:
            raise ValueError("the two QD descriptors must differ")
        return self


class SearchObservation(_Frozen):
    metrics: tuple[SearchMetric, ...] = (
        "best_score",
        "generation_best",
        "generation_mean",
        "viable_fraction",
        "evaluations",
        "holdout_best",
        "sigma",
    )


class SearchASAL(_Frozen):
    """Foundation-model scoring (ASAL). Frames are sampled at the ends of ``frames`` equal
    chunks of the evaluation (ASAL ``time_sampling=(frames, True)``)."""

    model: Literal["openai_vit_b32"] = "openai_vit_b32"
    prompts: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=200, pattern=r"^[^\x00-\x1f]+$")]],
        Field(max_length=8),
    ] = []
    frames: Annotated[int, Field(ge=1, le=64)] = 1
    # None: the whole world is shown to CLIP; an int: a square of this side centred on the
    # circular centroid of the pattern (ASAL's Lenia "phenotype", 64 for a 128 world).
    crop: Annotated[int, Field(ge=8, le=MAX_WORLD_SIDE)] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if out.get("crop") is None:
            out.pop("crop", None)  # specs written before ``crop`` existed keep their digest
        return out


class SearchExperiment(_Frozen):
    """Optimisation run: proposes target-world specs, runs them headless and scores them with a
    pre-declared objective. The target substrate itself is unchanged (no reward inside it)."""

    name: _NAME
    mode: Literal["exploratory"] = "exploratory"
    substrate: Literal["search"]
    seed: _SEED
    # The base world; its seed is replaced by the evaluation seeds.
    target: Annotated[
        LeniaExperiment | LeniaMCExperiment | CreatureExperiment | ConnectomeExperiment,
        Field(discriminator="substrate"),
    ]
    space: Annotated[list[SearchParam], Field(min_length=1, max_length=MAX_SEARCH_PARAMS)]
    dynamics: SearchDynamics = SearchDynamics()
    # Optional, additive section: omitted from the canonical dump when absent so specs written
    # before it existed keep their digest.
    qd: SearchQD | None = None
    asal: SearchASAL | None = None
    explore: SearchExplore | None = None
    runner: RunnerSpec = RunnerSpec(max_steps=25, checkpoint_interval=5)
    observations: SearchObservation = SearchObservation()

    @model_validator(mode="after")
    def _check(self) -> SearchExperiment:
        paths = [p.path for p in self.space]
        if len(set(paths)) != len(paths):
            raise ValueError("search parameters must be distinct")
        for p in self.space:
            if search_path_kind(self.target, p.path) is None:
                raise ValueError(f"{p.path!r} is not searchable for {self.target.substrate!r}")
        d = self.dynamics
        clip = d.objective in CLIP_OBJECTIVES
        if clip != (self.asal is not None):
            raise ValueError("CLIP objectives need an asal section (and only they may have one)")
        expected_policy = (
            "float32_clip_torch"
            if clip
            else "float32_connectome"
            if self.target.substrate == "connectome"
            else "float64_cpu"
        )
        if d.numeric_policy != expected_policy:
            raise ValueError(f"numeric_policy must be {expected_policy!r} for this search target")
        if self.asal is not None:
            a = self.asal
            if d.objective == "lenia_clip_target":
                if not a.prompts or a.frames % len(a.prompts):
                    raise ValueError("lenia_clip_target needs prompts and frames = k x prompts")
            elif a.prompts or a.frames < 2:
                raise ValueError("lenia_clip_open_endedness takes no prompts and >= 2 frames")
            if d.eval_steps < a.frames:
                raise ValueError("eval_steps must be >= the number of sampled frames")
        if d.algorithm == "map_elites" and self.qd is None:
            raise ValueError("map_elites needs a qd section (descriptor grid)")
        if d.algorithm in ("novelty", "imgep") and self.explore is None:
            raise ValueError(f"{d.algorithm} needs an explore section")
        if d.algorithm == "imgep" and self.qd is None:
            raise ValueError("imgep samples goals in the qd descriptor box: it needs qd")
        if d.algorithm == "novelty" and self.qd is None and self.asal is None:
            raise ValueError("novelty needs a behaviour space: qd descriptors or CLIP (asal)")
        if "mean_novelty" in self.observations.metrics and d.algorithm != "novelty":
            raise ValueError("mean_novelty is reported by the novelty algorithm only")
        if "illumination" in self.observations.metrics and self.asal is None:
            raise ValueError("illumination is defined for CLIP searches (asal) only")
        if "blank_score" in self.observations.metrics and d.objective not in CLIP_OBJECTIVES:
            raise ValueError("blank_score is defined for CLIP objectives only")
        qd_metrics = {"coverage", "qd_score", "archive_size"}
        if self.qd is None and qd_metrics & set(self.observations.metrics):
            raise ValueError("QD metrics need a qd section")
        if self.target.substrate not in OBJECTIVE_TARGETS[d.objective]:
            raise ValueError(f"objective {d.objective!r} is not defined for this target")
        crop = self.asal.crop if self.asal is not None else None
        world = getattr(self.target, "world", None)
        if crop is not None and world is not None and crop > min(world.width, world.height):
            raise ValueError("asal.crop must fit in the target world")
        evaluations = d.population * d.eval_seeds * self.runner.max_steps
        if evaluations * d.eval_steps > 2_000_000_000:
            raise ValueError("search budget too large (population*seeds*generations*steps)")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_sections(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        for key in ("qd", "asal", "explore"):
            if out.get(key) is None:
                out.pop(key, None)
        return out


# --------------------------------------------------------------------------- #
# Connectome (spiking LIF whole-brain connectome, F1)
# --------------------------------------------------------------------------- #


ConnectomeDatasetId = Literal["flywire_783", "synthetic_tiny", "synthetic_mb"]
ConnectomeWiring = Literal["reference", "shuffled_control"]


class ConnectomeStimulusSpec(_Frozen):
    port: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")]
    rate_hz: Annotated[float, Field(ge=0.0, le=1000.0)] = 150.0


class ConnectomePlasticitySpec(_Frozen):
    """Simplified phenomenological KC -> MBON dopamine-modulated plasticity (F4).

    Modifies ONLY excitatory KC -> MBON synapses based on presynaptic KC eligibility traces
    and compartmental DAN activity derived from the anatomical connectome.
    """

    enabled: bool = False
    tau_elig_ms: Annotated[float, Field(gt=1.0, le=10_000.0)] = 200.0
    tau_dan_ms: Annotated[float, Field(gt=1.0, le=10_000.0)] = 150.0
    dan_threshold: Annotated[float, Field(ge=0.0, le=500.0)] = 18.0
    lr_ltd: Annotated[float, Field(ge=0.0, le=5.0)] = 0.00022
    lr_ext: Annotated[float, Field(ge=0.0, le=5.0)] = 0.03
    w_min: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    w_max: Annotated[float, Field(ge=1.0, le=10.0)] = 2.0
    min_dan_synapses: Annotated[int, Field(ge=1, le=1000)] = 5


class ConnectomeDynamicsSpec(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float32_deterministic"] = "float32_deterministic"
    v_0: Annotated[float, Field(ge=-120.0, le=0.0)] = -52.0
    v_rst: Annotated[float, Field(ge=-120.0, le=0.0)] = -52.0
    v_th: Annotated[float, Field(ge=-100.0, le=20.0)] = -45.0
    t_mbr_ms: Annotated[float, Field(gt=0.1, le=500.0)] = 20.0
    tau_ms: Annotated[float, Field(gt=0.1, le=500.0)] = 5.0
    t_rfc_ms: Annotated[float, Field(ge=0.0, le=50.0)] = 2.2
    t_dly_ms: Annotated[float, Field(ge=0.1, le=20.0)] = 1.8
    w_syn_mv: Annotated[float, Field(gt=0.0, le=50.0)] = 0.275
    f_poi: Annotated[float, Field(ge=0.0, le=2000.0)] = 250.0
    dt_ms: Annotated[float, Field(ge=0.01, le=1.0)] = 0.1
    substeps_per_tick: Annotated[int, Field(ge=1, le=100)] = 10
    stim_no_refrac: bool = True

    @model_validator(mode="after")
    def _check_thresholds(self) -> ConnectomeDynamicsSpec:
        if self.v_th <= self.v_rst:
            raise ValueError("v_th must be strictly greater than v_rst")
        if self.v_th <= self.v_0:
            raise ValueError("v_th must be strictly greater than v_0")
        return self


ConnectomeMetric = Literal[
    "spikes_tick",
    "active_neurons_tick",
    "total_spikes",
    "mean_firing_rate_hz",
    "mn9_spikes_tick",
    "mn9_rate_hz",
    "mbon11_rate_hz",
    "mbon01_rate_hz",
    "plastic_changed_edges",
    "plastic_mean_scale",
    "plastic_min_scale",
    "mean_v_mv",
    "time_ms",
]


class ConnectomeObservationSpec(_Frozen):
    metrics: tuple[ConnectomeMetric, ...] = (
        "spikes_tick",
        "active_neurons_tick",
        "mn9_spikes_tick",
        "mn9_rate_hz",
        "mean_firing_rate_hz",
        "time_ms",
    )


_CELL_TYPE_TOKEN = Annotated[
    str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-/ ()+]+$")
]


class ConnectomeExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "reference"
    substrate: Literal["connectome"]
    seed: _SEED
    dataset_id: ConnectomeDatasetId = "flywire_783"
    wiring: ConnectomeWiring = "reference"
    stimuli: tuple[ConnectomeStimulusSpec, ...] = ()
    silenced_ports: tuple[str, ...] = ()
    silenced_cell_types: tuple[_CELL_TYPE_TOKEN, ...] = ()
    plasticity: ConnectomePlasticitySpec | None = None
    dynamics: ConnectomeDynamicsSpec = ConnectomeDynamicsSpec()
    runner: RunnerSpec = RunnerSpec(max_steps=1000, checkpoint_interval=100)
    observations: ConnectomeObservationSpec = ConnectomeObservationSpec()

    @model_validator(mode="after")
    def _check_reference_mode(self) -> ConnectomeExperiment:
        if self.mode == "reference":
            if self.wiring != "reference":
                raise ValueError("reference mode for connectome requires wiring='reference'")
            if self.dynamics != ConnectomeDynamicsSpec():
                raise ValueError(
                    "reference mode for connectome requires Shiu et al. (2024) default parameters"
                )
            if self.plasticity is not None and self.plasticity.enabled:
                raise ValueError(
                    "reference mode for connectome requires plasticity disabled "
                    "(use mode='exploratory' for F4 mushroom body plasticity)"
                )
        return self

    @model_serializer(mode="wrap")
    def _omit_empty_optional_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if not out.get("silenced_cell_types"):
            out.pop("silenced_cell_types", None)
        if out.get("plasticity") is None:
            out.pop("plasticity", None)
        return out


# --------------------------------------------------------------------------- #
# Connectome System Identification (Phase F5: hidden-truth external fitting)
# --------------------------------------------------------------------------- #


ConnectomeSysIDDatasetId = Literal["flywire_783", "synthetic_olfactory"]
ConnectomeSubcircuitId = Literal["olfactory_mb_lh", "full"]


class ConnectomeSysIDMechanisms(_Frozen):
    tau_enabled: bool = True
    sign_flip_fraction: Annotated[float, Field(ge=0.0, le=0.5)] = 0.15
    neuromod_enabled: bool = True
    gap_junctions_enabled: bool = True
    gap_candidates: Annotated[int, Field(ge=0, le=2048)] = 64
    gap_active_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25


class ConnectomeSysIDObservation(_Frozen):
    obs_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25
    obs_classes: tuple[str, ...] = ()
    bin_ms: Annotated[float, Field(ge=2.0, le=100.0)] = 10.0
    noise_sd: Annotated[float, Field(ge=0.0, le=0.5)] = 0.015


class ConnectomeSysIDDynamics(_Frozen):
    clock_unit: Literal["epoch"] = "epoch"
    numeric_policy: Literal["float32_torch"] = "float32_torch"
    dt_ms: Annotated[float, Field(ge=0.5, le=10.0)] = 2.0
    sim_ms: Annotated[float, Field(ge=20.0, le=500.0)] = 100.0
    n_train_stimuli: Annotated[int, Field(ge=2, le=32)] = 8
    n_holdout_stimuli: Annotated[int, Field(ge=1, le=16)] = 4
    lr: Annotated[float, Field(gt=0.0, le=1.0)] = 0.06
    l1_gap: Annotated[float, Field(ge=0.0, le=1.0)] = 1e-3
    l2_reg: Annotated[float, Field(ge=0.0, le=1.0)] = 5e-4
    restart_index: Annotated[int, Field(ge=0, le=32)] = 0


ConnectomeSysIDMetric = Literal[
    "loss",
    "train_obs_r2",
    "heldout_all_r2",
    "heldout_obs_r2",
    "heldout_unobs_r2",
    "tau_pearson_r",
    "tau_mae_ms",
    "sign_flip_auroc",
    "sign_flip_accuracy",
    "neuromod_pearson_r",
    "gap_auroc",
    "gap_pearson_r",
]


class ConnectomeSysIDObservationSpec(_Frozen):
    metrics: tuple[ConnectomeSysIDMetric, ...] = (
        "loss",
        "train_obs_r2",
        "heldout_all_r2",
        "heldout_unobs_r2",
        "tau_pearson_r",
        "sign_flip_auroc",
        "neuromod_pearson_r",
        "gap_auroc",
    )


class ConnectomeSysIDExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["connectome_sysid"]
    seed: _SEED
    dataset_id: ConnectomeSysIDDatasetId = "synthetic_olfactory"
    subcircuit_id: ConnectomeSubcircuitId = "olfactory_mb_lh"
    wiring: ConnectomeWiring = "reference"
    mechanisms: ConnectomeSysIDMechanisms = ConnectomeSysIDMechanisms()
    observation: ConnectomeSysIDObservation = ConnectomeSysIDObservation()
    dynamics: ConnectomeSysIDDynamics = ConnectomeSysIDDynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=120, checkpoint_interval=30)
    observations: ConnectomeSysIDObservationSpec = ConnectomeSysIDObservationSpec()

    @model_validator(mode="after")
    def _check_mode(self) -> ConnectomeSysIDExperiment:
        if self.mode == "reference":
            raise ValueError(
                "connectome_sysid is an external hidden-truth fitting benchmark without a "
                "biological reference mode; use mode='exploratory'"
            )
        return self


# --------------------------------------------------------------------------- #
# Connectome Topology Task Learning (Phase F6: biological vs shuffled wiring)
# --------------------------------------------------------------------------- #


ConnectomeTaskName = Literal["odor_classification", "short_memory", "nonlinear_xor"]
ConnectomeTrainScope = Literal["readout_only", "io_only", "partial_weights"]


class ConnectomeTaskDynamics(_Frozen):
    clock_unit: Literal["epoch"] = "epoch"
    numeric_policy: Literal["float32_torch"] = "float32_torch"
    task: ConnectomeTaskName = "odor_classification"
    scope: ConnectomeTrainScope = "readout_only"
    dt_ms: Annotated[float, Field(ge=0.5, le=10.0)] = 2.0
    n_train: Annotated[int, Field(ge=16, le=512)] = 64
    n_test: Annotated[int, Field(ge=16, le=512)] = 64
    lr: Annotated[float, Field(gt=0.0, le=1.0)] = 0.08
    weight_decay: Annotated[float, Field(ge=0.0, le=1.0)] = 1e-3
    max_readout: Annotated[int, Field(ge=4, le=1024)] = 256


ConnectomeTaskMetric = Literal[
    "train_loss",
    "test_loss",
    "train_acc",
    "test_acc",
    "chance_acc",
    "test_margin",
    "readout_norm",
]


class ConnectomeTaskObservationSpec(_Frozen):
    metrics: tuple[ConnectomeTaskMetric, ...] = (
        "train_loss",
        "test_loss",
        "train_acc",
        "test_acc",
        "chance_acc",
        "test_margin",
    )


class ConnectomeTaskExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["connectome_task"]
    seed: _SEED
    dataset_id: ConnectomeSysIDDatasetId = "synthetic_olfactory"
    subcircuit_id: ConnectomeSubcircuitId = "olfactory_mb_lh"
    wiring: ConnectomeWiring = "reference"
    dynamics: ConnectomeTaskDynamics = ConnectomeTaskDynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=80, checkpoint_interval=20)
    observations: ConnectomeTaskObservationSpec = ConnectomeTaskObservationSpec()

    @model_validator(mode="after")
    def _check_mode(self) -> ConnectomeTaskExperiment:
        if self.mode == "reference":
            raise ValueError(
                "connectome_task is an external topology-learning machine without a biological "
                "reference mode; use mode='exploratory'"
            )
        return self


# --------------------------------------------------------------------------- #
# Embodied Drosophila Brain <-> Body (F8 Level 1)
# --------------------------------------------------------------------------- #

EmbodiedFlyDatasetId = Literal[
    "flywire_783",
    "banc_888",
    "male_cns_v1",
    "synthetic_fly_embodied",
    "synthetic_cns_embodied",
    "synthetic_male_aggression",
    "synthetic_ecosystem_male",
    "synthetic_ecosystem_female",
]
EmbodiedControlMode = Literal["level1_cpg", "level2_vnc", "level2_vnc_tuned"]
EmbodiedFoodKind = Literal["sugar", "bitter"]


class EmbodiedFoodSpec(_Frozen):
    x: Annotated[float, Field(ge=-14.5, le=14.5)] = 1.25
    y: Annotated[float, Field(ge=-14.5, le=14.5)] = 0.25
    radius: Annotated[float, Field(ge=0.15, le=1.0)] = 0.36
    kind: EmbodiedFoodKind = "sugar"


class EmbodiedArenaSpec(_Frozen):
    half_size: Annotated[float, Field(ge=2.0, le=15.0)] = 6.0
    foods: Annotated[tuple[EmbodiedFoodSpec, ...], Field(max_length=12)] = (
        EmbodiedFoodSpec(x=1.25, y=0.25, radius=0.36, kind="sugar"),
    )


class FlyEmbodiedDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float32_lif_float64_mujoco"] = "float32_lif_float64_mujoco"
    control_mode: EmbodiedControlMode = "level1_cpg"
    explore_drive_hz: Annotated[float, Field(ge=0.0, le=300.0)] = 110.0
    vnc_synapse_scale: Annotated[float, Field(gt=0.05, le=10.0)] = 1.0
    vnc_proprio_gain_hz: Annotated[float, Field(ge=0.0, le=300.0)] = 95.0
    vnc_mn_norm_hz: Annotated[float, Field(gt=1.0, le=300.0)] = 30.0
    vnc_coxa_gain: Annotated[float, Field(ge=0.0, le=2.0)] = 0.45
    vnc_femur_gain: Annotated[float, Field(ge=0.0, le=2.0)] = 0.55
    vnc_tibia_gain: Annotated[float, Field(ge=0.0, le=2.0)] = 0.50
    # Optional learned per-hemilineage ALIF profile name inside configs/vnc_profiles/<name>.json.
    vnc_profile: Annotated[str | None, Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")] = None
    # F12d: optional 24-float evolved fly_world descendant genome for full-brain + MuJoCo replay
    evolved_genome: Annotated[tuple[float, ...], Field(min_length=24, max_length=24)] | None = None
    evolved_fly_id: int | None = None
    evolved_generation: int | None = None
    evolved_fly_distance: float | None = None

    @model_serializer(mode="wrap")
    def _omit_empty_evolved(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        for key in (
            "evolved_genome",
            "evolved_fly_id",
            "evolved_generation",
            "evolved_fly_distance",
        ):
            if out.get(key) is None:
                out.pop(key, None)
        return out


FlyEmbodiedMetric = Literal[
    "cmd_forward",
    "cmd_turn",
    "cmd_proboscis",
    "cmd_grooming",
    "distance_cm",
    "feeding_ms",
    "grooming_ms",
    "realtime_factor",
    "vnc_mn_mean_hz",
    "vnc_tripod_alternation",
]


class FlyEmbodiedObservationSpec(_Frozen):
    metrics: tuple[FlyEmbodiedMetric, ...] = (
        "cmd_forward",
        "cmd_turn",
        "cmd_proboscis",
        "cmd_grooming",
        "distance_cm",
        "feeding_ms",
        "grooming_ms",
        "realtime_factor",
        "vnc_mn_mean_hz",
        "vnc_tripod_alternation",
    )


class FlyEmbodiedExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["fly_embodied"]
    seed: _SEED
    dataset_id: EmbodiedFlyDatasetId = "synthetic_fly_embodied"
    wiring: ConnectomeWiring = "reference"
    silenced_ports: tuple[str, ...] = ()
    arena: EmbodiedArenaSpec = EmbodiedArenaSpec()
    dynamics: FlyEmbodiedDynamics = FlyEmbodiedDynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=4000, checkpoint_interval=500)
    observations: FlyEmbodiedObservationSpec = FlyEmbodiedObservationSpec()

    @model_validator(mode="after")
    def _check_mode(self) -> FlyEmbodiedExperiment:
        if self.mode == "reference":
            raise ValueError(
                "fly_embodied Level 1 uses a hand-crafted tripod CPG and is exploratory only; "
                "use mode='exploratory'"
            )
        return self


# --------------------------------------------------------------------------- #
# Collective Whole-Brain Fly Swarm (F11 Stage 3: Ramdya 2015 & Social Swarm)
# --------------------------------------------------------------------------- #

FlySwarmSocialMode = Literal[
    "ramdya_odor_avoidance",
    "mixed_courtship_swarm",
    "male_aggression_arena",
    "dual_sex_ecosystem",
]
FlySwarmMetric = Literal[
    "clean_zone_fraction",
    "encounter_rate",
    "mean_speed_cm_s",
    "active_singers",
    "total_lunges",
    "active_threats",
    "active_feeders",
    "female_auditory_hz",
    "realtime_factor",
]


class FlySwarmDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float32_lif_float64_arena"] = "float32_lif_float64_arena"
    social_mode: FlySwarmSocialMode = "ramdya_odor_avoidance"
    enable_touch: bool = True
    odor_enabled: bool = True
    arena_half_size: Annotated[float, Field(ge=2.0, le=12.0)] = 4.5
    explore_drive_hz: Annotated[float, Field(ge=0.0, le=220.0)] = 55.0


class FlySwarmObservationSpec(_Frozen):
    metrics: tuple[FlySwarmMetric, ...] = (
        "clean_zone_fraction",
        "encounter_rate",
        "mean_speed_cm_s",
        "active_singers",
        "realtime_factor",
    )


class FlySwarmExperiment(_Frozen):
    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["fly_swarm"]
    seed: _SEED
    dataset_id: EmbodiedFlyDatasetId = "synthetic_fly_embodied"
    female_dataset_id: EmbodiedFlyDatasetId | None = None
    num_flies: Annotated[int, Field(ge=1, le=64)] = 16
    wiring: ConnectomeWiring = "reference"
    dynamics: FlySwarmDynamics = FlySwarmDynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=15000, checkpoint_interval=5000)
    observations: FlySwarmObservationSpec = FlySwarmObservationSpec()

    @model_validator(mode="after")
    def _check_mode(self) -> FlySwarmExperiment:
        if self.mode == "reference":
            raise ValueError("fly_swarm is exploratory only; use mode='exploratory'")
        return self


# --------------------------------------------------------------------------- #
# Phase F12: fly_world — Artificial Life Starting From Fly Connectome
# --------------------------------------------------------------------------- #

FlyWorldMetric = Literal[
    "population",
    "mean_generation",
    "fly_distance",
    "signal_mi_bits",
    "total_births",
    "total_deaths",
]


class FlyWorldDynamics(_Frozen):
    clock_unit: Literal["tick"] = "tick"
    numeric_policy: Literal["float32_compressed_brain"] = "float32_compressed_brain"
    brain_resolution: Literal["cell_type_compressed"] = "cell_type_compressed"
    arena_half_size: Annotated[float, Field(ge=2.5, le=12.0)] = 5.0
    signal_channel_enabled: bool = True
    predator_enabled: bool = True
    predator_interval: Annotated[int, Field(ge=40, le=2000)] = 220
    signal_cost: Annotated[float, Field(ge=0.0, le=0.05)] = 0.0018
    mutation_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.14
    mutation_sigma: Annotated[float, Field(ge=0.0, le=1.0)] = 0.22
    food_patch_count: Annotated[int, Field(ge=1, le=8)] = 3
    kin_cluster_init: bool = True


class FlyWorldObservationSpec(_Frozen):
    metrics: tuple[FlyWorldMetric, ...] = (
        "population",
        "mean_generation",
        "fly_distance",
        "signal_mi_bits",
        "total_births",
        "total_deaths",
    )


class FlyWorldExperiment(_Frozen):
    """Autonomous open-ended ALife from ancestral fly connectome (NO external fitness score)."""

    name: _NAME
    mode: Literal["reference", "exploratory"] = "exploratory"
    substrate: Literal["fly_world"]
    seed: _SEED
    dataset_id: EmbodiedFlyDatasetId = "synthetic_fly_embodied"
    initial_population: Annotated[int, Field(ge=4, le=64)] = 24
    max_population: Annotated[int, Field(ge=8, le=64)] = 48
    dynamics: FlyWorldDynamics = FlyWorldDynamics()
    runner: RunnerSpec = RunnerSpec(max_steps=30000, checkpoint_interval=5000)
    observations: FlyWorldObservationSpec = FlyWorldObservationSpec()

    @model_validator(mode="after")
    def _check_mode(self) -> FlyWorldExperiment:
        if self.mode == "reference":
            raise ValueError(
                "fly_world is artificial life (exploratory only); use mode='exploratory'"
            )
        if self.initial_population > self.max_population:
            raise ValueError("initial_population must be <= max_population")
        return self


SearchExperiment.model_rebuild()


AnyExperiment = Annotated[
    ConwayExperiment
    | BoidsExperiment
    | ParticleLifeExperiment
    | BFFExperiment
    | LeniaExperiment
    | FlowLeniaExperiment
    | GrayScottExperiment
    | PhysarumExperiment
    | ReactionNetworkExperiment
    | NCATrainExperiment
    | NCAExperiment
    | PDNCAExperiment
    | ProtocellExperiment
    | AgentWorldExperiment
    | LeniaMCExperiment
    | CreatureExperiment
    | SearchExperiment
    | ConnectomeExperiment
    | ConnectomeSysIDExperiment
    | ConnectomeTaskExperiment
    | FlyEmbodiedExperiment
    | FlySwarmExperiment
    | FlyWorldExperiment,
    Field(discriminator="substrate"),
]


class ExperimentSpec(_Frozen):
    schema_version: Literal[1] = 1
    experiment: AnyExperiment

    def digest(self) -> str:
        """SHA-256 of the canonical JSON of the fully-resolved scientific spec."""
        return sha256_json(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# ExecutionProfile
# --------------------------------------------------------------------------- #

ExecutionRole = Literal[
    "cpu_dev", "local_interactive", "remote_compute", "multi_gpu", "cluster_batch"
]


class TargetSelector(_Frozen):
    role: ExecutionRole
    backend: str = "auto"
    gpu_count: Annotated[int, Field(ge=0, le=64)] = 0
    scheduler: str | None = None
    require_explicit_target: bool = False


class MemoryPolicy(_Frozen):
    policy: Literal["probe_available"] = "probe_available"
    max_fraction_of_available: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
    reserve_fraction: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.15


class SchedulingPolicy(_Frozen):
    mode: Literal["interactive", "batch"] = "interactive"
    max_active_jobs: Annotated[int, Field(ge=1, le=64)] | None = None
    max_active_gpu_jobs: Annotated[int, Field(ge=0, le=64)] | None = None
    adaptive_microbatch: bool = False
    checkpoint_and_exit: str | None = None
    job_arrays: str | None = None


class PreviewPolicy(_Frozen):
    max_fps: Annotated[float, Field(gt=0.0, le=120.0)] = 60.0
    adaptive_quality: bool = False


class TransportPolicy(_Frozen):
    mode: str
    artifact_transfer: Literal["checksum_verified"] = "checksum_verified"


class ResourceLimits(_Frozen):
    max_walltime_seconds: Annotated[int, Field(ge=1, le=30 * 24 * 3600)] = 3600
    max_disk_mb: Annotated[int, Field(ge=1, le=10_000_000)] = 1024


class SitePolicy(_Frozen):
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    notes: str | None = None


class ExecutionBody(_Frozen):
    id: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_\-]+$")]
    target_selector: TargetSelector
    memory: MemoryPolicy = MemoryPolicy()
    scheduling: SchedulingPolicy = SchedulingPolicy()
    preview: PreviewPolicy = PreviewPolicy()
    transport: TransportPolicy | None = None
    limits: ResourceLimits = ResourceLimits()
    site_policy: SitePolicy | None = None


class ExecutionProfile(_Frozen):
    schema_version: Literal[1] = 1
    execution: ExecutionBody

    def digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_spec(path: Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate(_load_yaml(path))


def load_execution_profile(path: Path) -> ExecutionProfile:
    return ExecutionProfile.model_validate(_load_yaml(path))
