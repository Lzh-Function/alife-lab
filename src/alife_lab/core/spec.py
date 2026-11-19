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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    | NCAExperiment,
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
    max_fps: Annotated[float, Field(gt=0.0, le=60.0)] = 5.0
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
