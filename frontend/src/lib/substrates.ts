// Per-substrate UI descriptors: intervention tools, create-form fields and frame summaries.
// Substrates differ (grid vs particles); the UI is driven by these small tables rather than
// by assuming a single state representation.
import type { Frame } from "./api";
import { popcount } from "./api";

export type SubstrateId = "conway" | "boids" | "particle_life" | "bff" | "lenia" | "flow_lenia" | "gray_scott" | "physarum" | "reaction_network" | "nca_train" | "nca" | "pd_nca" | "protocell" | "agent_world" | "lenia_mc" | "creature" | "search" | "connectome" | "connectome_sysid";

/** Substrates rendered as continuous particles (vs. grids). */
export const PARTICLE_SUBSTRATES = new Set(["boids", "particle_life"]);
/** Substrates rendered as a continuous scalar field (codec field_u8). */
export const FIELD_SUBSTRATES = new Set(["lenia", "flow_lenia", "gray_scott", "physarum", "reaction_network", "nca_train", "nca", "pd_nca", "protocell", "agent_world", "lenia_mc", "search", "connectome_sysid"]);
/** Non-spatial substrates whose display is a species x time image (rows = species). */
export const ROW_DISPLAY_SUBSTRATES = new Set(["reaction_network"]);

/** World size used for display/pointer mapping; non-spatial substrates get a display size. */
export function displayWorld(substrate: string, experiment: Record<string, unknown>): { width: number; height: number } {
  const world = experiment.world as { width?: number; height?: number; size?: number } | undefined;
  if (world?.size) return { width: world.size, height: world.size }; // square worlds (lenia_mc)
  if (world?.width && world.height) return { width: world.width, height: world.height };
  if (substrate === "nca_train" || substrate === "nca") {
    const target = experiment.target as { size?: number } | undefined;
    const pad = (substrate === "nca" ? experiment.pad : (experiment.dynamics as { pad?: number } | undefined)?.pad) as number | undefined;
    const side = (target?.size ?? 40) + 2 * (pad ?? 8);
    return { width: side, height: side };
  }
  if (substrate === "search") {
    // Parameter map (side = target height) + the best candidate's world, side by side.
    // With a QD grid, its archive panel (same side) sits between the two.
    const target = experiment.target as { world?: { width?: number; height?: number; size?: number } } | undefined;
    const tw = target?.world;
    const w = tw?.size ? { width: tw.size, height: tw.size } : { width: tw?.width ?? 64, height: tw?.height ?? 64 };
    const panels = experiment.qd ? 2 : 1;
    return { width: panels * w.height + w.width, height: w.height };
  }
  if (substrate === "reaction_network") {
    const species = ((experiment.dynamics as { species?: unknown[] } | undefined)?.species ?? []).length;
    return { width: 256, height: Math.max(1, species) };
  }
  if (substrate === "connectome_sysid") {
    return { width: 160, height: 120 };
  }
  return { width: 64, height: 64 };
}

export interface ToolState {
  mode: string; // "off" or an intervention mode
  radius: number;
  count: number;
  program?: string; // BFF: program text to inject
  value?: number; // Lenia: value painted into the field
  channel?: number | null; // Flow-Lenia: painted channel (null = all channels)
}

export interface ToolDescriptor {
  modes: string[]; // first entry is "off"
  radius: { min: number; max: number; step: number; initial: number };
  count?: { min: number; max: number; initial: number };
  value?: { initial: number };
  channelSelect?: boolean;
  channelLabelKey?: string; // label of the channel/agent selector
  noteKey: string;
}

export const TOOLS: Record<SubstrateId, ToolDescriptor> = {
  conway: {
    modes: ["off", "inspect", "draw", "erase"],
    radius: { min: 0, max: 8, step: 1, initial: 0 },
    noteKey: "lab.brush.note",
  },
  boids: {
    modes: ["off", "inspect", "spawn", "remove"],
    radius: { min: 1, max: 40, step: 1, initial: 8 },
    count: { min: 1, max: 500, initial: 30 },
    noteKey: "lab.tool.boids_note",
  },
  bff: {
    modes: ["off", "inspect", "inject", "zero"],
    radius: { min: 0, max: 8, step: 1, initial: 0 },
    noteKey: "lab.tool.bff_note",
  },
  lenia: {
    modes: ["off", "inspect", "paint", "erase"],
    radius: { min: 1, max: 30, step: 1, initial: 5 },
    value: { initial: 1 },
    noteKey: "lab.tool.lenia_note",
  },
  flow_lenia: {
    modes: ["off", "inspect", "paint", "erase"],
    radius: { min: 1, max: 30, step: 1, initial: 4 },
    value: { initial: 1 },
    channelSelect: true,
    noteKey: "lab.tool.flow_lenia_note",
  },
  creature: {
    modes: ["off", "inspect"],
    radius: { min: 0, max: 0, step: 1, initial: 0 },
    noteKey: "lab.tool.creature_note",
  },
  lenia_mc: {
    modes: ["off", "inspect", "paint", "erase"],
    radius: { min: 1, max: 30, step: 1, initial: 4 },
    value: { initial: 1 },
    channelSelect: true,
    noteKey: "lab.tool.lenia_mc_note",
  },
  gray_scott: {
    modes: ["off", "inspect", "seed", "erase"],
    radius: { min: 1, max: 40, step: 1, initial: 6 },
    noteKey: "lab.tool.gray_scott_note",
  },
  physarum: {
    modes: ["off", "inspect", "attract", "erase"],
    radius: { min: 1, max: 30, step: 1, initial: 4 },
    noteKey: "lab.tool.physarum_note",
  },
  reaction_network: {
    modes: ["off", "inspect", "add", "remove"],
    radius: { min: 0, max: 0, step: 1, initial: 0 },
    count: { min: 1, max: 100000, initial: 100 },
    noteKey: "lab.tool.reaction_network_note",
  },
  agent_world: {
    modes: ["off", "inspect", "add_food", "remove"],
    radius: { min: 1, max: 40, step: 1, initial: 6 },
    noteKey: "lab.tool.agent_world_note",
  },
  protocell: {
    modes: ["off", "inspect", "add_food", "add_cell", "kill"],
    radius: { min: 1, max: 16, step: 1, initial: 3 },
    count: { min: 1, max: 100000, initial: 500 },
    noteKey: "lab.tool.protocell_note",
  },
  pd_nca: {
    modes: ["off", "inspect", "seed", "erase"],
    radius: { min: 0, max: 16, step: 1, initial: 2 },
    channelSelect: true,
    channelLabelKey: "lab.tool.agent",
    noteKey: "lab.tool.pd_nca_note",
  },
  nca_train: {
    modes: ["off", "inspect"],
    radius: { min: 0, max: 0, step: 1, initial: 0 },
    noteKey: "lab.tool.nca_train_note",
  },
  nca: {
    modes: ["off", "inspect", "damage", "seed"],
    radius: { min: 1, max: 24, step: 1, initial: 6 },
    noteKey: "lab.tool.nca_note",
  },
  search: {
    modes: ["off", "inspect"],
    radius: { min: 0, max: 0, step: 1, initial: 0 },
    noteKey: "lab.tool.search_note",
  },
  particle_life: {
    modes: ["off", "inspect", "spawn", "remove"],
    radius: { min: 1, max: 40, step: 1, initial: 6 },
    count: { min: 1, max: 500, initial: 30 },
    noteKey: "lab.tool.particles_note",
  },
  connectome: {
    modes: ["off"],
    radius: { min: 0, max: 0, step: 1, initial: 0 },
    noteKey: "lab.tool.connectome_note",
  },
  connectome_sysid: {
    modes: ["off", "inspect"],
    radius: { min: 0, max: 0, step: 1, initial: 0 },
    noteKey: "lab.tool.connectome_sysid_note",
  },
};

export function initialTool(substrate: string): ToolState {
  const d = TOOLS[(substrate as SubstrateId) in TOOLS ? (substrate as SubstrateId) : "conway"];
  return { mode: "off", radius: d.radius.initial, count: d.count?.initial ?? 0, value: d.value?.initial };
}

export interface FormField {
  path: string; // dotted path under experiment.*
  labelKey: string;
  kind: "int" | "float" | "select" | "textlist"; // textlist: "a; b" <-> ["a", "b"]
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ value: string; labelKey: string }>;
}

const RUNNER: FormField[] = [
  { path: "runner.max_steps", labelKey: "form.max_steps", kind: "int", min: 1 },
  { path: "runner.checkpoint_interval", labelKey: "form.checkpoint_interval", kind: "int", min: 1 },
];

export const FORM_FIELDS: Record<SubstrateId, FormField[]> = {
  search: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    {
      path: "dynamics.algorithm",
      labelKey: "form.search.algorithm",
      kind: "select",
      options: [
        { value: "cma_es", labelKey: "form.search.algorithm.cma_es" },
        { value: "map_elites", labelKey: "form.search.algorithm.map_elites" },
        { value: "novelty", labelKey: "form.search.algorithm.novelty" },
        { value: "imgep", labelKey: "form.search.algorithm.imgep" },
        { value: "random", labelKey: "form.search.algorithm.random" },
      ],
    },
    { path: "dynamics.population", labelKey: "form.search.population", kind: "int", min: 2, max: 256 },
    { path: "dynamics.sigma0", labelKey: "form.search.sigma0", kind: "float", min: 0.01, max: 1, step: 0.05 },
    { path: "dynamics.eval_steps", labelKey: "form.search.eval_steps", kind: "int", min: 10, max: 20000 },
    { path: "dynamics.eval_seeds", labelKey: "form.search.eval_seeds", kind: "int", min: 1, max: 16 },
    { path: "dynamics.holdout_seeds", labelKey: "form.search.holdout_seeds", kind: "int", min: 0, max: 16 },
    { path: "asal.prompts", labelKey: "form.search.prompts", kind: "textlist" },
    { path: "asal.frames", labelKey: "form.search.frames", kind: "int", min: 1, max: 64 },
    { path: "runner.max_steps", labelKey: "form.search.generations", kind: "int", min: 1 },
    { path: "runner.checkpoint_interval", labelKey: "form.checkpoint_interval", kind: "int", min: 1 },
  ],
  conway: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "world.width", labelKey: "form.width", kind: "int", min: 8, max: 1024 },
    { path: "world.height", labelKey: "form.height", kind: "int", min: 8, max: 1024 },
    {
      path: "world.boundary",
      labelKey: "form.boundary",
      kind: "select",
      options: [
        { value: "periodic", labelKey: "form.boundary.periodic" },
        { value: "dead", labelKey: "form.boundary.dead" },
      ],
    },
    ...RUNNER,
  ],
  boids: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "initialization.count", labelKey: "form.boids.count", kind: "int", min: 0, max: 10000 },
    { path: "world.width", labelKey: "form.width", kind: "float", min: 10, step: 10 },
    { path: "world.height", labelKey: "form.height", kind: "float", min: 10, step: 10 },
    {
      path: "world.boundary",
      labelKey: "form.boundary",
      kind: "select",
      options: [
        { value: "periodic", labelKey: "form.boundary.periodic" },
        { value: "reflect", labelKey: "form.boundary.reflect" },
      ],
    },
    { path: "dynamics.separation_weight", labelKey: "form.boids.separation_weight", kind: "float", min: 0, step: 0.005 },
    { path: "dynamics.alignment_weight", labelKey: "form.boids.alignment_weight", kind: "float", min: 0, step: 0.005 },
    { path: "dynamics.cohesion_weight", labelKey: "form.boids.cohesion_weight", kind: "float", min: 0, step: 0.001 },
    { path: "dynamics.perception_radius", labelKey: "form.boids.perception_radius", kind: "float", min: 0.1, step: 0.5 },
    { path: "dynamics.noise", labelKey: "form.boids.noise", kind: "float", min: 0, step: 0.01 },
    ...RUNNER,
  ],
  bff: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "soup.num_programs", labelKey: "form.bff.num_programs", kind: "int", min: 2, step: 2 },
    {
      path: "dynamics.variant",
      labelKey: "form.bff.variant",
      kind: "select",
      options: [
        { value: "bff_noheads", labelKey: "form.bff.variant.noheads" },
        { value: "bff", labelKey: "form.bff.variant.heads" },
      ],
    },
    { path: "dynamics.mutation_prob", labelKey: "form.bff.mutation_prob", kind: "int", min: 0 },
    { path: "dynamics.step_limit", labelKey: "form.bff.step_limit", kind: "int", min: 0 },
    ...RUNNER,
  ],
  lenia: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "world.width", labelKey: "form.width", kind: "int", min: 16, max: 1024 },
    { path: "world.height", labelKey: "form.height", kind: "int", min: 16, max: 1024 },
    {
      path: "initialization.preset",
      labelKey: "form.lenia.init",
      kind: "select",
      options: [
        { value: "pattern", labelKey: "form.lenia.init.pattern" },
        { value: "random", labelKey: "form.lenia.init.random" },
        { value: "empty", labelKey: "form.lenia.init.empty" },
      ],
    },
    { path: "dynamics.T", labelKey: "form.lenia.T", kind: "int", min: 1, max: 1000 },
    { path: "dynamics.kernel.R", labelKey: "form.lenia.R", kind: "int", min: 2, max: 128 },
    { path: "dynamics.growth.m", labelKey: "form.lenia.m", kind: "float", min: 0, max: 1, step: 0.001 },
    { path: "dynamics.growth.s", labelKey: "form.lenia.s", kind: "float", min: 0.0001, max: 1, step: 0.001 },
    ...RUNNER,
  ],
  flow_lenia: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "world.width", labelKey: "form.width", kind: "int", min: 16, max: 1024 },
    { path: "world.height", labelKey: "form.height", kind: "int", min: 16, max: 1024 },
    {
      path: "world.boundary",
      labelKey: "form.boundary",
      kind: "select",
      options: [
        { value: "wall", labelKey: "form.boundary.wall" },
        { value: "torus", labelKey: "form.boundary.periodic" },
      ],
    },
    { path: "initialization.size", labelKey: "form.flow_lenia.init_size", kind: "int", min: 1, max: 1024 },
    { path: "dynamics.dt", labelKey: "form.flow_lenia.dt", kind: "float", min: 0.001, max: 10, step: 0.01 },
    { path: "dynamics.sigma", labelKey: "form.flow_lenia.sigma", kind: "float", min: 0.01, max: 5, step: 0.05 },
    ...RUNNER,
  ],
  creature: [
    { path: "dynamics.controller.frequency_hz", labelKey: "form.creature.frequency", kind: "float", min: 0.05, max: 5, step: 0.1 },
    { path: "dynamics.gear", labelKey: "form.creature.gear", kind: "float", min: 1, max: 500, step: 5 },
    { path: "dynamics.joint_stiffness", labelKey: "form.creature.stiffness", kind: "float", min: 0, max: 1000, step: 5 },
    { path: "dynamics.friction", labelKey: "form.creature.friction", kind: "float", min: 0.05, max: 5, step: 0.1 },
    { path: "dynamics.body.torso_radius", labelKey: "form.creature.torso", kind: "float", min: 0.05, max: 0.6, step: 0.05 },
    {
      path: "dynamics.controller.kind",
      labelKey: "form.creature.controller",
      kind: "select",
      options: [
        { value: "cpg", labelKey: "form.creature.controller.cpg" },
        { value: "off", labelKey: "form.creature.controller.off" },
      ],
    },
    ...RUNNER,
  ],
  lenia_mc: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "world.size", labelKey: "form.lenia_mc.size", kind: "int", min: 32, max: 1024 },
    {
      path: "initialization.preset",
      labelKey: "form.lenia.init",
      kind: "select",
      options: [
        { value: "pattern", labelKey: "form.lenia_mc.init.pattern" },
        { value: "random", labelKey: "form.lenia.init.random" },
        { value: "empty", labelKey: "form.lenia.init.empty" },
      ],
    },
    { path: "dynamics.T", labelKey: "form.lenia_mc.T", kind: "float", min: 0.1, max: 1000, step: 0.5 },
    ...RUNNER,
  ],
  gray_scott: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "world.width", labelKey: "form.gray_scott.size", kind: "int", min: 16, max: 1024 },
    { path: "world.height", labelKey: "form.gray_scott.size_h", kind: "int", min: 16, max: 1024 },
    { path: "world.length", labelKey: "form.gray_scott.length", kind: "float", min: 0.01, step: 0.01 },
    { path: "dynamics.F", labelKey: "form.gray_scott.F", kind: "float", min: 0, max: 1, step: 0.0001 },
    { path: "dynamics.k", labelKey: "form.gray_scott.k", kind: "float", min: 0, max: 1, step: 0.0001 },
    { path: "dynamics.dt", labelKey: "form.gray_scott.dt", kind: "float", min: 0.001, step: 0.1 },
    ...RUNNER,
  ],
  physarum: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "world.width", labelKey: "form.width", kind: "int", min: 16, max: 1024 },
    { path: "world.height", labelKey: "form.height", kind: "int", min: 16, max: 1024 },
    { path: "initialization.population_fraction", labelKey: "form.physarum.population", kind: "float", min: 0.001, max: 1, step: 0.01 },
    { path: "dynamics.sensor_angle_deg", labelKey: "form.physarum.sa", kind: "float", min: 0, max: 180, step: 1 },
    { path: "dynamics.rotation_angle_deg", labelKey: "form.physarum.ra", kind: "float", min: 0, max: 180, step: 1 },
    { path: "dynamics.sensor_offset", labelKey: "form.physarum.so", kind: "float", min: 0.1, max: 64, step: 0.5 },
    { path: "dynamics.deposit", labelKey: "form.physarum.deposit", kind: "float", min: 0, step: 0.5 },
    { path: "dynamics.decay", labelKey: "form.physarum.decay", kind: "float", min: 0, max: 1, step: 0.01 },
    {
      path: "dynamics.sensing",
      labelKey: "form.physarum.sensing",
      kind: "select",
      options: [
        { value: "true", labelKey: "form.on" },
        { value: "false", labelKey: "form.off_control" },
      ],
    },
    ...RUNNER,
  ],
  reaction_network: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "dynamics.time_per_tick", labelKey: "form.reaction_network.time_per_tick", kind: "float", min: 0.000001, step: 0.001 },
    ...RUNNER,
  ],
  nca_train: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    {
      path: "dynamics.experiment",
      labelKey: "form.nca.experiment",
      kind: "select",
      options: [
        { value: "growing", labelKey: "form.nca.experiment.growing" },
        { value: "persistent", labelKey: "form.nca.experiment.persistent" },
        { value: "regenerating", labelKey: "form.nca.experiment.regenerating" },
      ],
    },
    {
      path: "target.preset",
      labelKey: "form.nca.target",
      kind: "select",
      options: [
        { value: "heart", labelKey: "form.nca.target.heart" },
        { value: "disc", labelKey: "form.nca.target.disc" },
        { value: "ring", labelKey: "form.nca.target.ring" },
        { value: "letter_a", labelKey: "form.nca.target.letter_a" },
      ],
    },
    { path: "target.size", labelKey: "form.nca.target_size", kind: "int", min: 8, max: 64 },
    { path: "dynamics.batch_size", labelKey: "form.nca.batch_size", kind: "int", min: 1, max: 64 },
    { path: "dynamics.steps_min", labelKey: "form.nca.steps_min", kind: "int", min: 1, max: 512 },
    { path: "dynamics.steps_max", labelKey: "form.nca.steps_max", kind: "int", min: 1, max: 512 },
    { path: "dynamics.learning_rate", labelKey: "form.nca.lr", kind: "float", min: 0.00001, max: 1, step: 0.0001 },
    ...RUNNER,
  ],
  nca: [{ path: "seed", labelKey: "form.seed", kind: "int", min: 0 }, ...RUNNER],
  connectome: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    {
      path: "wiring",
      labelKey: "form.connectome.wiring",
      kind: "select",
      options: [
        { value: "reference", labelKey: "form.connectome.wiring.reference" },
        { value: "shuffled_control", labelKey: "form.connectome.wiring.shuffled_control" },
      ],
    },
    ...RUNNER,
  ],
  agent_world: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "initialization.count", labelKey: "form.agent_world.count", kind: "int", min: 0, max: 4000 },
    {
      path: "dynamics.controller.kind",
      labelKey: "form.agent_world.controller",
      kind: "select",
      options: [
        { value: "mlp", labelKey: "form.agent_world.controller.mlp" },
        { value: "baseline", labelKey: "form.agent_world.controller.baseline" },
        { value: "random", labelKey: "form.agent_world.controller.random" },
      ],
    },
    {
      path: "dynamics.controller.mode",
      labelKey: "form.agent_world.mode",
      kind: "select",
      options: [
        { value: "evolution", labelKey: "form.agent_world.mode.evolution" },
        { value: "lifetime", labelKey: "form.agent_world.mode.lifetime" },
        { value: "fixed", labelKey: "form.agent_world.mode.fixed" },
      ],
    },
    { path: "dynamics.food.regrowth", labelKey: "form.agent_world.regrowth", kind: "float", min: 0, max: 1, step: 0.001 },
    ...RUNNER,
  ],
  protocell: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "world.width", labelKey: "form.width", kind: "int", min: 8, max: 512 },
    { path: "world.height", labelKey: "form.height", kind: "int", min: 8, max: 512 },
    { path: "initialization.count", labelKey: "form.protocell.count", kind: "int", min: 0, max: 4000 },
    { path: "dynamics.environment.inflow_per_cell", labelKey: "form.protocell.inflow", kind: "float", min: 0, max: 100, step: 0.05 },
    { path: "dynamics.mutation_sigma", labelKey: "form.protocell.mutation", kind: "float", min: 0, max: 2, step: 0.01 },
    {
      path: "dynamics.partition",
      labelKey: "form.protocell.partition",
      kind: "select",
      options: [
        { value: "binomial", labelKey: "form.protocell.partition.binomial" },
        { value: "random_fraction", labelKey: "form.protocell.partition.random" },
      ],
    },
    ...RUNNER,
  ],
  pd_nca: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "world.width", labelKey: "form.width", kind: "int", min: 8, max: 256 },
    { path: "world.height", labelKey: "form.height", kind: "int", min: 8, max: 256 },
    { path: "dynamics.n_ncas", labelKey: "form.pd_nca.n_ncas", kind: "int", min: 1, max: 16 },
    {
      path: "dynamics.learning",
      labelKey: "form.pd_nca.learning",
      kind: "select",
      options: [
        { value: "online", labelKey: "form.pd_nca.learning.online" },
        { value: "frozen", labelKey: "form.pd_nca.learning.frozen" },
      ],
    },
    {
      path: "dynamics.competition",
      labelKey: "form.pd_nca.competition",
      kind: "select",
      options: [
        { value: "cosine", labelKey: "form.pd_nca.competition.cosine" },
        { value: "uniform", labelKey: "form.pd_nca.competition.uniform" },
      ],
    },
    { path: "dynamics.learning_rate", labelKey: "form.nca.lr", kind: "float", min: 0.00001, max: 10, step: 0.001 },
    ...RUNNER,
  ],
  particle_life: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    { path: "initialization.count", labelKey: "form.particles.count", kind: "int", min: 0, max: 10000 },
    { path: "world.width", labelKey: "form.width", kind: "float", min: 10 },
    { path: "world.height", labelKey: "form.height", kind: "float", min: 10 },
    {
      path: "world.boundary",
      labelKey: "form.boundary",
      kind: "select",
      options: [
        { value: "periodic", labelKey: "form.boundary.periodic" },
        { value: "reflect", labelKey: "form.boundary.reflect" },
      ],
    },
    { path: "dynamics.r_max", labelKey: "form.particles.r_max", kind: "float", min: 0.1 },
    { path: "dynamics.force_factor", labelKey: "form.particles.force_factor", kind: "float", min: 0 },
    { path: "dynamics.friction_half_life", labelKey: "form.particles.friction_half_life", kind: "float", min: 0.001 },
    ...RUNNER,
  ],
  connectome_sysid: [
    { path: "seed", labelKey: "form.seed", kind: "int", min: 0 },
    {
      path: "wiring",
      labelKey: "form.connectome.wiring",
      kind: "select",
      options: [
        { value: "reference", labelKey: "form.connectome.wiring.reference" },
        { value: "shuffled_control", labelKey: "form.connectome.wiring.shuffled_control" },
      ],
    },
    { path: "observation.obs_fraction", labelKey: "form.sysid.obs_fraction", kind: "float", min: 0.01, max: 1, step: 0.05 },
    { path: "observation.bin_ms", labelKey: "form.sysid.bin_ms", kind: "float", min: 2, max: 50, step: 2 },
    { path: "observation.noise_sd", labelKey: "form.sysid.noise_sd", kind: "float", min: 0, max: 0.2, step: 0.005 },
    { path: "dynamics.lr", labelKey: "form.nca.lr", kind: "float", min: 0.001, max: 0.5, step: 0.01 },
    ...RUNNER,
  ],
};

export function getPath(obj: Record<string, unknown>, path: string): unknown {
  return path.split(".").reduce<unknown>((o, k) => (o && typeof o === "object" ? (o as Record<string, unknown>)[k] : undefined), obj);
}

export function setPath(obj: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const keys = path.split(".");
  const root: Record<string, unknown> = { ...obj };
  let cur = root;
  for (let i = 0; i < keys.length - 1; i++) {
    const next = cur[keys[i]];
    cur[keys[i]] = next && typeof next === "object" ? { ...(next as Record<string, unknown>) } : {};
    cur = cur[keys[i]] as Record<string, unknown>;
  }
  cur[keys[keys.length - 1]] = value;
  return root;
}

/** Population shown for a display frame (alive cells or number of particles). */
export function framePopulation(frame: Frame): number {
  if (frame.header.codec === "bff_soup") return (frame.header as { programs?: number }).programs ?? 0;
  if (frame.header.codec === "field_u8") {
    const c = (frame.header as { channels?: number }).channels ?? 1;
    let n = 0;
    for (let i = 0; i < frame.bits.length; i += c) {
      let sum = 0;
      for (let k = 0; k < c; k++) sum += frame.bits[i + k];
      if (sum > 2) n++;
    }
    return n;
  }
  if (frame.header.codec === "particles_f32" || frame.header.codec === "particles_typed_f32") {
    return frame.header.count ?? 0;
  }
  return popcount(frame.bits);
}
