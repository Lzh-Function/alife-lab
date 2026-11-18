// Per-substrate UI descriptors: intervention tools, create-form fields and frame summaries.
// Substrates differ (grid vs particles); the UI is driven by these small tables rather than
// by assuming a single state representation.
import type { Frame } from "./api";
import { popcount } from "./api";

export type SubstrateId = "conway" | "boids";

export interface ToolState {
  mode: string; // "off" or an intervention mode
  radius: number;
  count: number;
}

export interface ToolDescriptor {
  modes: string[]; // first entry is "off"
  radius: { min: number; max: number; step: number; initial: number };
  count?: { min: number; max: number; initial: number };
  noteKey: string;
}

export const TOOLS: Record<SubstrateId, ToolDescriptor> = {
  conway: {
    modes: ["off", "draw", "erase"],
    radius: { min: 0, max: 8, step: 1, initial: 0 },
    noteKey: "lab.brush.note",
  },
  boids: {
    modes: ["off", "spawn", "remove"],
    radius: { min: 1, max: 40, step: 1, initial: 8 },
    count: { min: 1, max: 500, initial: 30 },
    noteKey: "lab.tool.boids_note",
  },
};

export function initialTool(substrate: string): ToolState {
  const d = TOOLS[(substrate as SubstrateId) in TOOLS ? (substrate as SubstrateId) : "conway"];
  return { mode: "off", radius: d.radius.initial, count: d.count?.initial ?? 0 };
}

export interface FormField {
  path: string; // dotted path under experiment.*
  labelKey: string;
  kind: "int" | "float" | "select";
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
  if (frame.header.codec === "particles_f32") return frame.header.count ?? 0;
  return popcount(frame.bits);
}
