// Typed client for the ALife Lab REST API (same-origin).

export interface Envelope<T> {
  api_version: string;
  server_version: string;
  data: T;
}

export interface ApiErrorBody {
  code: string;
  message: string;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

export interface Checkpoint {
  id: string;
  run_id: string;
  epoch: number;
  simulation_clock: number;
  clock_unit: string;
  logical_path: string;
  sha256: string;
  size_bytes: number;
  reason: string;
  created_at: number;
}

export interface RunSummary {
  id: string;
  name: string;
  label: string | null;
  experiment_id: string;
  parent_run_id: string | null;
  parent_checkpoint_id: string | null;
  children: string[];
  comparisons: string[];
  substrate: string;
  declared_mode: "reference" | "exploratory";
  effective_mode: "reference" | "exploratory";
  intervened: boolean;
  state: string;
  revision: number;
  seed: number;
  backend: string | null;
  target_id: string;
  profile_id: string;
  spec_hash: string;
  profile_hash: string;
  spec: { experiment: { world: { width: number; height: number; boundary: string }; runner: { max_steps: number; checkpoint_interval: number }; observations: { metrics: string[] } } };
  clock: { unit: string; value: number };
  epoch: number;
  ticks_per_second: number;
  created_at: number;
  finish_reason: string | null;
  last_error: string | null;
  latest_checkpoint: Checkpoint | null;
  restorable: boolean;
  live: null | {
    job_id: string;
    steps_per_second: number;
    worker_rss_mb: number | null;
    worker_peak_rss_mb: number | null;
    worker_cpu_percent: number | null;
  };
}

export interface RunListItem {
  id: string;
  state: string;
  tick: number;
  created_at: number;
  effective_mode: string;
  intervened: number;
  profile_id: string;
  name: string;
  substrate: string;
  spec_hash: string;
  parent_run_id: string | null;
}

export interface Preset {
  id: string;
  title_key: string;
  description_key: string;
  substrate: string;
  spec: { experiment: Record<string, unknown> & { name: string; seed: number; world: { width: number; height: number; boundary: string }; runner: { max_steps: number; checkpoint_interval: number } } };
  spec_hash: string;
  expected_observation: string;
  negative_control: string | null;
}

export interface ProfileInfo {
  id: string;
  role: string;
  implemented: boolean;
  profile_hash: string;
}

export interface RunEvent {
  run_id: string;
  seq: number;
  clock_unit: string;
  clock_value: number | null;
  observed_at_utc: number;
  source: string;
  type: string;
  severity: string;
  payload: Record<string, unknown>;
}

export interface MetricRow {
  tick: number;
  epoch: number;
  values: Record<string, number | null>;
  post_intervention?: boolean;
}

export interface Intervention {
  id: string;
  command_id: string;
  kind: string;
  requested_clock: number | null;
  applied_clock: number | null;
  status: string;
  author: string;
  result: Record<string, unknown> | null;
}

export interface CommandResult {
  command_id: string;
  status: string;
  ack: null | { applied_clock: number | null; status: string; error?: string | null; result?: unknown };
  run?: RunSummary;
}

let cachedToken: string | null = null;

export interface BrushIntervention {
  kind: "brush";
  payload: Record<string, unknown>;
}

export interface ComparisonSummary {
  id: string;
  name: string;
  design: string;
  parent_run_id: string | null;
  parent_checkpoint_id: string | null;
  created_at: number;
  members: Array<{ arm: string; run: RunSummary }>;
}

export interface DivergencePoint {
  tick: number;
  hamming_to_first: Array<number | null>;
  alive: number[];
  reasons: string[];
}

export interface ComparisonReport {
  comparison: ComparisonSummary;
  divergence: { definition: string; common_ticks: number; sampled_ticks: number; shape_mismatch: boolean; points: DivergencePoint[] };
  metrics: Record<string, MetricRow[]>;
}

export function sessionToken(): string {
  if (cachedToken !== null) return cachedToken;
  const meta = document.querySelector('meta[name="alife-token"]');
  return meta?.getAttribute("content") ?? "";
}

/** The token rotates on every server start; re-read it from the same-origin index page. */
export async function refreshSessionToken(): Promise<string> {
  const resp = await fetch("/", { cache: "no-store" });
  const html = await resp.text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  cachedToken = doc.querySelector('meta[name="alife-token"]')?.getAttribute("content") ?? "";
  return cachedToken;
}

async function request<T>(method: string, path: string, body?: unknown, headers?: Record<string, string>, retried = false): Promise<T> {
  const init: RequestInit = {
    method,
    headers: {
      Accept: "application/json",
      ...(method !== "GET" ? { "X-ALife-Token": sessionToken() } : {}),
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  };
  const resp = await fetch(`/api/v1${path}`, init);
  if (!resp.ok) {
    let err: ApiErrorBody = { code: "http_error", message: resp.statusText };
    try {
      const parsed = (await resp.json()) as { error?: ApiErrorBody };
      if (parsed.error) err = parsed.error;
    } catch {
      /* not JSON */
    }
    if (resp.status === 403 && err.code === "token_required" && !retried) {
      await refreshSessionToken();
      return request<T>(method, path, body, headers, true);
    }
    throw new ApiError(resp.status, err.code, err.message);
  }
  const parsed = (await resp.json()) as Envelope<T>;
  return parsed.data;
}

export function newCommandId(): string {
  return crypto.randomUUID();
}

export const api = {
  presets: () => request<Preset[]>("GET", "/presets"),
  profiles: () => request<ProfileInfo[]>("GET", "/execution-profiles"),
  substrates: () => request<Array<{ slug: string; plugin_version: string; display_name_key: string; backends: string[] }>>("GET", "/substrates"),
  resources: () => request<Record<string, unknown>>("GET", "/system/resources"),
  capabilities: () => request<Record<string, unknown>>("GET", "/system/capabilities"),
  runs: () => request<RunListItem[]>("GET", "/experiments"),
  run: (id: string) => request<RunSummary>("GET", `/experiments/${id}`),
  validate: (body: unknown) => request<{ report: { ok: boolean; errors: string[]; warnings: string[] }; spec_hash: string; profile_hash: string }>("POST", "/experiments/validate", body),
  create: (body: unknown) =>
    request<RunSummary>("POST", "/experiments", body, { "Idempotency-Key": `create-${newCommandId()}` }),
  control: (id: string, action: "pause" | "resume" | "step" | "checkpoints", body: Record<string, unknown> = {}) =>
    request<CommandResult>("POST", `/experiments/${id}/${action}`, { command_id: newCommandId(), ...body }),
  speed: (id: string, tps: number) =>
    request<CommandResult>("POST", `/experiments/${id}/speed`, { command_id: newCommandId(), ticks_per_second: tps }),
  intervene: (id: string, payload: Record<string, unknown>, requestedClock: number) =>
    request<CommandResult>("POST", `/experiments/${id}/interventions`, {
      command_id: newCommandId(),
      kind: "brush",
      payload,
      requested_clock: requestedClock,
    }),
  restore: (id: string) => request<RunSummary>("POST", `/experiments/${id}/restore`, { checkpoint_id: "latest", start_paused: true }),
  cancel: (id: string) => request<RunSummary>("POST", `/experiments/${id}/cancel`),
  fork: (id: string, checkpointId: string, label?: string) =>
    request<RunSummary>("POST", `/experiments/${id}/fork`, { checkpoint_id: checkpointId, label, auto_start: true, start_paused: true }, { "Idempotency-Key": `fork-${newCommandId()}` }),
  pairedFork: (id: string, checkpointId = "now", treatment?: BrushIntervention) =>
    request<ComparisonSummary>(
      "POST",
      `/experiments/${id}/paired-fork`,
      { checkpoint_id: checkpointId, treatment_intervention: treatment ?? null, start: true, start_paused: true },
      { "Idempotency-Key": `pf-${newCommandId()}` },
    ),
  comparisons: () => request<Array<Omit<ComparisonSummary, "members">>>("GET", "/comparisons"),
  comparison: (id: string) => request<ComparisonSummary>("GET", `/comparisons/${id}`),
  comparisonReport: (id: string) => request<ComparisonReport>("GET", `/comparisons/${id}/report?max_points=1000`),
  checkpoints: (id: string) => request<Checkpoint[]>("GET", `/experiments/${id}/checkpoints`),
  events: (id: string, sinceSeq = 0) => request<RunEvent[]>("GET", `/experiments/${id}/events?since_seq=${sinceSeq}`),
  metrics: (id: string) => request<{ rows: MetricRow[]; total_points: number; stride: number }>("GET", `/experiments/${id}/metrics?max_points=2000`),
  interventions: (id: string) => request<Intervention[]>("GET", `/experiments/${id}/interventions`),
  checkpointFrame: async (id: string, checkpointId: string): Promise<Frame> => {
    const resp = await fetch(`/api/v1/experiments/${id}/checkpoints/${checkpointId}/frame`);
    if (!resp.ok) throw new ApiError(resp.status, "frame_error", resp.statusText);
    return unpackFrame(await resp.arrayBuffer());
  },
};

export interface Frame {
  header: { tick: number; width: number; height: number; codec: string; dropped_previews?: number };
  bits: Uint8Array;
}

export function unpackFrame(buffer: ArrayBuffer): Frame {
  const view = new DataView(buffer);
  const n = view.getUint32(0, false);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, n))) as Frame["header"];
  return { header, bits: new Uint8Array(buffer, 4 + n) };
}

/** Bit at index i of a numpy.packbits (big-endian bit order) buffer. */
export function bitAt(bits: Uint8Array, i: number): number {
  return (bits[i >> 3] >> (7 - (i & 7))) & 1;
}

/** Cells within a circular brush, clipped to the world (mirrors the server rule). */
export function circleCells(cx: number, cy: number, radius: number, width: number, height: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  const r2 = radius * radius;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy > r2) continue;
      const x = cx + dx;
      const y = cy + dy;
      if (x >= 0 && x < width && y >= 0 && y < height) out.push([x, y]);
    }
  }
  return out;
}
