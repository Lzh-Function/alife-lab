import { useT } from "../i18n";
import { MultiLine } from "./MultiLine";
import { SearchInspector, type SearchInspection } from "./SearchInspector";

// Read-only inspection results (conway cell / boids neighbourhood).
type Vec = { x: number; y: number; magnitude: number };
type Boid = { id: number; x: number; y: number; vx: number; vy: number; speed: number; heading_deg: number; distance: number };

export interface Inspection extends Omit<Partial<SearchInspection>, "region"> {
  substrate: string;
  tick: number;
  source?: { method: string; base_tick?: number; interventions_replayed?: number };
  // conway
  cell?: { alive: boolean; live_neighbors: number; next_tick: string };
  rule?: string | { R: number; kernels: unknown[] }; // conway rule string / flow_lenia rule
  patch?: number[][];
  patch_live_neighbors?: Array<Array<number | null>>;
  patch_origin?: { x: number; y: number };
  // boids
  count_in_radius?: number;
  boids?: Boid[];
  nearest?: (Boid & { steering: { neighbors: number; too_close: number; separation: Vec; alignment: Vec; cohesion: Vec } }) | null;
  query?: { x: number; y: number; radius: number };
  // bff
  index?: number;
  program_hex?: string;
  disassembly?: string;
  instruction_fraction?: number;
  exact_copies_in_soup?: number;
  trace?: import("./BffDebugger").BffTrace;
  self_pair_test?: { description: string; ops: number; first_half_unchanged: boolean; second_half_equals_program: boolean };
  // particle_life
  type_counts_in_radius?: number[];
  matrix?: number[][];
  nearest_particle?: { id: number; type: number; speed: number; neighbors_by_type: number[] } | null;
  // lenia
  value?: number;
  potential?: number;
  growth?: number;
  next_value?: number;
  patch_mass?: number;
  // gray_scott
  u?: number;
  v?: number;
  laplacian_u?: number;
  laplacian_v?: number;
  reaction_uvv?: number;
  du_dt?: number;
  dv_dt?: number;
  // physarum
  trail?: number;
  agents_in_radius?: number;
  nearest_agent?: {
    id: number;
    heading_deg: number;
    distance: number;
    sensors: { front: number; left: number; right: number };
    decision: string;
  } | null;
  // nca / nca_train
  rgba?: number[];
  hidden?: number[];
  alive?: boolean;
  target_rgba?: number[] | null;
  last_step?: { loss?: number; steps?: number; lr?: number };
  parameter_count?: number;
  // agent_world
  agent?: {
    id: number;
    parent: number;
    birth_tick: number;
    age: number;
    energy: number;
    heading_deg: number;
    inputs: string[];
    actions: string[];
    history: { inputs: number[][]; hidden: number[][]; probs: number[][]; action: number[]; reward: number[] };
    learned_delta_norm: number;
  } | null;
  // protocell
  field?: { food: number; waste: number };
  protocell?: {
    id: number;
    birth_tick: number;
    ancestors: number[];
    contents: Record<string, number>;
    genome: Record<string, number>;
    radius: number;
    pressure: number;
    distance: number;
  } | null;
  ledger?: Record<string, number>;
  ledger_residual?: number;
  // pd_nca
  aliveness?: number[];
  dominant?: number;
  territories?: number[];
  interaction?: Array<Array<number | null>>;
  // reaction_network
  time?: number;
  species?: Array<{ name: string; count: number; constant: boolean }>;
  selected_species?: { name: string; count: number; constant: boolean; consumed_by: string[]; produced_by: string[] } | null;
  reactions?: Array<{ id: string; label: string; equation: string; products: string; rate: number; enabled: boolean; propensity: number; fired: number }>;
  cycle_candidates?: { autocatalysis: Array<{ species: string; reaction: string }>; cycles: Array<{ species: string[]; reactions: string[] }>; truncated: boolean; rule: string };
  // creature
  torso?: { position: number[]; velocity: number[]; upright: number };
  legs?: Array<{ leg: number; hip_deg: number; knee_deg: number; hip_ctrl: number; knee_ctrl: number; foot_on_ground: boolean }>;
  mass_kg?: number;
  // lenia_mc
  values?: number[];
  next_values?: number[];
  kernels?: Array<{ index: number; c0: number; c1: number; potential: number; growth: number; m: number; s: number; h: number }>;
  // flow_lenia
  channels?: Array<{
    value: number;
    affinity: number;
    alpha: number;
    flow: { x: number; y: number };
    displacement: { x: number; y: number };
  }>;
  // connectome
  dataset_id?: string;
  port?: string;
  count?: number;
  spikes_tick?: number;
  total_spikes?: number;
  mean_rate_hz?: number;
  mean_v_mv?: number;
  cell_index?: number;
  source_id?: number;
  cell_type?: string;
  super_class?: string;
  cell_class?: string;
  region?: string;
  side?: string;
  role?: string;
  neurotransmitter?: string;
  position_nm?: number[];
  v_mv?: number;
  g_mv?: number;
  refrac_substeps?: number;
  stim_rate_hz?: number;
  silenced?: boolean;
  last_tick_spikes?: number;
}

const f = (v: number, d = 3) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(d));

export function InspectorPanel({
  data,
  error,
  onTrace,
  onAssay,
  assayPending = false,
  onReactionToggle,
  onOpenSpec,
  onRegister,
}: {
  data: Inspection | null;
  error: string | null;
  onTrace?: (mode: "trace_next_epoch" | "trace_self") => void;
  onAssay?: () => void;
  assayPending?: boolean;
  onReactionToggle?: (reactionId: string, enable: boolean) => void;
  onOpenSpec?: (spec: Record<string, unknown>) => void;
  onRegister?: (candidate: number, claimKind: string) => void;
}) {
  const t = useT();
  return (
    <div className="panel" data-testid="inspector">
      <h3>{t("inspector.title")}</h3>
      {error && <p className="error small">{error}</p>}
      {!data && !error && <p className="muted small">{t("inspector.hint")}</p>}
      {data && (
        <>
          <p className="small muted" data-testid="inspector-source">
            t={data.tick} · {t(`inspector.source.${data.source?.method ?? "live_worker"}`)}
            {data.source?.method === "deterministic_replay" && ` (t=${data.source.base_tick}, ${t("timeline.interventions_replayed")} ${data.source.interventions_replayed})`}
          </p>
          {data.substrate === "conway" && data.cell && (
            <>
              <dl className="stats small" data-testid="inspector-cell">
                <dt>{t("inspector.cell")}</dt>
                <dd>
                  ({data.query?.x}, {data.query?.y}) · {data.cell.alive ? t("inspector.alive") : t("inspector.dead")}
                </dd>
                <dt>{t("inspector.live_neighbors")}</dt>
                <dd>{data.cell.live_neighbors}</dd>
                <dt>{t("inspector.next")}</dt>
                <dd data-testid="inspector-next">{t(`inspector.outcome.${data.cell.next_tick}`)}</dd>
                <dt>rule</dt>
                <dd className="mono">{typeof data.rule === "string" ? data.rule : ""}</dd>
              </dl>
              <table className="patch" aria-label={t("inspector.patch")}>
                <tbody>
                  {(data.patch ?? []).map((row, yi) => (
                    <tr key={yi}>
                      {row.map((v, xi) => {
                        const center = yi === (data.patch!.length - 1) / 2 && xi === (row.length - 1) / 2;
                        const n = data.patch_live_neighbors?.[yi][xi];
                        return (
                          <td key={xi} className={`${v === 1 ? "on" : v === -1 ? "out" : "off"} ${center ? "center" : ""}`} title={n === null ? "" : `n=${n}`}>
                            {n ?? ""}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted small">{t("inspector.patch_note")}</p>
            </>
          )}
          {data.substrate === "bff" && data.disassembly !== undefined && (
            <>
              <dl className="stats small" data-testid="inspector-program">
                <dt>{t("inspector.program_index")}</dt>
                <dd className="mono">#{data.index}</dd>
                <dt>{t("inspector.copies")}</dt>
                <dd className="mono" data-testid="inspector-copies">{data.exact_copies_in_soup}</dd>
                <dt>{t("inspector.instruction_fraction")}</dt>
                <dd className="mono">{((data.instruction_fraction ?? 0) * 100).toFixed(0)}%</dd>
              </dl>
              <pre className="program" data-testid="inspector-disassembly">
                {data.disassembly.slice(0, 32)}
                {"\n"}
                {data.disassembly.slice(32)}
              </pre>
              <p className="muted small mono">{data.program_hex?.slice(0, 64)}…</p>
              {data.self_pair_test && (
                <p className="small">
                  {t("inspector.self_pair")}: ops {data.self_pair_test.ops},{" "}
                  {data.self_pair_test.second_half_equals_program ? t("inspector.self_pair.copy_kept") : t("inspector.self_pair.changed")}
                  <br />
                  <span className="muted">{t("inspector.self_pair_note")}</span>
                </p>
              )}
              {onTrace && (
                <div className="toolbar">
                  <button data-testid="btn-trace-next" onClick={() => onTrace("trace_next_epoch")}>
                    {t("debugger.trace_next")}
                  </button>
                  <button data-testid="btn-trace-self" onClick={() => onTrace("trace_self")}>
                    {t("debugger.trace_self")}
                  </button>
                  {onAssay && (
                    <button data-testid="btn-assay" onClick={onAssay} disabled={assayPending}>
                      {assayPending ? t("assay.running") : t("assay.run")}
                    </button>
                  )}
                </div>
              )}
            </>
          )}
          {data.substrate === "particle_life" && (
            <>
              <p className="small">
                {t("inspector.in_radius")}: {data.count_in_radius} · {t("inspector.by_type")}: [{(data.type_counts_in_radius ?? []).join(", ")}]
              </p>
              {data.nearest_particle ? (
                <dl className="stats small" data-testid="inspector-particle">
                  <dt>id / type</dt>
                  <dd className="mono">
                    {data.nearest_particle.id} / {data.nearest_particle.type}
                  </dd>
                  <dt>{t("inspector.speed")}</dt>
                  <dd className="mono">{f(data.nearest_particle.speed)}</dd>
                  <dt>{t("inspector.neighbors_by_type")}</dt>
                  <dd className="mono">[{data.nearest_particle.neighbors_by_type.join(", ")}]</dd>
                </dl>
              ) : (
                <p className="muted small">{t("inspector.none")}</p>
              )}
              {data.matrix && (
                <table className="matrix small" data-testid="inspector-matrix">
                  <caption className="muted">{t("inspector.matrix")}</caption>
                  <tbody>
                    {data.matrix.map((row, a) => (
                      <tr key={a}>
                        {row.map((v, b) => (
                          <td key={b} style={{ background: v >= 0 ? `rgba(18,122,99,${Math.abs(v)})` : `rgba(194,65,12,${Math.abs(v)})` }}>
                            {v.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          )}
          {data.substrate === "search" && data.axes && <SearchInspector data={data as SearchInspection} onOpenSpec={onOpenSpec} onRegister={onRegister} />}
          {data.substrate === "lenia" && data.value !== undefined && (
            <dl className="stats small" data-testid="inspector-field">
              <dt>{t("inspector.cell")}</dt>
              <dd className="mono">
                ({data.query?.x}, {data.query?.y}) · A = {data.value.toFixed(4)}
              </dd>
              <dt>{t("inspector.potential")}</dt>
              <dd className="mono">{data.potential?.toFixed(4)}</dd>
              <dt>{t("inspector.growth")}</dt>
              <dd className="mono">{data.growth?.toFixed(4)}</dd>
              <dt>{t("inspector.next_value")}</dt>
              <dd className="mono" data-testid="inspector-next">{data.next_value?.toFixed(4)}</dd>
              <dt>{t("inspector.patch_mass")}</dt>
              <dd className="mono">
                {data.patch_mass?.toFixed(3)} (r = {data.query?.radius})
              </dd>
            </dl>
          )}
          {data.substrate === "agent_world" && (
            <div data-testid="inspector-agent">
              {data.agent ? (
                <>
                  <p className="small mono">
                    #{data.agent.id} ← {data.agent.parent >= 0 ? `#${data.agent.parent}` : "—"} · {t("inspector.agent.age")} {data.agent.age} ·{" "}
                    {t("inspector.agent.energy")} {data.agent.energy.toFixed(2)} · {data.agent.heading_deg.toFixed(0)}° · |ΔW|{" "}
                    {data.agent.learned_delta_norm.toFixed(3)}
                  </p>
                  {data.agent.history.inputs.length > 0 && (
                    <>
                      <table className="small mono">
                        <tbody>
                          {data.agent.inputs.map((name, k) => (
                            <tr key={name}>
                              <td>{name}</td>
                              <td>{data.agent!.history.inputs[data.agent!.history.inputs.length - 1][k].toFixed(3)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <p className="small muted">{t("inspector.agent.probs")}</p>
                      <MultiLine
                        testId="inspector-agent-probs"
                        height={110}
                        series={data.agent.actions.map((name, k) => ({
                          label: name,
                          className: ["arm-a", "arm-b", "arm-c", "arm-d"][k],
                          points: data.agent!.history.probs.map((p, i) => [i, p[k]] as [number, number]),
                        }))}
                      />
                      <p className="small mono" data-testid="inspector-agent-actions">
                        {t("inspector.agent.actions")}: {data.agent.history.action.map((a) => data.agent!.actions[a][0]).join("")}
                      </p>
                    </>
                  )}
                </>
              ) : (
                <p className="muted small">{t("inspector.none")}</p>
              )}
            </div>
          )}
          {data.substrate === "protocell" && data.ledger && (
            <div data-testid="inspector-protocell">
              {data.protocell ? (
                <dl className="stats small">
                  <dt>{t("inspector.protocell.id")}</dt>
                  <dd className="mono">
                    #{data.protocell.id} · {t("inspector.protocell.born")} t={data.protocell.birth_tick} · r={data.protocell.radius.toFixed(2)}
                  </dd>
                  <dt>{t("inspector.protocell.contents")}</dt>
                  <dd className="mono">
                    {Object.entries(data.protocell.contents)
                      .map(([k, v]) => `${k} ${v}`)
                      .join(" · ")}
                  </dd>
                  <dt>{t("inspector.protocell.genome")}</dt>
                  <dd className="mono">
                    {Object.entries(data.protocell.genome)
                      .map(([k, v]) => `${k} ${v.toPrecision(3)}`)
                      .join(" · ")}
                  </dd>
                  <dt>{t("inspector.protocell.pressure")}</dt>
                  <dd className="mono">{data.protocell.pressure.toFixed(2)}</dd>
                  <dt>{t("inspector.protocell.ancestors")}</dt>
                  <dd className="mono" data-testid="inspector-ancestors">
                    {data.protocell.ancestors.length ? data.protocell.ancestors.join(" ← ") : "—"}
                  </dd>
                </dl>
              ) : (
                <p className="muted small">{t("inspector.none")}</p>
              )}
              <p className="small mono" data-testid="inspector-ledger">
                {t("inspector.protocell.ledger")}: {Object.entries(data.ledger)
                  .map(([k, v]) => `${k} ${v}`)
                  .join(" · ")}{" "}
                · residual {data.ledger_residual}
              </p>
            </div>
          )}
          {data.substrate === "pd_nca" && data.aliveness && (
            <div data-testid="inspector-pdnca">
              <p className="small">
                ({data.query?.x}, {data.query?.y}) · {t("inspector.dominant")}: {data.dominant === 0 ? t("inspector.sun") : `NCA ${(data.dominant ?? 1) - 1}`}
              </p>
              <p className="small mono">
                {t("inspector.aliveness")}: {data.aliveness.map((v) => v.toFixed(2)).join(" ")}
              </p>
              <p className="small mono">
                {t("inspector.territories")}: {(data.territories ?? []).map((v) => (v * 100).toFixed(1) + "%").join(" ")}
              </p>
              {data.interaction && (
                <table className="matrix small" data-testid="inspector-interaction">
                  <caption className="muted">{t("inspector.interaction")}</caption>
                  <tbody>
                    {data.interaction.map((row, a) => (
                      <tr key={a}>
                        <th>{a === 0 ? "☀" : a - 1}</th>
                        {row.map((v, b) => (
                          <td key={b} style={{ background: v == null ? undefined : v >= 0 ? `rgba(18,122,99,${Math.abs(v)})` : `rgba(194,65,12,${Math.abs(v)})` }}>
                            {v == null ? "·" : v.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
          {(data.substrate === "nca" || data.substrate === "nca_train") && data.rgba && (
            <dl className="stats small" data-testid="inspector-nca">
              <dt>{t("inspector.cell")}</dt>
              <dd className="mono">
                ({data.query?.x}, {data.query?.y}) · {data.alive ? t("inspector.alive") : t("inspector.dead")}
              </dd>
              <dt>RGBA</dt>
              <dd className="mono">{data.rgba.map((v) => v.toFixed(2)).join(" ")}</dd>
              {data.target_rgba && (
                <>
                  <dt>{t("inspector.target")}</dt>
                  <dd className="mono">{data.target_rgba.map((v) => v.toFixed(2)).join(" ")}</dd>
                </>
              )}
              <dt>{t("inspector.hidden")}</dt>
              <dd className="mono">{(data.hidden ?? []).map((v) => v.toFixed(1)).join(" ")}</dd>
              {data.last_step?.loss !== undefined && (
                <>
                  <dt>{t("metric.loss")}</dt>
                  <dd className="mono">
                    {data.last_step.loss.toExponential(3)} ({data.last_step.steps} steps) · {data.parameter_count} params
                  </dd>
                </>
              )}
            </dl>
          )}
          {data.substrate === "reaction_network" && data.reactions && (
            <div data-testid="inspector-reactions">
              {data.selected_species && (
                <p className="small">
                  <strong>{data.selected_species.name}</strong> = {data.selected_species.count}
                  {data.selected_species.constant && ` (${t("inspector.buffered")})`} · {t("inspector.consumed_by")}{" "}
                  {data.selected_species.consumed_by.join(", ") || "—"} · {t("inspector.produced_by")} {data.selected_species.produced_by.join(", ") || "—"}
                </p>
              )}
              <p className="small muted">t = {data.time?.toFixed(3)}</p>
              <table className="small mono">
                <thead>
                  <tr>
                    <th>id</th>
                    <th>{t("inspector.reaction")}</th>
                    <th>c</th>
                    <th>a</th>
                    <th>{t("inspector.fired")}</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {data.reactions.map((r) => (
                    <tr key={r.id} className={r.enabled ? "" : "muted"}>
                      <td title={r.label}>{r.id}</td>
                      <td>
                        {r.equation} → {r.products}
                      </td>
                      <td>{r.rate.toPrecision(3)}</td>
                      <td>{r.propensity.toPrecision(3)}</td>
                      <td>{r.fired}</td>
                      <td>
                        {onReactionToggle && (
                          <button className="small" data-testid={`toggle-${r.id}`} onClick={() => onReactionToggle(r.id, !r.enabled)}>
                            {r.enabled ? t("inspector.knock_out") : t("inspector.restore")}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {data.cycle_candidates && (
                <div className="small" data-testid="inspector-cycles">
                  <strong>{t("inspector.cycle_candidates")}</strong>{" "}
                  {data.cycle_candidates.cycles.length === 0 && t("inspector.no_cycles")}
                  <ul>
                    {data.cycle_candidates.cycles.map((c, i) => (
                      <li key={i} className="mono">
                        {c.species.join(" → ")} → {c.species[0]} ({c.reactions.join(", ")})
                      </li>
                    ))}
                  </ul>
                  {data.cycle_candidates.autocatalysis.length > 0 && (
                    <span className="muted">
                      {t("inspector.autocatalysis")}: {data.cycle_candidates.autocatalysis.map((a) => `${a.species} (${a.reaction})`).join(", ")}
                    </span>
                  )}
                  <br />
                  <span className="muted">{data.cycle_candidates.rule}</span>
                </div>
              )}
            </div>
          )}
          {data.substrate === "physarum" && data.trail !== undefined && (
            <dl className="stats small" data-testid="inspector-physarum">
              <dt>{t("inspector.trail")}</dt>
              <dd className="mono">{data.trail.toFixed(3)}</dd>
              <dt>{t("inspector.in_radius")}</dt>
              <dd className="mono">{data.agents_in_radius}</dd>
              {data.nearest_agent && (
                <>
                  <dt>{t("inspector.nearest_agent")}</dt>
                  <dd className="mono">
                    #{data.nearest_agent.id} · {data.nearest_agent.heading_deg.toFixed(0)}° · d={data.nearest_agent.distance.toFixed(1)}
                  </dd>
                  <dt>{t("inspector.sensors")}</dt>
                  <dd className="mono">
                    L {data.nearest_agent.sensors.left.toFixed(2)} · F {data.nearest_agent.sensors.front.toFixed(2)} · R{" "}
                    {data.nearest_agent.sensors.right.toFixed(2)}
                  </dd>
                  <dt>{t("inspector.decision")}</dt>
                  <dd className="mono" data-testid="inspector-decision">
                    {t(`inspector.decision.${data.nearest_agent.decision}`)}
                  </dd>
                </>
              )}
            </dl>
          )}
          {data.substrate === "gray_scott" && data.u !== undefined && (
            <dl className="stats small" data-testid="inspector-gray-scott">
              <dt>{t("inspector.cell")}</dt>
              <dd className="mono">
                ({data.query?.x}, {data.query?.y})
              </dd>
              <dt>u / v</dt>
              <dd className="mono">
                {data.u.toFixed(5)} / {data.v?.toFixed(5)}
              </dd>
              <dt>∇²u / ∇²v</dt>
              <dd className="mono">
                {data.laplacian_u?.toExponential(3)} / {data.laplacian_v?.toExponential(3)}
              </dd>
              <dt>uv²</dt>
              <dd className="mono">{data.reaction_uvv?.toExponential(3)}</dd>
              <dt>du/dt / dv/dt</dt>
              <dd className="mono" data-testid="inspector-rates">
                {data.du_dt?.toExponential(3)} / {data.dv_dt?.toExponential(3)}
              </dd>
            </dl>
          )}
          {data.substrate === "creature" && data.torso && (
            <div data-testid="inspector-creature">
              <dl className="stats small">
                <dt>{t("inspector.creature.position")}</dt>
                <dd className="mono">{data.torso.position.map((v) => v.toFixed(2)).join(", ")}</dd>
                <dt>{t("inspector.creature.velocity")}</dt>
                <dd className="mono">{data.torso.velocity.map((v) => v.toFixed(2)).join(", ")}</dd>
                <dt>{t("metric.upright")}</dt>
                <dd className="mono">{data.torso.upright.toFixed(3)}</dd>
                <dt>{t("inspector.creature.mass")}</dt>
                <dd className="mono">{data.mass_kg?.toFixed(1)} kg</dd>
              </dl>
              <table className="table small mono">
                <thead>
                  <tr>
                    <th>{t("inspector.creature.leg")}</th>
                    <th>{t("inspector.creature.hip")}</th>
                    <th>{t("inspector.creature.knee")}</th>
                    <th>{t("inspector.creature.ground")}</th>
                  </tr>
                </thead>
                <tbody>
                  {data.legs?.map((l) => (
                    <tr key={l.leg}>
                      <td>{l.leg}</td>
                      <td>
                        {l.hip_deg.toFixed(0)}° ({l.hip_ctrl.toFixed(2)})
                      </td>
                      <td>
                        {l.knee_deg.toFixed(0)}° ({l.knee_ctrl.toFixed(2)})
                      </td>
                      <td>{l.foot_on_ground ? "●" : "○"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {data.substrate === "lenia_mc" && data.values && (
            <div data-testid="inspector-lenia-mc">
              <dl className="stats small">
                <dt>{t("inspector.cell")}</dt>
                <dd className="mono">
                  ({data.query?.x}, {data.query?.y})
                </dd>
                <dt>{t("inspector.lenia_mc.values")}</dt>
                <dd className="mono">{data.values.map((v) => v.toFixed(3)).join(" / ")}</dd>
                <dt>{t("inspector.lenia_mc.next")}</dt>
                <dd className="mono">{data.next_values?.map((v) => v.toFixed(3)).join(" / ")}</dd>
              </dl>
              <p className="small muted">{t("inspector.lenia_mc.kernels")}</p>
              <table className="table small mono">
                <tbody>
                  {data.kernels?.map((k) => (
                    <tr key={k.index}>
                      <td>
                        #{k.index} {"RGB"[k.c0]}→{"RGB"[k.c1]}
                      </td>
                      <td>U {k.potential.toFixed(3)}</td>
                      <td>G {k.growth.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {data.substrate === "flow_lenia" && data.channels && (
            <>
              <p className="small">
                ({data.query?.x}, {data.query?.y}) · {t("inspector.patch_mass")} {data.patch_mass?.toFixed(3)} (r = {data.query?.radius}) · R ={" "}
                {typeof data.rule === "object" ? data.rule.R.toFixed(2) : "—"} · {t("inspector.kernels")}{" "}
                {typeof data.rule === "object" ? data.rule.kernels.length : "—"}
              </p>
              <table className="small mono" data-testid="inspector-channels">
                <thead>
                  <tr>
                    <th>ch</th>
                    <th>A</th>
                    <th>{t("inspector.affinity")}</th>
                    <th>α</th>
                    <th>{t("inspector.displacement")}</th>
                  </tr>
                </thead>
                <tbody>
                  {data.channels.map((c, i) => (
                    <tr key={i}>
                      <td>{i}</td>
                      <td>{c.value.toFixed(3)}</td>
                      <td>{c.affinity.toFixed(3)}</td>
                      <td>{c.alpha.toFixed(2)}</td>
                      <td>
                        ({c.displacement.x.toFixed(2)}, {c.displacement.y.toFixed(2)})
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          {data.substrate === "connectome" && (
            <div data-testid="inspector-connectome">
              {data.port ? (
                <dl className="stats small">
                  <dt>{t("connectome.col.port")}</dt>
                  <dd className="mono">{data.port}</dd>
                  <dt>{t("connectome.col.count")}</dt>
                  <dd className="mono">{data.count}</dd>
                  <dt>{t("metric.spikes_tick")}</dt>
                  <dd className="mono">{data.spikes_tick}</dd>
                  <dt>{t("metric.total_spikes")}</dt>
                  <dd className="mono">{data.total_spikes}</dd>
                  <dt>{t("connectome.inspector.mean_rate")}</dt>
                  <dd className="mono">{(data.mean_rate_hz ?? 0).toFixed(2)} Hz</dd>
                  <dt>{t("metric.mean_v_mv")}</dt>
                  <dd className="mono">{(data.mean_v_mv ?? 0).toFixed(2)} mV</dd>
                </dl>
              ) : (
                <dl className="stats small">
                  <dt>{t("connectome.col.cell_type")}</dt>
                  <dd className="mono">
                    {data.cell_type || "—"} (#{data.cell_index})
                  </dd>
                  <dt>{t("connectome.col.source_id")}</dt>
                  <dd className="mono">{data.source_id}</dd>
                  <dt>{t("connectome.col.region")}</dt>
                  <dd className="mono">
                    {data.region} {data.side ? `(${data.side})` : ""}
                  </dd>
                  <dt>{t("connectome.col.role")}</dt>
                  <dd>{data.role ? t(`connectome.role.${data.role}`) : "—"}</dd>
                  <dt>{t("connectome.col.nt")}</dt>
                  <dd className="mono">{data.neurotransmitter || "—"}</dd>
                  <dt>v / g</dt>
                  <dd className="mono">
                    {(data.v_mv ?? 0).toFixed(2)} / {(data.g_mv ?? 0).toFixed(2)} mV
                  </dd>
                  <dt>{t("connectome.stim_rate")}</dt>
                  <dd className="mono">{(data.stim_rate_hz ?? 0).toFixed(1)} Hz</dd>
                  <dt>{t("connectome.inspector.silenced")}</dt>
                  <dd>{data.silenced ? t("connectome.yes") : t("connectome.no")}</dd>
                  <dt>{t("metric.total_spikes")}</dt>
                  <dd className="mono">
                    {data.total_spikes} (tick: {data.last_tick_spikes})
                  </dd>
                </dl>
              )}
            </div>
          )}
          {data.substrate === "boids" && (
            <>
              <p className="small">
                {t("inspector.in_radius")}: {data.count_in_radius} (r = {data.query && f(data.query.radius)})
              </p>
              {data.nearest ? (
                <dl className="stats small" data-testid="inspector-boid">
                  <dt>id</dt>
                  <dd className="mono">{data.nearest.id}</dd>
                  <dt>{t("inspector.position")}</dt>
                  <dd className="mono">
                    ({f(data.nearest.x)}, {f(data.nearest.y)})
                  </dd>
                  <dt>{t("inspector.speed")}</dt>
                  <dd className="mono">{f(data.nearest.speed)}</dd>
                  <dt>{t("inspector.heading")}</dt>
                  <dd className="mono">{data.nearest.heading_deg.toFixed(0)}°</dd>
                  <dt>{t("inspector.neighbors")}</dt>
                  <dd className="mono">{data.nearest.steering.neighbors}</dd>
                  {(["separation", "alignment", "cohesion"] as const).map((k) => (
                    <div key={k} className="contents">
                      <dt>{t(`inspector.${k}`)}</dt>
                      <dd className="mono" data-testid={`inspector-${k}`}>
                        |{f(data.nearest!.steering[k].magnitude)}|
                      </dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p className="muted small">{t("inspector.none")}</p>
              )}
              <p className="muted small">{t("inspector.steering_note")}</p>
            </>
          )}
        </>
      )}
    </div>
  );
}
