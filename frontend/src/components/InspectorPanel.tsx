import { useT } from "../i18n";

// Read-only inspection results (conway cell / boids neighbourhood).
type Vec = { x: number; y: number; magnitude: number };
type Boid = { id: number; x: number; y: number; vx: number; vy: number; speed: number; heading_deg: number; distance: number };

export interface Inspection {
  substrate: string;
  tick: number;
  source?: { method: string; base_tick?: number; interventions_replayed?: number };
  // conway
  cell?: { alive: boolean; live_neighbors: number; next_tick: string };
  rule?: string;
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
  self_pair_test?: { description: string; ops: number; first_half_unchanged: boolean; second_half_equals_program: boolean };
  // particle_life
  type_counts_in_radius?: number[];
  matrix?: number[][];
  nearest_particle?: { id: number; type: number; speed: number; neighbors_by_type: number[] } | null;
}

const f = (v: number, d = 3) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(d));

export function InspectorPanel({ data, error }: { data: Inspection | null; error: string | null }) {
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
                <dd className="mono">{data.rule}</dd>
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
