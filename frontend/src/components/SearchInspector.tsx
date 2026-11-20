// Inspection of a search run: the clicked candidate (or the best one), its parameters, scores
// and a button that opens it as an ordinary run of the target world (a re-test outside the
// search, with the full spec).
import { useState } from "react";
import { useT } from "../i18n";

type Register = (candidate: number, claimKind: string) => void;

export interface SearchCandidate {
  index: number;
  generation: number;
  params: Record<string, number>;
  score: number;
  viable_fraction: number;
  alive_fraction: number;
  speed: number;
  mass?: number;
  holdout_score: number | null;
  lineage?: Array<{ index: number; generation: number; score: number }>;
  lineage_depth?: number;
  out_of_box: boolean;
  spec: Record<string, unknown> | null;
}

export interface SearchInspection {
  algorithm: string;
  objective: string;
  axes: string[];
  evaluated: number;
  search_seeds: number[];
  holdout_seeds: number[];
  region?: "map" | "archive" | "best";
  qd?: { descriptors: Array<{ name: string; low: number; high: number; bins: number }>; filled: number; cells: number; archive_version: number };
  archive_cell?: Record<string, number | number[]>;
  candidate?: SearchCandidate;
  best?: SearchCandidate;
  cma?: { mean: Record<string, number>; sigma: number };
}

const short = (path: string) => path.split(".").pop() ?? path;
const num = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(4));

function Candidate({ c, title, testId, onOpenSpec, onRegister }: { c: SearchCandidate; title: string; testId: string; onOpenSpec?: (spec: Record<string, unknown>) => void; onRegister?: Register }) {
  const t = useT();
  const [claim, setClaim] = useState("moving_localized");
  return (
    <div data-testid={testId}>
      <h4 className="small">
        {title} <span className="muted">#{c.index} · {t("search.generation")} {c.generation}</span>
      </h4>
      <dl className="stats small">
        {Object.entries(c.params).map(([k, v]) => (
          <div key={k} style={{ display: "contents" }}>
            <dt className="mono">{short(k)}</dt>
            <dd className="mono">{num(v)}</dd>
          </div>
        ))}
        <dt>{t("search.score")}</dt>
        <dd className="mono" data-testid={`${testId}-score`}>{c.score.toFixed(4)}</dd>
        <dt>{t("search.holdout")}</dt>
        <dd className="mono">{c.holdout_score === null ? "—" : c.holdout_score.toFixed(4)}</dd>
        <dt>{t("search.viable")}</dt>
        <dd className="mono">{(c.viable_fraction * 100).toFixed(0)}%</dd>
        <dt>{t("search.speed")}</dt>
        <dd className="mono">{c.speed.toFixed(4)}</dd>
        {c.mass !== undefined && Number.isFinite(c.mass) && (
          <>
            <dt>{t("search.mass")}</dt>
            <dd className="mono">{c.mass.toFixed(1)}</dd>
          </>
        )}
        <dt>{t("metric.alive_fraction")}</dt>
        <dd className="mono">{c.alive_fraction.toFixed(3)}</dd>
      </dl>
      {c.lineage !== undefined && (
        <p className="small muted" data-testid={`${testId}-lineage`}>
          {t("search.lineage")}:{" "}
          {c.lineage.length === 0
            ? t("search.no_parent")
            : c.lineage.map((a) => `#${a.index} (${t("search.generation")} ${a.generation}, ${a.score.toFixed(3)})`).join(" ← ") +
              ((c.lineage_depth ?? 0) > c.lineage.length ? " ← …" : "")}
        </p>
      )}
      {c.out_of_box && <p className="muted small">{t("search.out_of_box")}</p>}
      {onOpenSpec && c.spec && (
        <button type="button" className="small" data-testid={`${testId}-open`} onClick={() => onOpenSpec(c.spec!)}>
          {t("search.open_as_run")}
        </button>
      )}
      {onRegister && c.spec && c.viable_fraction > 0 && (
        <div className="register-row">
          <select value={claim} onChange={(e) => setClaim(e.target.value)} data-testid={`${testId}-claim`}>
            <option value="moving_localized">{t("discovery.claim.moving_localized")}</option>
            <option value="faster_than_base">{t("discovery.claim.faster_than_base")}</option>
          </select>
          <button type="button" className="small" data-testid={`${testId}-register`} onClick={() => onRegister(c.index, claim)}>
            {t("discovery.register")}
          </button>
        </div>
      )}
    </div>
  );
}

export function SearchInspector({ data, onOpenSpec, onRegister }: { data: SearchInspection; onOpenSpec?: (spec: Record<string, unknown>) => void; onRegister?: Register }) {
  const t = useT();
  return (
    <div className="search-inspector" data-testid="inspector-search">
      <p className="small muted">
        {t(`form.search.algorithm.${data.algorithm}`)} · {data.evaluated} {t("search.candidates")} · {t("search.axes")}:{" "}
        <span className="mono">
          {t("search.axis_x")} {short(data.axes[0])}
          {data.axes[1] ? ` / ${t("search.axis_y")} ${short(data.axes[1])}` : ""}
        </span>
      </p>
      {data.qd && (
        <p className="small muted" data-testid="search-qd">
          {t("search.archive_summary").replace("{filled}", String(data.qd.filled)).replace("{cells}", String(data.qd.cells))} (
          {data.qd.descriptors.map((d) => `${d.name} ${d.low}–${d.high}`).join(" × ")})
        </p>
      )}
      {data.region === "map" && !data.candidate && <p className="muted small">{t("search.no_candidate")}</p>}
      {data.region === "archive" && data.archive_cell && (
        <p className="small mono" data-testid="search-archive-cell">
          {t("search.archive_cell")}:{" "}
          {Object.entries(data.archive_cell)
            .filter(([, v]) => Array.isArray(v))
            .map(([k, v]) => `${k} ${(v as number[])[0].toFixed(2)}–${(v as number[])[1].toFixed(2)}`)
            .join(", ")}
          {!data.candidate && <span className="muted"> · {t("search.archive_empty")}</span>}
        </p>
      )}
      {data.candidate && <Candidate c={data.candidate} title={t("search.clicked")} testId="search-candidate" onOpenSpec={onOpenSpec} onRegister={onRegister} />}
      {data.best && <Candidate c={data.best} title={t("search.best")} testId="search-best" onOpenSpec={onOpenSpec} onRegister={onRegister} />}
      {data.cma && (
        <p className="small muted mono">
          CMA-ES mean: {Object.entries(data.cma.mean).map(([k, v]) => `${short(k)}=${num(v)}`).join(", ")} · σ={data.cma.sigma.toFixed(3)}
        </p>
      )}
      <p className="small muted">
        {t("search.seeds_note")} search {data.search_seeds.length} / holdout {data.holdout_seeds.length}
      </p>
    </div>
  );
}
