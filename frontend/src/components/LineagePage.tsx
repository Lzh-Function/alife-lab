// Lineage view of a search run: every candidate as a dot (x = generation, y = a chosen
// quantity), parent -> child links, and the ancestry of the selected candidate highlighted.
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import { CPU_PROFILE } from "../lib/compute";
import { useT } from "../i18n";
import { Candidate, type SearchCandidate } from "./SearchInspector";

const W = 900;
const H = 420;
const M = { left: 48, right: 16, top: 16, bottom: 36 };
const VIRIDIS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];

function colour(v: number): string {
  const x = Math.max(0, Math.min(1, v)) * (VIRIDIS.length - 1);
  return VIRIDIS[Math.round(x)];
}

const short = (path: string) => {
  const k = /^dynamics\.kernels\.(\d+)\.(\w+)$/.exec(path);
  return k ? `k${k[1]}.${k[2]}` : (path.split(".").pop() ?? path);
};

export function LineagePage({ runId }: { runId: string }) {
  const t = useT();
  const q = useQuery({ queryKey: ["lineage", runId], queryFn: () => api.lineage(runId) });
  const [selected, setSelected] = useState<number | null>(null);
  const [axisChoice, setAxis] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const data = q.data;
  // Default y axis: a behaviour (e.g. speed) when the run has QD descriptors, else the score.
  const axis = axisChoice ?? (data?.descriptors ? Object.keys(data.descriptors)[0] : "score");
  const sel = selected ?? data?.best.index ?? null;
  const detail = useQuery({
    queryKey: ["candidate", runId, sel],
    queryFn: () => api.candidate(runId, sel as number),
    enabled: sel !== null,
  });

  const table = data?.lineage_table;
  const n = table?.generation.length ?? 0;
  const yValues = useMemo(() => {
    if (!table) return [];
    if (axis === "u0" || axis === "u1") return table.u.map((u) => u[axis === "u0" ? 0 : 1] ?? 0.5);
    if (axis === "score") {
      const max = Math.max(...table.score, 1e-9);
      const min = Math.min(...table.score, 0);
      return table.score.map((s) => (s - min) / (max - min || 1));
    }
    const d = data?.descriptors?.[axis] ?? [];
    const max = Math.max(...d, 1e-9);
    return d.map((v) => v / max);
  }, [table, axis, data]);

  const ancestry = useMemo(() => {
    const out: number[] = [];
    if (!table || sel === null) return out;
    let i = sel;
    while (i >= 0 && out.length < 500) {
      out.push(i);
      i = table.parent[i];
    }
    return out;
  }, [table, sel]);

  if (q.error) return <p className="error">{q.error instanceof ApiError ? q.error.message : String(q.error)}</p>;
  if (!data || !table) return <p className="muted">…</p>;

  const gens = Math.max(...table.generation, 1);
  const px = (g: number) => M.left + ((g - 1) / Math.max(1, gens - 1)) * (W - M.left - M.right);
  const py = (v: number) => H - M.bottom - v * (H - M.top - M.bottom);
  const maxScore = Math.max(...table.score, 1e-9);
  const withParent = table.parent.filter((p) => p >= 0).length;
  const children = sel === null ? 0 : table.parent.filter((p) => p === sel).length;
  const axes = [
    { id: "u0", label: `${t("lineage.axis.param")} ${short(data.axis_labels[0] ?? "")}` },
    ...(data.axis_labels[1] ? [{ id: "u1", label: `${t("lineage.axis.param")} ${short(data.axis_labels[1])}` }] : []),
    { id: "score", label: t("search.score") },
    ...Object.keys(data.descriptors ?? {}).map((d) => ({ id: d, label: t(`lineage.axis.${d}`) })),
  ];

  const openSpec = async (spec: Record<string, unknown>) => {
    try {
      const run = await api.create({ spec, profile_id: CPU_PROFILE, ticks_per_second: 30, auto_start: true, start_paused: true });
      window.location.hash = `/lab/${run.id}`;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  };

  return (
    <div className="lineage-page" data-testid="lineage-page">
      <p>
        <a href={`#/lab/${runId}`}>← {t("lineage.back")}</a>
      </p>
      <h2>
        {t("lineage.title")} <span className="muted small mono">{runId}</span>
      </h2>
      <div className="substrate-intro">
        <p>{t("lineage.intro")}</p>
        <p className="small muted">
          {t(`form.search.algorithm.${data.algorithm}`)} · {n} {t("search.candidates")} · {t("lineage.with_parent")} {withParent}
          {withParent === 0 && <> · {t("lineage.no_parents")}</>}
        </p>
      </div>
      <div className="lineage-layout">
        <div className="panel">
          <label className="small">
            {t("lineage.y_axis")}{" "}
            <select value={axis} onChange={(e) => setAxis(e.target.value)} data-testid="lineage-axis">
              {axes.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.label}
                </option>
              ))}
            </select>
          </label>
          <svg viewBox={`0 0 ${W} ${H}`} className="lineage-svg" data-testid="lineage-svg" role="img" aria-label={t("lineage.title")}>
            <line x1={M.left} y1={H - M.bottom} x2={W - M.right} y2={H - M.bottom} className="axis" />
            <line x1={M.left} y1={M.top} x2={M.left} y2={H - M.bottom} className="axis" />
            <text x={(W + M.left) / 2} y={H - 8} className="axis-label" textAnchor="middle">
              {t("search.generation")} (1 … {gens})
            </text>
            {table.parent.map((p, i) =>
              p >= 0 ? <line key={`e${i}`} x1={px(table.generation[p])} y1={py(yValues[p])} x2={px(table.generation[i])} y2={py(yValues[i])} className="edge" /> : null,
            )}
            {ancestry.slice(0, -1).map((i, k) => {
              const p = ancestry[k + 1];
              return <line key={`a${i}`} x1={px(table.generation[p])} y1={py(yValues[p])} x2={px(table.generation[i])} y2={py(yValues[i])} className="edge-highlight" />;
            })}
            {table.generation.map((g, i) => (
              <circle
                key={i}
                cx={px(g)}
                cy={py(yValues[i])}
                r={i === sel ? 6 : ancestry.includes(i) ? 4.5 : table.viable[i] > 0 ? 3 : 2}
                fill={table.viable[i] > 0 ? colour(table.score[i] / maxScore) : "#7a8190"}
                className={i === sel ? "node selected" : "node"}
                data-testid={`node-${i}`}
                onClick={() => setSelected(i)}
              >
                <title>
                  #{i} · {t("search.generation")} {g} · {t("search.score")} {table.score[i].toFixed(4)}
                </title>
              </circle>
            ))}
          </svg>
          <p className="small muted">{t("lineage.legend")}</p>
        </div>
        <aside className="panel" data-testid="lineage-detail">
          {sel !== null && (
            <p className="small">
              {t("lineage.depth")} {ancestry.length - 1} · {t("lineage.children")} {children}
            </p>
          )}
          {detail.data?.candidate ? (
            <Candidate c={detail.data.candidate as SearchCandidate} title={t("lineage.selected")} testId="lineage-candidate" onOpenSpec={openSpec} />
          ) : (
            <p className="muted small">…</p>
          )}
          {error && <p className="error small">{error}</p>}
        </aside>
      </div>
    </div>
  );
}
