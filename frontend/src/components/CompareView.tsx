import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import { useT } from "../i18n";
import { MultiLine, type Series } from "./MultiLine";
import { RunPane } from "./RunPane";

const ARM_CLASSES = ["arm-a", "arm-b", "arm-c", "arm-d"];

export function CompareView({ comparisonId }: { comparisonId: string }) {
  const t = useT();
  const qc = useQueryClient();
  const [brushArm, setBrushArm] = useState<string | null>(null);
  const [brushMode, setBrushMode] = useState<"draw" | "erase">("draw");
  const [radius, setRadius] = useState(1);
  const [stepCount, setStepCount] = useState(10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const summary = useQuery({ queryKey: ["comparison", comparisonId], queryFn: () => api.comparison(comparisonId), refetchInterval: 3000 });
  const report = useQuery({ queryKey: ["comparison-report", comparisonId], queryFn: () => api.comparisonReport(comparisonId) });

  // Debounced refresh of the (checkpoint-based) report when any arm writes a checkpoint.
  const timer = useRef<number | undefined>(undefined);
  const refreshReport = useCallback(() => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => void qc.invalidateQueries({ queryKey: ["comparison-report", comparisonId] }), 400);
  }, [qc, comparisonId]);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const members = summary.data?.members ?? [];
  const all = async (fn: (runId: string) => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await Promise.all(members.map((m) => fn(m.run.id)));
      await qc.invalidateQueries({ queryKey: ["comparison", comparisonId] });
      refreshReport();
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!summary.data) return <p>{summary.isError ? t("common.error") : t("common.loading")}</p>;
  const states = members.map((m) => m.run.state);
  const allPaused = states.every((s) => s === "paused");
  const anyRunning = states.some((s) => s === "running" || s === "resuming");
  const div = report.data?.divergence;

  const hammingSeries: Series[] = members.slice(1).map((m, i) => ({
    label: `${m.arm} vs ${members[0].arm}`,
    className: ARM_CLASSES[i + 1],
    points: (div?.points ?? []).map((p) => [p.tick, p.hamming_to_first[i + 1]]),
  }));
  const aliveSeries: Series[] = members.map((m, i) => ({
    label: m.arm,
    className: ARM_CLASSES[i],
    points: (report.data?.metrics[m.run.id] ?? []).map((r) => [r.tick, r.values.alive_cell_count ?? null]),
  }));

  return (
    <div className="compare">
      <div className="lab-head">
        <h2 data-testid="comparison-name">{summary.data.name}</h2>
        <span className="badge">{summary.data.design}</span>
        {summary.data.parent_run_id && (
          <a className="small" href={`#/lab/${summary.data.parent_run_id}`}>
            {t("compare.parent")}: <span className="mono">{summary.data.parent_run_id}</span>
          </a>
        )}
      </div>
      <p className="muted small">{t("compare.note")}</p>
      <div className="toolbar">
        <button data-testid="btn-step-all" disabled={busy || !allPaused} onClick={() => all((id) => api.control(id, "step", { count: stepCount }))}>
          ⏭ {t("compare.step_all")}
        </button>
        <input type="number" min={1} max={10000} value={stepCount} onChange={(e) => setStepCount(Number(e.target.value))} style={{ width: "5rem" }} aria-label={t("compare.step_count")} />
        <button data-testid="btn-play-all" disabled={busy || !allPaused} onClick={() => all((id) => api.control(id, "resume"))}>
          ▶ {t("compare.play_all")}
        </button>
        <button data-testid="btn-pause-all" disabled={busy || !anyRunning} onClick={() => all((id) => api.control(id, "pause"))}>
          ❚❚ {t("compare.pause_all")}
        </button>
        <button data-testid="btn-checkpoint-all" disabled={busy || !allPaused} onClick={() => all((id) => api.control(id, "checkpoints"))}>
          💾 {t("compare.checkpoint_all")}
        </button>
      </div>
      <div className="toolbar">
        <strong>{t("lab.brush")}:</strong>
        <label className="inline">
          <input type="radio" name="cmp-brush" checked={brushArm === null} onChange={() => setBrushArm(null)} />
          {t("lab.brush.off")}
        </label>
        {members.map((m) => (
          <label key={m.arm} className="inline">
            <input type="radio" name="cmp-brush" data-testid={`brush-arm-${m.arm}`} checked={brushArm === m.arm} onChange={() => setBrushArm(m.arm)} />
            {m.arm}
          </label>
        ))}
        <select value={brushMode} onChange={(e) => setBrushMode(e.target.value as "draw" | "erase")}>
          <option value="draw">{t("lab.brush.draw")}</option>
          <option value="erase">{t("lab.brush.erase")}</option>
        </select>
        <label className="inline">
          {t("lab.brush.radius")}
          <input type="range" min={0} max={8} value={radius} onChange={(e) => setRadius(Number(e.target.value))} />
          {radius}
        </label>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="panes">
        {members.map((m) => (
          <RunPane
            key={m.run.id}
            runId={m.run.id}
            arm={m.arm}
            brushMode={brushArm === m.arm ? brushMode : "off"}
            brushRadius={radius}
            onCheckpoint={refreshReport}
          />
        ))}
      </div>
      <div className="grid2">
        <div className="panel">
          <h3>{t("compare.divergence")}</h3>
          <p className="muted small">{t("compare.divergence_note")}</p>
          <MultiLine series={hammingSeries} testId="divergence-chart" />
          <div className="table-wrap" style={{ maxHeight: "16rem", overflowY: "auto" }}>
            <table className="table small" data-testid="divergence-table">
              <thead>
                <tr>
                  <th>tick</th>
                  {members.slice(1).map((m) => (
                    <th key={m.arm}>Hamming {m.arm}</th>
                  ))}
                  {members.map((m) => (
                    <th key={m.arm}>alive {m.arm}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(div?.points ?? []).slice().reverse().map((p) => (
                  <tr key={p.tick}>
                    <td className="mono">{p.tick}</td>
                    {p.hamming_to_first.slice(1).map((h, i) => (
                      <td key={i} className="mono">{h ?? "—"}</td>
                    ))}
                    {p.alive.map((a, i) => (
                      <td key={`a${i}`} className="mono">{a}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">
            {t("compare.common_ticks")}: {div?.common_ticks ?? 0}
          </p>
        </div>
        <div className="panel">
          <h3>{t("metric.alive_cell_count")}</h3>
          <MultiLine series={aliveSeries} testId="alive-chart" />
          <button onClick={() => void qc.invalidateQueries({ queryKey: ["comparison-report", comparisonId] })}>{t("compare.refresh")}</button>
        </div>
      </div>
    </div>
  );
}

export function ComparisonList({ onOpen }: { onOpen: (id: string) => void }) {
  const t = useT();
  const list = useQuery({ queryKey: ["comparisons"], queryFn: api.comparisons, refetchInterval: 5000 });
  if (!list.data?.length) return <p className="muted">{t("compare.empty")}</p>;
  return (
    <ul className="list" data-testid="comparison-list">
      {list.data.map((c) => (
        <li key={c.id}>
          <button className="ghost" data-testid={`open-${c.id}`} onClick={() => onOpen(c.id)}>
            {c.name}
          </button>{" "}
          <span className="muted small">
            {c.design} · {new Date(c.created_at * 1000).toLocaleString()}
          </span>
        </li>
      ))}
    </ul>
  );
}
