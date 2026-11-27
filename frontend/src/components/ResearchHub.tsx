// Unified GUI Research Hub closing all CLI-only research workflows:
// 1. Connectome dataset manager (list / verify SHA-256 / summary & reference checks / fetch).
// 2. All 21 research assays, comparisons, chimeras, and cultural phase diagrams with rich viewers.
// 3. Persistent async job history with live polling and cancel support.
import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  ApiError,
  type ConnectomeDatasetInfo,
  type ConnectomeDatasetSummary,
  type ResearchJobDetail,
  type ResearchToolSpec,
} from "../lib/api";
import { useT } from "../i18n";

type HubTab = "tools" | "datasets" | "jobs";
type Category = "all" | ResearchToolSpec["category"];

const CATEGORIES: Category[] = [
  "all",
  "species_chimera",
  "circuits_plasticity",
  "learning_fit",
  "embodied_social",
  "cultural",
];

function fmtBytes(n: number): string {
  if (n <= 0) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// --------------------------------------------------------------------------- //
// Rich domain-specific result renderers
// --------------------------------------------------------------------------- //

function CompareSpeciesView({ data }: { data: Record<string, unknown> }) {
  const t = useT();
  const topology = (data.topology as Array<Record<string, unknown>> | undefined) ?? [];
  const probes = (data.sensorimotor_probe as Array<Record<string, unknown>> | undefined) ?? [];
  return (
    <div data-testid="result-compare-species">
      <h4>{t("research.view.topology_table")}</h4>
      <div className="table-wrap">
        <table className="table small mono">
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Cells</th>
              <th>Edges</th>
              <th>Synapses</th>
              <th>Density</th>
              <th>Out-Deg</th>
              <th>Hub 1%</th>
              <th>Recip.</th>
              <th>Gap %</th>
            </tr>
          </thead>
          <tbody>
            {topology.map((row) => (
              <tr key={String(row.dataset_id)}>
                <td>{String(row.dataset_id)}</td>
                <td>{Number(row.num_cells ?? 0).toLocaleString()}</td>
                <td>{Number(row.num_edges ?? 0).toLocaleString()}</td>
                <td>{Number(row.total_synapses ?? 0).toLocaleString()}</td>
                <td>{Number(row.density ?? 0).toFixed(5)}</td>
                <td>{Number(row.mean_out_degree ?? 0).toFixed(2)}</td>
                <td>{(Number(row.top1pct_hub_synapse_share ?? 0) * 100).toFixed(1)}%</td>
                <td>{Number(row.reciprocity ?? 0).toFixed(3)}</td>
                <td>{(Number(row.gap_junction_edge_fraction ?? 0) * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {probes.length > 0 && (
        <>
          <h4 style={{ marginTop: "0.8rem" }}>{t("research.view.physiology_table")}</h4>
          <div className="table-wrap">
            <table className="table small mono">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Mode (w_syn)</th>
                  <th>Sensory → Motor</th>
                  <th>Reference</th>
                  <th>Shuffled</th>
                  <th>Delta</th>
                  <th>Ratio</th>
                </tr>
              </thead>
              <tbody>
                {probes.map((p) => {
                  const delta = Number(p.wiring_delta ?? 0);
                  return (
                    <tr key={String(p.dataset_id)}>
                      <td>{String(p.dataset_id)}</td>
                      <td>
                        {String(p.dynamics_mode)} ({String(p.w_syn_mv)})
                      </td>
                      <td>
                        {String(p.sensory_port)} → {String(p.motor_port)}
                      </td>
                      <td>
                        {Number(p.reference_motor_response ?? 0).toFixed(2)} {String(p.unit ?? "")}
                      </td>
                      <td>
                        {Number(p.shuffled_motor_response ?? 0).toFixed(2)} {String(p.unit ?? "")}
                      </td>
                      <td style={{ color: delta >= 0 ? "var(--accent)" : "var(--danger)" }}>
                        {delta >= 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)}
                      </td>
                      <td>{Number(p.selectivity_ratio ?? 0).toFixed(2)}×</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function ChimeraView({ data }: { data: Record<string, unknown> }) {
  const t = useT();
  const prov = (data.provenance_breakdown as Record<string, number> | undefined) ?? {};
  const arms = (data.arms as Record<string, Record<string, unknown>> | undefined) ?? null;
  const checks = (data.checks as Record<string, boolean> | undefined) ?? null;
  const colors = ["#127a63", "#2563eb", "#d97706", "#7c3aed", "#dc2626"];
  const provEntries = Object.entries(prov);
  const provSum = provEntries.reduce((s, [, v]) => s + Math.max(0, Number(v)), 0) || 1;

  return (
    <div data-testid="result-chimera">
      {typeof data.framing_note === "string" && <p className="notice small">{data.framing_note}</p>}
      {provEntries.length > 0 && (
        <div style={{ margin: "0.6rem 0" }}>
          <div className="small muted" style={{ marginBottom: "0.25rem" }}>
            {t("research.view.provenance_bar")}
          </div>
          <div
            style={{
              display: "flex",
              height: "14px",
              borderRadius: "6px",
              overflow: "hidden",
              border: "1px solid var(--border)",
            }}
          >
            {provEntries.map(([k, v], i) => (
              <div
                key={k}
                title={`${k}: ${(Number(v) * 100).toFixed(1)}%`}
                style={{
                  width: `${(Math.max(0, Number(v)) / provSum) * 100}%`,
                  background: colors[i % colors.length],
                }}
              />
            ))}
          </div>
          <div className="toolbar small mono" style={{ marginTop: "0.3rem", gap: "0.8rem" }}>
            {provEntries.map(([k, v], i) => (
              <span key={k}>
                <span
                  style={{
                    display: "inline-block",
                    width: "8px",
                    height: "8px",
                    borderRadius: "2px",
                    background: colors[i % colors.length],
                    marginRight: "4px",
                  }}
                />
                {k}: {(Number(v) * 100).toFixed(1)}%
              </span>
            ))}
          </div>
        </div>
      )}
      {checks && (
        <div className="toolbar small" style={{ margin: "0.5rem 0", flexWrap: "wrap" }}>
          {Object.entries(checks).map(([k, ok]) => (
            <span
              key={k}
              className="badge"
              style={{
                background: ok ? "var(--accent)" : "var(--danger)",
                color: "#fff",
              }}
            >
              {ok ? "PASS" : "FAIL"} · {k}
            </span>
          ))}
        </div>
      )}
      {arms && (
        <div className="table-wrap" style={{ marginTop: "0.5rem" }}>
          <table className="table small mono">
            <thead>
              <tr>
                <th>Arm</th>
                {Object.keys(Object.values(arms)[0] ?? {})
                  .filter((k) => typeof (Object.values(arms)[0] ?? {})[k] !== "object")
                  .map((k) => (
                    <th key={k}>{k}</th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(arms).map(([armName, row]) => (
                <tr key={armName}>
                  <td>{armName}</td>
                  {Object.entries(row)
                    .filter(([, v]) => typeof v !== "object")
                    .map(([k, v]) => (
                      <td key={k}>{typeof v === "number" ? v.toFixed(4) : String(v)}</td>
                    ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function PhaseDiagramView({ data }: { data: Record<string, unknown> }) {
  const t = useT();
  const [metric, setMetric] = useState<"mean_utility" | "mean_depth" | "mean_retention" | "survival_rate">("mean_utility");
  const cells = (data.cells as Array<Record<string, unknown>> | undefined) ?? [];
  const periods = useMemo(
    () => Array.from(new Set(cells.map((c) => Number(c.regime_period)))).sort((a, b) => a - b),
    [cells],
  );
  const costs = useMemo(
    () => Array.from(new Set(cells.map((c) => Number(c.transmission_cost)))).sort((a, b) => a - b),
    [cells],
  );
  const cellMap = useMemo(() => {
    const m = new Map<string, Record<string, unknown>>();
    for (const c of cells) m.set(`${c.regime_period}:${c.transmission_cost}`, c);
    return m;
  }, [cells]);
  const vals = cells.map((c) => Number(c[metric] ?? 0)).filter(Number.isFinite);
  const vmin = vals.length ? Math.min(...vals) : 0;
  const vmax = vals.length ? Math.max(...vals) : 1;

  const cellColor = (v: number) => {
    const tNorm = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5;
    const hue = Math.round(190 - tNorm * 145);
    const light = Math.round(24 + tNorm * 32);
    return `hsl(${hue}, 68%, ${light}%)`;
  };

  return (
    <div data-testid="result-phase-diagram">
      <div className="toolbar small" style={{ marginBottom: "0.5rem" }}>
        <span>{t("research.view.phase_metric")}:</span>
        {(["mean_utility", "mean_depth", "mean_retention", "survival_rate"] as const).map((m) => (
          <button
            key={m}
            type="button"
            className={metric === m ? "small" : "small ghost"}
            onClick={() => setMetric(m)}
          >
            {m}
          </button>
        ))}
      </div>
      <div className="table-wrap">
        <table className="table small mono">
          <thead>
            <tr>
              <th>cost \ period</th>
              {periods.map((p) => (
                <th key={p}>{p} ticks</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {costs.map((cost) => (
              <tr key={cost}>
                <td>{cost}</td>
                {periods.map((p) => {
                  const c = cellMap.get(`${p}:${cost}`);
                  const v = c ? Number(c[metric] ?? 0) : NaN;
                  return (
                    <td
                      key={p}
                      style={{
                        background: Number.isFinite(v) ? cellColor(v) : "transparent",
                        color: "#fff",
                        textAlign: "center",
                        fontWeight: 600,
                      }}
                    >
                      {Number.isFinite(v) ? v.toFixed(3) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ResearchResultView({ job }: { job: ResearchJobDetail }) {
  const t = useT();
  const [showRaw, setShowRaw] = useState(false);
  const res = job.result;

  const checksObj =
    res && typeof res.checks === "object" && res.checks !== null && !Array.isArray(res.checks)
      ? (res.checks as Record<string, unknown>)
      : null;

  return (
    <div className="panel" data-testid="research-job-result" style={{ marginTop: "0.8rem" }}>
      <div className="lab-head">
        <div>
          <strong>{t(job.title_key as never)}</strong>{" "}
          <span className="badge mono">{job.status}</span>{" "}
          <span className="small muted mono">
            {job.id} · {job.elapsed_seconds.toFixed(1)}s
          </span>
        </div>
        <div className="toolbar small">
          {res && (
            <>
              <button type="button" className="small ghost" onClick={() => setShowRaw((s) => !s)}>
                {showRaw ? t("research.hide_json") : t("research.show_json")}
              </button>
              <button
                type="button"
                className="small ghost"
                onClick={() => downloadJson(`${job.tool_id}_${job.id}.json`, res)}
              >
                {t("research.download_json")}
              </button>
            </>
          )}
        </div>
      </div>

      {job.error && (
        <p className="error small" role="alert">
          {job.error}
        </p>
      )}

      {job.status === "running" && <p className="muted small">{t("research.running_hint")}</p>}

      {res && (
        <>
          {job.tool_id === "connectome_compare_species" && <CompareSpeciesView data={res} />}
          {job.tool_id === "connectome_chimera" && <ChimeraView data={res} />}
          {job.tool_id === "cultural_phase_diagram" && <PhaseDiagramView data={res} />}

          {job.tool_id !== "connectome_compare_species" &&
            job.tool_id !== "connectome_chimera" &&
            job.tool_id !== "cultural_phase_diagram" && (
              <div>
                {typeof res.framing_note === "string" && <p className="notice small">{res.framing_note}</p>}
                {checksObj && (
                  <div className="toolbar small" style={{ margin: "0.4rem 0", flexWrap: "wrap" }}>
                    {Object.entries(checksObj).map(([k, v]) => {
                      const ok = typeof v === "boolean" ? v : Boolean((v as { ok?: boolean } | null)?.ok);
                      return (
                        <span
                          key={k}
                          className="badge"
                          style={{ background: ok ? "var(--accent)" : "var(--danger)", color: "#fff" }}
                        >
                          {ok ? "PASS" : "FAIL"} · {k}
                        </span>
                      );
                    })}
                  </div>
                )}
                <dl className="stats small mono">
                  {Object.entries(res)
                    .filter(([, v]) => typeof v !== "object" || v === null)
                    .slice(0, 18)
                    .map(([k, v]) => (
                      <div key={k} className="contents">
                        <dt>{k}</dt>
                        <dd>{String(v)}</dd>
                      </div>
                    ))}
                </dl>
              </div>
            )}

          {(showRaw ||
            (job.tool_id !== "connectome_compare_species" &&
              job.tool_id !== "connectome_chimera" &&
              job.tool_id !== "cultural_phase_diagram")) && (
            <details open={showRaw} style={{ marginTop: "0.6rem" }}>
              <summary className="small muted">{t("research.raw_json")}</summary>
              <pre className="small mono" style={{ maxHeight: "22rem", overflow: "auto" }}>
                {JSON.stringify(res, null, 2)}
              </pre>
            </details>
          )}
        </>
      )}

      {!res && job.stdout_tail && (
        <pre className="small mono" style={{ maxHeight: "14rem", overflow: "auto" }}>
          {job.stdout_tail}
        </pre>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Main ResearchHub Page
// --------------------------------------------------------------------------- //

export function ResearchHub() {
  const t = useT();
  const qc = useQueryClient();
  const [tab, setTab] = useState<HubTab>("tools");
  const [category, setCategory] = useState<Category>("all");
  const [selectedToolId, setSelectedToolId] = useState<string>("connectome_compare_species");
  const [formValues, setFormValues] = useState<Record<string, unknown>>({});
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Dataset detail state
  const [dsSummary, setDsSummary] = useState<ConnectomeDatasetSummary | null>(null);
  const [dsVerify, setDsVerify] = useState<{ dataset_id: string; report: Record<string, unknown> } | null>(null);

  const catalogQ = useQuery({ queryKey: ["research-catalog"], queryFn: api.researchCatalog, staleTime: 60_000 });
  const datasetsQ = useQuery({ queryKey: ["connectome-datasets"], queryFn: api.connectomeDatasets, staleTime: 15_000 });
  const jobsQ = useQuery({
    queryKey: ["research-jobs"],
    queryFn: api.researchJobs,
    refetchInterval: 2000,
  });
  const activeJobQ = useQuery({
    queryKey: ["research-job", activeJobId],
    queryFn: () => api.researchJob(activeJobId!),
    enabled: activeJobId !== null,
    refetchInterval: (q) => (q.state.data?.status === "running" ? 1200 : false),
  });

  const tools = useMemo(() => catalogQ.data ?? [], [catalogQ.data]);
  const filteredTools = useMemo(
    () => (category === "all" ? tools : tools.filter((x) => x.category === category)),
    [tools, category],
  );
  const selectedTool = useMemo(
    () => tools.find((x) => x.id === selectedToolId) ?? filteredTools[0] ?? null,
    [tools, filteredTools, selectedToolId],
  );

  useEffect(() => {
    if (!selectedTool) return;
    const init: Record<string, unknown> = {};
    for (const p of selectedTool.params) init[p.name] = p.default;
    setFormValues(init);
  }, [selectedTool]);

  const runTool = async () => {
    if (!selectedTool) return;
    setBusy(true);
    setActionError(null);
    try {
      const job = await api.createResearchJob(selectedTool.id, formValues);
      setActiveJobId(job.id);
      void qc.invalidateQueries({ queryKey: ["research-jobs"] });
    } catch (e) {
      setActionError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const inspectDataset = async (ds: ConnectomeDatasetInfo) => {
    setActionError(null);
    setDsVerify(null);
    try {
      const sum = await api.connectomeDatasetSummary(ds.dataset_id);
      setDsSummary(sum);
    } catch (e) {
      setActionError(e instanceof ApiError ? e.message : String(e));
    }
  };

  const verifyDataset = async (ds: ConnectomeDatasetInfo) => {
    setActionError(null);
    setDsSummary(null);
    try {
      const rep = await api.verifyConnectomeDataset(ds.dataset_id);
      setDsVerify({ dataset_id: ds.dataset_id, report: rep });
    } catch (e) {
      setActionError(e instanceof ApiError ? e.message : String(e));
    }
  };

  const fetchDataset = async (ds: ConnectomeDatasetInfo) => {
    setActionError(null);
    try {
      const job = await api.fetchConnectomeDataset(ds.dataset_id);
      setActiveJobId(job.id);
      setTab("jobs");
      void qc.invalidateQueries({ queryKey: ["research-jobs"] });
    } catch (e) {
      setActionError(e instanceof ApiError ? e.message : String(e));
    }
  };

  const cancelJob = async (jobId: string) => {
    setActionError(null);
    try {
      await api.cancelResearchJob(jobId);
      void qc.invalidateQueries({ queryKey: ["research-jobs"] });
      void qc.invalidateQueries({ queryKey: ["research-job", jobId] });
    } catch (e) {
      setActionError(e instanceof ApiError ? e.message : String(e));
    }
  };

  return (
    <div className="research-hub" data-testid="research-hub">
      <div className="lab-head">
        <div>
          <h2>{t("research.title")}</h2>
          <p className="muted small">{t("research.subtitle")}</p>
        </div>
        <div className="toolbar">
          <button
            type="button"
            data-testid="research-tab-tools"
            className={tab === "tools" ? "" : "ghost"}
            onClick={() => setTab("tools")}
          >
            {t("research.tab.tools")} ({tools.length})
          </button>
          <button
            type="button"
            data-testid="research-tab-datasets"
            className={tab === "datasets" ? "" : "ghost"}
            onClick={() => setTab("datasets")}
          >
            {t("research.tab.datasets")} ({datasetsQ.data?.length ?? 0})
          </button>
          <button
            type="button"
            data-testid="research-tab-jobs"
            className={tab === "jobs" ? "" : "ghost"}
            onClick={() => setTab("jobs")}
          >
            {t("research.tab.jobs")} ({jobsQ.data?.length ?? 0})
          </button>
        </div>
      </div>

      {actionError && (
        <p className="error" role="alert">
          {actionError}
        </p>
      )}

      {tab === "tools" && (
        <div>
          <div className="toolbar small" style={{ marginBottom: "0.7rem", flexWrap: "wrap" }}>
            {CATEGORIES.map((c) => (
              <button
                key={c}
                type="button"
                className={category === c ? "small" : "small ghost"}
                onClick={() => setCategory(c)}
              >
                {t(`research.cat.${c}` as never)}
              </button>
            ))}
          </div>

          <div className="grid2" style={{ gridTemplateColumns: "minmax(240px, 340px) 1fr", alignItems: "start" }}>
            <div className="panel" style={{ maxHeight: "34rem", overflowY: "auto" }}>
              <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: "0.35rem" }}>
                {filteredTools.map((tool) => (
                  <li key={tool.id}>
                    <button
                      type="button"
                      data-testid={`tool-item-${tool.id}`}
                      className={selectedTool?.id === tool.id ? "" : "ghost"}
                      style={{ width: "100%", textAlign: "left", display: "block" }}
                      onClick={() => setSelectedToolId(tool.id)}
                    >
                      <div style={{ fontWeight: 600 }}>
                        {t(tool.title_key as never)}{" "}
                        {tool.gpu_optional && <span className="badge small">GPU</span>}
                      </div>
                      <div className="small muted mono">{tool.cli_command.join(" ")}</div>
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              {selectedTool && (
                <div className="panel" data-testid="research-tool-form">
                  <div className="lab-head">
                    <div>
                      <h3 style={{ margin: 0 }}>{t(selectedTool.title_key as never)}</h3>
                      <p className="muted small">{t(selectedTool.description_key as never)}</p>
                    </div>
                    <button
                      type="button"
                      data-testid="research-run-btn"
                      disabled={busy}
                      onClick={() => void runTool()}
                    >
                      {busy ? t("common.loading") : t("research.run")}
                    </button>
                  </div>

                  <div className="grid2" style={{ marginTop: "0.5rem" }}>
                    {selectedTool.params.map((p) => {
                      const val = formValues[p.name] ?? p.default;
                      return (
                        <label key={p.name} className="small" style={{ display: "grid", gap: "0.2rem" }}>
                          <span>{t(p.label_key as never)}</span>
                          {p.kind === "select" ? (
                            <select
                              value={String(val ?? "")}
                              data-testid={`param-${p.name}`}
                              onChange={(e) => setFormValues((prev) => ({ ...prev, [p.name]: e.target.value }))}
                            >
                              {p.choices.map((ch) => (
                                <option key={ch} value={ch}>
                                  {ch}
                                </option>
                              ))}
                            </select>
                          ) : p.kind === "bool" ? (
                            <input
                              type="checkbox"
                              checked={Boolean(val)}
                              data-testid={`param-${p.name}`}
                              onChange={(e) => setFormValues((prev) => ({ ...prev, [p.name]: e.target.checked }))}
                            />
                          ) : p.kind === "int" || p.kind === "float" ? (
                            <input
                              type="number"
                              value={String(val ?? "")}
                              min={p.min_value ?? undefined}
                              max={p.max_value ?? undefined}
                              step={p.step ?? (p.kind === "int" ? 1 : 0.01)}
                              data-testid={`param-${p.name}`}
                              onChange={(e) =>
                                setFormValues((prev) => ({
                                  ...prev,
                                  [p.name]: p.kind === "int" ? Number.parseInt(e.target.value, 10) : Number(e.target.value),
                                }))
                              }
                            />
                          ) : (
                            <input
                              type="text"
                              value={String(val ?? "")}
                              data-testid={`param-${p.name}`}
                              onChange={(e) => setFormValues((prev) => ({ ...prev, [p.name]: e.target.value }))}
                            />
                          )}
                        </label>
                      );
                    })}
                  </div>
                </div>
              )}

              {activeJobQ.data && <ResearchResultView job={activeJobQ.data} />}
            </div>
          </div>
        </div>
      )}

      {tab === "datasets" && (
        <div className="panel" data-testid="research-datasets-panel">
          <div className="lab-head">
            <h3>{t("research.datasets.title")}</h3>
            <button type="button" className="small ghost" onClick={() => void datasetsQ.refetch()}>
              {t("diag.refresh")}
            </button>
          </div>
          <div className="table-wrap">
            <table className="table small">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>{t("research.datasets.kind")}</th>
                  <th>{t("research.datasets.species")}</th>
                  <th>{t("research.datasets.status")}</th>
                  <th>{t("research.datasets.size")}</th>
                  <th>{t("connectome.col.actions")}</th>
                </tr>
              </thead>
              <tbody>
                {(datasetsQ.data ?? []).map((ds) => (
                  <tr key={ds.dataset_id} data-testid={`ds-row-${ds.dataset_id}`}>
                    <td className="mono">{ds.dataset_id}</td>
                    <td>{ds.kind === "empirical" ? t("research.datasets.empirical") : t("research.datasets.synthetic")}</td>
                    <td className="mono">{ds.species}</td>
                    <td>
                      <span
                        className="badge"
                        style={{
                          background: ds.installed ? "var(--accent)" : "var(--border)",
                          color: ds.installed ? "#fff" : "var(--text)",
                        }}
                      >
                        {ds.installed ? t("research.datasets.installed") : t("research.datasets.not_installed")}
                      </span>
                    </td>
                    <td className="mono">{fmtBytes(ds.local_bytes || ds.expected_bytes)}</td>
                    <td>
                      <button
                        type="button"
                        className="small ghost"
                        disabled={!ds.installed}
                        data-testid={`ds-summary-${ds.dataset_id}`}
                        onClick={() => void inspectDataset(ds)}
                      >
                        {t("research.datasets.summary")}
                      </button>{" "}
                      {ds.kind === "empirical" && (
                        <>
                          <button
                            type="button"
                            className="small ghost"
                            disabled={!ds.installed}
                            data-testid={`ds-verify-${ds.dataset_id}`}
                            onClick={() => void verifyDataset(ds)}
                          >
                            SHA-256
                          </button>{" "}
                          <button
                            type="button"
                            className="small"
                            data-testid={`ds-fetch-${ds.dataset_id}`}
                            onClick={() => void fetchDataset(ds)}
                          >
                            {t("research.datasets.fetch")}
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {dsSummary && (
            <div className="panel" data-testid="ds-summary-view" style={{ marginTop: "0.8rem" }}>
              <h4>
                {dsSummary.dataset_id} — {t("research.datasets.summary")}
              </h4>
              {dsSummary.reference_checks.length > 0 && (
                <div className="toolbar small" style={{ marginBottom: "0.5rem", flexWrap: "wrap" }}>
                  {dsSummary.reference_checks.map((c) => (
                    <span
                      key={c.name}
                      className="badge"
                      style={{ background: c.ok ? "var(--accent)" : "var(--danger)", color: "#fff" }}
                    >
                      {c.ok ? "PASS" : "FAIL"} · {c.name}
                    </span>
                  ))}
                </div>
              )}
              <pre className="small mono" style={{ maxHeight: "18rem", overflow: "auto" }}>
                {JSON.stringify(dsSummary.summary, null, 2)}
              </pre>
            </div>
          )}

          {dsVerify && (
            <div className="panel" data-testid="ds-verify-view" style={{ marginTop: "0.8rem" }}>
              <h4>
                {dsVerify.dataset_id} — SHA-256{" "}
                <span className="badge">{dsVerify.report.ok ? "OK" : "FAILED"}</span>
              </h4>
              <pre className="small mono" style={{ maxHeight: "16rem", overflow: "auto" }}>
                {JSON.stringify(dsVerify.report, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {tab === "jobs" && (
        <div className="panel" data-testid="research-jobs-panel">
          <h3>{t("research.tab.jobs")}</h3>
          {!jobsQ.data?.length ? (
            <p className="muted small">{t("research.jobs.empty")}</p>
          ) : (
            <div className="table-wrap">
              <table className="table small">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Tool</th>
                    <th>Status</th>
                    <th>Time</th>
                    <th>{t("connectome.col.actions")}</th>
                  </tr>
                </thead>
                <tbody>
                  {jobsQ.data.map((j) => (
                    <tr key={j.id}>
                      <td className="mono">{j.id}</td>
                      <td>{t(j.title_key as never)}</td>
                      <td className="mono">{j.status}</td>
                      <td className="mono">{j.elapsed_seconds.toFixed(1)}s</td>
                      <td>
                        <button
                          type="button"
                          className="small ghost"
                          onClick={() => {
                            setActiveJobId(j.id);
                          }}
                        >
                          {t("research.jobs.view")}
                        </button>{" "}
                        {j.status === "running" && (
                          <button type="button" className="small" onClick={() => void cancelJob(j.id)}>
                            {t("research.jobs.cancel")}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {activeJobQ.data && <ResearchResultView job={activeJobQ.data} />}
        </div>
      )}
    </div>
  );
}
