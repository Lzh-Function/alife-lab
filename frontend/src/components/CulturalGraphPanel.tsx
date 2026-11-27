// CulturalWorld P6-P7 panel: Knowledge DAG & Trade Network SVG visualizer + social KPI strip.
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  api,
  ApiError,
  type CulturalKnowledgeNode,
  type CulturalSocialSummary,
  type CulturalTradeNode,
} from "../lib/api";
import { useT } from "../i18n";

const ROLE_COLORS: Record<string, string> = {
  A: "#f59e0b", // amber
  dual: "#10b981", // emerald
  B: "#3b82f6", // blue
  generalist: "#64748b", // slate
};

const RESOURCE_COLORS: Record<"A" | "B" | "dual", string> = {
  A: "#d97706",
  B: "#2563eb",
  dual: "#10b981",
};

function resourceKind(tr: string | number): "A" | "B" | "dual" {
  if (tr === 3 || tr === "dual_a_b" || tr === "dual" || tr === "AB") return "dual";
  if (tr === 1 || tr === "material_a" || tr === "A" || tr === "a") return "A";
  return "B";
}

function formatResource(tr: string | number): string {
  const k = resourceKind(tr);
  if (k === "dual") return "dual (A+B)";
  return k === "A" ? "material A" : "material B";
}

const ROLE_CENTERS: Record<string, { cx: number; cy: number; labelKey: string }> = {
  A: { cx: 145, cy: 175, labelKey: "cultural.graph.role_a" },
  dual: { cx: 360, cy: 86, labelKey: "cultural.graph.role_dual" },
  B: { cx: 575, cy: 175, labelKey: "cultural.graph.role_b" },
  generalist: { cx: 360, cy: 266, labelKey: "cultural.graph.role_generalist" },
};

export function CulturalGraphPanel({
  runId,
  tick,
  live,
}: {
  runId: string;
  tick: number;
  live: boolean;
}) {
  const t = useT();
  const [tab, setTab] = useState<"dag" | "trade">("dag");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedSkillId, setSelectedSkillId] = useState<number | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null);

  const q = useQuery({
    queryKey: ["cultural-graph", runId],
    queryFn: () => api.inspectCulturalGraph(runId),
    refetchInterval: autoRefresh && live ? 2500 : false,
  });

  const { refetch } = q;
  const lastRefetchRef = useRef<{ tick: number; at: number }>({ tick: -1, at: 0 });
  useEffect(() => {
    if (!autoRefresh) return;
    if (tick === lastRefetchRef.current.tick) return;
    const now = Date.now();
    if (lastRefetchRef.current.tick < 0) {
      lastRefetchRef.current = { tick, at: now };
      return;
    }
    if (now - lastRefetchRef.current.at >= 400) {
      lastRefetchRef.current = { tick, at: now };
      void refetch();
    }
  }, [tick, autoRefresh, refetch]);

  const graph = q.data?.cultural_graph;
  const dagNodes = useMemo(() => graph?.knowledge_dag?.nodes ?? [], [graph]);
  const dagEdges = useMemo(() => graph?.knowledge_dag?.edges ?? [], [graph]);
  const tradeNodes = useMemo(() => graph?.trade_network?.nodes ?? [], [graph]);
  const tradeEdges = useMemo(() => graph?.trade_network?.edges ?? [], [graph]);

  const socialSummary: CulturalSocialSummary | undefined = q.data?.social_summary;
  const exchangeEnabled = q.data?.exchange_enabled ?? socialSummary?.exchange_enabled ?? false;

  // Compute Knowledge DAG layered layout by depth
  const dagLayout = useMemo(() => {
    const byDepth = new Map<number, CulturalKnowledgeNode[]>();
    for (const n of dagNodes) {
      const d = Number.isFinite(n.depth) ? n.depth : 0;
      const arr = byDepth.get(d);
      if (arr) arr.push(n);
      else byDepth.set(d, [n]);
    }
    const depths = Array.from(byDepth.keys()).sort((a, b) => a - b);
    let maxLayerSize = 1;
    for (const d of depths) {
      const layer = byDepth.get(d)!;
      layer.sort((a, b) => a.skill_id - b.skill_id);
      if (layer.length > maxLayerSize) maxLayerSize = layer.length;
    }
    const W = Math.max(720, depths.length * 130 + 80);
    const H = Math.max(290, Math.min(500, maxLayerSize * 36 + 76));
    const pos = new Map<number, { x: number; y: number; node: CulturalKnowledgeNode }>();
    depths.forEach((d, colIdx) => {
      const layer = byDepth.get(d)!;
      const x = depths.length === 1 ? W / 2 : 68 + (colIdx / (depths.length - 1)) * (W - 136);
      layer.forEach((node, rowIdx) => {
        const y = layer.length === 1 ? H / 2 : 48 + (rowIdx / (layer.length - 1)) * (H - 88);
        pos.set(node.skill_id, { x, y, node });
      });
    });
    return { W, H, depths, pos };
  }, [dagNodes]);

  // Compute Trade Network layout grouped by role
  const tradeLayout = useMemo(() => {
    const W = 720;
    const H = 340;
    const groups: Record<string, CulturalTradeNode[]> = { A: [], dual: [], B: [], generalist: [] };
    for (const n of tradeNodes) {
      const r = n.role in groups ? n.role : "generalist";
      groups[r].push(n);
    }
    const pos = new Map<number, { x: number; y: number; node: CulturalTradeNode }>();
    for (const [role, list] of Object.entries(groups)) {
      list.sort((a, b) => a.agent_id - b.agent_id);
      const center = ROLE_CENTERS[role];
      list.forEach((node, i) => {
        if (list.length === 1) {
          pos.set(node.agent_id, { x: center.cx, y: center.cy, node });
          return;
        }
        const angle = i * 2.399963229728653;
        const dist = Math.min(72, 16 + Math.sqrt(i) * 16);
        const x = Math.max(24, Math.min(W - 24, center.cx + Math.cos(angle) * dist * 1.15));
        const y = Math.max(28, Math.min(H - 22, center.cy + Math.sin(angle) * dist * 0.72));
        pos.set(node.agent_id, { x, y, node });
      });
    }
    const maxVol = Math.max(1e-6, ...tradeEdges.map((e) => e.volume));
    return { W, H, groups, pos, maxVol };
  }, [tradeNodes, tradeEdges]);

  const selectedSkill = useMemo(
    () => dagNodes.find((n) => n.skill_id === selectedSkillId) ?? null,
    [dagNodes, selectedSkillId],
  );
  const selectedAgent = useMemo(
    () => tradeNodes.find((n) => n.agent_id === selectedAgentId) ?? null,
    [tradeNodes, selectedAgentId],
  );

  return (
    <div className="panel" data-testid="cultural-graph-panel">
      <div className="lab-head" style={{ justifyContent: "space-between" }}>
        <div className="toolbar" style={{ marginBottom: 0 }}>
          <h3 style={{ margin: 0 }}>{t("cultural.graph.title")}</h3>
          <button
            type="button"
            data-testid="cultural-tab-dag"
            className={tab === "dag" ? "small" : "small ghost"}
            onClick={() => setTab("dag")}
          >
            {t("cultural.graph.tab_dag")} ({dagNodes.length})
          </button>
          <button
            type="button"
            data-testid="cultural-tab-trade"
            className={tab === "trade" ? "small" : "small ghost"}
            onClick={() => setTab("trade")}
          >
            {t("cultural.graph.tab_trade")} ({tradeNodes.length})
          </button>
        </div>
        <div className="toolbar small" style={{ marginBottom: 0 }}>
          <label className="inline">
            <input
              type="checkbox"
              data-testid="cultural-graph-autorefresh"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            {t("cultural.graph.auto_refresh")}
          </label>
          <button
            type="button"
            className="small ghost"
            data-testid="cultural-graph-refresh"
            disabled={q.isFetching}
            onClick={() => void q.refetch()}
          >
            ↻ {t("diag.refresh")}
          </button>
        </div>
      </div>

      {/* Social KPI summary strip */}
      {socialSummary && (
        <div className="toolbar small mono" data-testid="cultural-social-kpis" style={{ gap: "0.55rem" }}>
          <span
            className="badge"
            style={{
              background: exchangeEnabled ? "var(--accent)" : "var(--border)",
              color: exchangeEnabled ? "#fff" : "var(--text)",
              marginLeft: 0,
            }}
          >
            {exchangeEnabled ? t("cultural.graph.exchange_on") : t("cultural.graph.exchange_off")}
          </span>
          <span>
            {t("metric.trade_count")}: <strong>{socialSummary.trade_count}</strong>
          </span>
          <span>
            {t("metric.trade_volume")}: <strong>{socialSummary.trade_volume.toFixed(2)}</strong>
          </span>
          <span>
            {t("metric.specialization_index")}: <strong>{socialSummary.specialization_index.toFixed(3)}</strong>
          </span>
          <span>
            {t("metric.division_of_labor_index")}: <strong>{socialSummary.division_of_labor_index.toFixed(3)}</strong>
          </span>
          <span>
            {t("metric.interdependence_index")}: <strong>{socialSummary.interdependence_index.toFixed(3)}</strong>
          </span>
          <span>
            {t("metric.repeat_partner_rate")}: <strong>{(socialSummary.repeat_partner_rate * 100).toFixed(1)}%</strong>
          </span>
        </div>
      )}

      {q.isError && (
        <p className="error small">
          {q.error instanceof ApiError ? q.error.message : String(q.error)}
        </p>
      )}
      {q.isLoading && <p className="muted small">{t("common.loading")}</p>}

      {tab === "dag" && !q.isLoading && (
        <div data-testid="cultural-dag-view">
          {dagNodes.length === 0 ? (
            <p className="muted small">{t("cultural.graph.dag_empty")}</p>
          ) : (
            <>
              <div style={{ overflowX: "auto" }}>
                <svg
                  viewBox={`0 0 ${dagLayout.W} ${dagLayout.H}`}
                  className="lineage-svg"
                  data-testid="cultural-dag-svg"
                  role="img"
                  aria-label={t("cultural.graph.tab_dag")}
                  style={{ maxHeight: "360px", background: "var(--bg)", borderRadius: "6px", border: "1px solid var(--border)" }}
                >
                  <defs>
                    <marker id="dag-arrow-refine" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
                      <path d="M 0 1 L 7 4 L 0 7 z" fill="#64748b" />
                    </marker>
                    <marker id="dag-arrow-compose" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
                      <path d="M 0 1 L 7 4 L 0 7 z" fill="#10b981" />
                    </marker>
                  </defs>

                  {dagLayout.depths.map((d, colIdx) => {
                    const x =
                      dagLayout.depths.length === 1
                        ? dagLayout.W / 2
                        : 68 + (colIdx / (dagLayout.depths.length - 1)) * (dagLayout.W - 136);
                    return (
                      <g key={d}>
                        <line x1={x} y1={26} x2={x} y2={dagLayout.H - 16} stroke="var(--border)" strokeDasharray="2 4" />
                        <text x={x} y={18} textAnchor="middle" className="axis-label">
                          d={d}
                        </text>
                      </g>
                    );
                  })}

                  {dagEdges.map((e, idx) => {
                    const src = dagLayout.pos.get(e.source);
                    const dst = dagLayout.pos.get(e.target);
                    if (!src || !dst || e.source === e.target) return null;
                    const highlighted = selectedSkillId !== null && (e.source === selectedSkillId || e.target === selectedSkillId);
                    const isCompose = e.kind === "compose";
                    const mx = (src.x + dst.x) / 2;
                    const curveOffset = isCompose ? -22 : src.x === dst.x ? 20 : 0;
                    const my = (src.y + dst.y) / 2 + curveOffset;
                    const dPath = `M ${src.x} ${src.y} Q ${mx} ${my} ${dst.x} ${dst.y}`;
                    return (
                      <path
                        key={`${e.source}-${e.target}-${e.kind}-${idx}`}
                        d={dPath}
                        fill="none"
                        stroke={isCompose ? "#10b981" : "#64748b"}
                        strokeWidth={highlighted ? 2.4 : isCompose ? 1.7 : 1.4}
                        strokeOpacity={highlighted ? 1 : 0.72}
                        strokeDasharray={isCompose ? "5 3" : undefined}
                        markerEnd={isCompose ? "url(#dag-arrow-compose)" : "url(#dag-arrow-refine)"}
                      />
                    );
                  })}

                  {dagNodes.map((n) => {
                    const p = dagLayout.pos.get(n.skill_id);
                    if (!p) return null;
                    const r = Math.max(6, Math.min(14, 6 + Math.sqrt(Math.max(0, n.alive_holders)) * 1.7));
                    const rk = resourceKind(n.target_resource);
                    const fill = RESOURCE_COLORS[rk];
                    const selected = selectedSkillId === n.skill_id;
                    return (
                      <g
                        key={n.skill_id}
                        style={{ cursor: "pointer" }}
                        data-testid={`dag-node-${n.skill_id}`}
                        onClick={() => setSelectedSkillId(n.skill_id)}
                      >
                        {n.inventor_dead && (
                          <circle
                            cx={p.x}
                            cy={p.y}
                            r={r + 4}
                            fill="none"
                            stroke="#ef4444"
                            strokeWidth={1.8}
                            strokeDasharray="3 2"
                          />
                        )}
                        {selected && (
                          <circle
                            cx={p.x}
                            cy={p.y}
                            r={r + 6.5}
                            fill="none"
                            stroke="#f59e0b"
                            strokeWidth={2}
                          />
                        )}
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r={r}
                          fill={fill}
                          fillOpacity={n.alive_holders > 0 ? 0.95 : 0.45}
                          stroke="var(--panel)"
                          strokeWidth={1.2}
                        >
                          <title>
                            {`s#${n.skill_id} d=${n.depth} ${formatResource(n.target_resource)} eff=${n.efficiency.toFixed(2)} holders=${n.alive_holders}${n.inventor_dead ? " (inventor dead)" : ""}\n[${n.ops.join(" -> ")}]`}
                          </title>
                        </circle>
                        {(dagNodes.length <= 36 || selected) && (
                          <text
                            x={p.x}
                            y={p.y + r + 11}
                            textAnchor="middle"
                            fontSize="9"
                            fill="var(--text)"
                            className="mono"
                          >
                            #{n.skill_id}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </svg>
              </div>
              <p className="muted small" style={{ marginTop: "0.35rem" }}>
                {t("cultural.graph.dag_legend")}
              </p>
              {selectedSkill ? (
                <div className="notice small mono" data-testid="cultural-dag-detail" style={{ marginTop: "0.4rem" }}>
                  <strong>
                    {t("cultural.graph.skill_detail")} #{selectedSkill.skill_id}
                  </strong>{" "}
                  · {t("cultural.graph.depth")}={selectedSkill.depth} · {formatResource(selectedSkill.target_resource)} ·{" "}
                  {t("cultural.graph.efficiency")}={selectedSkill.efficiency.toFixed(3)} ·{" "}
                  {t("cultural.graph.alive_holders")}={selectedSkill.alive_holders} · {t("cultural.graph.inventor")}=#
                  {selectedSkill.inventor_id} (
                  {selectedSkill.inventor_dead
                    ? t("cultural.graph.inventor_dead")
                    : t("cultural.graph.inventor_alive")}
                  )
                  <br />
                  {t("cultural.graph.parents")}: p1={selectedSkill.parent_skill_id >= 0 ? `#${selectedSkill.parent_skill_id}` : "—"}
                  {selectedSkill.secondary_parent_id !== null && selectedSkill.secondary_parent_id >= 0
                    ? `, p2=#${selectedSkill.secondary_parent_id}`
                    : ""}
                  <br />
                  {t("cultural.graph.ops")}: [{selectedSkill.ops.join(" → ")}]
                </div>
              ) : (
                <p className="muted small" data-testid="cultural-dag-detail">
                  {t("cultural.graph.dag_hint")}
                </p>
              )}
            </>
          )}
        </div>
      )}

      {tab === "trade" && !q.isLoading && (
        <div data-testid="cultural-trade-view">
          {tradeNodes.length === 0 ? (
            <p className="muted small">{t("cultural.graph.trade_empty")}</p>
          ) : (
            <>
              <div style={{ overflowX: "auto" }}>
                <svg
                  viewBox={`0 0 ${tradeLayout.W} ${tradeLayout.H}`}
                  className="lineage-svg"
                  data-testid="cultural-trade-svg"
                  role="img"
                  aria-label={t("cultural.graph.tab_trade")}
                  style={{ maxHeight: "360px", background: "var(--bg)", borderRadius: "6px", border: "1px solid var(--border)" }}
                >
                  {Object.entries(ROLE_CENTERS).map(([role, cfg]) => (
                    <g key={role}>
                      <ellipse
                        cx={cfg.cx}
                        cy={cfg.cy}
                        rx={106}
                        ry={62}
                        fill={ROLE_COLORS[role]}
                        fillOpacity={0.07}
                        stroke={ROLE_COLORS[role]}
                        strokeOpacity={0.35}
                        strokeDasharray="4 3"
                      />
                      <text
                        x={cfg.cx}
                        y={cfg.cy - 68}
                        textAnchor="middle"
                        fontSize="11"
                        fontWeight={600}
                        fill={ROLE_COLORS[role]}
                      >
                        {t(cfg.labelKey)} ({tradeLayout.groups[role]?.length ?? 0})
                      </text>
                    </g>
                  ))}

                  {tradeEdges.map((e, idx) => {
                    const src = tradeLayout.pos.get(e.source);
                    const dst = tradeLayout.pos.get(e.target);
                    if (!src || !dst) return null;
                    const ratio = e.volume / tradeLayout.maxVol;
                    const sw = Math.max(1, Math.min(6, 1 + 5 * ratio));
                    const highlighted = selectedAgentId !== null && (e.source === selectedAgentId || e.target === selectedAgentId);
                    const complementary = src.node.role !== dst.node.role;
                    return (
                      <line
                        key={`${e.source}-${e.target}-${idx}`}
                        x1={src.x}
                        y1={src.y}
                        x2={dst.x}
                        y2={dst.y}
                        stroke={complementary ? "#10b981" : "#94a3b8"}
                        strokeWidth={highlighted ? sw + 1.2 : sw}
                        strokeOpacity={highlighted ? 0.95 : 0.35 + 0.5 * ratio}
                      >
                        <title>
                          {`#${e.source} (${src.node.role}) <-> #${e.target} (${dst.node.role}): count=${e.count}, volume=${e.volume.toFixed(3)}`}
                        </title>
                      </line>
                    );
                  })}

                  {tradeNodes.map((n) => {
                    const p = tradeLayout.pos.get(n.agent_id);
                    if (!p) return null;
                    const r = Math.max(5, Math.min(13, 5 + Math.sqrt(Math.max(0, n.trades)) * 1.4));
                    const fill = ROLE_COLORS[n.role] ?? ROLE_COLORS.generalist;
                    const selected = selectedAgentId === n.agent_id;
                    return (
                      <g
                        key={n.agent_id}
                        style={{ cursor: "pointer" }}
                        data-testid={`trade-node-${n.agent_id}`}
                        onClick={() => setSelectedAgentId(n.agent_id)}
                      >
                        {selected && (
                          <circle cx={p.x} cy={p.y} r={r + 4} fill="none" stroke="var(--text)" strokeWidth={2} />
                        )}
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r={r}
                          fill={fill}
                          stroke="var(--panel)"
                          strokeWidth={1.2}
                        >
                          <title>
                            {`#${n.agent_id} [${n.role}] E=${n.energy.toFixed(2)} inv=(A:${n.inv_a.toFixed(2)}, B:${n.inv_b.toFixed(2)}) trades=${n.trades}`}
                          </title>
                        </circle>
                      </g>
                    );
                  })}
                </svg>
              </div>
              {tradeEdges.length === 0 && (
                <p className="muted small" style={{ marginTop: "0.3rem" }}>
                  {t("cultural.graph.trade_no_edges")}
                </p>
              )}
              <p className="muted small" style={{ marginTop: "0.3rem" }}>
                {t("cultural.graph.trade_legend")}
              </p>
              {selectedAgent && (
                <div className="notice small mono" data-testid="cultural-trade-detail" style={{ marginTop: "0.35rem" }}>
                  #{selectedAgent.agent_id} · {t("inspector.cultural.role")}={selectedAgent.role} · E=
                  {selectedAgent.energy.toFixed(3)} · A={selectedAgent.inv_a.toFixed(3)} · B=
                  {selectedAgent.inv_b.toFixed(3)} · {t("cultural.graph.trades")}={selectedAgent.trades}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
