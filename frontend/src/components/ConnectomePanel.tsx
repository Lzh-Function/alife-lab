// Connectome browser controls:
// 1. Named IO port stimulation / silencing / clear-all controls.
// 2. Neuron search by cell_type / root_id / brain region / role / IO port with per-cell actions.
// 3. Highlighted population firing-rate history chart (MN9 & population spikes).
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type ConnectomePlasticitySummary, type MetricRow } from "../lib/api";
import { useT } from "../i18n";
import { MultiLine, type Series } from "./MultiLine";

interface Props {
  datasetId: string;
  live: boolean;
  replay: boolean;
  rows: MetricRow[];
  plasticity?: ConnectomePlasticitySummary | null;
  onIntervene: (kind: string, payload: Record<string, unknown>) => void;
  onInspectCell: (cellIndex: number) => void;
  onInspectPort: (portId: string) => void;
}

const ROLES = ["sensory", "interneuron", "motor", "other"] as const;

export function ConnectomePanel({
  datasetId,
  live,
  replay,
  rows,
  plasticity,
  onIntervene,
  onInspectCell,
  onInspectPort,
}: Props) {
  const t = useT();
  const [rateHz, setRateHz] = useState(150);
  const [q, setQ] = useState("");
  const [region, setRegion] = useState("");
  const [role, setRole] = useState("");
  const [portFilter, setPortFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const limit = 20;

  const meta = useQuery({
    queryKey: ["connectome-meta", datasetId],
    queryFn: () => api.connectomeMeta(datasetId),
    staleTime: 60_000,
  });

  const search = useQuery({
    queryKey: ["connectome-cells", datasetId, q, region, role, portFilter, offset],
    queryFn: () =>
      api.connectomeCells(datasetId, {
        q,
        region,
        role,
        port: portFilter,
        limit,
        offset,
      }),
    staleTime: 15_000,
  });

  const canIntervene = live && !replay;
  const ports = meta.data?.ports ?? [];
  const regions = meta.data?.regions ?? [];

  const hasMbon = rows.some((r) => r.values.mbon11_rate_hz !== undefined || r.values.mbon01_rate_hz !== undefined);
  const hasMn9Port = ports.some((p) => p.port_id === "mn9_proboscis_extension");
  const hasMeanV = rows.some((r) => r.values.mean_v_mv !== undefined);
  const hasActive = rows.some((r) => r.values.active_neurons_tick !== undefined);
  const rateSeries: Series[] = hasMbon
    ? [
        {
          label: t("metric.mbon11_rate_hz"),
          className: "arm-a",
          points: rows.map((r) => [r.tick, r.values.mbon11_rate_hz ?? null]),
        },
        {
          label: t("metric.mbon01_rate_hz"),
          className: "arm-b",
          points: rows.map((r) => [r.tick, r.values.mbon01_rate_hz ?? null]),
        },
      ]
    : hasMn9Port && rows.some((r) => r.values.mn9_rate_hz !== undefined)
      ? [
          {
            label: t("metric.mn9_rate_hz"),
            className: "arm-b",
            points: rows.map((r) => [r.tick, r.values.mn9_rate_hz ?? null]),
          },
          {
            label: t("metric.mean_firing_rate_hz"),
            className: "arm-a",
            points: rows.map((r) => [r.tick, r.values.mean_firing_rate_hz ?? null]),
          },
        ]
      : [
          {
            label: t("metric.mean_firing_rate_hz"),
            className: "arm-a",
            points: rows.map((r) => [r.tick, r.values.mean_firing_rate_hz ?? null]),
          },
          ...(hasMeanV
            ? [
                {
                  label: t("metric.mean_v_mv"),
                  className: "arm-b",
                  points: rows.map((r) => [r.tick, r.values.mean_v_mv ?? null] as [number, number | null]),
                },
              ]
            : hasActive
              ? [
                  {
                    label: t("metric.active_neurons_tick"),
                    className: "arm-b",
                    points: rows.map((r) => [r.tick, r.values.active_neurons_tick ?? null] as [number, number | null]),
                  },
                ]
              : []),
        ];

  return (
    <div className="connectome-controls">
      {/* Honest framing notice */}
      <p className="notice small" data-testid="connectome-framing">
        {t("connectome.framing_note")}
      </p>

      {/* Named IO Ports: quick stimulation / silencing */}
      <div className="panel" data-testid="connectome-ports">
        <div className="lab-head">
          <h3>{t("connectome.ports.title")}</h3>
          <label className="inline small">
            {t("connectome.stim_rate")}
            <input
              type="number"
              min={0}
              max={1000}
              step={10}
              value={rateHz}
              data-testid="connectome-rate-input"
              onChange={(e) => setRateHz(Math.max(0, Math.min(1000, Number(e.target.value))))}
              style={{ width: "5.5rem" }}
            />
            Hz
          </label>
          <button
            type="button"
            data-testid="btn-clear-stimuli"
            disabled={!canIntervene}
            onClick={() => onIntervene("clear_stimuli", {})}
          >
            {t("connectome.clear_stimuli")}
          </button>
        </div>
        {meta.isError && <p className="error small">{String(meta.error)}</p>}
        <div className="table-wrap" style={{ maxHeight: "15rem", overflowY: "auto" }}>
          <table className="table small">
            <thead>
              <tr>
                <th>{t("connectome.col.port")}</th>
                <th>{t("connectome.col.direction")}</th>
                <th>{t("connectome.col.count")}</th>
                <th>{t("connectome.col.actions")}</th>
              </tr>
            </thead>
            <tbody>
              {ports.map((p) => (
                <tr key={p.port_id} data-testid={`port-row-${p.port_id}`}>
                  <td title={p.description}>
                    <strong>{p.label}</strong> <span className="muted mono">({p.port_id})</span>
                  </td>
                  <td>
                    <span className="badge">{t(`connectome.role.${p.direction === "circuit" ? "interneuron" : p.direction}`)}</span>
                  </td>
                  <td className="mono">{p.count}</td>
                  <td>
                    <button
                      type="button"
                      className="small"
                      data-testid={`btn-stim-${p.port_id}`}
                      disabled={!canIntervene}
                      onClick={() => onIntervene("stimulate", { port: p.port_id, rate_hz: rateHz })}
                    >
                      ⚡ {t("connectome.stimulate")}
                    </button>{" "}
                    <button
                      type="button"
                      className="small"
                      data-testid={`btn-silence-${p.port_id}`}
                      disabled={!canIntervene}
                      onClick={() => onIntervene("silence", { port: p.port_id, silenced: true })}
                    >
                      🔇 {t("connectome.silence")}
                    </button>{" "}
                    <button
                      type="button"
                      className="small ghost"
                      data-testid={`btn-unsilence-${p.port_id}`}
                      disabled={!canIntervene}
                      onClick={() => onIntervene("silence", { port: p.port_id, silenced: false })}
                    >
                      {t("connectome.unsilence")}
                    </button>{" "}
                    <button
                      type="button"
                      className="small ghost"
                      data-testid={`btn-inspect-port-${p.port_id}`}
                      onClick={() => onInspectPort(p.port_id)}
                    >
                      {t("lab.tool.inspect")}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Highlighted population firing-rate chart */}
      {rows.length > 0 && (
        <div className="panel" data-testid="connectome-rate-chart">
          <h3>{t("connectome.chart.title")}</h3>
          <MultiLine series={rateSeries} testId="connectome-rates-svg" />
        </div>
      )}

      {/* Changed KC -> MBON plastic synapses (F4) */}
      {plasticity?.enabled && (
        <div className="panel" data-testid="connectome-plasticity">
          <h3>{t("connectome.plasticity.title")}</h3>
          <p className="small muted">{t("connectome.plasticity.note")}</p>
          <p className="small mono">
            {t("connectome.plasticity.changed")}: <strong>{plasticity.changed_edges.toLocaleString()}</strong> /{" "}
            {plasticity.total_edges.toLocaleString()} · {t("connectome.plasticity.mean_scale")}:{" "}
            {plasticity.mean_scale.toFixed(3)} · {t("connectome.plasticity.min_scale")}: {plasticity.min_scale.toFixed(3)}
          </p>
          {plasticity.top_changed.length > 0 ? (
            <div className="table-wrap" style={{ maxHeight: "12rem", overflowY: "auto" }}>
              <table className="table small">
                <thead>
                  <tr>
                    <th>KC → MBON</th>
                    <th>{t("connectome.plasticity.base_eff")}</th>
                    <th>{t("connectome.plasticity.scale")}</th>
                  </tr>
                </thead>
                <tbody>
                  {plasticity.top_changed.map((e) => (
                    <tr key={`${e.pre_index}-${e.post_index}`}>
                      <td className="mono">
                        {e.pre_type} (#{e.pre_index}) → {e.post_type} (#{e.post_index})
                      </td>
                      <td className="mono">
                        {e.base_weight} → {e.eff_weight}
                      </td>
                      <td className="mono">
                        {(e.scale * 100).toFixed(0)}% ({e.delta_pct >= 0 ? `+${e.delta_pct}` : e.delta_pct}%)
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="small muted">{t("connectome.plasticity.none_yet")}</p>
          )}
        </div>
      )}

      {/* Neuron search */}
      <div className="panel" data-testid="connectome-search">
        <h3>{t("connectome.search.title")}</h3>
        <div className="toolbar">
          <input
            type="search"
            value={q}
            data-testid="connectome-search-q"
            placeholder={t("connectome.search.placeholder")}
            onChange={(e) => {
              setQ(e.target.value);
              setOffset(0);
            }}
            style={{ minWidth: "14rem" }}
          />
          <select
            value={role}
            data-testid="connectome-search-role"
            onChange={(e) => {
              setRole(e.target.value);
              setOffset(0);
            }}
          >
            <option value="">{t("connectome.search.all_roles")}</option>
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {t(`connectome.role.${r}`)}
              </option>
            ))}
          </select>
          <select
            value={region}
            data-testid="connectome-search-region"
            onChange={(e) => {
              setRegion(e.target.value);
              setOffset(0);
            }}
          >
            <option value="">{t("connectome.search.all_regions")}</option>
            {regions.map((r) => (
              <option key={r.region} value={r.region}>
                {r.region} ({r.count})
              </option>
            ))}
          </select>
          <select
            value={portFilter}
            data-testid="connectome-search-port"
            onChange={(e) => {
              setPortFilter(e.target.value);
              setOffset(0);
            }}
          >
            <option value="">{t("connectome.search.all_ports")}</option>
            {ports.map((p) => (
              <option key={p.port_id} value={p.port_id}>
                {p.label} ({p.count})
              </option>
            ))}
          </select>
          {search.data && (
            <span className="small muted" data-testid="connectome-search-count">
              {search.data.total.toLocaleString()} {t("connectome.search.hits")}
            </span>
          )}
        </div>

        <div className="table-wrap" style={{ maxHeight: "16rem", overflowY: "auto" }}>
          <table className="table small" data-testid="connectome-search-results">
            <thead>
              <tr>
                <th>#</th>
                <th>{t("connectome.col.cell_type")}</th>
                <th>{t("connectome.col.region")}</th>
                <th>{t("connectome.col.role")}</th>
                <th>{t("connectome.col.nt")}</th>
                <th>{t("connectome.col.source_id")}</th>
                <th>{t("connectome.col.actions")}</th>
              </tr>
            </thead>
            <tbody>
              {(search.data?.items ?? []).map((cell) => (
                <tr key={cell.index} data-testid={`cell-row-${cell.index}`}>
                  <td className="mono">{cell.index}</td>
                  <td className="mono">{cell.cell_type || "—"}</td>
                  <td className="mono">{cell.region}</td>
                  <td>{t(`connectome.role.${cell.role}`)}</td>
                  <td className="mono">{cell.neurotransmitter || "—"}</td>
                  <td className="mono">{cell.source_id}</td>
                  <td>
                    <button
                      type="button"
                      className="small ghost"
                      data-testid={`btn-inspect-cell-${cell.index}`}
                      onClick={() => onInspectCell(cell.index)}
                    >
                      {t("lab.tool.inspect")}
                    </button>{" "}
                    <button
                      type="button"
                      className="small"
                      data-testid={`btn-stim-cell-${cell.index}`}
                      disabled={!canIntervene}
                      onClick={() => onIntervene("stimulate", { cell_indices: [cell.index], rate_hz: rateHz })}
                    >
                      ⚡
                    </button>{" "}
                    <button
                      type="button"
                      className="small"
                      data-testid={`btn-silence-cell-${cell.index}`}
                      disabled={!canIntervene}
                      onClick={() => onIntervene("silence", { cell_indices: [cell.index], silenced: true })}
                    >
                      🔇
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {search.data && search.data.total > limit && (
          <div className="toolbar small" style={{ marginTop: "0.4rem" }}>
            <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
              ← {t("connectome.prev")}
            </button>
            <span className="muted">
              {offset + 1}–{Math.min(offset + limit, search.data.total)} / {search.data.total.toLocaleString()}
            </span>
            <button
              type="button"
              disabled={offset + limit >= search.data.total}
              onClick={() => setOffset(offset + limit)}
            >
              {t("connectome.next")} →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
