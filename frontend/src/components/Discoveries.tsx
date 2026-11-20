// Discovery ledger: candidates registered from searches, their pre-declared assay plan,
// results, status history and evidence level. Verdicts come only from the plan's rules.
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, type DiscoveryDetail } from "../lib/api";
import { CPU_PROFILE } from "../lib/compute";
import { useT } from "../i18n";

const when = (t: number) => new Date(t * 1000).toLocaleString();

export function StatusChip({ status }: { status: string }) {
  const t = useT();
  return (
    <span className={`badge discovery-${status}`} data-testid="discovery-status" data-status={status}>
      {t(`discovery.status.${status}`)}
    </span>
  );
}

export function DiscoveryList({ onOpen }: { onOpen: (id: string) => void }) {
  const t = useT();
  const list = useQuery({ queryKey: ["discoveries"], queryFn: api.discoveries, refetchInterval: 3000 });
  return (
    <div className="discoveries">
      <h2>{t("discovery.title")}</h2>
      <p className="muted small">{t("discovery.intro")}</p>
      {!list.data?.length ? (
        <p className="muted" data-testid="discovery-empty">{t("discovery.empty")}</p>
      ) : (
        <div className="table-wrap">
          <table className="table" data-testid="discovery-list">
            <thead>
              <tr>
                <th>{t("discovery.col.title")}</th>
                <th>{t("discovery.col.claim")}</th>
                <th>{t("discovery.col.status")}</th>
                <th>{t("discovery.col.evidence")}</th>
                <th>{t("discovery.col.updated")}</th>
              </tr>
            </thead>
            <tbody>
              {list.data.map((d) => (
                <tr key={d.id}>
                  <td>
                    <button className="ghost" data-testid={`open-${d.id}`} onClick={() => onOpen(d.id)}>
                      {d.title}
                    </button>
                  </td>
                  <td>{t(`discovery.claim.${d.claim_kind}`)}</td>
                  <td>
                    <StatusChip status={d.status} /> {d.running && <span className="muted small">{t("discovery.running")}</span>}
                  </td>
                  <td className="mono">{d.evidence_level}</td>
                  <td className="small muted">{when(d.updated_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function CheckRow({ d, check, index }: { d: DiscoveryDetail; check: DiscoveryDetail["assays"][number]["plan"]["checks"][number]; index: number }) {
  const t = useT();
  const latest = d.assays[d.assays.length - 1];
  const res = latest?.result?.results[check.id];
  const verdict = latest?.result?.judgement.verdicts[check.id];
  const world = (check.spec.experiment as { world: { width: number; height: number } }).world;
  const open = async () => {
    const run = await api.create({ spec: check.spec, profile_id: CPU_PROFILE, ticks_per_second: 30, auto_start: true, start_paused: true });
    window.location.hash = `/lab/${run.id}`;
  };
  return (
    <tr data-testid={`check-${check.id}`}>
      <td>
        {index + 1}. <strong>{t(`discovery.check.${check.id}`)}</strong>
        <div className="small muted">
          {world.width}×{world.height} · {check.steps} tick
        </div>
      </td>
      <td className="small">{t(`discovery.rule.${check.id}`)}</td>
      <td className="small mono">
        {res?.error
          ? res.error
          : res
            ? `${t("discovery.viable")} ${(res.viable_fraction! * 100).toFixed(0)}% · ${t("search.speed")} ${res.mean_speed!.toFixed(4)}`
            : "—"}
      </td>
      <td data-testid={`verdict-${check.id}`}>
        {verdict ? (verdict.passed === null ? t("discovery.verdict.na") : verdict.passed ? "✓ " + t("discovery.verdict.pass") : "✗ " + t("discovery.verdict.fail")) : "—"}
        {verdict && <div className="small muted">{verdict.reason}</div>}
      </td>
      <td>
        <button type="button" className="small" onClick={open}>
          {t("discovery.open_check")}
        </button>
      </td>
    </tr>
  );
}

export function DiscoveryDetailView({ id }: { id: string }) {
  const t = useT();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["discovery", id], queryFn: () => api.discovery(id), refetchInterval: (query) => (query.state.data?.running ? 1000 : 5000) });
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const d = q.data;
  if (!d) return <p className="muted">{q.error ? String(q.error) : "…"}</p>;
  const latest = d.assays[d.assays.length - 1];
  const plan = latest.plan;
  const act = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
      await qc.invalidateQueries({ queryKey: ["discovery", id] });
      await qc.invalidateQueries({ queryKey: ["discoveries"] });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  };
  const exportRecord = async () => {
    const rec = await api.discoveryRecord(id);
    const blob = new Blob([JSON.stringify(rec, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const openCandidate = async () => {
    const run = await api.create({ spec: d.candidate_spec, profile_id: CPU_PROFILE, ticks_per_second: 30, auto_start: true, start_paused: true });
    window.location.hash = `/lab/${run.id}`;
  };
  return (
    <div className="discovery-detail" data-testid="discovery-detail">
      <p>
        <a href="#/discoveries">← {t("discovery.back")}</a>
      </p>
      <h2>
        {d.title} <StatusChip status={d.status} /> <span className="badge mono" data-testid="discovery-evidence">{d.evidence_level}</span>
      </h2>
      <div className="panel">
        <h3>{t("discovery.claim_heading")}</h3>
        <p data-testid="discovery-claim">{t(`discovery.claim_text.${d.claim_kind}`)}</p>
        <dl className="stats small">
          <dt>{t("discovery.source")}</dt>
          <dd>
            <a href={`#/lab/${d.source.run_id}`}>{d.source.run_id}</a> · {t("search.generation")} {d.source.generation} · #{d.source.candidate_index} ({d.source.algorithm})
          </dd>
          {Object.entries(d.source.params ?? {}).map(([k, v]) => (
            <div key={k} style={{ display: "contents" }}>
              <dt className="mono">{k.split(".").pop()}</dt>
              <dd className="mono">{Number(v).toPrecision(4)}</dd>
            </div>
          ))}
          <dt>{t("search.score")}</dt>
          <dd className="mono">{d.recorded.score?.toFixed(4)}</dd>
          <dt>{t("search.speed")}</dt>
          <dd className="mono">{d.recorded.speed?.toFixed(4)}</dd>
          <dt>{t("discovery.spec_hash")}</dt>
          <dd className="mono small">{d.candidate_spec_hash.slice(0, 16)}…</dd>
        </dl>
        <div className="toolbar">
          <button type="button" onClick={openCandidate}>
            {t("search.open_as_run")}
          </button>
          <button type="button" onClick={exportRecord} data-testid="discovery-export">
            {t("discovery.export")}
          </button>
        </div>
      </div>

      <div className="panel">
        <h3>{t("discovery.plan_heading")}</h3>
        <p className="small muted">
          {t("discovery.plan_note")} <span className="mono">plan {plan.plan_hash.slice(0, 12)}</span> ·{" "}
          {plan.seed_free_initial_state ? t("discovery.seed_free") : `${plan.seeds.length} seeds`} · {t("discovery.max_level")} {plan.max_evidence_level}
        </p>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>{t("discovery.col.check")}</th>
                <th>{t("discovery.col.rule")}</th>
                <th>{t("discovery.col.result")}</th>
                <th>{t("discovery.col.verdict")}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {plan.checks.map((c, i) => (
                <CheckRow key={c.id} d={d} check={c} index={i} />
              ))}
            </tbody>
          </table>
        </div>
        <div className="toolbar">
          <button type="button" className="primary" data-testid="btn-assay" disabled={d.running} onClick={() => act(() => api.startAssay(id))}>
            {d.running ? t("discovery.running") : latest.state === "completed" ? t("discovery.rerun") : t("discovery.run")}
          </button>
          {d.status === "new" && (
            <button type="button" data-testid="btn-triage" onClick={() => act(() => api.reviewDiscovery(id, { status: "triaged" }))}>
              {t("discovery.triage")}
            </button>
          )}
        </div>
        <p className="small muted">{t("discovery.levels_note")}</p>
      </div>

      <div className="panel">
        <h3>{t("discovery.history")}</h3>
        <ul className="small" data-testid="discovery-history">
          {d.events.map((e, i) => (
            <li key={i}>
              <span className="muted">{when(e.at)}</span> · {t(`discovery.event.${e.kind}`)}
              {e.kind === "status_changed" && (
                <>
                  {" "}
                  {t(`discovery.status.${String(e.payload.from)}`)} → {t(`discovery.status.${String(e.payload.to)}`)} ({String(e.payload.evidence_level)})
                </>
              )}
              {e.kind === "note" && <> : {String(e.payload.text)}</>} <span className="muted">({e.author})</span>
            </li>
          ))}
        </ul>
        <div className="toolbar">
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder={t("discovery.note_placeholder")} maxLength={2000} style={{ flex: 1 }} />
          <button
            type="button"
            disabled={!note.trim()}
            onClick={() =>
              act(async () => {
                await api.reviewDiscovery(id, { note });
                setNote("");
              })
            }
          >
            {t("discovery.add_note")}
          </button>
        </div>
        {error && <p className="error small">{error}</p>}
      </div>
    </div>
  );
}
