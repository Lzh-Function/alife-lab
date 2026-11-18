import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, type Frame, type MetricRow, type RunEvent } from "../lib/api";
import { useRunStream } from "../lib/stream";
import { useT } from "../i18n";
import { WorldCanvas, type WorldCanvasHandle } from "./WorldCanvas";
import { Sparkline } from "./Sparkline";
import { StateBadge } from "./StateBadge";
import { Timeline } from "./Timeline";

const LIVE = new Set(["starting", "running", "pausing", "paused", "resuming"]);
const MAX_ROWS = 4000;

export function LiveLab({ runId, onOpenComparison }: { runId: string; onOpenComparison: (id: string) => void }) {
  const t = useT();
  const qc = useQueryClient();
  const canvas = useRef<WorldCanvasHandle>(null);
  const [tick, setTick] = useState<number | null>(null);
  const [rows, setRows] = useState<MetricRow[]>([]);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [simRate, setSimRate] = useState(0);
  const [uiFps, setUiFps] = useState(0);
  const [dropped, setDropped] = useState(0);
  const [worker, setWorker] = useState<{ rss: number | null; cpu: number | null }>({ rss: null, cpu: null });
  const [brushMode, setBrushMode] = useState<"off" | "draw" | "erase">("off");
  const [radius, setRadius] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState<string>("");
  // Replay mode: the canvas shows replayed states; live previews are not drawn.
  const [replay, setReplay] = useState(false);
  const replayRef = useRef(false);
  replayRef.current = replay;
  const lastLive = useRef<Frame | null>(null);

  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.run(runId), refetchInterval: 2000 });
  const checkpoints = useQuery({ queryKey: ["checkpoints", runId], queryFn: () => api.checkpoints(runId) });
  const interventions = useQuery({ queryKey: ["interventions", runId], queryFn: () => api.interventions(runId) });

  const resync = useCallback(async () => {
    try {
      const [m, ev] = await Promise.all([api.metrics(runId), api.events(runId)]);
      setRows(m.rows.slice(-MAX_ROWS));
      setEvents(ev);
      if (m.rows.length) setTick(m.rows[m.rows.length - 1].tick);
      await qc.invalidateQueries({ queryKey: ["run", runId] });
      await qc.invalidateQueries({ queryKey: ["checkpoints", runId] });
      await qc.invalidateQueries({ queryKey: ["interventions", runId] });
    } catch (e) {
      setError(String(e));
    }
  }, [runId, qc]);

  useEffect(() => {
    setRows([]);
    setEvents([]);
    setTick(null);
    void resync();
  }, [runId, resync]);

  const conn = useRunStream(runId, {
    onFrame: (frame: Frame) => {
      lastLive.current = frame;
      setTick(frame.header.tick);
      if (!replayRef.current) canvas.current?.draw(frame);
      if (frame.header.dropped_previews !== undefined) setDropped(frame.header.dropped_previews);
    },
    onMetrics: (newRows, sps) => {
      setRows((prev) => {
        const merged = [...prev];
        for (const r of newRows) {
          while (merged.length && merged[merged.length - 1].tick >= r.tick && !r.post_intervention) merged.pop();
          if (r.post_intervention && merged.length && merged[merged.length - 1].tick === r.tick) merged.pop();
          merged.push(r);
        }
        return merged.slice(-MAX_ROWS);
      });
      setSimRate(sps);
    },
    onEvent: (ev) => {
      setEvents((prev) => (prev.some((p) => p.seq === ev.seq) ? prev : [...prev, ev]));
      if (ev.type === "checkpoint_ready") void qc.invalidateQueries({ queryKey: ["checkpoints", runId] });
      if (ev.type.startsWith("intervention")) void qc.invalidateQueries({ queryKey: ["interventions", runId] });
    },
    onRunState: () => void qc.invalidateQueries({ queryKey: ["run", runId] }),
    onHeartbeat: (hb) => {
      setSimRate(hb.steps_per_second);
      setWorker({ rss: hb.worker_rss_mb, cpu: hb.worker_cpu_percent });
    },
    onResync: () => void resync(),
  });

  const summary = run.data;
  const live = summary ? LIVE.has(summary.state) : false;

  // Show the latest checkpoint when no worker is streaming (e.g. after a restart).
  const latestId = summary?.latest_checkpoint?.id;
  useEffect(() => {
    if (!summary || live || !latestId) return;
    api.checkpointFrame(runId, latestId).then((f) => {
      lastLive.current = f;
      if (!replayRef.current) canvas.current?.draw(f);
      setTick(f.header.tick);
    }).catch(() => undefined);
  }, [runId, latestId, live, summary]);

  const onReplayActive = useCallback((active: boolean) => {
    setReplay(active);
    if (!active && lastLive.current) canvas.current?.draw(lastLive.current);
  }, []);
  const onReplayFrame = useCallback((frame: Frame) => canvas.current?.draw(frame), []);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      const res = (await fn()) as { status?: string; ack?: { error?: string | null } | null } | undefined;
      if (res && res.status === "rejected") setError(res.ack?.error ?? "rejected");
      await qc.invalidateQueries({ queryKey: ["run", runId] });
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onStroke = useCallback(
    (cells: Array<[number, number]>, value: 0 | 1) => {
      void act(() => api.intervene(runId, { shape: "cells", cells, value }, tick ?? 0));
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [runId, tick],
  );

  if (!summary) return <p>{run.isError ? t("common.error") : t("common.loading")}</p>;
  const world = summary.spec.experiment.world;
  const last = rows.length ? rows[rows.length - 1] : null;
  const metricIds = summary.spec.experiment.observations.metrics;

  return (
    <div className="lab">
      <section className="lab-main">
        <div className="lab-head" data-testid="lab-head">
          <h2 data-testid="run-name">{summary.name}</h2>
          <StateBadge state={summary.state} />
          <span className={`mode mode-${summary.effective_mode}`} data-testid="run-mode">
            {t(`lab.mode.${summary.effective_mode}`)}
          </span>
          <span className={`conn conn-${conn}`}>{t(`conn.${conn}`)}</span>
        </div>
        <div className="toolbar" role="toolbar">
          {summary.state === "paused" || summary.state === "pausing" ? (
            <button data-testid="btn-play" disabled={busy} onClick={() => act(() => api.control(runId, "resume"))}>
              ▶ {t("lab.play")}
            </button>
          ) : (
            <button data-testid="btn-pause" disabled={busy || !live} onClick={() => act(() => api.control(runId, "pause"))}>
              ❚❚ {t("lab.pause")}
            </button>
          )}
          <button
            data-testid="btn-step"
            disabled={busy || summary.state !== "paused"}
            onClick={() => act(() => api.control(runId, "step", { count: 1 }))}
          >
            ⏭ {t("lab.step")}
          </button>
          <button data-testid="btn-checkpoint" disabled={busy || !live} onClick={() => act(() => api.control(runId, "checkpoints"))}>
            💾 {t("lab.checkpoint")}
          </button>
          <button
            data-testid="btn-paired-fork"
            title={t("lab.paired_fork_note")}
            disabled={busy || !(live || summary.latest_checkpoint)}
            onClick={async () => {
              setBusy(true);
              setError(null);
              try {
                const cmp = await api.pairedFork(runId, live ? "now" : "latest");
                onOpenComparison(cmp.id);
              } catch (e) {
                setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
              } finally {
                setBusy(false);
              }
            }}
          >
            ⑂ {t("lab.paired_fork")}
          </button>
          {summary.restorable && (
            <button data-testid="btn-restore" className="primary" disabled={busy} onClick={() => act(() => api.restore(runId))}>
              ⟲ {t("lab.restore")}
            </button>
          )}
          <button data-testid="btn-cancel" className="danger" disabled={busy || !(live || summary.state === "interrupted")} onClick={() => act(() => api.cancel(runId))}>
            ■ {t("lab.cancel")}
          </button>
          <label className="inline">
            {t("lab.speed")}
            <input
              type="number"
              min={0}
              max={10000}
              step="any"
              value={speed}
              placeholder={String(summary.ticks_per_second)}
              onChange={(e) => setSpeed(e.target.value)}
              style={{ width: "6rem" }}
            />
            <button disabled={busy || !live || speed === ""} onClick={() => act(() => api.speed(runId, Number(speed)))}>
              {t("lab.speed.apply")}
            </button>
          </label>
        </div>
        <div className="toolbar brush" role="group" aria-label={t("lab.brush")}>
          <strong>{t("lab.brush")}:</strong>
          {(["off", "draw", "erase"] as const).map((m) => (
            <label key={m} className="inline">
              <input type="radio" name="brush" data-testid={`brush-${m}`} checked={brushMode === m} disabled={!live} onChange={() => setBrushMode(m)} />
              {t(`lab.brush.${m}`)}
            </label>
          ))}
          <label className="inline">
            {t("lab.brush.radius")}
            <input type="range" min={0} max={8} value={radius} onChange={(e) => setRadius(Number(e.target.value))} />
            {radius}
          </label>
          <small className="muted">{t("lab.brush.note")}</small>
        </div>
        {error && <p className="error" role="alert">{t("common.error")}: {error}</p>}
        {summary.restorable && <p className="notice">{t("lab.restorable")}</p>}
        {!live && !summary.restorable && <p className="muted">{t("lab.not_live")}</p>}
        <div className="canvas-wrap">
          <WorldCanvas
            ref={canvas}
            width={world.width}
            height={world.height}
            brushMode={live && !replay ? brushMode : "off"}
            brushRadius={radius}
            onStroke={onStroke}
            onDisplayFps={setUiFps}
          />
        </div>
        <p className="muted small">{t("lab.preview_note")} · {t("lab.evidence")}</p>
        {replay && <p className="notice small" data-testid="replay-banner">{t("timeline.banner")}</p>}
        <Timeline runId={runId} liveTick={tick} active={replay} onActiveChange={onReplayActive} onFrame={onReplayFrame} />
      </section>

      <aside className="lab-side">
        <div className="panel">
          <h3>{t("lab.metrics")}</h3>
          <dl className="stats">
            <dt>{t("lab.tick")}</dt>
            <dd data-testid="tick">{tick ?? summary.clock.value}</dd>
            {metricIds.map((m) => (
              <MetricStat key={m} id={m} value={last?.values[m] ?? null} />
            ))}
            <dt>{t("lab.sim_rate")}</dt>
            <dd>{simRate.toFixed(1)} tick/s</dd>
            <dt>{t("lab.ui_fps")}</dt>
            <dd>{uiFps}</dd>
            <dt>{t("lab.dropped")}</dt>
            <dd>{dropped}</dd>
            <dt>{t("lab.worker")}</dt>
            <dd>{worker.rss !== null ? `${worker.rss} MB RSS, CPU ${worker.cpu ?? "?"}%` : "—"}</dd>
          </dl>
          {metricIds.map((m) => (
            <div key={m} className="spark">
              <small>{t(`metric.${m}`)}</small>
              <Sparkline values={rows.map((r) => r.values[m] ?? null)} />
            </div>
          ))}
        </div>

        <div className="panel">
          <h3>{t("lab.provenance")}</h3>
          <dl className="stats small">
            <dt>run</dt>
            <dd className="mono">{summary.id}</dd>
            <dt>spec</dt>
            <dd className="mono" title={summary.spec_hash}>{summary.spec_hash.slice(0, 16)}…</dd>
            <dt>profile</dt>
            <dd className="mono" title={summary.profile_hash}>{summary.profile_id} · {summary.profile_hash.slice(0, 10)}…</dd>
            <dt>seed</dt>
            <dd>{summary.seed}</dd>
            <dt>backend</dt>
            <dd>{summary.backend} @ {summary.target_id}</dd>
            <dt>epoch</dt>
            <dd>{summary.epoch}</dd>
          </dl>
          {summary.intervened && <p className="notice small">{t("lab.mode.changed")}</p>}
        </div>

        {(summary.parent_run_id || summary.children.length > 0 || summary.comparisons.length > 0) && (
          <div className="panel" data-testid="lineage">
            <h3>{t("lab.lineage")}</h3>
            <dl className="stats small">
              {summary.parent_run_id && (
                <>
                  <dt>{t("lab.parent")}</dt>
                  <dd>
                    <a className="mono" href={`#/lab/${summary.parent_run_id}`}>{summary.parent_run_id}</a>
                  </dd>
                </>
              )}
              {summary.children.length > 0 && (
                <>
                  <dt>{t("lab.children")}</dt>
                  <dd>
                    {summary.children.map((c) => (
                      <div key={c}>
                        <a className="mono" href={`#/lab/${c}`}>{c}</a>
                      </div>
                    ))}
                  </dd>
                </>
              )}
              {summary.comparisons.length > 0 && (
                <>
                  <dt>{t("lab.comparisons")}</dt>
                  <dd>
                    {summary.comparisons.map((c) => (
                      <div key={c}>
                        <a className="mono" href={`#/compare/${c}`}>{c}</a>
                      </div>
                    ))}
                  </dd>
                </>
              )}
            </dl>
          </div>
        )}

        <div className="panel">
          <h3>{t("lab.checkpoints")} ({checkpoints.data?.length ?? 0})</h3>
          <ul className="list small" data-testid="checkpoint-list">
            {(checkpoints.data ?? []).slice(-8).reverse().map((c) => (
              <li key={c.id}>
                <span className="mono">t={c.simulation_clock}</span> · {c.reason} · e{c.epoch} ·{" "}
                <span className="mono" title={c.sha256}>{c.sha256.slice(0, 8)}</span>{" "}
                <button
                  className="ghost small"
                  data-testid={`fork-${c.id}`}
                  disabled={busy}
                  onClick={() =>
                    act(async () => {
                      const child = await api.fork(runId, c.id, `fork@t${c.simulation_clock}`);
                      window.location.hash = `/lab/${child.id}`;
                    })
                  }
                >
                  ⑂ {t("lab.fork")}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="panel">
          <h3>{t("lab.interventions")} ({interventions.data?.length ?? 0})</h3>
          <ul className="list small" data-testid="intervention-list">
            {(interventions.data ?? []).slice(-6).reverse().map((i) => (
              <li key={i.id}>
                {i.kind} · {i.status} · req t={i.requested_clock ?? "?"} → applied t={i.applied_clock ?? "—"}
              </li>
            ))}
          </ul>
        </div>

        <div className="panel">
          <h3>{t("lab.events")}</h3>
          <ul className="list small events" data-testid="event-list">
            {events.slice(-12).reverse().map((e) => (
              <li key={e.seq} className={`sev-${e.severity}`}>
                <span className="mono">#{e.seq} t={e.clock_value ?? "?"}</span> {e.type}
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  );
}

function MetricStat({ id, value }: { id: string; value: number | null }) {
  const t = useT();
  return (
    <>
      <dt>{t(`metric.${id}`)}</dt>
      <dd data-testid={`metric-${id}`}>{value === null ? "null" : Number.isInteger(value) ? value : value.toFixed(3)}</dd>
    </>
  );
}
