import { useCallback, useEffect, useRef, useState } from "react";
import { GaitPanel, type GaitPanelHandle } from "./GaitPanel";
import { useElementWidth, useViewMode, useWindowHeight, ViewModeBar } from "./ViewModeBar";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, type ConnectomeFrameHeader, type ConnectomePlasticitySummary, type Frame, type MetricRow, type RunEvent } from "../lib/api";
import { useRunStream, type GpuMemory } from "../lib/stream";
import { useT } from "../i18n";
import { ToolBar, WorldView, type ViewMode, type WorldViewHandle } from "./WorldView";
import { Help, metricHelp } from "./Help";
import { CPU_PROFILE, GPU_PROFILE, useCompute } from "../lib/compute";
import { SUBSTRATE_GUIDE } from "../guide/content";
import { displayWorld, initialTool, type ToolState } from "../lib/substrates";
import { Sparkline } from "./Sparkline";
import { StateBadge } from "./StateBadge";
import { Timeline } from "./Timeline";
import { InspectorPanel, type Inspection } from "./InspectorPanel";
import { BffDebugger, type BffTrace } from "./BffDebugger";
import { AssayResult, type AssayResultData } from "./AssayResult";
import { SpeciesChart } from "./SpeciesChart";
import { ConnectomePanel } from "./ConnectomePanel";

const LIVE = new Set(["starting", "running", "pausing", "paused", "resuming"]);
const MAX_ROWS = 600;

export function LiveLab({ runId, onOpenComparison }: { runId: string; onOpenComparison: (id: string) => void }) {
  const t = useT();
  const qc = useQueryClient();
  const canvas = useRef<WorldViewHandle>(null);
  const gait = useRef<GaitPanelHandle>(null);
  const viewHost = useRef<HTMLDivElement>(null);
  const [viewWidth, mainRef] = useElementWidth();
  const windowHeight = useWindowHeight();
  // As large as the column allows, but a square view should still fit on the screen
  // below the toolbars (never smaller than the previous fixed 720 px, never above 1400 px).
  const viewPixels = Math.min(viewWidth - 4, Math.max(720, windowHeight - 320), 1400);
  // Wide (16:10) 3D views: as wide as the column, as long as their height fits the screen.
  const widePixels = Math.min(viewWidth - 4, Math.max(720, Math.round((windowHeight - 320) * 1.6)), 1400);
  const [tick, setTick] = useState<number | null>(null);
  const [rows, setRows] = useState<MetricRow[]>([]);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [simRate, setSimRate] = useState(0);
  const [uiFps, setUiFps] = useState(0);
  const [dropped, setDropped] = useState(0);
  const [worker, setWorker] = useState<{ rss: number | null; cpu: number | null; gpu: GpuMemory | null }>({ rss: null, cpu: null, gpu: null });
  const [tool, setTool] = useState<ToolState | null>(null);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const shownReplayTick = useRef<number | null>(null);
  const currentToolRef = useRef<ToolState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState<string>("");
  // Replay mode: the canvas shows replayed states; live previews are not drawn.
  const [replay, setReplay] = useState(false);
  const replayRef = useRef(false);
  replayRef.current = replay;
  const lastLive = useRef<Frame | null>(null);
  const lastTickUiRef = useRef<number>(0);
  const [plasticity, setPlasticity] = useState<ConnectomePlasticitySummary | null>(null);

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
      if (!replayRef.current) {
        canvas.current?.draw(frame);
        gait.current?.push(frame.header as unknown as Record<string, unknown>);
        if (frame.header.codec === "connectome_v1") {
          const ch = frame.header as unknown as ConnectomeFrameHeader;
          setPlasticity(ch.plasticity ?? null);
        }
      }
      // Throttle React tick/dropped re-renders to ~10 Hz so 60 FPS 3D views never stall on DOM diffing.
      const nowMs = performance.now();
      if (nowMs - lastTickUiRef.current >= 100 || frame.header.tick === 0) {
        lastTickUiRef.current = nowMs;
        setTick(frame.header.tick);
        if (frame.header.dropped_previews !== undefined) setDropped(frame.header.dropped_previews);
      }
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
      setWorker({ rss: hb.worker_rss_mb, cpu: hb.worker_cpu_percent, gpu: hb.gpu_memory ?? null });
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
  const onReplayFrame = useCallback((frame: Frame) => {
    canvas.current?.draw(frame);
    if (frame.header.codec === "connectome_v1") {
      const ch = frame.header as unknown as ConnectomeFrameHeader;
      setPlasticity(ch.plasticity ?? null);
    }
  }, []);

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

  const onIntervene = useCallback(
    (kind: string, payload: Record<string, unknown>) => {
      void act(() => api.intervene(runId, kind, payload, tick ?? 0));
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [runId, tick],
  );

  const lastInspect = useRef<{ x: number; y: number; r: number; tick: number | undefined } | null>(null);
  const [trace, setTrace] = useState<BffTrace | null>(null);
  const [assay, setAssay] = useState<AssayResultData | null>(null);
  const [assayPending, setAssayPending] = useState(false);
  const onAssay = () => {
    const idx = inspection?.index;
    if (idx === undefined) return;
    setAssayPending(true);
    api
      .bffAssay({ run_id: runId, index: idx, tick: lastInspect.current?.tick })
      .then((r) => {
        setAssay(r);
        setInspectError(null);
      })
      .catch((e) => setInspectError(e instanceof ApiError ? e.message : String(e)))
      .finally(() => setAssayPending(false));
  };
  const onTrace = (mode: "trace_next_epoch" | "trace_self") => {
    const q = lastInspect.current;
    if (!q) return;
    api
      .inspect(runId, q.x, q.y, q.r, q.tick, mode)
      .then((d) => {
        setTrace(d.trace ?? null);
        setInspectError(null);
      })
      .catch((e) => setInspectError(e instanceof ApiError ? e.message : String(e)));
  };

  // search: open a candidate as an ordinary run of the target world (paused, on the CPU).
  const onOpenSpec = (spec: Record<string, unknown>) => {
    api
      .create({ spec, profile_id: CPU_PROFILE, ticks_per_second: 20, auto_start: true, start_paused: true })
      .then((run) => {
        window.location.hash = `/lab/${run.id}`;
      })
      .catch((e) => setInspectError(e instanceof ApiError ? e.message : String(e)));
  };

  // search: register a candidate in the discovery ledger (the server re-derives it by index).
  const onRegister = (candidate: number, claimKind: string) => {
    api
      .createDiscovery({ run_id: runId, candidate, tick: lastInspect.current?.tick, claim_kind: claimKind, title: `${summary?.name ?? runId} #${candidate}` })
      .then((d) => {
        window.location.hash = `/discoveries/${d.id}`;
      })
      .catch((e) => setInspectError(e instanceof ApiError ? e.message : String(e)));
  };

  const computeMode = useCompute((s) => s.mode);

  // nca_train: checkpoint the current weights, export them into an nca spec and start a run.
  const runTrainedModel = async () => {
    try {
      const current = tick ?? summary?.clock.value ?? 0;
      if (live) await api.control(runId, "checkpoints");
      let exported = await api.exportNca(runId);
      for (let i = 0; i < 20 && live && exported.checkpoint.tick < current; i++) {
        await new Promise((r) => window.setTimeout(r, 250));
        exported = await api.exportNca(runId);
      }
      const run = await api.create({ spec: exported.spec, profile_id: "local_interactive", ticks_per_second: 20, auto_start: true });
      window.location.hash = `/lab/${run.id}`;
    } catch (e) {
      setInspectError(e instanceof ApiError ? e.message : String(e));
    }
  };

  // Phase F12d: checkpoint current fly_world run, export selected descendant into full-brain + MuJoCo fly_embodied, and launch!
  const replayFullBrain = async (flyId?: number) => {
    setBusy(true);
    setError(null);
    try {
      const current = tick ?? summary?.clock.value ?? 0;
      if (live) await api.control(runId, "checkpoints");
      let exported = await api.exportFlyWorldReplay(runId, flyId);
      for (let i = 0; i < 20 && live && exported.checkpoint.tick < current; i++) {
        await new Promise((r) => window.setTimeout(r, 250));
        exported = await api.exportFlyWorldReplay(runId, flyId);
      }
      const dsId = exported.descendant.dataset_id;
      const profileId =
        dsId !== "synthetic_fly_embodied" && (summary?.profile_id === GPU_PROFILE || computeMode === "gpu")
          ? GPU_PROFILE
          : CPU_PROFILE;
      const newRun = await api.create({
        spec: exported.spec,
        profile_id: profileId,
        ticks_per_second: 60,
        auto_start: true,
      });
      window.location.hash = `/lab/${newRun.id}`;
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onInspect = (x: number, y: number) => {
    const tickArg = replay && shownReplayTick.current !== null ? shownReplayTick.current : undefined;
    const toolRadius = currentToolRef.current?.radius ?? 2;
    const r = summary?.substrate === "conway" ? Math.max(2, toolRadius) : toolRadius;
    lastInspect.current = { x: Number(x.toFixed(3)), y: Number(y.toFixed(3)), r, tick: tickArg };
    api
      .inspect(runId, Number(x.toFixed(3)), Number(y.toFixed(3)), r, tickArg)
      .then((d) => {
        setInspection(d);
        setInspectError(null);
      })
      .catch((e) => setInspectError(e instanceof ApiError ? e.message : String(e)));
  };

  const onInspectConnectome = (opts: { cellIndex?: number; port?: string }) => {
    const tickArg = replay && shownReplayTick.current !== null ? shownReplayTick.current : undefined;
    api
      .inspectConnectome(runId, { ...opts, tick: tickArg })
      .then((d) => {
        setInspection(d);
        setInspectError(null);
      })
      .catch((e) => setInspectError(e instanceof ApiError ? e.message : String(e)));
  };

  if (!summary) return <p>{run.isError ? t("common.error") : t("common.loading")}</p>;
  const currentTool = tool ?? initialTool(summary.substrate);
  currentToolRef.current = currentTool;
  const world = displayWorld(summary.substrate, summary.spec.experiment as Record<string, unknown>);
  const last = rows.length ? rows[rows.length - 1] : null;
  const metricIds = summary.spec.experiment.observations.metrics;
  const toolForView = currentTool.mode === "inspect" ? currentTool : live && !replay ? currentTool : null;

  return (
    <div className="lab">
      <section className="lab-main" ref={mainRef}>
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
            ⏭ {t("lab.step_unit").replace("{unit}", t(`clock.${summary.clock.unit}`))}
          </button>
          <button data-testid="btn-checkpoint" disabled={busy || !live} onClick={() => act(() => api.control(runId, "checkpoints"))}>
            💾 {t("lab.checkpoint")}
          </button>
          <GallerySave runId={runId} tick={tick ?? summary.clock.value} defaultTitle={summary.name} />
          {summary.substrate === "search" && (
            <a className="button-link" data-testid="btn-lineage" href={`#/lineage/${runId}`}>
              🌳 {t("lineage.open")}
            </a>
          )}
          {summary.substrate === "nca_train" && (
            <button data-testid="btn-run-model" disabled={busy} onClick={() => void runTrainedModel()}>
              🌱 {t("lab.nca.run_model")}
            </button>
          )}
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
            {t("lab.speed_unit").replace("{unit}", t(`clock.${summary.clock.unit}`))}
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
        <ToolBar substrate={summary.substrate} tool={currentTool} setTool={setTool} disabled={!live} channels={summary.spec.experiment.dynamics?.channels ?? summary.spec.experiment.dynamics?.n_ncas ?? 1} />
        {summary.substrate === "creature" && (
          <div className="toolbar" role="group" aria-label={t("lab.push")} data-testid="push-panel">
            <strong>{t("lab.push")}:</strong>
            {(
              [
                ["+x", 1.5, 0],
                ["-x", -1.5, 0],
                ["+y", 0, 1.5],
                ["-y", 0, -1.5],
                ["up", 0, 0],
              ] as const
            ).map(([label, vx, vy]) => (
              <button
                key={label}
                type="button"
                data-testid={`push-${label}`}
                disabled={!live}
                onClick={() => onIntervene("push", label === "up" ? { vz: 2.5 } : { vx, vy })}
              >
                {t(`lab.push.${label}`)}
              </button>
            ))}
            <span className="small muted">{t("lab.push.note")}</span>
          </div>
        )}
        {summary.substrate === "cultural_world" && (
          <div className="toolbar" role="group" aria-label="Cultural interventions" data-testid="cultural-world-bar">
            <button
              type="button"
              data-testid="cultural-erase-culture"
              disabled={!live}
              onClick={() => onIntervene("erase_culture", {})}
            >
              {t("cultural.act.erase_culture")}
            </button>
            <button
              type="button"
              data-testid="cultural-toggle-transmission"
              disabled={!live}
              onClick={() => onIntervene("set_transmission", {})}
            >
              {t("cultural.act.toggle_transmission")}
            </button>
            <button
              type="button"
              data-testid="cultural-shock-env"
              disabled={!live}
              onClick={() => onIntervene("shock_environment", { phase_shift: 0.5 })}
            >
              {t("cultural.act.shock_env")}
            </button>
          </div>
        )}
        {summary.substrate === "worm_embodied" && (
          <div className="toolbar" role="group" aria-label="C. elegans controls" data-testid="worm-embodied-bar">
            <button
              type="button"
              data-testid="worm-poke-nose"
              disabled={!live}
              onClick={() => onIntervene("poke_nose", { duration_ticks: 25 })}
            >
              {t("worm.act.poke_nose")}
            </button>
            <button
              type="button"
              data-testid="worm-poke-tail"
              disabled={!live}
              onClick={() => onIntervene("poke_tail", { duration_ticks: 25 })}
            >
              {t("worm.act.poke_tail")}
            </button>
            <button
              type="button"
              data-testid="worm-toggle-gap"
              disabled={!live}
              onClick={() => onIntervene("toggle_gap_junctions", {})}
            >
              {t("worm.act.toggle_gap")}
            </button>
            <button
              type="button"
              data-testid="worm-clear-food"
              disabled={!live}
              onClick={() => onIntervene("clear_food", {})}
            >
              {t("worm.act.clear_food")}
            </button>
          </div>
        )}
        {error && <p className="error" role="alert">{t("common.error")}: {error}</p>}
        {summary.restorable && <p className="notice">{t("lab.restorable")}</p>}
        {!live && !summary.restorable && <p className="muted">{t("lab.not_live")}</p>}
        <LabViewModes substrate={summary.substrate} viewHost={viewHost}>
          {(viewMode) => (
        <div className={`canvas-wrap ${viewMode !== "2d" ? "canvas-wrap-wide" : ""}`} ref={viewHost}>
          <WorldView
            ref={canvas}
            substrate={summary.substrate}
            world={world}
            datasetId={summary.spec.experiment.dataset_id}
            live={live}
            viewMode={viewMode}
            frameSource={replay ? "replay" : "live"}
            maxPixels={viewMode === "terrain" || viewMode === "spacetime" || viewMode === "brain3d" || summary.substrate === "creature" || summary.substrate === "connectome" || summary.substrate === "fly_embodied" || summary.substrate === "fly_swarm" || summary.substrate === "fly_world" ? widePixels : viewPixels}
            tool={viewMode === "2d" || viewMode === "glow" ? toolForView : null}
            onIntervene={onIntervene}
            onInspect={onInspect}
            onDisplayFps={setUiFps}
            onReplayFullBrain={summary.substrate === "fly_world" ? replayFullBrain : undefined}
            replayBusy={busy}
            runId={runId}
          />
        </div>
          )}
        </LabViewModes>
        {summary.substrate === "creature" && (
          <GaitPanel ref={gait} hasReflex={Boolean((summary.spec.experiment.dynamics as Record<string, unknown> | undefined)?.reflex)} />
        )}
        {summary.substrate === "connectome" && (
          <ConnectomePanel
            datasetId={summary.spec.experiment.dataset_id ?? "flywire_783"}
            live={live}
            replay={replay}
            rows={rows}
            plasticity={plasticity}
            onIntervene={onIntervene}
            onInspectCell={(cellIndex) => onInspectConnectome({ cellIndex })}
            onInspectPort={(port) => onInspectConnectome({ port })}
          />
        )}
        <p className="muted small">{t("lab.preview_note")} · {t("lab.evidence")}</p>
        {replay && <p className="notice small" data-testid="replay-banner">{t("timeline.banner")}</p>}
        <Timeline
          runId={runId}
          liveTick={tick}
          active={replay}
          onActiveChange={onReplayActive}
          onFrame={onReplayFrame}
          onShownTick={(v) => (shownReplayTick.current = v)}
        />
        {summary.substrate === "reaction_network" && <SpeciesChart rows={rows} experiment={summary.spec.experiment as Record<string, unknown>} />}
        {summary.substrate === "pd_nca" && (
          <SpeciesChart
            rows={rows}
            experiment={summary.spec.experiment as Record<string, unknown>}
            keys={["territory.sun", ...Array.from({ length: summary.spec.experiment.dynamics?.n_ncas ?? 0 }, (_, k) => `territory.nca_${k}`)]}
            titleKey="lab.territories"
          />
        )}
        {trace && <BffDebugger trace={trace} onClose={() => setTrace(null)} />}
        {assay && <AssayResult result={assay} onClose={() => setAssay(null)} />}
      </section>

      <aside className="lab-side">
        <InspectorPanel
          data={inspection}
          error={inspectError}
          onTrace={summary.substrate === "bff" ? onTrace : undefined}
          onAssay={summary.substrate === "bff" ? onAssay : undefined}
          assayPending={assayPending}
          onOpenSpec={summary.substrate === "search" ? onOpenSpec : undefined}
          onRegister={summary.substrate === "search" ? onRegister : undefined}
          onReactionToggle={
            summary.substrate === "reaction_network" && live
              ? (id, enable) => {
                  onIntervene(enable ? "enable_reaction" : "disable_reaction", { reaction: id });
                  const q = lastInspect.current;
                  if (q) window.setTimeout(() => onInspect(q.x, q.y), 300);
                }
              : undefined
          }
        />
        <AboutSubstrate substrate={summary.substrate} />
        <div className="panel">
          <h3>{t("lab.metrics")}</h3>
          <dl className="stats">
            <dt>{t(`clock.${summary.clock.unit}`)}</dt>
            <dd data-testid="tick">{tick ?? summary.clock.value}</dd>
            {metricIds.map((m) => (
              <MetricStat key={m} id={m} substrate={summary.substrate} value={last?.values[m] ?? null} />
            ))}
            <dt>{t("lab.sim_rate")}</dt>
            <dd>{simRate.toFixed(1)} tick/s</dd>
            <dt>{t("lab.ui_fps")}</dt>
            <dd>{uiFps}</dd>
            <dt>{t("lab.dropped")}</dt>
            <dd>{dropped}</dd>
            <dt>{t("lab.worker")}</dt>
            <dd>{worker.rss !== null ? `${worker.rss} MB RSS, CPU ${worker.cpu ?? "?"}%` : "—"}</dd>
            {worker.gpu && (
              <>
                <dt>{t("lab.gpu_memory")}</dt>
                <dd data-testid="gpu-memory" title={t("lab.gpu_memory_note")}>
                  {worker.gpu.pool_held_mib.toFixed(1)} MiB ({t("lab.gpu_peak")} {worker.gpu.peak_pool_held_mib.toFixed(1)}) ·{" "}
                  {t("lab.gpu_device")} {Math.round(worker.gpu.device_used_mib)}/{Math.round(worker.gpu.device_total_mib)} MiB
                </dd>
              </>
            )}
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

/** Saves what the canvas shows now (PNG) with the tick, a title and a note to the gallery. */
function GallerySave({ runId, tick, defaultTitle }: { runId: string; tick: number; defaultTitle: string }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [note, setNote] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const save = async () => {
    const canvas = document.querySelector<HTMLCanvasElement>("canvas[data-testid='world-canvas']");
    if (!canvas) {
      setMsg(t("gallery.no_canvas"));
      return;
    }
    const dataUrl = canvas.toDataURL("image/png");
    try {
      await api.addGallery({ run_id: runId, tick, title: title || `${defaultTitle} t=${tick}`, note, png_base64: dataUrl.split(",")[1] });
      setMsg(t("gallery.saved"));
      setOpen(false);
      setTitle("");
      setNote("");
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : String(e));
    }
  };
  return (
    <span className="gallery-save">
      <button type="button" data-testid="btn-gallery" onClick={() => setOpen((o) => !o)}>
        ☆ {t("gallery.save")}
      </button>
      {msg && !open && (
        <span className="small muted" data-testid="gallery-msg">
          {" "}
          {msg} <a href="#/gallery">{t("gallery.view")}</a>
        </span>
      )}
      {open && (
        <span className="gallery-form panel">
          <input data-testid="gallery-title" placeholder={`${defaultTitle} t=${tick}`} value={title} onChange={(e) => setTitle(e.target.value)} maxLength={200} />
          <textarea data-testid="gallery-note" placeholder={t("gallery.note_placeholder")} value={note} onChange={(e) => setNote(e.target.value)} maxLength={2000} rows={2} />
          <span className="small muted">{t("gallery.save_note")}</span>
          <button type="button" className="primary" data-testid="gallery-confirm" onClick={save}>
            {t("gallery.save_confirm")}
          </button>
        </span>
      )}
    </span>
  );
}

function AboutSubstrate({ substrate }: { substrate: string }) {
  const t = useT();
  const g = SUBSTRATE_GUIDE[substrate];
  if (!g) return null;
  return (
    <details className="panel about" data-testid="lab-about">
      <summary>
        <strong>{t("lab.about")}</strong>: {g.title}
      </summary>
      <p>{g.oneLine}</p>
      <h4>ここを見よう</h4>
      <ul>
        {g.watch.map((w) => (
          <li key={w}>{w}</li>
        ))}
      </ul>
      <h4>やってみよう</h4>
      <ul>
        {g.tryThis.map((w) => (
          <li key={w}>{w}</li>
        ))}
      </ul>
      <p className="small">
        <strong>まだ言えないこと：</strong>
        {g.cannotSay}
      </p>
      <a href={`#/guide/${substrate}`}>{t("lab.about.more")} →</a>
    </details>
  );
}

function MetricStat({ id, substrate, value }: { id: string; substrate: string; value: number | null }) {
  const t = useT();
  return (
    <>
      <dt>
        {t(`metric.${id}`)} <Help text={metricHelp(substrate, id)} testId={`metric-help-${id}`} />
      </dt>
      <dd data-testid={`metric-${id}`}>{value === null ? "null" : Number.isInteger(value) ? value : value.toFixed(3)}</dd>
    </>
  );
}

// Keeps the view mode next to the world view so switching it re-renders only this part.
function LabViewModes({ substrate, viewHost, children }: { substrate: string; viewHost: React.RefObject<HTMLDivElement | null>; children: (m: ViewMode) => React.ReactNode }) {
  const [mode, setMode] = useViewMode(substrate);
  return (
    <>
      <ViewModeBar substrate={substrate} mode={mode} setMode={setMode} fullscreenTarget={viewHost} />
      {children(mode)}
    </>
  );
}
