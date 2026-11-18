import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError, type ReplayFrame, type ReplayReport } from "../lib/api";
import { framePopulation } from "../lib/substrates";
import { useT } from "../i18n";

interface Props {
  runId: string;
  liveTick: number | null;
  active: boolean;
  onActiveChange: (active: boolean) => void;
  onFrame: (frame: ReplayFrame) => void;
}

const PLAY_FPS = 15;
const BATCH = 30;

// Timeline: checkpoint / intervention markers, seek by deterministic replay, replay
// verification and fork-from-any-tick. Never changes the live run's state.
export function Timeline({ runId, liveTick, active, onActiveChange, onFrame }: Props) {
  const t = useT();
  const timeline = useQuery({ queryKey: ["timeline", runId], queryFn: () => api.timeline(runId), refetchInterval: 3000 });
  // `tick` is the requested target; `frameInfo.tick` is the state actually on the canvas.
  const [tick, setTick] = useState<number | null>(null);
  const [pending, setPending] = useState(false);
  const [frameInfo, setFrameInfo] = useState<ReplayFrame["header"] | null>(null);
  const [alive, setAlive] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<ReplayReport | null>(null);
  const [busy, setBusy] = useState(false);
  const requestSeq = useRef(0);

  const tl = timeline.data;
  const min = tl?.range[0] ?? 0;
  const max = Math.max(min, liveTick ?? tl?.range[1] ?? 0);
  const exactOnly = tl?.replay.mode !== "exact_recompute";

  const show = useCallback(
    (frame: ReplayFrame) => {
      onFrame(frame);
      setFrameInfo(frame.header);
      setAlive(framePopulation(frame));
      setTick(frame.header.tick);
    },
    [onFrame],
  );

  const seek = useCallback(
    async (target: number) => {
      const seq = ++requestSeq.current;
      setError(null);
      setPending(true);
      setTick(target);
      try {
        const [frame] = await api.replayFrames(runId, target, 1, 1);
        if (seq === requestSeq.current && frame) show(frame);
      } catch (e) {
        if (seq === requestSeq.current) setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        if (seq === requestSeq.current) setPending(false);
      }
    },
    [runId, show],
  );

  // Debounced seeking while dragging the slider.
  const pendingSeek = useRef<number | undefined>(undefined);
  const scheduleSeek = (target: number) => {
    setTick(target);
    setPending(true);
    window.clearTimeout(pendingSeek.current);
    pendingSeek.current = window.setTimeout(() => void seek(target), 120);
  };
  useEffect(() => () => window.clearTimeout(pendingSeek.current), []);

  useEffect(() => {
    if (active && tick === null) {
      const start = Math.max(min, max);
      void seek(start);
    }
    if (!active) {
      setPlaying(false);
      setTick(null);
      setFrameInfo(null);
    }
  }, [active, tick, min, max, seek]);

  // Playback: fetch batches and paint them at a fixed display rate.
  useEffect(() => {
    if (!playing || tick === null) return;
    let cancelled = false;
    let queue: ReplayFrame[] = [];
    let next = tick + 1;
    let fetching = false;
    const fill = async () => {
      if (fetching || next > max) return;
      fetching = true;
      const count = Math.min(BATCH, max - next + 1);
      try {
        const frames = await api.replayFrames(runId, next, count, 1);
        if (!cancelled) {
          queue = queue.concat(frames);
          next += count;
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof ApiError ? e.message : String(e));
          setPlaying(false);
        }
      } finally {
        fetching = false;
      }
    };
    void fill();
    const id = window.setInterval(() => {
      const frame = queue.shift();
      if (frame) show(frame);
      else if (next > max && !fetching) setPlaying(false);
      if (queue.length < BATCH / 2) void fill();
    }, 1000 / PLAY_FPS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  const verify = async () => {
    setBusy(true);
    setError(null);
    try {
      setReport(await api.verifyReplay(runId));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const forkHere = async () => {
    if (tick === null) return;
    setBusy(true);
    setError(null);
    try {
      const child = await api.forkAtTick(runId, tick);
      window.location.hash = `/lab/${child.id}`;
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
    } finally {
      setBusy(false);
    }
  };

  const pct = (x: number) => (max > min ? ((x - min) / (max - min)) * 100 : 0);

  return (
    <div className="panel timeline" data-testid="timeline">
      <div className="lab-head">
        <h3>{t("timeline.title")}</h3>
        <label className="inline">
          <input type="checkbox" data-testid="replay-toggle" checked={active} disabled={!tl || tl.range[0] === null} onChange={(e) => onActiveChange(e.target.checked)} />
          {t("timeline.replay_mode")}
        </label>
        <span className="muted small">
          t={min}…{max}
        </span>
      </div>
      <div className="track" aria-hidden="true">
        {(tl?.checkpoints ?? []).map((c) => (
          <span key={c.id} className="mark mark-ckpt" style={{ left: `${pct(c.tick)}%` }} title={`checkpoint t=${c.tick} (${c.reason})`} />
        ))}
        {(tl?.interventions ?? [])
          .filter((i) => i.applied_clock !== null)
          .map((i) => (
            <span key={i.id} className={`mark mark-intv status-${i.status}`} style={{ left: `${pct(i.applied_clock ?? 0)}%` }} title={`${i.kind} t=${i.applied_clock} (${i.status})`} />
          ))}
        {active && tick !== null && <span className="mark mark-cursor" style={{ left: `${pct(tick)}%` }} />}
      </div>
      {active && (
        <>
          <input
            type="range"
            className="scrubber"
            data-testid="replay-slider"
            min={min}
            max={max}
            value={tick ?? max}
            onChange={(e) => scheduleSeek(Number(e.target.value))}
            aria-label={t("timeline.seek")}
          />
          <div className="toolbar">
            <button data-testid="replay-back" disabled={tick === null || tick <= min || playing} onClick={() => tick !== null && void seek(tick - 1)}>
              ◀ 1
            </button>
            <button data-testid="replay-forward" disabled={tick === null || tick >= max || playing} onClick={() => tick !== null && void seek(tick + 1)}>
              1 ▶
            </button>
            <button data-testid="replay-play" disabled={tick === null || tick >= max || exactOnly} onClick={() => setPlaying((p) => !p)}>
              {playing ? `❚❚ ${t("timeline.stop")}` : `▶ ${t("timeline.play")}`}
            </button>
            <label className="inline">
              tick
              <input
                type="number"
                data-testid="replay-tick-input"
                min={min}
                max={max}
                value={tick ?? ""}
                onChange={(e) => scheduleSeek(Math.min(max, Math.max(min, Number(e.target.value))))}
                style={{ width: "6rem" }}
              />
            </label>
            <button data-testid="replay-fork" disabled={busy || tick === null || exactOnly} onClick={() => void forkHere()}>
              ⑂ {t("timeline.fork_here")}
            </button>
          </div>
          <p className="small" data-testid="replay-source" data-pending={pending ? "true" : "false"}>
            {pending && <span className="muted">{t("common.loading")} </span>}
            <strong>t={frameInfo?.tick ?? "—"}</strong> · {t("timeline.population")} {alive ?? "—"} ·{" "}
            {frameInfo
              ? frameInfo.source === "checkpoint_tick" && frameInfo.steps_from_base === 0
                ? t("timeline.source.exact")
                : `${t("timeline.source.replay")} (t=${frameInfo.base_tick} + ${frameInfo.steps_from_base} tick, ${t("timeline.interventions_replayed")} ${frameInfo.interventions_replayed})`
              : "—"}
          </p>
        </>
      )}
      <div className="toolbar">
        <button data-testid="replay-verify" disabled={busy || exactOnly} onClick={() => void verify()}>
          ✓ {t("timeline.verify")}
        </button>
        {report && (
          <span className={report.ok ? "ok-text small" : "error small"} data-testid="replay-verify-result">
            {t(`timeline.verify.${report.status}`)} ({report.segments_checked}/{report.segments_total})
          </span>
        )}
      </div>
      {error && <p className="error small">{error}</p>}
      <p className="muted small">{tl ? t(exactOnly ? "timeline.note.visual" : "timeline.note.exact") : ""}</p>
    </div>
  );
}
