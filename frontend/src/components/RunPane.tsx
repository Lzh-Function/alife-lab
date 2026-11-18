import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import { useRunStream } from "../lib/stream";
import { useT } from "../i18n";
import { StateBadge } from "./StateBadge";
import { WorldCanvas, type WorldCanvasHandle } from "./WorldCanvas";

const LIVE = new Set(["starting", "running", "pausing", "paused", "resuming"]);

// Compact live view of one run (used side by side in the comparison view).
export function RunPane({
  runId,
  arm,
  brushMode,
  brushRadius,
  onCheckpoint,
}: {
  runId: string;
  arm: string;
  brushMode: "off" | "draw" | "erase";
  brushRadius: number;
  onCheckpoint: () => void;
}) {
  const t = useT();
  const qc = useQueryClient();
  const canvas = useRef<WorldCanvasHandle>(null);
  const [tick, setTick] = useState<number | null>(null);
  const [alive, setAlive] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.run(runId), refetchInterval: 2000 });
  const summary = run.data;
  const live = summary ? LIVE.has(summary.state) : false;

  const conn = useRunStream(runId, {
    onFrame: (f) => {
      canvas.current?.draw(f);
      setTick(f.header.tick);
    },
    onMetrics: (rows) => {
      const last = rows[rows.length - 1];
      if (last) setAlive(last.values.alive_cell_count ?? null);
    },
    onEvent: (ev) => {
      if (ev.type === "checkpoint_ready") onCheckpoint();
      if (ev.type.startsWith("intervention")) void qc.invalidateQueries({ queryKey: ["run", runId] });
    },
    onRunState: () => void qc.invalidateQueries({ queryKey: ["run", runId] }),
    onHeartbeat: () => undefined,
    onResync: () => void qc.invalidateQueries({ queryKey: ["run", runId] }),
  });

  const latestId = summary?.latest_checkpoint?.id;
  useEffect(() => {
    if (!summary || live || !latestId) return;
    api.checkpointFrame(runId, latestId).then((f) => {
      canvas.current?.draw(f);
      setTick(f.header.tick);
    }).catch(() => undefined);
  }, [runId, latestId, live, summary]);

  const onStroke = useCallback(
    (cells: Array<[number, number]>, value: 0 | 1) => {
      api
        .intervene(runId, { shape: "cells", cells, value }, tick ?? 0)
        .then((r) => setError(r.status === "rejected" ? (r.ack?.error ?? "rejected") : null))
        .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    },
    [runId, tick],
  );

  if (!summary) return <div className="panel">{t("common.loading")}</div>;
  const world = summary.spec.experiment.world;
  return (
    <div className="panel run-pane" data-testid={`pane-${arm}`}>
      <div className="lab-head">
        <strong className={`arm arm-${arm}`}>{arm}</strong>
        <a href={`#/lab/${runId}`} className="mono small">{runId}</a>
        <StateBadge state={summary.state} />
        <span className={`mode mode-${summary.effective_mode}`}>{summary.effective_mode}</span>
        <span className={`conn conn-${conn}`}>{t(`conn.${conn}`)}</span>
      </div>
      <div className="small">
        {t("lab.tick")} <span data-testid={`tick-${arm}`} className="mono">{tick ?? summary.clock.value}</span> · {t("metric.alive_cell_count")}{" "}
        <span className="mono">{alive ?? "—"}</span>
      </div>
      {error && <p className="error small">{error}</p>}
      <div className="canvas-wrap">
        <WorldCanvas
          ref={canvas}
          width={world.width}
          height={world.height}
          brushMode={live ? brushMode : "off"}
          brushRadius={brushRadius}
          onStroke={onStroke}
          onDisplayFps={() => undefined}
          maxPixels={420}
          testId={`canvas-${arm}`}
        />
      </div>
    </div>
  );
}
