import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import { useRunStream } from "../lib/stream";
import { useT } from "../i18n";
import { StateBadge } from "./StateBadge";
import { WorldView, type WorldViewHandle } from "./WorldView";
import type { ToolState } from "../lib/substrates";

const LIVE = new Set(["starting", "running", "pausing", "paused", "resuming"]);

// Compact live view of one run (used side by side in the comparison view).
export function RunPane({
  runId,
  arm,
  tool,
  onCheckpoint,
}: {
  runId: string;
  arm: string;
  tool: ToolState | null;
  onCheckpoint: () => void;
}) {
  const t = useT();
  const qc = useQueryClient();
  const canvas = useRef<WorldViewHandle>(null);
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
      if (last) setAlive(last.values.alive_cell_count ?? last.values.population ?? null);
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

  const onIntervene = useCallback(
    (kind: string, payload: Record<string, unknown>) => {
      api
        .intervene(runId, kind, payload, tick ?? 0)
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
        {t("lab.tick")} <span data-testid={`tick-${arm}`} className="mono">{tick ?? summary.clock.value}</span> · {t("timeline.population")}{" "}
        <span className="mono">{alive ?? "—"}</span>
      </div>
      {error && <p className="error small">{error}</p>}
      <div className="canvas-wrap">
        <WorldView
          ref={canvas}
          substrate={summary.substrate}
          world={world}
          tool={live ? tool : null}
          onIntervene={onIntervene}
          maxPixels={420}
          testId={`canvas-${arm}`}
        />
      </div>
    </div>
  );
}
