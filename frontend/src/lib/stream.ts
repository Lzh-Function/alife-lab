// WebSocket subscription to a run. Previews go to a mutable ref (never React state)
// so rendering stays decoupled from React re-renders; durable data is refetched via
// REST after reconnects or `resync_required`.
import { useEffect, useRef, useState } from "react";
import { refreshSessionToken, sessionToken, unpackFrame, type Frame, type MetricRow, type RunEvent } from "./api";

export type ConnState = "connecting" | "connected" | "disconnected";

export interface StreamHandlers {
  onFrame: (frame: Frame) => void;
  onMetrics: (rows: MetricRow[], stepsPerSecond: number) => void;
  onEvent: (event: RunEvent) => void;
  onRunState: (state: string) => void;
  onHeartbeat: (info: { steps_per_second: number; worker_rss_mb: number | null; worker_cpu_percent: number | null }) => void;
  onResync: () => void;
}

export function useRunStream(runId: string | null, handlers: StreamHandlers): ConnState {
  const [conn, setConn] = useState<ConnState>("connecting");
  const ref = useRef(handlers);
  ref.current = handlers;

  useEffect(() => {
    if (!runId) return;
    let ws: WebSocket | null = null;
    let closed = false;
    let retry = 0;
    let timer: number | undefined;
    let pinger: number | undefined;

    const connect = () => {
      setConn("connecting");
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const url = `${proto}://${window.location.host}/api/v1/stream/experiments/${runId}?token=${encodeURIComponent(sessionToken())}`;
      ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        retry = 0;
        setConn("connected");
        ref.current.onResync();
        pinger = window.setInterval(() => ws?.readyState === WebSocket.OPEN && ws.send("ping"), 15000);
      };
      ws.onmessage = (ev: MessageEvent) => {
        if (ev.data instanceof ArrayBuffer) {
          ref.current.onFrame(unpackFrame(ev.data));
          return;
        }
        const msg = JSON.parse(ev.data as string) as Record<string, unknown>;
        switch (msg.type) {
          case "metrics":
            ref.current.onMetrics(msg.rows as MetricRow[], msg.steps_per_second as number);
            break;
          case "event":
            ref.current.onEvent(msg.event as RunEvent);
            break;
          case "run_state":
            ref.current.onRunState(msg.state as string);
            break;
          case "heartbeat":
            ref.current.onHeartbeat(msg as never);
            break;
          case "resync_required":
            ref.current.onResync();
            break;
          default:
            break;
        }
      };
      ws.onclose = () => {
        window.clearInterval(pinger);
        if (closed) return;
        setConn("disconnected");
        retry = Math.min(retry + 1, 6);
        timer = window.setTimeout(() => {
          // The server may have restarted with a new session token.
          refreshSessionToken().catch(() => undefined).finally(() => {
            if (!closed) connect();
          });
        }, 250 * 2 ** retry);
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      window.clearInterval(pinger);
      ws?.close();
    };
  }, [runId]);

  return conn;
}
