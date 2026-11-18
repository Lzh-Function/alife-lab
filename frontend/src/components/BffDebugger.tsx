import { useEffect, useMemo, useState } from "react";
import { useT } from "../i18n";
import { OP_COLORS, OP_NAMES, glyph, hexToBytes } from "../lib/bff";

// Step-by-step view of one BFF pair evaluation. The trace comes from the server's VM (the same
// semantics as the simulation); the browser only replays the recorded write log.
export interface BffTrace {
  pairing: "self" | "next_epoch" | "manual";
  variant: string;
  step_limit: number;
  initial_tape: string;
  final_tape: string;
  steps: number;
  ops: number;
  halt: "step_limit" | "end_of_tape" | "unmatched_open" | "unmatched_close";
  pc: number[];
  cmd: number[];
  head0: number[];
  head1: number[];
  write_addr: number[];
  write_old: number[];
  write_new: number[];
  next_pc: number[];
  epoch?: number;
  pair?: number;
  first?: number;
  second?: number;
  side?: string;
  mutated_positions?: number[];
  summary?: {
    first_half_unchanged: boolean;
    second_half_unchanged: boolean;
    second_half_equals_first_input: boolean;
    first_half_equals_second_input: boolean;
    bytes_changed: number;
  };
}

const CHECKPOINT = 256; // tape snapshots every N steps for fast seeking

export function BffDebugger({ trace, onClose }: { trace: BffTrace; onClose?: () => void }) {
  const t = useT();
  const tf = (key: string, vars: Record<string, number>) =>
    Object.entries(vars).reduce((acc, [name, v]) => acc.replace(`{${name}}`, String(v)), t(key));
  const [k, setK] = useState(0); // number of executed steps shown
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(20);

  const snapshots = useMemo(() => {
    const snaps: Uint8Array[] = [];
    const cur = hexToBytes(trace.initial_tape);
    for (let s = 0; s <= trace.steps; s++) {
      if (s % CHECKPOINT === 0) snaps.push(cur.slice());
      if (s < trace.steps && trace.write_addr[s] >= 0) cur[trace.write_addr[s]] = trace.write_new[s];
    }
    return snaps;
  }, [trace]);
  const tape = useMemo(() => {
    const cur = snapshots[Math.floor(k / CHECKPOINT)].slice();
    for (let s = Math.floor(k / CHECKPOINT) * CHECKPOINT; s < k; s++) {
      if (trace.write_addr[s] >= 0) cur[trace.write_addr[s]] = trace.write_new[s];
    }
    return cur;
  }, [k, snapshots, trace]);

  useEffect(() => setK(0), [trace]);
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setK((v) => {
        if (v >= trace.steps) {
          setPlaying(false);
          return v;
        }
        return v + 1;
      });
    }, 1000 / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, trace.steps]);

  const done = k >= trace.steps;
  const s = done ? trace.steps - 1 : k; // the step about to execute (or the last one)
  const pc = done ? trace.next_pc[trace.steps - 1] ?? -1 : trace.pc[k];
  const h0 = trace.steps ? trace.head0[Math.max(0, s)] : 0;
  const h1 = trace.steps ? trace.head1[Math.max(0, s)] : 0;
  const lastWrite = k > 0 ? trace.write_addr[k - 1] : -1;
  const mutated = new Set(trace.mutated_positions ?? []);
  const seekWrite = (dir: 1 | -1) => {
    for (let v = k + dir; v >= 0 && v <= trace.steps; v += dir) {
      if (v > 0 && trace.write_addr[v - 1] >= 0) return setK(v);
    }
    setK(dir > 0 ? trace.steps : 0);
  };

  return (
    <div className="panel bff-debugger" data-testid="bff-debugger">
      <h3>
        {t("debugger.title")}
        {onClose && (
          <button className="link" onClick={onClose} style={{ float: "right" }}>
            ×
          </button>
        )}
      </h3>
      <p className="small" data-testid="debugger-context">
        {trace.pairing === "next_epoch" && (
          <>
            {tf("debugger.next_epoch", { epoch: trace.epoch ?? 0, pair: trace.pair ?? 0 })} · A = #{trace.first}, B = #{trace.second}
            {mutated.size > 0 && ` · ${t("debugger.mutated")} ${[...mutated].join(", ")}`}
          </>
        )}
        {trace.pairing === "self" && t("debugger.self")}
        {trace.pairing === "manual" && t("debugger.manual")}
        {" · "}
        {trace.variant} · {tf("debugger.result", { steps: trace.steps, ops: trace.ops })} ·{" "}
        <span data-testid="debugger-halt">{t(`debugger.halt.${trace.halt}`)}</span>
      </p>
      <div className="toolbar">
        <button onClick={() => setK(0)} data-testid="dbg-first">⏮</button>
        <button onClick={() => setK(Math.max(0, k - 1))} data-testid="dbg-prev">◀</button>
        <button onClick={() => setPlaying(!playing)} data-testid="dbg-play">{playing ? "⏸" : "▶"}</button>
        <button onClick={() => setK(Math.min(trace.steps, k + 1))} data-testid="dbg-next">▶|</button>
        <button onClick={() => setK(trace.steps)} data-testid="dbg-last">⏭</button>
        <button onClick={() => seekWrite(-1)}>{t("debugger.prev_write")}</button>
        <button onClick={() => seekWrite(1)} data-testid="dbg-next-write">{t("debugger.next_write")}</button>
        <input
          type="range"
          min={0}
          max={trace.steps}
          value={k}
          onChange={(e) => setK(Number(e.target.value))}
          data-testid="dbg-slider"
          style={{ flex: 1, minWidth: "10rem" }}
        />
        <span className="mono" data-testid="dbg-step">
          {k}/{trace.steps}
        </span>
        <label className="inline small">
          {t("debugger.speed")}
          <input type="number" min={1} max={1000} value={speed} onChange={(e) => setSpeed(Math.max(1, Number(e.target.value)))} style={{ width: "4rem" }} />
        </label>
      </div>
      <div className="bff-tape" role="table" aria-label="tape">
        {[0, 1].map((half) => (
          <div className="bff-row" key={half} role="row">
            <span className="bff-label">{half === 0 ? "A" : "B"}</span>
            {Array.from({ length: 64 }, (_, j) => {
              const addr = half * 64 + j;
              const b = tape[addr];
              const c = OP_COLORS[b];
              const cls = [
                "bff-cell",
                addr === pc ? "pc" : "",
                addr === h0 ? "h0" : "",
                addr === h1 ? "h1" : "",
                addr === lastWrite ? "written" : "",
                mutated.has(addr) ? "mutated" : "",
              ].join(" ");
              return (
                <span
                  key={j}
                  className={cls}
                  role="cell"
                  data-testid={`cell-${addr}`}
                  title={`[${addr}] = ${b} (0x${b.toString(16).padStart(2, "0")})`}
                  style={c ? { color: `rgb(${c[0]},${c[1]},${c[2]})` } : undefined}
                >
                  {glyph(b)}
                </span>
              );
            })}
          </div>
        ))}
      </div>
      <p className="small mono" data-testid="debugger-current">
        {done
          ? t("debugger.finished")
          : `pc=${trace.pc[k]} '${glyph(trace.cmd[k])}' ${OP_NAMES[trace.cmd[k]] ?? t("debugger.noop")} · head0=${trace.head0[k]} head1=${trace.head1[k]}`}
        {k > 0 && trace.write_addr[k - 1] >= 0 && ` · ${t("debugger.last_write")} [${trace.write_addr[k - 1]}] ${trace.write_old[k - 1]}→${trace.write_new[k - 1]}`}
      </p>
      <p className="small muted">
        <span className="bff-cell pc">pc</span> <span className="bff-cell h0">head0</span> <span className="bff-cell h1">head1</span>{" "}
        <span className="bff-cell written">{t("debugger.legend_written")}</span> <span className="bff-cell mutated">{t("debugger.legend_mutated")}</span>
      </p>
      {done && trace.summary && (
        <dl className="stats small" data-testid="debugger-summary">
          <dt>{t("debugger.bytes_changed")}</dt>
          <dd className="mono">{trace.summary.bytes_changed}</dd>
          <dt>{t("debugger.b_equals_a")}</dt>
          <dd className="mono" data-testid="debugger-b-equals-a">{String(trace.summary.second_half_equals_first_input)}</dd>
          <dt>{t("debugger.a_equals_b")}</dt>
          <dd className="mono">{String(trace.summary.first_half_equals_second_input)}</dd>
        </dl>
      )}
    </div>
  );
}
