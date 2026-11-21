// Live gait panel for creature runs, fed from frame headers (no extra requests):
// - gait diagram (footfall pattern): one row per leg, a bar while that foot touches the ground;
// - torso tilt over the same time window;
// - the current knee reflex per leg (only for bodies with a reflex).
// Frames arrive at up to ~30/s; the panel re-renders at most 5 times a second.
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { useT } from "../i18n";

export interface GaitSample {
  time: number;
  contacts: boolean[];
  tilt: number;
  reflex: number[];
}

export interface GaitPanelHandle {
  push: (header: Record<string, unknown>) => void;
}

const WINDOW_S = 10;
const W = 640;
const LABEL = 44;
const ROW = 16;
const TILT_H = 70;

export const GaitPanel = forwardRef<GaitPanelHandle, { hasReflex: boolean }>(function GaitPanel({ hasReflex }, ref) {
  const t = useT();
  const buffer = useRef<GaitSample[]>([]);
  const [samples, setSamples] = useState<GaitSample[]>([]);
  const [hover, setHover] = useState<number | null>(null); // x in svg units

  useImperativeHandle(ref, () => ({
    push: (h) => {
      if (h.codec !== "scene3d_f32" || !Array.isArray(h.contacts)) return;
      const s: GaitSample = {
        time: Number(h.time ?? 0),
        contacts: h.contacts as boolean[],
        tilt: Number(h.tilt_deg ?? 0),
        reflex: (h.reflex as number[] | undefined) ?? [],
      };
      const b = buffer.current;
      if (b.length && s.time < b[b.length - 1].time) b.length = 0; // restored or replayed
      b.push(s);
      while (b.length && b[0].time < s.time - WINDOW_S) b.shift();
    },
  }));

  useEffect(() => {
    const id = window.setInterval(() => setSamples(buffer.current.slice()), 200);
    return () => window.clearInterval(id);
  }, []);

  if (samples.length < 2) return <p className="muted small" data-testid="gait-panel">{t("gait.waiting")}</p>;
  const legs = samples[samples.length - 1].contacts.length;
  const tEnd = samples[samples.length - 1].time;
  const tStart = tEnd - WINDOW_S;
  const x = (time: number) => LABEL + ((time - tStart) / WINDOW_S) * (W - LABEL - 4);
  const gaitH = legs * ROW + 4;

  // Contact runs per leg -> rectangles.
  const bars: Array<{ leg: number; x0: number; x1: number }> = [];
  for (let leg = 0; leg < legs; leg++) {
    let start: number | null = null;
    samples.forEach((s, i) => {
      const on = s.contacts[leg];
      if (on && start === null) start = s.time;
      const last = i === samples.length - 1;
      if (start !== null && (!on || last)) {
        bars.push({ leg, x0: x(start), x1: x(s.time) });
        start = null;
      }
    });
  }
  const duty = Array.from({ length: legs }, (_, leg) => samples.filter((s) => s.contacts[leg]).length / samples.length);
  const tiltMax = Math.max(30, ...samples.map((s) => s.tilt));
  const ty = (v: number) => TILT_H - 14 - (v / tiltMax) * (TILT_H - 24);
  const tiltPath = samples.map((s, i) => `${i ? "L" : "M"}${x(s.time).toFixed(1)},${ty(s.tilt).toFixed(1)}`).join(" ");
  const now = samples[samples.length - 1];

  // Hover: nearest sample to the pointer.
  const hovered =
    hover === null
      ? null
      : samples.reduce((best, s) => (Math.abs(x(s.time) - hover) < Math.abs(x(best.time) - hover) ? s : best), samples[0]);
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    setHover(px >= LABEL ? px : null);
  };

  return (
    <div className="gait-panel panel" data-testid="gait-panel">
      <h4>{t("gait.title")}</h4>
      <p className="small muted">{t("gait.intro")}</p>
      <svg viewBox={`0 0 ${W} ${gaitH + TILT_H}`} className="gait-svg" data-testid="gait-diagram" onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label={t("gait.title")}>
        {Array.from({ length: legs }, (_, leg) => (
          <g key={leg}>
            <text x={2} y={leg * ROW + 12} className="axis-label">
              {t("gait.leg")}
              {leg}
            </text>
            <line x1={LABEL} x2={W - 4} y1={leg * ROW + 8} y2={leg * ROW + 8} className="grid-line" />
          </g>
        ))}
        {bars.map((b, i) => (
          <rect key={i} x={b.x0} y={b.leg * ROW + 2} width={Math.max(1, b.x1 - b.x0)} height={ROW - 4} rx={2} className="gait-bar" />
        ))}
        <g transform={`translate(0 ${gaitH})`}>
          <text x={2} y={TILT_H / 2} className="axis-label">{t("gait.tilt")}</text>
          <text x={2} y={TILT_H - 14} className="axis-label">0°</text>
          <text x={LABEL - 4} y={ty(tiltMax) + 4} className="axis-label" textAnchor="end">{tiltMax.toFixed(0)}°</text>
          <line x1={LABEL} x2={W - 4} y1={ty(0)} y2={ty(0)} className="grid-line" />
          <path d={tiltPath} className="gait-tilt" />
          <text x={W - 4} y={TILT_H - 2} className="axis-label" textAnchor="end">
            {t("gait.time_axis")}
          </text>
        </g>
        {hovered && (
          <g pointerEvents="none">
            <line x1={x(hovered.time)} x2={x(hovered.time)} y1={0} y2={gaitH + TILT_H - 14} className="crosshair" />
          </g>
        )}
      </svg>
      <p className="small mono" data-testid="gait-readout">
        {hovered ? (
          <>
            t = {hovered.time.toFixed(2)} s · {t("gait.feet_down")} {hovered.contacts.filter(Boolean).length}/{legs} · {t("gait.tilt")} {hovered.tilt.toFixed(1)}°
          </>
        ) : (
          <>
            {t("gait.duty")}: {duty.map((d, i) => `${t("gait.leg")}${i} ${(d * 100).toFixed(0)}%`).join(" · ")}
          </>
        )}
      </p>
      {hasReflex && (
        <div data-testid="reflex-bars">
          <p className="small">
            <strong>{t("gait.reflex")}</strong> <span className="muted">{t("gait.reflex_note")}</span>
          </p>
          <svg viewBox={`0 0 ${W} ${legs * ROW + 4}`} className="gait-svg">
            <line x1={(LABEL + W) / 2} x2={(LABEL + W) / 2} y1={0} y2={legs * ROW} className="axis" />
            {now.reflex.map((v, leg) => {
              const mid = (LABEL + W) / 2;
              const half = (W - LABEL) / 2 - 4;
              const len = Math.max(-1, Math.min(1, v)) * half;
              return (
                <g key={leg}>
                  <text x={2} y={leg * ROW + 12} className="axis-label">
                    {t("gait.leg")}
                    {leg}
                  </text>
                  <rect x={len < 0 ? mid + len : mid} y={leg * ROW + 3} width={Math.abs(len)} height={ROW - 6} rx={2} className={len < 0 ? "reflex-bar neg" : "reflex-bar"} />
                  <text x={W - 4} y={leg * ROW + 12} className="axis-label" textAnchor="end">
                    {v >= 0 ? "+" : ""}
                    {v.toFixed(2)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}
    </div>
  );
});
