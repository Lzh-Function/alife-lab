import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";
import type { Frame } from "../lib/api";

export interface ParticleCanvasHandle {
  draw: (frame: Frame) => void;
}

interface Props {
  worldWidth: number;
  worldHeight: number;
  toolRadius: number | null; // world units; null = no tool
  onPoint: (x: number, y: number) => void;
  onDisplayFps: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
}

// Canvas2D renderer for `particles_f32` frames: [x, y, vx, vy] float32 little-endian per
// particle. Each boid is a short segment along its heading, coloured by heading angle.
export const ParticleCanvas = forwardRef<ParticleCanvasHandle, Props>(function ParticleCanvas(
  { worldWidth, worldHeight, toolRadius, onPoint, onDisplayFps, maxPixels = 720, testId = "world-canvas" },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pending = useRef<Frame | null>(null);
  const last = useRef<Frame | null>(null);
  const cursor = useRef<[number, number] | null>(null);
  const raf = useRef<number | null>(null);
  const frames = useRef(0);
  const scale = Math.max(0.5, Math.min(maxPixels / worldWidth, maxPixels / worldHeight));
  const pw = Math.round(worldWidth * scale);
  const ph = Math.round(worldHeight * scale);

  const paint = useCallback(() => {
    raf.current = null;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const frame = pending.current ?? last.current;
    if (pending.current) frames.current += 1;
    pending.current = null;
    const styles = getComputedStyle(canvas);
    ctx.fillStyle = styles.getPropertyValue("--cell-dead").trim() || "#10141c";
    ctx.fillRect(0, 0, pw, ph);
    if (frame && frame.header.codec === "particles_typed_f32") {
      // [x, y, vx, vy, type]: one dot per particle, coloured by type.
      last.current = frame;
      const view = new DataView(frame.bits.buffer, frame.bits.byteOffset, frame.bits.byteLength);
      const n = Math.min(frame.header.count ?? 0, Math.floor(frame.bits.byteLength / 20));
      const k = Math.max(1, frame.header.types ?? 1);
      const radius = Math.max(1.2, scale * 0.6);
      for (let i = 0; i < n; i++) {
        const o = i * 20;
        const x = view.getFloat32(o, true) * scale;
        const y = view.getFloat32(o + 4, true) * scale;
        const type = view.getFloat32(o + 16, true);
        ctx.fillStyle = `hsl(${((type * 360) / k + 15).toFixed(0)}, 85%, 60%)`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();
      }
    } else if (frame && frame.header.codec === "particles_f32") {
      last.current = frame;
      const view = new DataView(frame.bits.buffer, frame.bits.byteOffset, frame.bits.byteLength);
      const n = Math.min(frame.header.count ?? 0, Math.floor(frame.bits.byteLength / 16));
      const len = Math.max(3, scale * 1.6);
      ctx.lineWidth = Math.max(1, scale * 0.35);
      for (let i = 0; i < n; i++) {
        const o = i * 16;
        const x = view.getFloat32(o, true) * scale;
        const y = view.getFloat32(o + 4, true) * scale;
        const vx = view.getFloat32(o + 8, true);
        const vy = view.getFloat32(o + 12, true);
        const sp = Math.hypot(vx, vy) || 1;
        const hue = ((Math.atan2(vy, vx) * 180) / Math.PI + 360) % 360;
        ctx.strokeStyle = `hsl(${hue.toFixed(0)}, 80%, 62%)`;
        ctx.beginPath();
        ctx.moveTo(x - (vx / sp) * len, y - (vy / sp) * len);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    }
    if (cursor.current && toolRadius !== null) {
      ctx.strokeStyle = "rgba(250, 200, 60, 0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cursor.current[0] * scale, cursor.current[1] * scale, toolRadius * scale, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }, [pw, ph, scale, toolRadius]);

  const schedule = useCallback(() => {
    if (raf.current === null) raf.current = requestAnimationFrame(paint);
  }, [paint]);

  useImperativeHandle(ref, () => ({ draw: (f: Frame) => ((pending.current = f), schedule()) }), [schedule]);

  useEffect(() => {
    const id = window.setInterval(() => {
      onDisplayFps(frames.current);
      frames.current = 0;
    }, 1000);
    return () => window.clearInterval(id);
  }, [onDisplayFps]);
  useEffect(() => schedule(), [schedule]);

  const toWorld = (ev: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * worldWidth;
    const y = ((ev.clientY - rect.top) / rect.height) * worldHeight;
    return [Math.min(worldWidth, Math.max(0, x)), Math.min(worldHeight, Math.max(0, y))];
  };

  return (
    <canvas
      ref={canvasRef}
      data-testid={testId}
      className={`world-canvas ${toolRadius !== null ? "brush-active" : ""}`}
      width={pw}
      height={ph}
      onPointerMove={(ev) => {
        if (toolRadius === null) return;
        cursor.current = toWorld(ev);
        schedule();
      }}
      onPointerLeave={() => {
        cursor.current = null;
        schedule();
      }}
      onPointerDown={(ev) => {
        if (toolRadius === null) return;
        const [x, y] = toWorld(ev);
        onPoint(x, y);
      }}
    />
  );
});
