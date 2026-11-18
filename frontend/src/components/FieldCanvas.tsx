import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";
import type { Frame } from "../lib/api";

export interface FieldCanvasHandle {
  draw: (frame: Frame) => void;
}

// Continuous scalar field (codec field_u8: one byte per cell, 0..255 = 0..1) drawn with a
// perceptually ordered colormap. The bytes are a display quantisation, not the state.
const STOPS: Array<[number, [number, number, number]]> = [
  [0, [10, 10, 24]],
  [0.25, [59, 40, 120]],
  [0.5, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1, [253, 231, 37]],
];
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let v = 0; v < 256; v++) {
    const x = v / 255;
    let i = 0;
    while (i < STOPS.length - 2 && x > STOPS[i + 1][0]) i++;
    const [x0, c0] = STOPS[i];
    const [x1, c1] = STOPS[i + 1];
    const f = (x - x0) / (x1 - x0);
    for (let k = 0; k < 3; k++) lut[v * 3 + k] = c0[k] + f * (c1[k] - c0[k]);
  }
  return lut;
})();

interface Props {
  width: number;
  height: number;
  toolRadius: number | null; // null = pointer input disabled
  onPoint: (x: number, y: number) => void;
  onDisplayFps: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
}

export const FieldCanvas = forwardRef<FieldCanvasHandle, Props>(function FieldCanvas(
  { width, height, toolRadius, onPoint, onDisplayFps, maxPixels = 720, testId = "world-canvas" },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const off = useRef<HTMLCanvasElement | null>(null);
  const pending = useRef<Frame | null>(null);
  const last = useRef<Frame | null>(null);
  const frames = useRef(0);
  const raf = useRef<number | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const scale = Math.max(1, Math.floor(maxPixels / Math.max(width, height)));

  const paint = useCallback(() => {
    raf.current = null;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const frame = pending.current ?? last.current;
    if (pending.current) frames.current += 1;
    pending.current = null;
    if (!canvas || !ctx || !frame || frame.header.codec !== "field_u8") return;
    last.current = frame;
    const w = frame.header.width ?? width;
    const h = frame.header.height ?? height;
    if (!off.current) off.current = document.createElement("canvas");
    off.current.width = w;
    off.current.height = h;
    const octx = off.current.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(w, h);
    const bytes = frame.bits;
    for (let i = 0; i < w * h; i++) {
      const v = bytes[i] * 3;
      img.data[i * 4] = LUT[v];
      img.data[i * 4 + 1] = LUT[v + 1];
      img.data[i * 4 + 2] = LUT[v + 2];
      img.data[i * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off.current, 0, 0, canvas.width, canvas.height);
  }, [width, height]);

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
  useEffect(() => schedule(), [width, height, schedule]);

  const cellAt = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - rect.left) / rect.width) * width);
    const y = Math.floor(((ev.clientY - rect.top) / rect.height) * height);
    return { x: Math.min(width - 1, Math.max(0, x)), y: Math.min(height - 1, Math.max(0, y)) };
  };

  return (
    <canvas
      ref={canvasRef}
      data-testid={testId}
      className={`world-canvas ${toolRadius !== null ? "brush-active" : ""}`}
      width={width * scale}
      height={height * scale}
      onPointerDown={(ev) => {
        if (toolRadius === null) return;
        const c = cellAt(ev);
        drag.current = c;
        onPoint(c.x, c.y);
      }}
      onPointerMove={(ev) => {
        if (toolRadius === null || !drag.current || ev.buttons === 0) return;
        const c = cellAt(ev);
        // One intervention per brush-radius of travel keeps the command log small.
        if (Math.hypot(c.x - drag.current.x, c.y - drag.current.y) >= Math.max(1, toolRadius)) {
          drag.current = c;
          onPoint(c.x, c.y);
        }
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerLeave={() => (drag.current = null)}
    />
  );
});
