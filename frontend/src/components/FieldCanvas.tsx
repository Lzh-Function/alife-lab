import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";
import type { Frame } from "../lib/api";

export interface FieldCanvasHandle {
  draw: (frame: Frame) => void;
}

// Continuous field (codec field_u8: `channels` bytes per cell, 0..255 = 0..1). One channel is
// drawn with a perceptually ordered colormap, several channels as RGB. The bytes are a display quantisation, not the state.
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

const LUT_U32 = (() => {
  const out = new Uint32Array(256);
  for (let v = 0; v < 256; v++) {
    out[v] = (255 << 24) | (LUT[v * 3 + 2] << 16) | (LUT[v * 3 + 1] << 8) | LUT[v * 3];
  }
  return out;
})();

interface Props {
  width: number;
  height: number;
  toolRadius: number | null; // null = pointer input disabled
  onPoint: (x: number, y: number) => void;
  onDisplayFps: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
  minRowPixels?: number;
}

export const FieldCanvas = forwardRef<FieldCanvasHandle, Props>(function FieldCanvas(
  { width, height, toolRadius, onPoint, onDisplayFps, maxPixels = 720, testId = "world-canvas", minRowPixels = 0 },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const off = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<{ w: number; h: number; img: ImageData; u32: Uint32Array } | null>(null);
  const pending = useRef<Frame | null>(null);
  const last = useRef<Frame | null>(null);
  const frames = useRef(0);
  const raf = useRef<number | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const scale = Math.max(1, Math.floor(maxPixels / Math.max(width, height)));
  // Short, wide images (e.g. species x time) get taller rows so they stay readable.
  const rowPx = Math.max(scale, minRowPixels);

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
    if (off.current.width !== w || off.current.height !== h) {
      off.current.width = w;
      off.current.height = h;
      imgRef.current = null;
    }
    const octx = off.current.getContext("2d");
    if (!octx) return;
    if (!imgRef.current || imgRef.current.w !== w || imgRef.current.h !== h) {
      const img = octx.createImageData(w, h);
      imgRef.current = { w, h, img, u32: new Uint32Array(img.data.buffer) };
    }
    const { img, u32 } = imgRef.current;
    const bytes = frame.bits;
    const c = (frame.header as { channels?: number }).channels ?? 1;
    const n = w * h;
    if (c === 1) {
      for (let i = 0; i < n; i++) u32[i] = LUT_U32[bytes[i]];
    } else if (c >= 3) {
      for (let i = 0; i < n; i++) {
        const o = i * c;
        u32[i] = (255 << 24) | (bytes[o + 2] << 16) | (bytes[o + 1] << 8) | bytes[o];
      }
    } else {
      for (let i = 0; i < n; i++) {
        const o = i * c;
        u32[i] = (255 << 24) | (bytes[o + 1] << 8) | bytes[o];
      }
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
      height={height * rowPx}
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
