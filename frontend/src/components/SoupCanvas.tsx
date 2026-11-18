import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";
import type { Frame } from "../lib/api";
import { OP_COLORS } from "../lib/bff";

export interface SoupCanvasHandle {
  draw: (frame: Frame) => void;
}

// BFF soup: each 64-byte program is an 8x8 tile, tiles in a grid of `columns`.
// Instruction bytes have distinct colours, NUL is black, other bytes are dim grey.
const PIX = 8; // program tile edge in bytes

interface Props {
  onTile: (col: number, row: number, columns: number, programs: number) => void;
  onDisplayFps: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
  toolRadius: number | null;
}

export const SoupCanvas = forwardRef<SoupCanvasHandle, Props>(function SoupCanvas(
  { onTile, onDisplayFps, maxPixels = 720, testId = "world-canvas", toolRadius },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const off = useRef<HTMLCanvasElement | null>(null);
  const pending = useRef<Frame | null>(null);
  const last = useRef<Frame | null>(null);
  const frames = useRef(0);
  const raf = useRef<number | null>(null);
  const columns = useRef(64);
  const programs = useRef(0);

  const paint = useCallback(() => {
    raf.current = null;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const frame = pending.current ?? last.current;
    if (pending.current) frames.current += 1;
    pending.current = null;
    if (!canvas || !ctx || !frame || frame.header.codec !== "bff_soup") return;
    last.current = frame;
    const n = (frame.header as { programs?: number }).programs ?? 0;
    const cols = (frame.header as { columns?: number }).columns ?? 64;
    columns.current = cols;
    programs.current = n;
    const rows = Math.ceil(n / cols);
    const w = cols * PIX;
    const h = rows * PIX;
    if (!off.current) off.current = document.createElement("canvas");
    off.current.width = w;
    off.current.height = h;
    const octx = off.current.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(w, h);
    const bytes = frame.bits;
    for (let p = 0; p < n; p++) {
      const tx = (p % cols) * PIX;
      const ty = Math.floor(p / cols) * PIX;
      for (let b = 0; b < 64; b++) {
        const v = bytes[p * 64 + b];
        const c = OP_COLORS[v] ?? (v === 0 ? [0, 0, 0] : [40 + (v >> 3), 40 + (v >> 3), 48 + (v >> 3)]);
        const x = tx + (b % PIX);
        const y = ty + Math.floor(b / PIX);
        const o = (y * w + x) * 4;
        img.data[o] = c[0];
        img.data[o + 1] = c[1];
        img.data[o + 2] = c[2];
        img.data[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    const scale = Math.max(1, Math.floor(maxPixels / Math.max(w, h)));
    if (canvas.width !== w * scale || canvas.height !== h * scale) {
      canvas.width = w * scale;
      canvas.height = h * scale;
    }
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off.current, 0, 0, canvas.width, canvas.height);
  }, [maxPixels]);

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

  return (
    <canvas
      ref={canvasRef}
      data-testid={testId}
      className={`world-canvas ${toolRadius !== null ? "brush-active" : ""}`}
      width={512}
      height={512}
      onPointerDown={(ev) => {
        if (toolRadius === null) return;
        const rect = ev.currentTarget.getBoundingClientRect();
        const tilePx = rect.width / columns.current;
        onTile(
          Math.floor((ev.clientX - rect.left) / tilePx),
          Math.floor((ev.clientY - rect.top) / tilePx),
          columns.current,
          programs.current,
        );
      }}
    />
  );
});
