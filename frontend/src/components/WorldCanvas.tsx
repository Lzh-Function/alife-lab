import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";
import { bitAt, circleCells, type Frame } from "../lib/api";

export interface WorldCanvasHandle {
  draw: (frame: Frame) => void;
}

interface Props {
  width: number;
  height: number;
  brushMode: "off" | "draw" | "erase";
  brushRadius: number;
  onStroke: (cells: Array<[number, number]>, value: 0 | 1) => void;
  onDisplayFps: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
  onCellClick?: (x: number, y: number) => void;
}

const MAX_STROKE_CELLS = 65536;

// Canvas2D renderer: decoupled from React; frames are painted via requestAnimationFrame.
export const WorldCanvas = forwardRef<WorldCanvasHandle, Props>(function WorldCanvas(
  { width, height, brushMode, brushRadius, onStroke, onDisplayFps, maxPixels = 720, testId = "world-canvas", onCellClick },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreen = useRef<HTMLCanvasElement | null>(null);
  const imgCache = useRef<{ w: number; h: number; img: ImageData; u32: Uint32Array } | null>(null);
  const colorsCache = useRef<{ alive: number; dead: number } | null>(null);
  const pending = useRef<Frame | null>(null);
  const lastFrame = useRef<Frame | null>(null);
  const stroke = useRef<Map<string, [number, number]> | null>(null);
  const lastCell = useRef<[number, number] | null>(null);
  const frameCount = useRef(0);
  const raf = useRef<number | null>(null);

  const paint = useCallback(() => {
    raf.current = null;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const frame = pending.current ?? lastFrame.current;
    pending.current = null;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (!offscreen.current) offscreen.current = document.createElement("canvas");
    const off = offscreen.current;
    if (off.width !== width || off.height !== height) {
      off.width = width;
      off.height = height;
      imgCache.current = null;
    }
    const octx = off.getContext("2d");
    if (!octx) return;
    if (!imgCache.current || imgCache.current.w !== width || imgCache.current.h !== height) {
      const img = octx.createImageData(width, height);
      imgCache.current = { w: width, h: height, img, u32: new Uint32Array(img.data.buffer) };
    }
    if (!colorsCache.current) {
      const styles = getComputedStyle(canvas);
      const a = parseColor(styles.getPropertyValue("--cell-alive"), [32, 201, 151]);
      const d = parseColor(styles.getPropertyValue("--cell-dead"), [16, 20, 28]);
      colorsCache.current = {
        alive: (255 << 24) | (a[2] << 16) | (a[1] << 8) | a[0],
        dead: (255 << 24) | (d[2] << 16) | (d[1] << 8) | d[0],
      };
    }
    const { img, u32 } = imgCache.current;
    const { alive, dead } = colorsCache.current;
    const total = width * height;
    if (frame && frame.header.width === width && frame.header.height === height) {
      for (let i = 0; i < total; i++) {
        u32[i] = bitAt(frame.bits, i) ? alive : dead;
      }
      lastFrame.current = frame;
      frameCount.current += 1;
    } else {
      u32.fill(dead);
    }
    if (stroke.current) {
      const sCol = brushMode === "erase" ? (255 << 24) | (87 << 16) | (87 << 8) | 235 : (255 << 24) | (60 << 16) | (200 << 8) | 250;
      for (const [x, y] of stroke.current.values()) {
        u32[y * width + x] = sCol;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }, [width, height, brushMode]);

  const schedule = useCallback(() => {
    if (raf.current === null) raf.current = requestAnimationFrame(paint);
  }, [paint]);

  useImperativeHandle(ref, () => ({
    draw: (frame: Frame) => {
      pending.current = frame;
      schedule();
    },
  }), [schedule]);

  useEffect(() => {
    const id = window.setInterval(() => {
      onDisplayFps(frameCount.current);
      frameCount.current = 0;
    }, 1000);
    return () => window.clearInterval(id);
  }, [onDisplayFps]);

  useEffect(() => {
    schedule();
  }, [schedule]);

  const cellAt = (ev: React.PointerEvent<HTMLCanvasElement>): [number, number] | null => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - rect.left) / rect.width) * width);
    const y = Math.floor(((ev.clientY - rect.top) / rect.height) * height);
    if (x < 0 || y < 0 || x >= width || y >= height) return null;
    return [x, y];
  };

  const addCells = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const cell = cellAt(ev);
    if (!cell || !stroke.current) return;
    // Interpolate from the previous pointer cell so fast drags produce continuous strokes.
    const from = lastCell.current ?? cell;
    const n = Math.max(Math.abs(cell[0] - from[0]), Math.abs(cell[1] - from[1]), 1);
    for (let i = 0; i <= n; i++) {
      const x = Math.round(from[0] + ((cell[0] - from[0]) * i) / n);
      const y = Math.round(from[1] + ((cell[1] - from[1]) * i) / n);
      for (const c of circleCells(x, y, brushRadius, width, height)) {
        if (stroke.current.size >= MAX_STROKE_CELLS) break;
        stroke.current.set(`${c[0]},${c[1]}`, c);
      }
    }
    lastCell.current = cell;
    schedule();
  };

  const scale = Math.max(1, Math.floor(Math.min(maxPixels / width, maxPixels / height)));

  return (
    <canvas
      ref={canvasRef}
      data-testid={testId}
      className={`world-canvas ${brushMode !== "off" ? "brush-active" : ""}`}
      width={width * scale}
      height={height * scale}
      onPointerDown={(ev) => {
        if (brushMode === "off") {
          const cell = cellAt(ev);
          if (cell && onCellClick) onCellClick(cell[0], cell[1]);
          return;
        }
        ev.currentTarget.setPointerCapture(ev.pointerId);
        stroke.current = new Map();
        lastCell.current = null;
        addCells(ev);
      }}
      onPointerMove={(ev) => {
        if (stroke.current) addCells(ev);
      }}
      onPointerUp={() => {
        const cells = stroke.current ? [...stroke.current.values()] : [];
        stroke.current = null;
        if (cells.length) onStroke(cells, brushMode === "erase" ? 0 : 1);
        schedule();
      }}
      onPointerCancel={() => {
        stroke.current = null;
        schedule();
      }}
    />
  );
});

function parseColor(value: string, fallback: number[]): number[] {
  const m = value.trim().match(/^#([0-9a-f]{6})$/i);
  if (!m) return fallback;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
