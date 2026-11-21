// 2D connectome measurement view (codec connectome_v1):
// Left: rolling spike raster (neuron index vs time ticks).
// Right: brain-region firing-intensity bars + highlighted IO port firing rates (MN9, sugar, bitter...).
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import type { ConnectomeFrameHeader, Frame } from "../lib/api";

export interface ConnectomeViewHandle {
  draw: (frame: Frame) => void;
}

interface Props {
  onDisplayFps?: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
  frameSource?: string;
}

interface RasterEntry {
  tick: number;
  timeMs: number;
  numCells: number;
  fired: Int32Array;
  spikesTick: number;
}

const MAX_HISTORY = 240;

function heatColor(intensity: number): string {
  const v = Math.max(0, Math.min(1, intensity));
  if (v < 0.01) return "#1e293b";
  if (v < 0.5) {
    const t = v / 0.5;
    const r = Math.round(20 + t * 220);
    const g = Math.round(90 + t * 90);
    const b = Math.round(130 - t * 90);
    return `rgb(${r},${g},${b})`;
  }
  const t = (v - 0.5) / 0.5;
  const r = Math.round(240 + t * 15);
  const g = Math.round(180 + t * 70);
  const b = Math.round(40 + t * 180);
  return `rgb(${r},${g},${b})`;
}

export const ConnectomeView = forwardRef<ConnectomeViewHandle, Props>(function ConnectomeView(
  { onDisplayFps, maxPixels = 760, testId = "world-canvas", frameSource = "live" },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<RasterEntry[]>([]);
  const lastHeaderRef = useRef<ConnectomeFrameHeader | null>(null);
  const lastSourceRef = useRef<string>(frameSource);
  const framesRef = useRef(0);

  const width = Math.max(360, Math.min(maxPixels, 1280));
  const height = Math.max(280, Math.round(width * 0.58));

  const renderCanvas = () => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (cv.width !== Math.round(width * dpr) || cv.height !== Math.round(height * dpr)) {
      cv.width = Math.round(width * dpr);
      cv.height = Math.round(height * dpr);
    }
    ctx.save();
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = "#0b101b";
    ctx.fillRect(0, 0, width, height);

    const header = lastHeaderRef.current;
    const history = historyRef.current;

    // Header status strip
    const topH = 26;
    ctx.fillStyle = "#121a2a";
    ctx.fillRect(0, 0, width, topH);
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    if (header) {
      const summary = `${header.dataset_id} (${header.num_cells.toLocaleString()} cells) · t=${header.tick} (${header.time_ms.toFixed(1)} ms) · spikes/tick=${header.spikes_tick} · stim=${header.stimulated_count} · sil=${header.silenced_count}`;
      ctx.fillText(summary, 10, 17);
    } else {
      ctx.fillText("Waiting for connectome frames…", 10, 17);
      ctx.restore();
      return;
    }

    const compact = width < 520;
    const pad = 10;
    const contentTop = topH + pad;
    const contentH = height - contentTop - pad;
    const splitX = compact ? width - pad : Math.round(width * 0.62);

    // ---- Left: Spike Raster Plot ----
    const rx = pad;
    const ry = contentTop + 16;
    const rw = splitX - rx - pad;
    const rh = contentH - 24;

    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("Spike raster (neurons × time)", rx, contentTop + 11);

    ctx.fillStyle = "#060911";
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeRect(rx + 0.5, ry + 0.5, rw - 1, rh - 1);

    // Subtle horizontal grid lines
    ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
    for (let g = 1; g < 4; g++) {
      const gy = ry + (rh * g) / 4;
      ctx.beginPath();
      ctx.moveTo(rx, gy);
      ctx.lineTo(rx + rw, gy);
      ctx.stroke();
    }

    const nCells = Math.max(1, header.num_cells);
    const cols = history.length;
    if (cols > 0) {
      const colW = rw / MAX_HISTORY;
      const xOffset = rx + rw - cols * colW;
      const dotW = Math.max(1.5, Math.min(6, colW));
      const dotH = nCells <= 32 ? Math.max(4, Math.min(18, (rh / nCells) * 0.55)) : 2;

      for (let c = 0; c < cols; c++) {
        const entry = history[c];
        const cx = xOffset + c * colW;
        const fired = entry.fired;
        const stride = fired.length > 600 ? Math.ceil(fired.length / 600) : 1;
        ctx.fillStyle = c === cols - 1 ? "#fde047" : "#38bdf8";
        for (let k = 0; k < fired.length; k += stride) {
          const idx = fired[k];
          const frac = nCells > 1 ? idx / (nCells - 1) : 0.5;
          const cy = ry + 3 + frac * (rh - 6 - dotH);
          ctx.fillRect(cx, cy, dotW, dotH);
        }
      }
      // Tick axis labels
      ctx.fillStyle = "#64748b";
      ctx.font = "10px ui-monospace, monospace";
      const firstTick = history[0].tick;
      const lastTick = history[cols - 1].tick;
      ctx.fillText(`t=${firstTick}`, rx + 2, ry + rh + 12);
      const endLabel = `t=${lastTick}`;
      ctx.fillText(endLabel, rx + rw - ctx.measureText(endLabel).width - 2, ry + rh + 12);
    }

    // ---- Right: Regions heatmap + IO Ports ----
    if (!compact) {
      const px = splitX + 4;
      const pw = width - px - pad;

      // Top half: Region firing intensity bars
      const regions = header.regions ?? [];
      const ports = header.ports ?? [];
      const regH = Math.round(contentH * 0.54);
      ctx.fillStyle = "#94a3b8";
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText("Brain regions (spikes / tick)", px, contentTop + 11);

      const maxRegs = Math.min(regions.length, Math.max(4, Math.floor((regH - 22) / 15)));
      for (let i = 0; i < maxRegs; i++) {
        const r = regions[i];
        const y = contentTop + 18 + i * 15;
        const labelW = 66;
        const barW = Math.max(20, pw - labelW - 36);
        ctx.fillStyle = "#cbd5e1";
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(r.region.slice(0, 10), px, y + 10);

        ctx.fillStyle = "#162033";
        ctx.fillRect(px + labelW, y + 1, barW, 11);
        const fillW = Math.round(barW * Math.max(0, Math.min(1, r.intensity)));
        if (fillW > 0) {
          ctx.fillStyle = heatColor(r.intensity);
          ctx.fillRect(px + labelW, y + 1, fillW, 11);
        }
        ctx.fillStyle = r.spikes_tick > 0 ? "#fde047" : "#64748b";
        ctx.fillText(String(r.spikes_tick), px + labelW + barW + 4, y + 10);
      }

      // Bottom half: Highlighted IO ports (MN9, sugar, bitter, grooming...)
      const portTop = contentTop + regH + 6;
      ctx.fillStyle = "#94a3b8";
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText("IO ports (mean Hz)", px, portTop + 10);

      // Prioritize MN9, active/stimulated ports, then others.
      const sortedPorts = [...ports].sort((a, b) => {
        const score = (p: typeof a) =>
          (p.port_id.includes("mn9") ? 1000 : 0) +
          (p.stim_rate_hz > 0 ? 500 : 0) +
          (p.rate_hz > 0 ? 100 : 0);
        return score(b) - score(a);
      });
      const availH = height - pad - (portTop + 18);
      const maxPorts = Math.min(sortedPorts.length, Math.max(3, Math.floor(availH / 16)));
      for (let i = 0; i < maxPorts; i++) {
        const p = sortedPorts[i];
        const y = portTop + 18 + i * 16;
        const isMn9 = p.port_id.includes("mn9");
        const labelW = 92;
        const barW = Math.max(20, pw - labelW - 52);

        ctx.fillStyle = isMn9 ? "#fda4af" : p.stim_rate_hz > 0 ? "#38bdf8" : "#cbd5e1";
        ctx.font = isMn9 ? "bold 10px ui-monospace, monospace" : "10px ui-monospace, monospace";
        const shortName = p.port_id.replace("_proboscis_extension", "").replace("_mechanosensory", "").slice(0, 14);
        ctx.fillText(shortName, px, y + 10);

        ctx.fillStyle = "#162033";
        ctx.fillRect(px + labelW, y + 1, barW, 11);
        const frac = Math.min(1, p.rate_hz / 150);
        const fillW = Math.round(barW * frac);
        if (fillW > 0) {
          ctx.fillStyle = isMn9 ? "#fb7185" : p.stim_rate_hz > 0 ? "#38bdf8" : "#34d399";
          ctx.fillRect(px + labelW, y + 1, fillW, 11);
        }
        ctx.fillStyle = p.silenced_count > 0 ? "#f87171" : "#e2e8f0";
        ctx.fillText(`${p.rate_hz.toFixed(1)}Hz`, px + labelW + barW + 4, y + 10);
      }
    }

    ctx.restore();
  };

  useEffect(() => {
    if (frameSource !== lastSourceRef.current) {
      historyRef.current = [];
      lastSourceRef.current = frameSource;
    }
    renderCanvas();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height, frameSource]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      onDisplayFps?.(framesRef.current);
      framesRef.current = 0;
    }, 1000);
    return () => window.clearInterval(timer);
  }, [onDisplayFps]);

  useImperativeHandle(ref, () => ({
    draw: (frame: Frame) => {
      if (frame.header.codec !== "connectome_v1") return;
      const h = frame.header as unknown as ConnectomeFrameHeader;
      // Copy to aligned buffer before Int32Array view.
      const fired = new Int32Array(frame.bits.slice().buffer);
      const hist = historyRef.current;
      // Reset rolling history if tick went backwards (e.g. timeline rewind/scrub).
      if (hist.length > 0 && h.tick < hist[hist.length - 1].tick) {
        historyRef.current = [];
      } else if (hist.length > 0 && h.tick === hist[hist.length - 1].tick) {
        hist.pop();
      }
      historyRef.current.push({
        tick: h.tick,
        timeMs: h.time_ms,
        numCells: h.num_cells,
        fired,
        spikesTick: h.spikes_tick,
      });
      if (historyRef.current.length > MAX_HISTORY) {
        historyRef.current.splice(0, historyRef.current.length - MAX_HISTORY);
      }
      lastHeaderRef.current = h;
      framesRef.current += 1;
      renderCanvas();
    },
  }));

  return (
    <canvas
      ref={canvasRef}
      data-testid={testId}
      className="world-canvas connectome-canvas"
      style={{ width, height }}
    />
  );
});
