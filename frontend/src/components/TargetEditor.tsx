import { useEffect, useRef, useState } from "react";
import { useT } from "../i18n";

// Draw an NCA training target (RGBA, straight alpha) in the browser. The bytes are sent as
// base64 in the ExperimentSpec (target.kind = "custom"); the server validates the size.
const PX = 8;

function toB64(bytes: Uint8ClampedArray): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s);
}

export function TargetEditor({
  size,
  active,
  onChange,
}: {
  size: number;
  active: boolean;
  onChange: (target: Record<string, unknown> | null) => void;
}) {
  const t = useT();
  const canvas = useRef<HTMLCanvasElement>(null);
  const pixels = useRef(new Uint8ClampedArray(size * size * 4));
  const [color, setColor] = useState("#e0303c");
  const [radius, setRadius] = useState(2);
  const [erase, setErase] = useState(false);
  const drawing = useRef(false);

  const redraw = () => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, size * PX, size * PX);
    const img = new ImageData(new Uint8ClampedArray(pixels.current), size, size);
    const off = document.createElement("canvas");
    off.width = size;
    off.height = size;
    off.getContext("2d")!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, size * PX, size * PX);
    ctx.strokeStyle = "rgba(0,0,0,0.06)";
    for (let i = 0; i <= size; i++) {
      ctx.beginPath();
      ctx.moveTo(i * PX, 0);
      ctx.lineTo(i * PX, size * PX);
      ctx.moveTo(0, i * PX);
      ctx.lineTo(size * PX, i * PX);
      ctx.stroke();
    }
  };

  useEffect(() => {
    pixels.current = new Uint8ClampedArray(size * size * 4);
    redraw();
    if (active) onChange(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size]);

  const emit = () => {
    if (active) onChange({ kind: "custom", size, rgba_b64: toB64(pixels.current) });
  };

  const paint = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const cx = Math.floor(((ev.clientX - rect.left) / rect.width) * size);
    const cy = Math.floor(((ev.clientY - rect.top) / rect.height) * size);
    const rgb = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
    for (let y = cy - radius; y <= cy + radius; y++) {
      for (let x = cx - radius; x <= cx + radius; x++) {
        if (x < 0 || y < 0 || x >= size || y >= size || (x - cx) ** 2 + (y - cy) ** 2 > radius * radius + 0.5) continue;
        const o = (y * size + x) * 4;
        if (erase) pixels.current.fill(0, o, o + 4);
        else pixels.current.set([...rgb, 255], o);
      }
    }
    redraw();
  };

  return (
    <div className="target-editor" data-testid="target-editor">
      <label className="inline">
        <input
          type="checkbox"
          data-testid="target-custom"
          checked={active}
          onChange={(e) => onChange(e.target.checked ? { kind: "custom", size, rgba_b64: toB64(pixels.current) } : null)}
        />
        {t("target.use_drawing")}
      </label>
      <div className="toolbar">
        <input type="color" value={color} onChange={(e) => setColor(e.target.value)} data-testid="target-color" />
        <label className="inline">
          {t("lab.brush.radius")}
          <input type="range" min={0} max={6} value={radius} onChange={(e) => setRadius(Number(e.target.value))} />
          {radius}
        </label>
        <label className="inline">
          <input type="checkbox" checked={erase} onChange={(e) => setErase(e.target.checked)} />
          {t("target.erase")}
        </label>
        <button
          type="button"
          onClick={() => {
            pixels.current.fill(0);
            redraw();
            emit();
          }}
        >
          {t("target.clear")}
        </button>
      </div>
      <canvas
        ref={canvas}
        width={size * PX}
        height={size * PX}
        data-testid="target-canvas"
        className="target-canvas"
        onPointerDown={(ev) => {
          drawing.current = true;
          paint(ev);
        }}
        onPointerMove={(ev) => drawing.current && ev.buttons && paint(ev)}
        onPointerUp={() => {
          drawing.current = false;
          emit();
        }}
        onPointerLeave={() => {
          if (drawing.current) emit();
          drawing.current = false;
        }}
      />
      <p className="muted small">{t("target.note")}</p>
    </div>
  );
}
