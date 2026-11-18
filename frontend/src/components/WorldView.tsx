import { forwardRef, useCallback, useImperativeHandle, useRef } from "react";
import type { Frame } from "../lib/api";
import { TOOLS, type SubstrateId, type ToolState } from "../lib/substrates";
import { useT } from "../i18n";
import { ParticleCanvas, type ParticleCanvasHandle } from "./ParticleCanvas";
import { WorldCanvas, type WorldCanvasHandle } from "./WorldCanvas";

export interface WorldViewHandle {
  draw: (frame: Frame) => void;
}

interface Props {
  substrate: string;
  world: { width: number; height: number };
  tool: ToolState | null; // null = interventions disabled
  onIntervene: (kind: string, payload: Record<string, unknown>) => void;
  onDisplayFps?: (fps: number) => void;
  maxPixels?: number;
  testId?: string;
}

const noop = () => undefined;

// Chooses the renderer and maps pointer input to the substrate's intervention payloads.
export const WorldView = forwardRef<WorldViewHandle, Props>(function WorldView(
  { substrate, world, tool, onIntervene, onDisplayFps = noop, maxPixels, testId },
  ref,
) {
  const grid = useRef<WorldCanvasHandle>(null);
  const particles = useRef<ParticleCanvasHandle>(null);
  useImperativeHandle(ref, () => ({
    draw: (frame: Frame) => (substrate === "boids" ? particles.current?.draw(frame) : grid.current?.draw(frame)),
  }), [substrate]);
  const active = tool !== null && tool.mode !== "off";

  const onStroke = useCallback(
    (cells: Array<[number, number]>, value: 0 | 1) => onIntervene("brush", { shape: "cells", cells, value }),
    [onIntervene],
  );
  const onPoint = useCallback(
    (x: number, y: number) => {
      if (!tool || tool.mode === "off") return;
      const payload: Record<string, unknown> = { x: Number(x.toFixed(3)), y: Number(y.toFixed(3)), radius: tool.radius };
      if (tool.mode === "spawn") payload.count = tool.count;
      onIntervene(tool.mode, payload);
    },
    [tool, onIntervene],
  );

  if (substrate === "boids") {
    return (
      <ParticleCanvas
        ref={particles}
        worldWidth={world.width}
        worldHeight={world.height}
        toolRadius={active ? tool!.radius : null}
        onPoint={onPoint}
        onDisplayFps={onDisplayFps}
        maxPixels={maxPixels}
        testId={testId}
      />
    );
  }
  return (
    <WorldCanvas
      ref={grid}
      width={world.width}
      height={world.height}
      brushMode={active ? (tool!.mode as "draw" | "erase") : "off"}
      brushRadius={tool?.radius ?? 0}
      onStroke={onStroke}
      onDisplayFps={onDisplayFps}
      maxPixels={maxPixels}
      testId={testId}
    />
  );
});

export function ToolBar({
  substrate,
  tool,
  setTool,
  disabled,
  name = "tool",
}: {
  substrate: string;
  tool: ToolState;
  setTool: (t: ToolState) => void;
  disabled: boolean;
  name?: string;
}) {
  const t = useT();
  const d = TOOLS[(substrate as SubstrateId) in TOOLS ? (substrate as SubstrateId) : "conway"];
  return (
    <div className="toolbar brush" role="group" aria-label={t("lab.brush")}>
      <strong>{t("lab.brush")}:</strong>
      {d.modes.map((m) => (
        <label key={m} className="inline">
          <input
            type="radio"
            name={name}
            data-testid={`brush-${m}`}
            checked={tool.mode === m}
            disabled={disabled}
            onChange={() => setTool({ ...tool, mode: m })}
          />
          {t(`lab.tool.${m}`)}
        </label>
      ))}
      <label className="inline">
        {t("lab.brush.radius")}
        <input
          type="range"
          min={d.radius.min}
          max={d.radius.max}
          step={d.radius.step}
          value={tool.radius}
          onChange={(e) => setTool({ ...tool, radius: Number(e.target.value) })}
        />
        {tool.radius}
      </label>
      {d.count && (
        <label className="inline">
          {t("lab.tool.count")}
          <input
            type="number"
            data-testid="tool-count"
            min={d.count.min}
            max={d.count.max}
            value={tool.count}
            onChange={(e) => setTool({ ...tool, count: Math.max(d.count!.min, Math.min(d.count!.max, Number(e.target.value))) })}
            style={{ width: "5rem" }}
          />
        </label>
      )}
      <small className="muted">{t(d.noteKey)}</small>
    </div>
  );
}
