import { forwardRef, useCallback, useImperativeHandle, useRef } from "react";
import type { Frame } from "../lib/api";
import { FIELD_SUBSTRATES, PARTICLE_SUBSTRATES, TOOLS, type SubstrateId, type ToolState } from "../lib/substrates";
import { useT } from "../i18n";
import { ParticleCanvas, type ParticleCanvasHandle } from "./ParticleCanvas";
import { WorldCanvas, type WorldCanvasHandle } from "./WorldCanvas";
import { SoupCanvas, type SoupCanvasHandle } from "./SoupCanvas";
import { FieldCanvas, type FieldCanvasHandle } from "./FieldCanvas";

export interface WorldViewHandle {
  draw: (frame: Frame) => void;
}

interface Props {
  substrate: string;
  world: { width: number; height: number };
  tool: ToolState | null; // null = interventions disabled
  onIntervene: (kind: string, payload: Record<string, unknown>) => void;
  onDisplayFps?: (fps: number) => void;
  onInspect?: (x: number, y: number) => void;
  maxPixels?: number;
  testId?: string;
}

const noop = () => undefined;

// Chooses the renderer and maps pointer input to the substrate's intervention payloads.
export const WorldView = forwardRef<WorldViewHandle, Props>(function WorldView(
  { substrate, world, tool, onIntervene, onDisplayFps = noop, onInspect, maxPixels, testId },
  ref,
) {
  const grid = useRef<WorldCanvasHandle>(null);
  const particles = useRef<ParticleCanvasHandle>(null);
  const soup = useRef<SoupCanvasHandle>(null);
  const field = useRef<FieldCanvasHandle>(null);
  useImperativeHandle(ref, () => ({
    draw: (frame: Frame) =>
      substrate === "bff"
        ? soup.current?.draw(frame)
        : FIELD_SUBSTRATES.has(substrate)
          ? field.current?.draw(frame)
          : PARTICLE_SUBSTRATES.has(substrate)
            ? particles.current?.draw(frame)
            : grid.current?.draw(frame),
  }), [substrate]);
  const inspecting = tool !== null && tool.mode === "inspect";
  const active = tool !== null && tool.mode !== "off" && !inspecting;

  const onStroke = useCallback(
    (cells: Array<[number, number]>, value: 0 | 1) => onIntervene("brush", { shape: "cells", cells, value }),
    [onIntervene],
  );
  const onPoint = useCallback(
    (x: number, y: number) => {
      if (!tool || tool.mode === "off") return;
      if (tool.mode === "inspect") {
        onInspect?.(x, y);
        return;
      }
      const payload: Record<string, unknown> = { x: Number(x.toFixed(3)), y: Number(y.toFixed(3)), radius: tool.radius };
      if (tool.mode === "spawn") payload.count = tool.count;
      if (tool.mode === "paint") payload.value = tool.value ?? 1;
      if ((tool.mode === "paint" || tool.mode === "erase") && tool.channel != null) payload.channel = tool.channel;
      onIntervene(tool.mode, payload);
    },
    [tool, onIntervene],
  );

  if (substrate === "bff") {
    return (
      <SoupCanvas
        ref={soup}
        toolRadius={active || inspecting ? (tool?.radius ?? 0) : null}
        onTile={(col, row, columns, programs) => {
          if (!tool) return;
          if (tool.mode === "inspect") {
            onInspect?.(col, row);
            return;
          }
          // Explicit program indices (a square block of tiles) keep the intervention replayable.
          const indices: number[] = [];
          for (let dy = -tool.radius; dy <= tool.radius; dy++) {
            for (let dx = -tool.radius; dx <= tool.radius; dx++) {
              const c = col + dx;
              const r = row + dy;
              const idx = r * columns + c;
              if (c >= 0 && c < columns && r >= 0 && idx < programs) indices.push(idx);
            }
          }
          if (!indices.length) return;
          onIntervene(tool.mode === "inject" ? "inject_program" : "zero_programs", {
            indices,
            ...(tool.mode === "inject" ? { program: tool.program ?? "", encoding: "text" } : {}),
          });
        }}
        onDisplayFps={onDisplayFps}
        maxPixels={maxPixels}
        testId={testId}
      />
    );
  }
  if (FIELD_SUBSTRATES.has(substrate)) {
    return (
      <FieldCanvas
        ref={field}
        width={world.width}
        height={world.height}
        toolRadius={active || inspecting ? tool!.radius : null}
        onPoint={onPoint}
        onDisplayFps={onDisplayFps}
        maxPixels={maxPixels}
        testId={testId}
      />
    );
  }
  if (PARTICLE_SUBSTRATES.has(substrate)) {
    return (
      <ParticleCanvas
        ref={particles}
        worldWidth={world.width}
        worldHeight={world.height}
        toolRadius={active || inspecting ? tool!.radius : null}
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
      onCellClick={inspecting ? onInspect : undefined}
    />
  );
});

export function ToolBar({
  substrate,
  tool,
  setTool,
  disabled,
  name = "tool",
  channels = 1,
}: {
  substrate: string;
  channels?: number;
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
            disabled={disabled && m !== "off" && m !== "inspect"}
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
      {substrate === "bff" && (
        <label className="inline">
          {t("lab.tool.program")}
          <input
            className="mono"
            data-testid="tool-program"
            maxLength={64}
            value={tool.program ?? ""}
            placeholder="[[{.>]-]"
            onChange={(e) => setTool({ ...tool, program: e.target.value })}
            style={{ width: "16rem" }}
          />
        </label>
      )}
      {d.value && (
        <label className="inline">
          {t("lab.tool.value")}
          <input
            type="range"
            data-testid="tool-value"
            min={0}
            max={1}
            step={0.05}
            value={tool.value ?? d.value.initial}
            onChange={(e) => setTool({ ...tool, value: Number(e.target.value) })}
          />
          {(tool.value ?? d.value.initial).toFixed(2)}
        </label>
      )}
      {d.channelSelect && channels > 1 && (
        <label className="inline">
          {t("lab.tool.channel")}
          <select
            data-testid="tool-channel"
            value={tool.channel == null ? "all" : String(tool.channel)}
            onChange={(e) => setTool({ ...tool, channel: e.target.value === "all" ? null : Number(e.target.value) })}
          >
            <option value="all">{t("lab.tool.channel_all")}</option>
            {Array.from({ length: channels }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
      )}
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
