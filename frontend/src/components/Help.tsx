// Small "?" badge that shows a plain-language explanation on hover, focus or click.
import { useState } from "react";
import { METRIC_HELP, PARAM_HELP } from "../guide/content";

export function paramHelp(substrate: string, path: string): string | undefined {
  return PARAM_HELP[`${substrate}:${path}`] ?? PARAM_HELP[`*:${path}`];
}

export function metricHelp(substrate: string, id: string): string | undefined {
  if (METRIC_HELP[`${substrate}:${id}`]) return METRIC_HELP[`${substrate}:${id}`];
  if (METRIC_HELP[id]) return METRIC_HELP[id];
  const dot = id.indexOf(".");
  if (dot > 0) {
    const [group, name] = [id.slice(0, dot), id.slice(dot + 1)];
    if (group === "count") return `化学種 ${name} の分子の数。`;
    if (group === "territory") return `NCA ${name} が占めている陣地の割合。`;
  }
  return undefined;
}

export function Help({ text, testId }: { text: string | undefined; testId?: string }) {
  const [open, setOpen] = useState(false);
  if (!text) return null;
  return (
    <span
      className={`help ${open ? "open" : ""}`}
      tabIndex={0}
      role="button"
      aria-label="説明"
      aria-expanded={open}
      data-testid={testId}
      onClick={(e) => {
        // Inside a <label>, a click would otherwise focus the input.
        e.preventDefault();
        e.stopPropagation();
        setOpen((o) => !o);
      }}
      onBlur={() => setOpen(false)}
      onKeyDown={(e) => {
        if (e.key === "Escape") setOpen(false);
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          setOpen((o) => !o);
        }
      }}
    >
      ?
      <span className="help-pop" role="tooltip">
        {text}
      </span>
    </span>
  );
}
