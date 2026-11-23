import { memo } from "react";

// Minimal SVG line chart; downsamples for display while the full series stays in memory.
export const Sparkline = memo(function Sparkline({ values, height = 48 }: { values: Array<number | null>; height?: number }) {
  const width = 280;
  const step = Math.max(1, Math.ceil(values.length / width));
  const sampled: Array<number | null> = [];
  for (let i = 0; i < values.length; i += step) sampled.push(values[i]);
  const finite = sampled.filter((v): v is number => v !== null && Number.isFinite(v));
  if (finite.length < 2) return <svg className="sparkline" width={width} height={height} aria-hidden="true" />;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;
  let d = "";
  let pen = false;
  sampled.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) {
      pen = false;
      return;
    }
    const x = (i / Math.max(1, sampled.length - 1)) * width;
    const y = height - 4 - ((v - min) / span) * (height - 8);
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)} `;
    pen = true;
  });
  return (
    <svg className="sparkline" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`min ${min} max ${max}`}>
      <path d={d} fill="none" stroke="currentColor" strokeWidth={1.5} />
      <text x={2} y={10} className="spark-label">{max.toPrecision(4)}</text>
      <text x={2} y={height - 2} className="spark-label">{min.toPrecision(4)}</text>
    </svg>
  );
});
