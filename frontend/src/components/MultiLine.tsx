// Multi-series SVG line chart on a shared x axis (simulation tick). Display only.
export interface Series {
  label: string;
  className: string;
  points: Array<[number, number | null]>;
}

export function MultiLine({ series, height = 140, testId }: { series: Series[]; height?: number; testId?: string }) {
  const width = 560;
  const pad = { l: 44, r: 8, t: 8, b: 20 };
  const all = series.flatMap((s) => s.points).filter((p): p is [number, number] => p[1] !== null && Number.isFinite(p[1]));
  if (all.length < 2) return <svg className="multiline" width="100%" viewBox={`0 0 ${width} ${height}`} data-testid={testId} aria-hidden="true" />;
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(0, ...ys), Math.max(...ys)];
  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (width - pad.l - pad.r);
  const sy = (y: number) => height - pad.b - ((y - y0) / (y1 - y0 || 1)) * (height - pad.t - pad.b);
  const path = (pts: Series["points"]) => {
    let d = "";
    let pen = false;
    const step = Math.max(1, Math.ceil(pts.length / 600));
    for (let i = 0; i < pts.length; i += step) {
      const [x, y] = pts[i];
      if (y === null || !Number.isFinite(y)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)} `;
      pen = true;
    }
    return d;
  };
  return (
    <figure className="multiline-fig">
      <svg className="multiline" width="100%" viewBox={`0 0 ${width} ${height}`} role="img" data-testid={testId}>
        <line x1={pad.l} x2={width - pad.r} y1={sy(y0)} y2={sy(y0)} className="axis" />
        <line x1={pad.l} x2={pad.l} y1={pad.t} y2={height - pad.b} className="axis" />
        <text x={pad.l - 4} y={sy(y1) + 4} className="tick-label" textAnchor="end">{y1.toPrecision(4)}</text>
        <text x={pad.l - 4} y={sy(y0)} className="tick-label" textAnchor="end">{y0.toPrecision(3)}</text>
        <text x={pad.l} y={height - 4} className="tick-label">t={x0}</text>
        <text x={width - pad.r} y={height - 4} className="tick-label" textAnchor="end">t={x1}</text>
        {series.map((s) => (
          <path key={s.label} d={path(s.points)} className={`series ${s.className}`} fill="none" strokeWidth={1.6} />
        ))}
        {series.map((s) =>
          s.points.length <= 60
            ? s.points
                .filter((p): p is [number, number] => p[1] !== null && Number.isFinite(p[1]))
                .map(([x, y]) => <circle key={`${s.label}-${x}`} cx={sx(x)} cy={sy(y)} r={2.5} className={`dot ${s.className}`} />)
            : null,
        )}
      </svg>
      <figcaption className="legend">
        {series.map((s) => (
          <span key={s.label} className={`legend-item ${s.className}`}>
            <i /> {s.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
