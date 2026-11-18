// One dot per seeded run per condition, with the condition mean as a bar. Display only.
const CLASSES = ["arm-a", "arm-b", "arm-c", "arm-d"];

export function StripPlot({
  groups,
  height = 180,
  testId,
}: {
  groups: Array<{ label: string; values: number[]; mean: number | null; reference?: boolean }>;
  height?: number;
  testId?: string;
}) {
  const width = 520;
  const pad = { l: 52, r: 10, t: 10, b: 34 };
  const all = groups.flatMap((g) => g.values);
  if (!all.length) return <svg className="multiline" viewBox={`0 0 ${width} ${height}`} data-testid={testId} aria-hidden="true" />;
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (hi === lo) {
    lo -= 1;
    hi += 1;
  }
  const sy = (v: number) => height - pad.b - ((v - lo) / (hi - lo)) * (height - pad.t - pad.b);
  const band = (width - pad.l - pad.r) / groups.length;
  return (
    <svg className="multiline" width="100%" viewBox={`0 0 ${width} ${height}`} role="img" data-testid={testId}>
      <line x1={pad.l} x2={pad.l} y1={pad.t} y2={height - pad.b} className="axis" />
      <text x={pad.l - 4} y={sy(hi) + 4} className="tick-label" textAnchor="end">{hi.toPrecision(3)}</text>
      <text x={pad.l - 4} y={sy(lo)} className="tick-label" textAnchor="end">{lo.toPrecision(3)}</text>
      {groups.map((g, gi) => {
        const cx = pad.l + band * (gi + 0.5);
        return (
          <g key={g.label} className={CLASSES[gi % CLASSES.length]}>
            {g.values.map((v, i) => {
              // Deterministic horizontal jitter so overlapping values stay visible.
              const jitter = ((i * 37) % 21 - 10) * (band / 80);
              return <circle key={i} cx={cx + jitter} cy={sy(v)} r={3.5} className="dot" opacity={0.85} />;
            })}
            {g.mean !== null && <line x1={cx - band * 0.25} x2={cx + band * 0.25} y1={sy(g.mean)} y2={sy(g.mean)} className="mean-bar" />}
            <text x={cx} y={height - 12} className="tick-label" textAnchor="middle">
              {g.label}
              {g.reference ? " *" : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
