import { useT } from "../i18n";
import type { MetricRow } from "../lib/api";
import { MultiLine, type Series } from "./MultiLine";

// Species counts over time for reaction networks (metrics keys `count.<species>`).
const CLASSES = ["arm-a", "arm-b", "arm-c", "arm-d", "sp-4", "sp-5", "sp-6", "sp-7"];

export function SpeciesChart({ rows, experiment }: { rows: MetricRow[]; experiment: Record<string, unknown> }) {
  const t = useT();
  const species = ((experiment.dynamics as { species?: Array<{ name: string; constant?: boolean }> })?.species ?? []).filter((s) => !s.constant);
  const series: Series[] = species.slice(0, CLASSES.length).map((s, i) => ({
    label: s.name,
    className: CLASSES[i],
    points: rows.map((r) => [r.tick, r.values[`count.${s.name}`] ?? null]),
  }));
  return (
    <div className="panel" data-testid="species-chart">
      <h3>{t("lab.species_counts")}</h3>
      <MultiLine series={series} height={180} />
      {species.length > CLASSES.length && <p className="muted small">+{species.length - CLASSES.length}</p>}
    </div>
  );
}
