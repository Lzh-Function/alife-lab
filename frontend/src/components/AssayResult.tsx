import { useT } from "../i18n";
import { glyph, hexToBytes } from "../lib/bff";
import { MultiLine, type Series } from "./MultiLine";

// Result of the BFF replicator assay (computed on copies of a program; the run is untouched).
interface PartnerStats {
  n: number;
  copy_rate: number;
  intact_rate: number;
  copy_and_intact_rate: number;
  near_copy_rate: number;
  mean_partner_hamming_to_program: number;
  mean_ops: number;
}
interface PairSide {
  nul_partner: { partner_became_copy: boolean; program_intact: boolean };
  random_partners: PartnerStats;
}
interface Condition {
  exact: number[];
  near: number[];
  final_exact: number;
}
export interface AssayResultData {
  program_hex: string;
  scrambled_hex: string;
  backend: string;
  elapsed_seconds: number;
  params: { copies: number; soup_size: number; epochs: number; replicates: number; near_distance: number; mutation_prob: number; variant: string; growth_factor: number };
  pair_assay: { program_first: PairSide; program_second: PairSide; self_pair: { first_half_intact: boolean; second_half_intact: boolean; hamming_first: number; hamming_second: number; ops: number } };
  soup_assay: { replicates: Array<{ assay_seed: number; conditions: Record<"program" | "scrambled" | "background", Condition> }> };
  verdict: { outcome: "replicates_in_assay" | "no_evidence" | "inconclusive"; reason: string; rule: string };
  source?: { kind: string; run_id?: string; index?: number; epoch?: number };
  artifact?: { logical_path: string; sha256: string };
}

const CLASSES = { program: "arm-a", scrambled: "arm-b", background: "arm-c" } as const;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const show = (hex: string) => Array.from(hexToBytes(hex), glyph).join("");

export function AssayResult({ result, onClose }: { result: AssayResultData; onClose?: () => void }) {
  const t = useT();
  const p = result.params;
  const series: Series[] = result.soup_assay.replicates.flatMap((rep, r) =>
    (Object.keys(CLASSES) as Array<keyof typeof CLASSES>).map((name) => ({
      label: `${t(`assay.cond.${name}`)} #${r + 1}`,
      className: CLASSES[name],
      points: rep.conditions[name].exact.map((v, e) => [e, v] as [number, number]),
    })),
  );
  return (
    <div className="panel" data-testid="assay-result">
      <h3>
        {t("assay.title")}
        {onClose && (
          <button className="link" onClick={onClose} style={{ float: "right" }}>
            ×
          </button>
        )}
      </h3>
      <p>
        <strong data-testid="assay-outcome" className={`verdict ${result.verdict.outcome}`}>
          {t(`assay.outcome.${result.verdict.outcome}`)}
        </strong>{" "}
        <span className="small muted">({result.verdict.reason})</span>
      </p>
      <p className="small muted">
        {t("assay.rule")}: {result.verdict.rule}
        <br />
        {t("assay.note")}
      </p>
      <p className="small">
        {result.source?.kind === "run" && `${result.source.run_id} #${result.source.index} @ ${result.source.epoch} · `}
        {p.variant} · {t("assay.soup")} {p.soup_size} · {t("assay.copies")} {p.copies} · {p.epochs} {t("clock.epoch")} · mutation {p.mutation_prob} · ×
        {p.replicates} · {result.backend} · {result.elapsed_seconds}s
        {result.artifact && (
          <>
            {" · "}
            <span className="mono">{result.artifact.logical_path}</span>
          </>
        )}
      </p>
      <pre className="program">
        P {show(result.program_hex)}
        {"\n"}S {show(result.scrambled_hex)}
      </pre>
      <h4>{t("assay.soup_title")}</h4>
      <MultiLine series={series} testId="assay-chart" />
      <div className="legend">
        {(Object.keys(CLASSES) as Array<keyof typeof CLASSES>).map((name) => (
          <span key={name} className={`legend-item ${CLASSES[name]}`}>
            <i /> {t(`assay.cond.${name}`)}
          </span>
        ))}
      </div>
      <table className="small mono" data-testid="assay-finals">
        <thead>
          <tr>
            <th>#</th>
            <th>{t("assay.cond.program")}</th>
            <th>{t("assay.cond.scrambled")}</th>
            <th>{t("assay.cond.background")}</th>
            <th>{t("assay.near")} ≤{p.near_distance}</th>
          </tr>
        </thead>
        <tbody>
          {result.soup_assay.replicates.map((rep, r) => (
            <tr key={r}>
              <td>{r + 1}</td>
              <td>
                {rep.conditions.program.exact[0]} → {rep.conditions.program.final_exact}
              </td>
              <td>
                {rep.conditions.scrambled.exact[0]} → {rep.conditions.scrambled.final_exact}
              </td>
              <td>{rep.conditions.background.final_exact}</td>
              <td>{rep.conditions.program.near[rep.conditions.program.near.length - 1]}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h4>{t("assay.pair_title")}</h4>
      <table className="small mono">
        <thead>
          <tr>
            <th />
            <th>{t("assay.copy_rate")}</th>
            <th>{t("assay.near_copy_rate")}</th>
            <th>{t("assay.intact_rate")}</th>
            <th>ops</th>
          </tr>
        </thead>
        <tbody>
          {(["program_first", "program_second"] as const).map((side) => {
            const s = result.pair_assay[side].random_partners;
            return (
              <tr key={side}>
                <td>{t(`assay.${side}`)}</td>
                <td>{pct(s.copy_rate)}</td>
                <td>{pct(s.near_copy_rate)}</td>
                <td>{pct(s.intact_rate)}</td>
                <td>{s.mean_ops.toFixed(0)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="small">
        {t("assay.self_pair")}: {t("assay.hamming")} A {result.pair_assay.self_pair.hamming_first}, B {result.pair_assay.self_pair.hamming_second} · ops{" "}
        {result.pair_assay.self_pair.ops}
      </p>
    </div>
  );
}
