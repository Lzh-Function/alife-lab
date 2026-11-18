import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, type SweepPlan } from "../lib/api";
import { FORM_FIELDS, type SubstrateId } from "../lib/substrates";
import { useT } from "../i18n";
import { MultiLine, type Series } from "./MultiLine";
import { StripPlot } from "./StripPlot";

const ARM = ["arm-a", "arm-b", "arm-c", "arm-d"];

function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined) return "—";
  if (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toString();
}

function parseValues(text: string): Array<number | string> {
  return text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => (Number.isFinite(Number(s)) ? Number(s) : s));
}

export function SweepList({ onOpen }: { onOpen: (id: string) => void }) {
  const t = useT();
  const list = useQuery({ queryKey: ["sweeps"], queryFn: api.sweeps, refetchInterval: 3000 });
  return (
    <div className="catalog">
      <section>
        <h2>{t("sweep.title")}</h2>
        <p className="muted small">{t("sweep.intro")}</p>
        {!list.data?.length ? (
          <p className="muted">{t("sweep.empty")}</p>
        ) : (
          <ul className="list" data-testid="sweep-list">
            {list.data.map((s) => (
              <li key={s.id}>
                <button className="ghost" data-testid={`open-${s.id}`} onClick={() => onOpen(s.id)}>
                  {s.name}
                </button>{" "}
                <span className={`badge state-${s.state}`}>{s.state}</span> <span className="muted small">{s.design}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
      <section>
        <SweepForm onCreated={onOpen} />
      </section>
    </div>
  );
}

function SweepForm({ onCreated }: { onCreated: (id: string) => void }) {
  const t = useT();
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const substrates = useQuery({ queryKey: ["substrates"], queryFn: api.substrates });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const [preset, setPreset] = useState("boids_flocking");
  const [name, setName] = useState("my_sweep");
  const [seedStart, setSeedStart] = useState(0);
  const [seedCount, setSeedCount] = useState(6);
  const [factorPath, setFactorPath] = useState("dynamics.alignment_weight");
  const [values, setValues] = useState("0.05, 0.0");
  const [labels, setLabels] = useState("alignment, no_alignment");
  const [reference, setReference] = useState(1);
  const [metric, setMetric] = useState("polarization");
  const [at, setAt] = useState("final");
  const [maxSteps, setMaxSteps] = useState(600);
  const [design, setDesign] = useState<"exploratory" | "confirmatory">("exploratory");
  const [hypothesis, setHypothesis] = useState("");
  const [profile, setProfile] = useState("cpu_dev");
  const [plan, setPlan] = useState<SweepPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const presetObj = presets.data?.find((p) => p.id === preset);
  const substrate = presetObj?.substrate ?? "conway";
  const factorFields = FORM_FIELDS[(substrate as SubstrateId) in FORM_FIELDS ? (substrate as SubstrateId) : "conway"].filter(
    (f) => /^(dynamics|world|initialization)\./.test(f.path) && f.kind !== "select",
  );
  const metrics = substrates.data?.find((s) => s.slug === substrate)?.metrics ?? [];

  useEffect(() => {
    if (factorFields.length && !factorFields.some((f) => f.path === factorPath)) setFactorPath(factorFields[0].path);
    if (metrics.length && !metrics.some((m) => m.id === metric)) setMetric(metrics[0].id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [substrate, metrics.length]);

  const spec = useMemo(() => {
    const vals = parseValues(values);
    const labs = labels.split(",").map((s) => s.trim()).filter(Boolean);
    const observed = metrics.map((m) => m.id);
    return {
      schema_version: 1,
      sweep: {
        name,
        base_preset: preset,
        overrides: {
          runner: { max_steps: maxSteps, checkpoint_interval: maxSteps },
          observations: { metrics: substrate === "conway" ? ["alive_cell_count", "centroid_displacement"] : observed },
        },
        seeds: Array.from({ length: Math.max(0, seedCount) }, (_, i) => seedStart + i),
        factors: vals.length ? [{ path: factorPath, values: vals, ...(labs.length === vals.length ? { labels: labs } : {}) }] : [],
        reference_condition: Math.min(reference, Math.max(0, vals.length - 1)),
        endpoint: { metric, at: at === "final" ? "final" : Number(at) },
        design,
        hypothesis: hypothesis.trim() || null,
        profile_id: profile,
      },
    };
  }, [name, preset, maxSteps, substrate, metrics, seedStart, seedCount, values, labels, factorPath, reference, metric, at, design, hypothesis, profile]);

  useEffect(() => {
    const h = window.setTimeout(() => {
      api
        .validateSweep(spec)
        .then((p) => {
          setPlan(p);
          setError(null);
        })
        .catch((e) => {
          setPlan(null);
          setError(e instanceof ApiError ? e.message : String(e));
        });
    }, 300);
    return () => window.clearTimeout(h);
  }, [spec]);

  const submit = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setBusy(true);
    try {
      const created = await api.createSweep(spec);
      onCreated(created.id);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="form" onSubmit={submit} data-testid="sweep-form">
      <h2>{t("sweep.create")}</h2>
      <fieldset>
        <legend>SweepSpec</legend>
        <label>
          {t("form.name")}
          <input data-testid="sweep-name" value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label>
          {t("form.preset")}
          <select data-testid="sweep-preset" value={preset} onChange={(e) => setPreset(e.target.value)}>
            {(presets.data ?? []).map((p) => (
              <option key={p.id} value={p.id}>
                {t(p.title_key)}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t("sweep.seed_start")}
          <input type="number" min={0} value={seedStart} onChange={(e) => setSeedStart(Number(e.target.value))} />
        </label>
        <label>
          {t("sweep.seed_count")}
          <input data-testid="sweep-seed-count" type="number" min={1} max={100} value={seedCount} onChange={(e) => setSeedCount(Number(e.target.value))} />
        </label>
        <label>
          {t("sweep.factor")}
          <select data-testid="sweep-factor" value={factorPath} onChange={(e) => setFactorPath(e.target.value)}>
            {factorFields.map((f) => (
              <option key={f.path} value={f.path}>
                {t(f.labelKey)} ({f.path})
              </option>
            ))}
          </select>
        </label>
        <label>
          {t("sweep.values")}
          <input data-testid="sweep-values" value={values} onChange={(e) => setValues(e.target.value)} />
        </label>
        <label>
          {t("sweep.labels")}
          <input data-testid="sweep-labels" value={labels} onChange={(e) => setLabels(e.target.value)} />
        </label>
        <label>
          {t("sweep.reference")}
          <input type="number" min={0} value={reference} onChange={(e) => setReference(Number(e.target.value))} />
        </label>
        <label>
          {t("sweep.metric")}
          <select data-testid="sweep-metric" value={metric} onChange={(e) => setMetric(e.target.value)}>
            {metrics.map((m) => (
              <option key={m.id} value={m.id}>
                {t(m.display_name_key)}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t("sweep.at")}
          <input value={at} onChange={(e) => setAt(e.target.value)} placeholder="final" />
        </label>
        <label>
          {t("form.max_steps")}
          <input data-testid="sweep-max-steps" type="number" min={1} value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} />
        </label>
        <label>
          {t("sweep.design")}
          <select data-testid="sweep-design" value={design} onChange={(e) => setDesign(e.target.value as "exploratory" | "confirmatory")}>
            <option value="exploratory">exploratory</option>
            <option value="confirmatory">confirmatory</option>
          </select>
        </label>
        <label>
          {t("form.profile")}
          <select value={profile} onChange={(e) => setProfile(e.target.value)}>
            {(profiles.data ?? []).filter((p) => p.implemented).map((p) => (
              <option key={p.id} value={p.id}>
                {p.id}
              </option>
            ))}
          </select>
        </label>
      </fieldset>
      <label className="block">
        {t("sweep.hypothesis")}
        <textarea data-testid="sweep-hypothesis" rows={3} value={hypothesis} onChange={(e) => setHypothesis(e.target.value)} />
      </label>
      <p className="muted small">{t("sweep.design_note")}</p>
      {plan && (
        <div className="notice small" data-testid="sweep-plan">
          {t("sweep.plan")}: {plan.runs} runs · {plan.conditions.length} × {seedCount} · {plan.total_ticks.toLocaleString()} ticks
          <div>{plan.conditions.map((c) => `#${c.index} ${c.label}`).join(" / ")}</div>
          <div className="mono">sweep {plan.sweep_hash.slice(0, 16)}…</div>
        </div>
      )}
      {error && <p className="error small" role="alert">{error}</p>}
      <button type="submit" className="primary" data-testid="sweep-create" disabled={busy || !plan}>
        {t("sweep.run")}
      </button>
    </form>
  );
}

export function SweepDetail({ sweepId }: { sweepId: string }) {
  const t = useT();
  const qc = useQueryClient();
  const summary = useQuery({ queryKey: ["sweep", sweepId], queryFn: () => api.sweep(sweepId), refetchInterval: 1500 });
  const running = summary.data?.state === "running";
  const report = useQuery({
    queryKey: ["sweep-report", sweepId, summary.data?.state],
    queryFn: () => api.sweepReport(sweepId),
    enabled: !!summary.data && !running,
  });
  const [busy, setBusy] = useState(false);
  if (!summary.data) return <p>{summary.isError ? t("common.error") : t("common.loading")}</p>;
  const s = summary.data;
  const done = Object.entries(s.progress.by_state)
    .filter(([k]) => ["completed", "failed", "cancelled"].includes(k))
    .reduce((a, [, v]) => a + v, 0);
  const r = report.data;
  const series: Series[] = (r?.timecourse ?? []).map((tc, i) => ({
    label: r!.conditions[tc.condition_index].label,
    className: ARM[i % ARM.length],
    points: tc.ticks.map((tick, j) => [tick, tc.mean[j]]),
  }));
  return (
    <div className="sweep-detail">
      <div className="lab-head">
        <h2 data-testid="sweep-title">{s.name}</h2>
        <span className={`badge state-${s.state}`} data-testid="sweep-state" data-state={s.state}>
          {s.state}
        </span>
        <span className="badge">{s.design}</span>
        <span className="mono small">{s.spec_hash.slice(0, 12)}</span>
        {running && (
          <button
            className="danger"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              await api.cancelSweep(sweepId).catch(() => undefined);
              await qc.invalidateQueries({ queryKey: ["sweep", sweepId] });
              setBusy(false);
            }}
          >
            ■ {t("lab.cancel")}
          </button>
        )}
      </div>
      <div className="progress" aria-label="progress">
        <div style={{ width: `${(100 * done) / Math.max(1, s.progress.total)}%` }} />
      </div>
      <p className="small" data-testid="sweep-progress">
        {done} / {s.progress.total} · {Object.entries(s.progress.by_state).map(([k, v]) => `${k}: ${v}`).join(", ")}
      </p>
      {s.spec.sweep.hypothesis && (
        <p className="small">
          <strong>{t("sweep.hypothesis")}:</strong> {String(s.spec.sweep.hypothesis)}
        </p>
      )}
      {r && (
        <>
          <div className="grid2">
            <div className="panel">
              <h3>
                {t("sweep.endpoint")}: {t(`metric.${r.endpoint.metric}`)} @ {String(r.endpoint.at)}
              </h3>
              <StripPlot
                testId="strip-plot"
                groups={r.conditions.map((c) => ({ label: c.label, values: c.values, mean: c.summary.mean, reference: c.is_reference }))}
              />
              <table className="table small" data-testid="sweep-conditions">
                <thead>
                  <tr>
                    <th>{t("sweep.condition")}</th>
                    <th>n</th>
                    <th>mean</th>
                    <th>sd</th>
                    <th>median</th>
                    <th>min–max</th>
                  </tr>
                </thead>
                <tbody>
                  {r.conditions.map((c) => (
                    <tr key={c.index}>
                      <td>
                        {c.label}
                        {c.is_reference ? ` (${t("sweep.reference_short")})` : ""}
                      </td>
                      <td>{c.summary.n}</td>
                      <td className="mono">{fmt(c.summary.mean)}</td>
                      <td className="mono">{fmt(c.summary.sd, 3)}</td>
                      <td className="mono">{fmt(c.summary.median)}</td>
                      <td className="mono">
                        {fmt(c.summary.min, 3)}–{fmt(c.summary.max, 3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="panel">
              <h3>{t("sweep.timecourse")}</h3>
              <MultiLine series={series} testId="sweep-timecourse" />
              <p className="muted small">{t("sweep.timecourse_note")}</p>
            </div>
          </div>
          <div className="panel">
            <h3>{t("sweep.comparisons")}</h3>
            <div className="table-wrap">
              <table className="table small" data-testid="sweep-comparisons">
                <thead>
                  <tr>
                    <th>{t("sweep.condition")}</th>
                    <th>Δ mean</th>
                    <th>95% CI (bootstrap)</th>
                    <th>Cliff's δ</th>
                    <th>p (permutation)</th>
                    <th>p (Holm)</th>
                  </tr>
                </thead>
                <tbody>
                  {r.comparisons.map((c) => (
                    <tr key={c.condition_index}>
                      <td>
                        {c.condition} − {c.reference}
                      </td>
                      <td className="mono">{fmt(c.mean_difference)}</td>
                      <td className="mono">{c.mean_difference_ci95 ? `[${fmt(c.mean_difference_ci95[0], 3)}, ${fmt(c.mean_difference_ci95[1], 3)}]` : "—"}</td>
                      <td className="mono">{fmt(c.cliffs_delta, 3)}</td>
                      <td className="mono">
                        {fmt(c.permutation.p_value, 3)} <span className="muted">({c.permutation.method})</span>
                      </td>
                      <td className="mono">{fmt(c.p_value_holm, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="small" data-testid="sweep-notes">
              {r.notes.map((n) => (
                <li key={n.code} data-code={n.code}>
                  {t(`sweep.note.${n.code}`).replace("{count}", String(n.count ?? ""))}
                </li>
              ))}
            </ul>
            <a href={`/api/v1/sweeps/${sweepId}/report`} target="_blank" rel="noreferrer" className="small">
              {t("sweep.report_json")}
            </a>
          </div>
        </>
      )}
      <div className="panel">
        <h3>{t("sweep.runs")}</h3>
        <div className="table-wrap" style={{ maxHeight: "18rem", overflowY: "auto" }}>
          <table className="table small" data-testid="sweep-runs">
            <thead>
              <tr>
                <th>{t("sweep.condition")}</th>
                <th>seed</th>
                <th>{t("col.state")}</th>
                <th>{t("col.tick")}</th>
                <th>{t("sweep.endpoint")}</th>
                <th>{t("col.run")}</th>
              </tr>
            </thead>
            <tbody>
              {s.members.map((m) => {
                const pr = r?.per_run.find((x) => x.run_id === m.run_id);
                return (
                  <tr key={m.run_id}>
                    <td>{m.condition_label}</td>
                    <td>{m.seed}</td>
                    <td>{m.state}</td>
                    <td>{m.tick}</td>
                    <td className="mono">{fmt(pr?.endpoint_value)}</td>
                    <td>
                      <a className="mono" href={`#/lab/${m.run_id}`}>
                        {m.run_id}
                      </a>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
