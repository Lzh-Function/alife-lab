import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError, type Preset } from "../lib/api";
import { FORM_FIELDS, getPath, setPath, type FormField, type SubstrateId } from "../lib/substrates";
import { useT } from "../i18n";

interface FormState {
  preset: Preset;
  name: string;
  values: Record<string, unknown>; // dotted path -> value (substrate-specific fields)
  profile_id: string;
  tps: number;
  start_paused: boolean;
}

function fieldsFor(substrate: string): FormField[] {
  return FORM_FIELDS[(substrate as SubstrateId) in FORM_FIELDS ? (substrate as SubstrateId) : "conway"];
}

function fromPreset(p: Preset, prev?: FormState): FormState {
  const exp = p.spec.experiment as Record<string, unknown>;
  const values: Record<string, unknown> = {};
  for (const f of fieldsFor(p.substrate)) values[f.path] = getPath(exp, f.path);
  return {
    preset: p,
    name: String(exp.name),
    values,
    profile_id: prev?.profile_id ?? "local_interactive",
    tps: prev?.tps ?? (p.substrate === "boids" ? 30 : 10),
    start_paused: prev?.start_paused ?? false,
  };
}

export function Catalog({ onOpen }: { onOpen: (runId: string) => void }) {
  const t = useT();
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const substrates = useQuery({ queryKey: ["substrates"], queryFn: api.substrates });
  const [form, setForm] = useState<FormState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validation, setValidation] = useState<{ ok: boolean; errors: string[]; warnings: string[]; spec_hash?: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!form && presets.data?.length) {
      const glider = presets.data.find((p) => p.id === "conway_glider") ?? presets.data[0];
      setForm(fromPreset(glider));
    }
  }, [presets.data, form]);

  const body = useMemo(() => {
    if (!form) return null;
    let overrides: Record<string, unknown> = { name: form.name };
    for (const [path, value] of Object.entries(form.values)) overrides = setPath(overrides, path, value);
    return { preset_id: form.preset.id, overrides, profile_id: form.profile_id };
  }, [form]);

  useEffect(() => {
    if (!body) return;
    const handle = window.setTimeout(() => {
      api
        .validate(body)
        .then((r) => setValidation({ ...r.report, spec_hash: r.spec_hash }))
        .catch((e) => setValidation({ ok: false, errors: [e instanceof ApiError ? e.message : String(e)], warnings: [] }));
    }, 250);
    return () => window.clearTimeout(handle);
  }, [body]);

  const submit = async (ev: React.FormEvent) => {
    ev.preventDefault();
    if (!form || !body) return;
    setBusy(true);
    setError(null);
    try {
      const run = await api.create({ ...body, ticks_per_second: form.tps, auto_start: true, start_paused: form.start_paused });
      onOpen(run.id);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
    } finally {
      setBusy(false);
    }
  };

  const setValue = (path: string, value: unknown) =>
    setForm((f) => (f ? { ...f, values: { ...f.values, [path]: value } } : f));

  const bySubstrate = new Map<string, Preset[]>();
  for (const p of presets.data ?? []) bySubstrate.set(p.substrate, [...(bySubstrate.get(p.substrate) ?? []), p]);

  return (
    <div className="catalog">
      <section>
        <h2>{t("catalog.presets")}</h2>
        {(substrates.data ?? []).map((s) => (
          <div key={s.slug} className="substrate-group">
            <h3>
              {t(s.display_name_key)} <span className="muted small">v{s.plugin_version} · {s.backends.join(", ")}</span>
            </h3>
            <div className="cards">
              {(bySubstrate.get(s.slug) ?? []).map((p) => (
                <button
                  key={p.id}
                  type="button"
                  data-testid={`preset-${p.id}`}
                  className={`card ${form?.preset.id === p.id ? "selected" : ""}`}
                  onClick={() => setForm(fromPreset(p, form ?? undefined))}
                >
                  <strong>{t(p.title_key)}</strong>
                  <span>{t(p.description_key)}</span>
                  <small className="muted">
                    {t("catalog.expected")}: {p.expected_observation}
                  </small>
                  {p.negative_control && (
                    <small className="muted">
                      {t("catalog.negative_control")}: {p.negative_control}
                    </small>
                  )}
                </button>
              ))}
            </div>
          </div>
        ))}
      </section>
      {form && (
        <section>
          <h2>{t("catalog.create")}</h2>
          <form className="form" onSubmit={submit} data-testid="create-form">
            <fieldset>
              <legend>
                ExperimentSpec · {t(`substrate.${form.preset.substrate}`)} ·{" "}
                <span className="mono">{String((form.preset.spec.experiment as Record<string, unknown>).mode)}</span>
              </legend>
              <label>
                {t("form.name")}
                <input data-testid="input-name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} maxLength={120} />
              </label>
              {fieldsFor(form.preset.substrate).map((f) => (
                <label key={f.path}>
                  {t(f.labelKey)}
                  {f.kind === "select" ? (
                    <select data-testid={`field-${f.path}`} value={String(form.values[f.path])} onChange={(e) => setValue(f.path, e.target.value)}>
                      {(f.options ?? []).map((o) => (
                        <option key={o.value} value={o.value}>
                          {t(o.labelKey)}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      data-testid={`field-${f.path}`}
                      min={f.min}
                      max={f.max}
                      step={f.kind === "int" ? 1 : "any"}
                      value={String(form.values[f.path] ?? "")}
                      onChange={(e) => setValue(f.path, e.target.value === "" ? "" : Number(e.target.value))}
                    />
                  )}
                </label>
              ))}
            </fieldset>
            <fieldset>
              <legend>ExecutionProfile</legend>
              <label>
                {t("form.profile")}
                <select data-testid="input-profile" value={form.profile_id} onChange={(e) => setForm({ ...form, profile_id: e.target.value })}>
                  {(profiles.data ?? []).map((p) => (
                    <option key={p.id} value={p.id} disabled={!p.implemented}>
                      {p.id} ({p.role}){p.implemented ? "" : ` — ${t("form.profile.unimplemented")}`}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                {t("form.tps")}
                <input data-testid="input-tps" type="number" min={0} max={10000} step="any" value={form.tps} onChange={(e) => setForm({ ...form, tps: Number(e.target.value) })} />
              </label>
              <label className="inline">
                <input data-testid="input-start-paused" type="checkbox" checked={form.start_paused} onChange={(e) => setForm({ ...form, start_paused: e.target.checked })} />
                {t("form.start_paused")}
              </label>
            </fieldset>
            <p className="muted small">{t("form.science_note")}</p>
            {validation && (
              <div className={validation.ok ? "notice small" : "error small"} data-testid="validation">
                <strong>{t("form.validation")}:</strong> {validation.ok ? "OK" : validation.errors.join("; ")}
                {validation.warnings.map((w) => (
                  <div key={w}>⚠ {w}</div>
                ))}
                {validation.spec_hash && <div className="mono">spec {validation.spec_hash.slice(0, 16)}…</div>}
              </div>
            )}
            {error && <p className="error" role="alert">{error}</p>}
            <button type="submit" className="primary" data-testid="btn-create" disabled={busy || validation?.ok === false}>
              {t("form.submit")}
            </button>
          </form>
        </section>
      )}
    </div>
  );
}
