import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError, type Preset } from "../lib/api";
import { useT } from "../i18n";

interface FormState {
  preset_id: string;
  name: string;
  seed: number;
  width: number;
  height: number;
  boundary: string;
  max_steps: number;
  checkpoint_interval: number;
  profile_id: string;
  tps: number;
  start_paused: boolean;
}

function fromPreset(p: Preset, prev?: FormState): FormState {
  const e = p.spec.experiment;
  return {
    preset_id: p.id,
    name: e.name,
    seed: e.seed,
    width: e.world.width,
    height: e.world.height,
    boundary: e.world.boundary,
    max_steps: e.runner.max_steps,
    checkpoint_interval: e.runner.checkpoint_interval,
    profile_id: prev?.profile_id ?? "local_interactive",
    tps: prev?.tps ?? 10,
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
    return {
      preset_id: form.preset_id,
      overrides: {
        name: form.name,
        seed: form.seed,
        world: { width: form.width, height: form.height, boundary: form.boundary },
        runner: { max_steps: form.max_steps, checkpoint_interval: form.checkpoint_interval },
      },
      profile_id: form.profile_id,
    };
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

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) => setForm((f) => (f ? { ...f, [key]: value } : f));
  const num = (key: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement>) => set(key, Number(e.target.value) as never);

  return (
    <div className="catalog">
      <section>
        <h2>{t("catalog.presets")}</h2>
        {(substrates.data ?? []).map((s) => (
          <p key={s.slug} className="muted">
            <strong>{t(s.display_name_key)}</strong> · v{s.plugin_version} · {s.backends.join(", ")}
          </p>
        ))}
        <div className="cards">
          {(presets.data ?? []).map((p) => (
            <button
              key={p.id}
              type="button"
              data-testid={`preset-${p.id}`}
              className={`card ${form?.preset_id === p.id ? "selected" : ""}`}
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
      </section>
      {form && (
        <section>
          <h2>{t("catalog.create")}</h2>
          <form className="form" onSubmit={submit} data-testid="create-form">
            <fieldset>
              <legend>ExperimentSpec</legend>
              <label>
                {t("form.name")}
                <input data-testid="input-name" value={form.name} onChange={(e) => set("name", e.target.value)} maxLength={120} />
              </label>
              <label>
                {t("form.seed")}
                <input type="number" min={0} value={form.seed} onChange={num("seed")} />
              </label>
              <label>
                {t("form.width")}
                <input type="number" min={8} max={1024} value={form.width} onChange={num("width")} />
              </label>
              <label>
                {t("form.height")}
                <input type="number" min={8} max={1024} value={form.height} onChange={num("height")} />
              </label>
              <label>
                {t("form.boundary")}
                <select value={form.boundary} onChange={(e) => set("boundary", e.target.value)}>
                  <option value="periodic">{t("form.boundary.periodic")}</option>
                  <option value="dead">{t("form.boundary.dead")}</option>
                </select>
              </label>
              <label>
                {t("form.max_steps")}
                <input data-testid="input-max-steps" type="number" min={1} value={form.max_steps} onChange={num("max_steps")} />
              </label>
              <label>
                {t("form.checkpoint_interval")}
                <input type="number" min={1} value={form.checkpoint_interval} onChange={num("checkpoint_interval")} />
              </label>
            </fieldset>
            <fieldset>
              <legend>ExecutionProfile</legend>
              <label>
                {t("form.profile")}
                <select data-testid="input-profile" value={form.profile_id} onChange={(e) => set("profile_id", e.target.value)}>
                  {(profiles.data ?? []).map((p) => (
                    <option key={p.id} value={p.id} disabled={!p.implemented}>
                      {p.id} ({p.role}){p.implemented ? "" : ` — ${t("form.profile.unimplemented")}`}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                {t("form.tps")}
                <input data-testid="input-tps" type="number" min={0} max={10000} step="any" value={form.tps} onChange={num("tps")} />
              </label>
              <label className="inline">
                <input data-testid="input-start-paused" type="checkbox" checked={form.start_paused} onChange={(e) => set("start_paused", e.target.checked)} />
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
