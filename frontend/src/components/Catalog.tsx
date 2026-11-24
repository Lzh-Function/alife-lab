// World catalog: three panes. Left = substrates by category (+ search), middle = presets of the
// selected substrate (or search hits across all substrates), right = sticky create form.
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError, type Preset } from "../lib/api";
import { FORM_FIELDS, getPath, setPath, type FormField, type SubstrateId } from "../lib/substrates";
import { CPU_PROFILE, GPU_BACKENDS, GPU_PROFILE, OVERRIDABLE_POLICIES, gpuBackendFor, requiredPolicy, useCompute } from "../lib/compute";
import { CATEGORIES, SUBSTRATE_GUIDE } from "../guide/content";
import { useT } from "../i18n";
import { TargetEditor } from "./TargetEditor";
import { Help, paramHelp } from "./Help";
import { Stars } from "./GuidePage";

interface FormState {
  preset: Preset;
  name: string;
  values: Record<string, unknown>; // dotted path -> value (substrate-specific fields)
  profile_override: string | null; // null = follow the top-level CPU/GPU switch
  tps: number;
  start_paused: boolean;
}

function fieldsFor(substrate: string): FormField[] {
  return FORM_FIELDS[(substrate as SubstrateId) in FORM_FIELDS ? (substrate as SubstrateId) : "conway"];
}

function defaultTpsFor(substrate: string): number {
  if (substrate === "cultural_world") return 20;
  if (substrate === "fly_embodied" || substrate === "fly_swarm") return 1000;
  if (substrate === "worm_embodied") return 30;
  if (substrate === "fly_world") return 60;
  if (substrate === "connectome") return 200;
  if (substrate === "boids") return 30;
  return 10;
}

function fromPreset(p: Preset, prev?: FormState): FormState {
  const exp = p.spec.experiment as Record<string, unknown>;
  const values: Record<string, unknown> = {};
  for (const f of fieldsFor(p.substrate)) values[f.path] = getPath(exp, f.path);
  const sameSubstrate = prev?.preset.substrate === p.substrate;
  return {
    preset: p,
    name: String(exp.name),
    values,
    profile_override: prev?.profile_override ?? null,
    tps: sameSubstrate && prev ? prev.tps : defaultTpsFor(p.substrate),
    start_paused: prev?.start_paused ?? false,
  };
}

const CONTROL_ID = /(control|_no_|_zero_|frozen|_random$|_fixed|shuffle|non_metabolic|off_niche|random_partition)/;

function takeInitialSubstrate(): string | null {
  try {
    const s = sessionStorage.getItem("alife.catalog.substrate");
    sessionStorage.removeItem("alife.catalog.substrate");
    return s;
  } catch {
    return null;
  }
}

export function Catalog({ onOpen }: { onOpen: (runId: string) => void }) {
  const t = useT();
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const substrates = useQuery({ queryKey: ["substrates"], queryFn: api.substrates });
  const targets = useQuery({ queryKey: ["compute-targets"], queryFn: api.computeTargets });
  const computeMode = useCompute((s) => s.mode);
  const [form, setForm] = useState<FormState | null>(null);
  const [selected, setSelected] = useState<string>(() => takeInitialSubstrate() ?? "conway");
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [validation, setValidation] = useState<{ ok: boolean; errors: string[]; warnings: string[]; spec_hash?: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const presetList = useMemo(() => presets.data ?? [], [presets.data]);
  const substrateList = useMemo(() => substrates.data ?? [], [substrates.data]);
  const targetBackends = targets.data?.targets[0]?.capabilities.backends ?? [];

  useEffect(() => {
    if (!form && presetList.length) {
      const first = presetList.find((p) => p.substrate === selected) ?? presetList.find((p) => p.id === "conway_glider") ?? presetList[0];
      setForm(fromPreset(first));
      setSelected(first.substrate);
    }
  }, [presetList, form, selected]);

  // Toggling CPU/GPU in the top bar drops a manual profile choice.
  useEffect(() => {
    setForm((f) => (f && f.profile_override !== null ? { ...f, profile_override: null } : f));
  }, [computeMode]);

  const controls = useMemo(() => {
    const ids = new Set(presetList.map((p) => p.id));
    const out = new Set<string>();
    for (const p of presetList) {
      if (CONTROL_ID.test(p.id)) out.add(p.id);
      for (const word of (p.negative_control ?? "").split(/[^a-z0-9_]+/)) if (ids.has(word)) out.add(word);
    }
    return out;
  }, [presetList]);

  const manifest = form ? substrateList.find((s) => s.slug === form.preset.substrate) : undefined;
  const presetPolicy = form ? (getPath(form.preset.spec.experiment as Record<string, unknown>, "dynamics.numeric_policy") as string | undefined) : undefined;
  const anyGpuBackend = manifest ? gpuBackendFor(manifest.backends, targetBackends) : null;
  // A GPU backend is usable if it accepts the preset's numeric policy, or a policy the preset
  // may be switched to without changing its meaning beyond float rounding (JAX CPU -> CUDA).
  const gpuPolicy = requiredPolicy(manifest?.backend_numeric_policies, anyGpuBackend);
  const gpuCompatible = !gpuPolicy || gpuPolicy === presetPolicy || OVERRIDABLE_POLICIES[presetPolicy ?? ""] === gpuPolicy;
  const gpuBackend = gpuCompatible ? anyGpuBackend : null;
  const autoProfile = computeMode === "gpu" && gpuBackend ? GPU_PROFILE : CPU_PROFILE;
  const profileId = form?.profile_override ?? autoProfile;
  const policy = profileId === GPU_PROFILE ? requiredPolicy(manifest?.backend_numeric_policies, gpuBackend) : null;
  const policyOverride = policy && presetPolicy !== policy ? policy : null;

  const body = useMemo(() => {
    if (!form) return null;
    let overrides: Record<string, unknown> = { name: form.name };
    for (const [path, value] of Object.entries(form.values)) if (value !== undefined) overrides = setPath(overrides, path, value);
    if (policyOverride) overrides = setPath(overrides, "dynamics.numeric_policy", policyOverride);
    return { preset_id: form.preset.id, overrides, profile_id: profileId };
  }, [form, profileId, policyOverride]);

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

  const choosePreset = (p: Preset) => {
    setForm(fromPreset(p, form ?? undefined));
    setSelected(p.substrate);
  };

  const chooseSubstrate = (slug: string) => {
    setSelected(slug);
    setQuery("");
    if (form?.preset.substrate !== slug) {
      const first = presetList.find((p) => p.substrate === slug);
      if (first) setForm(fromPreset(first, form ?? undefined));
    }
  };

  const titleOf = (slug: string) => {
    const m = substrateList.find((s) => s.slug === slug);
    return m ? t(m.display_name_key) : slug;
  };

  const q = query.trim().toLowerCase();
  const hits = q
    ? presetList.filter((p) =>
        [p.id, t(p.title_key), t(p.description_key), p.substrate, titleOf(p.substrate), SUBSTRATE_GUIDE[p.substrate]?.title ?? ""]
          .join(" ")
          .toLowerCase()
          .includes(q),
      )
    : presetList.filter((p) => p.substrate === selected);

  const grouped = CATEGORIES.map((c) => ({
    ...c,
    items: substrateList.filter((s) => SUBSTRATE_GUIDE[s.slug]?.category === c.id),
  }));
  const others = substrateList.filter((s) => !SUBSTRATE_GUIDE[s.slug]);
  if (others.length) grouped.push({ id: "ca", title: t("catalog.other"), note: "", items: others });

  const guide = SUBSTRATE_GUIDE[selected];

  const card = (p: Preset) => (
    <button
      key={p.id}
      type="button"
      data-testid={`preset-${p.id}`}
      className={`card ${form?.preset.id === p.id ? "selected" : ""}`}
      onClick={() => choosePreset(p)}
    >
      <strong>
        {t(p.title_key)}
        {controls.has(p.id) && <span className="badge badge-control">{t("catalog.control_badge")}</span>}
        {String((p.spec.experiment as Record<string, unknown>).mode) === "reference" && <span className="badge">reference</span>}
      </strong>
      <span>{t(p.description_key)}</span>
      <small className="muted">
        {t("catalog.expected")}: {p.expected_observation}
      </small>
      {p.negative_control && (
        <small className="muted">
          {t("catalog.negative_control")}: {p.negative_control}
        </small>
      )}
      {q && <small className="muted mono">{titleOf(p.substrate)}</small>}
    </button>
  );

  return (
    <div className="catalog3">
      <nav className="catalog-nav" aria-label="substrates">
        <input
          type="search"
          className="catalog-search"
          data-testid="catalog-search"
          placeholder={t("catalog.search")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {grouped.map((c) => (
          <div key={c.title} className="catalog-cat">
            <div className="catalog-cat-title" title={c.note}>
              {c.title}
            </div>
            {c.items.map((s) => {
              const g = SUBSTRATE_GUIDE[s.slug];
              const gpu = s.backends.some((b) => b === "cupy" || b === "jax_cuda");
              return (
                <button
                  key={s.slug}
                  type="button"
                  data-testid={`substrate-${s.slug}`}
                  className={`substrate-item ${selected === s.slug && !q ? "active" : ""}`}
                  onClick={() => chooseSubstrate(s.slug)}
                >
                  <span>{g?.title ?? t(s.display_name_key)}</span>
                  <span className="substrate-meta">
                    {g && <Stars level={g.level} />}
                    {gpu && <span className="badge badge-gpu">GPU</span>}
                  </span>
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      <section className="catalog-presets">
        {q ? (
          <h2>
            {t("catalog.search_results")} <span className="muted small">({hits.length})</span>
          </h2>
        ) : (
          <>
            <h2>
              {guide?.title ?? titleOf(selected)}
              {guide && <Stars level={guide.level} />}
            </h2>
            {guide && (
              <div className="substrate-intro" data-testid="substrate-intro">
                <p>{guide.oneLine}</p>
                <a href={`#/guide/${selected}`}>{t("catalog.open_guide")} →</a>
              </div>
            )}
          </>
        )}
        <div className="cards">{hits.map(card)}</div>
        {q && hits.length === 0 && <p className="muted">{t("catalog.no_results")}</p>}
      </section>

      {form && (
        <section className="catalog-form">
          <h2>{t("catalog.create")}</h2>
          <form className="form" onSubmit={submit} data-testid="create-form">
            <fieldset>
              <legend>
                ExperimentSpec · {SUBSTRATE_GUIDE[form.preset.substrate]?.title ?? t(`substrate.${form.preset.substrate}`)} ·{" "}
                <span className="mono">{String((form.preset.spec.experiment as Record<string, unknown>).mode)}</span>
              </legend>
              <div className="muted small form-preset">{t(form.preset.title_key)}</div>
              <label>
                <span className="label-row">
                  {t("form.name")} <Help text={paramHelp(form.preset.substrate, "name")} />
                </span>
                <input data-testid="input-name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} maxLength={120} />
              </label>
              {fieldsFor(form.preset.substrate).filter((f) => form.values[f.path] !== undefined && !(f.path === "asal.prompts" && form.values["asal.frames"] !== undefined && getPath(form.preset.spec.experiment as Record<string, unknown>, "dynamics.objective") !== "lenia_clip_target")).map((f) => (
                <label key={f.path}>
                  <span className="label-row">
                    {t(f.labelKey)} <Help text={paramHelp(form.preset.substrate, f.path)} testId={`help-${f.path}`} />
                  </span>
                  {f.kind === "select" ? (
                    <select data-testid={`field-${f.path}`} value={String(form.values[f.path])} onChange={(e) => setValue(f.path, e.target.value)}>
                      {(f.options ?? []).map((o) => (
                        <option key={o.value} value={o.value}>
                          {t(o.labelKey)}
                        </option>
                      ))}
                    </select>
                  ) : f.kind === "textlist" ? (
                    <input
                      data-testid={`field-${f.path}`}
                      value={((form.values[f.path] as string[] | undefined) ?? []).join("; ")}
                      onChange={(e) =>
                        setValue(
                          f.path,
                          e.target.value
                            .split(";")
                            .map((v) => v.trim())
                            .filter(Boolean),
                        )
                      }
                    />
                  ) : f.kind === "bool" ? (
                    <input
                      type="checkbox"
                      data-testid={`field-${f.path}`}
                      checked={Boolean(form.values[f.path])}
                      onChange={(e) => setValue(f.path, e.target.checked)}
                    />
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
              {form.preset.substrate === "nca_train" && (
                <TargetEditor
                  size={Number(form.values["target.size"]) || 40}
                  active={form.values.target !== undefined}
                  onChange={(target) =>
                    setForm((f) => {
                      if (!f) return f;
                      const values = { ...f.values };
                      if (target) values.target = target;
                      else delete values.target;
                      return { ...f, values };
                    })
                  }
                />
              )}
            </fieldset>
            <fieldset>
              <legend>ExecutionProfile</legend>
              <div className="compute-note small" data-testid="compute-note">
                {profileId === GPU_PROFILE && gpuBackend
                  ? t("compute.note.gpu").replace("{backend}", gpuBackend)
                  : computeMode === "gpu" && !gpuBackend
                    ? anyGpuBackend && !gpuCompatible
                      ? t("compute.note.gpu_incompatible")
                      : manifest?.backends.some((b) => (GPU_BACKENDS as readonly string[]).includes(b))
                        ? t("compute.gpu_unavailable")
                        : t("compute.note.no_gpu")
                    : t("compute.note.cpu")}
                {policyOverride && <div className="notice">{t("compute.note.policy").split("{policy}").join(policyOverride)}</div>}
              </div>
              <label>
                <span className="label-row">
                  {t("form.profile")} <Help text={paramHelp(form.preset.substrate, "profile")} />
                </span>
                <select
                  data-testid="input-profile"
                  value={profileId}
                  onChange={(e) => setForm({ ...form, profile_override: e.target.value })}
                >
                  {(profiles.data ?? []).map((p) => (
                    <option key={p.id} value={p.id} disabled={!p.implemented}>
                      {p.id} ({p.role}){p.implemented ? "" : ` — ${t("form.profile.unimplemented")}`}
                    </option>
                  ))}
                </select>
                {form.profile_override !== null && (
                  <span className="small">
                    {t("compute.manual")}{" "}
                    <button type="button" className="linklike" onClick={() => setForm({ ...form, profile_override: null })}>
                      {t("compute.follow")}
                    </button>
                  </span>
                )}
              </label>
              <label>
                <span className="label-row">
                  {t("form.tps")} <Help text={paramHelp(form.preset.substrate, "tps")} />
                </span>
                <input data-testid="input-tps" type="number" min={0} max={10000} step="any" value={form.tps} onChange={(e) => setForm({ ...form, tps: Number(e.target.value) })} />
              </label>
              <label className="inline">
                <input data-testid="input-start-paused" type="checkbox" checked={form.start_paused} onChange={(e) => setForm({ ...form, start_paused: e.target.checked })} />
                {t("form.start_paused")} <Help text={paramHelp(form.preset.substrate, "start_paused")} />
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
