// Top-bar CPU/GPU switch. GPU is disabled when the local target offers no GPU backend.
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { GPU_BACKENDS, useCompute } from "../lib/compute";
import { useT } from "../i18n";

export function ComputeToggle() {
  const t = useT();
  const { mode, setMode } = useCompute();
  const targets = useQuery({ queryKey: ["compute-targets"], queryFn: api.computeTargets });
  const target = targets.data?.targets[0];
  const backends = target?.capabilities.backends ?? [];
  const gpuBackends = backends.filter((b) => (GPU_BACKENDS as readonly string[]).includes(b));
  const gpuOk = gpuBackends.length > 0;
  const device = target?.probe?.gpu?.devices?.[0];
  const gpuTitle = gpuOk
    ? `${device ? `${device.name} (${Math.round(device.memory_total_mib / 1024)} GB) · ` : ""}${gpuBackends.join(", ")}`
    : t("compute.gpu_unavailable");
  const effective = mode === "gpu" && !gpuOk ? "cpu" : mode;

  return (
    <div className="segmented" role="group" aria-label={t("compute.label")} data-testid="compute-toggle">
      <span className="muted small">{t("compute.label")}</span>
      <button type="button" data-testid="compute-cpu" className={effective === "cpu" ? "active" : ""} aria-pressed={effective === "cpu"} onClick={() => setMode("cpu")}>
        {t("compute.cpu")}
      </button>
      <button
        type="button"
        data-testid="compute-gpu"
        className={effective === "gpu" ? "active" : ""}
        aria-pressed={effective === "gpu"}
        disabled={!gpuOk}
        title={gpuTitle}
        onClick={() => setMode("gpu")}
      >
        {t("compute.gpu")}
      </button>
    </div>
  );
}
