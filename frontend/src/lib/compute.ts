// Top-level CPU/GPU preference. It only chooses the ExecutionProfile for new runs; it never
// changes the ExperimentSpec except for the numeric policy that a GPU backend requires
// (made visible in the create form).
import { create } from "zustand";

export type ComputeMode = "cpu" | "gpu";

/** Backends that run on a GPU (must match the controller's backend registry). */
export const GPU_BACKENDS = ["cupy", "jax_cuda", "torch_cuda"] as const;

/** Numeric policies a preset may be switched to for a GPU backend (only float rounding differs). */
export const OVERRIDABLE_POLICIES: Record<string, string> = { float32_jax_cpu: "float32_jax" };

export const CPU_PROFILE = "local_cpu"; // GPU-free local_interactive profile
export const GPU_PROFILE = "local_gpu";

function initialMode(): ComputeMode {
  try {
    return window.localStorage.getItem("alife.compute") === "gpu" ? "gpu" : "cpu";
  } catch {
    return "cpu";
  }
}

interface ComputeState {
  mode: ComputeMode;
  setMode: (mode: ComputeMode) => void;
}

export const useCompute = create<ComputeState>((set) => ({
  mode: initialMode(),
  setMode: (mode) => {
    try {
      window.localStorage.setItem("alife.compute", mode);
    } catch {
      /* storage unavailable */
    }
    set({ mode });
  },
}));

/** First GPU backend a substrate declares that the compute target also provides. */
export function gpuBackendFor(substrateBackends: string[], targetBackends: string[]): string | null {
  return substrateBackends.find((b) => (GPU_BACKENDS as readonly string[]).includes(b) && targetBackends.includes(b)) ?? null;
}

/** Numeric policy a backend requires, if the manifest restricts it (e.g. jax_cuda -> float32_jax). */
export function requiredPolicy(policies: Record<string, string[]> | undefined, backend: string | null): string | null {
  if (!backend || !policies?.[backend]?.length) return null;
  return policies[backend][0];
}
