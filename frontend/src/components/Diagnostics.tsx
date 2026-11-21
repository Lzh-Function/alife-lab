import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useT } from "../i18n";

export function Diagnostics() {
  const t = useT();
  const resources = useQuery({ queryKey: ["resources"], queryFn: api.resources, staleTime: 60_000 });
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: api.capabilities });
  return (
    <div className="diagnostics">
      <h2>{t("diag.title")}</h2>
      <p className="muted">{t("diag.measured_note")}</p>
      <button onClick={() => void resources.refetch()} disabled={resources.isFetching}>
        {t("diag.refresh")}
      </button>
      <div className="grid2">
        <div className="panel">
          <h3>{t("diag.resources")}</h3>
          <pre data-testid="diag-resources">{resources.data ? JSON.stringify(resources.data, null, 2) : t("common.loading")}</pre>
        </div>
        <div className="panel">
          <h3>{t("diag.capabilities")}</h3>
          <pre>{caps.data ? JSON.stringify(caps.data, null, 2) : t("common.loading")}</pre>
        </div>
      </div>
    </div>
  );
}
