import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useLang, useT } from "../i18n";
import { StateBadge } from "./StateBadge";

export function Library({ onOpen }: { onOpen: (runId: string) => void }) {
  const t = useT();
  const lang = useLang((s) => s.lang);
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs, refetchInterval: 3000 });
  if (runs.isLoading) return <p>{t("common.loading")}</p>;
  if (!runs.data?.length) return <p className="muted">{t("library.empty")}</p>;
  return (
    <div className="table-wrap">
      <table className="table" data-testid="run-table">
        <thead>
          <tr>
            <th>{t("col.run")}</th>
            <th>{t("col.name")}</th>
            <th>{t("col.state")}</th>
            <th>{t("col.tick")}</th>
            <th>{t("col.mode")}</th>
            <th>{t("col.created")}</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {runs.data.map((r) => (
            <tr key={r.id}>
              <td className="mono">{r.id}</td>
              <td>{r.name}</td>
              <td>
                <StateBadge state={r.state} />
              </td>
              <td>{r.tick}</td>
              <td>{r.effective_mode}</td>
              <td>{new Date(r.created_at * 1000).toLocaleString(lang)}</td>
              <td>
                <button data-testid={`open-${r.id}`} onClick={() => onOpen(r.id)}>
                  {t("library.open")}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
