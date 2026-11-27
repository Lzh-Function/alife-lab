import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import { useT } from "../i18n";

function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export function Diagnostics() {
  const t = useT();
  const qc = useQueryClient();
  const resources = useQuery({ queryKey: ["resources"], queryFn: api.resources, staleTime: 60_000 });
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: api.capabilities });
  const backups = useQuery({ queryKey: ["backups"], queryFn: api.backups, staleTime: 15_000 });

  const [backupBusy, setBackupBusy] = useState(false);
  const [backupError, setBackupError] = useState<string | null>(null);
  const [backupReport, setBackupReport] = useState<{ kind: string; data: Record<string, unknown> } | null>(null);

  const onCreateBackup = async () => {
    setBackupBusy(true);
    setBackupError(null);
    try {
      const res = await api.createBackup();
      setBackupReport({ kind: "create", data: res });
      void qc.invalidateQueries({ queryKey: ["backups"] });
    } catch (e) {
      setBackupError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBackupBusy(false);
    }
  };

  const onVerifyBackup = async (backupId: string) => {
    setBackupBusy(true);
    setBackupError(null);
    try {
      const res = await api.verifyBackup(backupId);
      setBackupReport({ kind: `verify (${backupId})`, data: res });
    } catch (e) {
      setBackupError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBackupBusy(false);
    }
  };

  const onTestRestore = async (backupId: string) => {
    setBackupBusy(true);
    setBackupError(null);
    try {
      const res = await api.testRestoreBackup(backupId);
      setBackupReport({ kind: `restore (${backupId})`, data: res });
    } catch (e) {
      setBackupError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBackupBusy(false);
    }
  };

  return (
    <div className="diagnostics">
      <h2>{t("diag.title")}</h2>
      <p className="muted">{t("diag.measured_note")}</p>
      <button onClick={() => void resources.refetch()} disabled={resources.isFetching}>
        {t("diag.refresh")}
      </button>

      {/* Managed Backup & Restore Panel (GUI parity for `alife backup`) */}
      <div className="panel" data-testid="diag-backups" style={{ marginTop: "0.8rem" }}>
        <div className="lab-head">
          <div>
            <h3 style={{ margin: 0 }}>{t("diag.backups.title")}</h3>
            <p className="small muted mono" style={{ margin: "0.2rem 0 0" }}>
              {backups.data?.backups_dir ?? ""}
            </p>
          </div>
          <button
            type="button"
            data-testid="btn-create-backup"
            disabled={backupBusy}
            onClick={() => void onCreateBackup()}
          >
            {backupBusy ? t("common.loading") : t("diag.backups.create")}
          </button>
        </div>

        {backupError && (
          <p className="error small" role="alert">
            {backupError}
          </p>
        )}

        {!backups.data?.items.length ? (
          <p className="small muted">{t("diag.backups.empty")}</p>
        ) : (
          <div className="table-wrap" style={{ marginTop: "0.5rem" }}>
            <table className="table small mono">
              <thead>
                <tr>
                  <th>Backup ID</th>
                  <th>{t("diag.backups.created")}</th>
                  <th>Runs</th>
                  <th>Files</th>
                  <th>Size</th>
                  <th>{t("connectome.col.actions")}</th>
                </tr>
              </thead>
              <tbody>
                {backups.data.items.map((b) => (
                  <tr key={b.backup_id} data-testid={`backup-row-${b.backup_id}`}>
                    <td>{b.backup_id}</td>
                    <td>{b.created_at ?? "—"}</td>
                    <td>{b.run_count}</td>
                    <td>{b.file_count}</td>
                    <td>{fmtSize(b.total_bytes)}</td>
                    <td>
                      <button
                        type="button"
                        className="small ghost"
                        disabled={backupBusy}
                        onClick={() => void onVerifyBackup(b.backup_id)}
                      >
                        {t("diag.backups.verify")}
                      </button>{" "}
                      <button
                        type="button"
                        className="small ghost"
                        disabled={backupBusy}
                        onClick={() => void onTestRestore(b.backup_id)}
                      >
                        {t("diag.backups.restore_test")}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {backupReport && (
          <div style={{ marginTop: "0.6rem" }} data-testid="backup-report">
            <div className="small muted">
              {t("diag.backups.last_result")}: <strong>{backupReport.kind}</strong>
            </div>
            <pre className="small mono" style={{ maxHeight: "14rem", overflow: "auto" }}>
              {JSON.stringify(backupReport.data, null, 2)}
            </pre>
          </div>
        )}
      </div>

      <div className="grid2" style={{ marginTop: "0.8rem" }}>
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
