import { useT } from "../i18n";

export function StateBadge({ state }: { state: string }) {
  const t = useT();
  return (
    <span className={`badge state-${state}`} data-testid="run-state" data-state={state}>
      {t(`state.${state}`)}
    </span>
  );
}
