import { useEffect, useState } from "react";
import { useLang, useT } from "./i18n";
import { Catalog } from "./components/Catalog";
import { Library } from "./components/Library";
import { LiveLab } from "./components/LiveLab";
import { Diagnostics } from "./components/Diagnostics";

type View = "catalog" | "library" | "lab" | "diagnostics";

function parseHash(): { view: View; runId: string | null } {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/");
  const view = (["catalog", "library", "lab", "diagnostics"] as const).find((v) => v === parts[0]) ?? "catalog";
  const runId = parts[1] && /^run_[0-9a-f]{16}$/.test(parts[1]) ? parts[1] : null;
  return { view, runId };
}

export function App() {
  const t = useT();
  const { lang, setLang } = useLang();
  const [route, setRoute] = useState(parseHash);

  useEffect(() => {
    const onHash = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHash);
    document.documentElement.lang = lang;
    return () => window.removeEventListener("hashchange", onHash);
  }, [lang]);

  const go = (view: View, runId: string | null = route.runId) => {
    window.location.hash = runId && view === "lab" ? `/lab/${runId}` : `/${view}`;
  };
  const open = (runId: string) => go("lab", runId);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <strong>{t("app.title")}</strong>
          <span className="muted">{t("app.subtitle")}</span>
        </div>
        <nav>
          {(["catalog", "library", "lab", "diagnostics"] as const).map((v) => (
            <button key={v} data-testid={`nav-${v}`} className={route.view === v ? "active" : ""} onClick={() => go(v)}>
              {t(`nav.${v}`)}
            </button>
          ))}
        </nav>
        <button className="ghost" data-testid="lang-toggle" onClick={() => setLang(lang === "ja" ? "en" : "ja")}>
          {t("lang.toggle")}
        </button>
      </header>
      <main>
        {route.view === "catalog" && <Catalog onOpen={open} />}
        {route.view === "library" && <Library onOpen={open} />}
        {route.view === "lab" && (route.runId ? <LiveLab key={route.runId} runId={route.runId} /> : <p className="muted">{t("lab.none")}</p>)}
        {route.view === "diagnostics" && <Diagnostics />}
      </main>
    </div>
  );
}
