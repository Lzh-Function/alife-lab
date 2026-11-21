import { useEffect, useState } from "react";
import { useLang, useT } from "./i18n";
import { Catalog } from "./components/Catalog";
import { Library } from "./components/Library";
import { LiveLab } from "./components/LiveLab";
import { Diagnostics } from "./components/Diagnostics";
import { CompareView, ComparisonList } from "./components/CompareView";
import { SweepDetail, SweepList } from "./components/Sweeps";
import { BffDebuggerPage } from "./components/BffDebuggerPage";
import { GuidePage } from "./components/GuidePage";
import { ComputeToggle } from "./components/ComputeToggle";
import { DiscoveryDetailView, DiscoveryList } from "./components/Discoveries";
import { GalleryPage } from "./components/Gallery";
import { LineagePage } from "./components/LineagePage";

type View = "guide" | "catalog" | "library" | "lab" | "compare" | "sweeps" | "discoveries" | "gallery" | "debugger" | "diagnostics" | "lineage";
const VIEWS = ["guide", "catalog", "library", "lab", "compare", "sweeps", "discoveries", "gallery", "debugger", "diagnostics", "lineage"] as const;
const NAV_HIDDEN = new Set<View>(["lineage"]); // reached from a run, not from the top bar

function parseHash(): { view: View; runId: string | null; comparisonId: string | null; sweepId: string | null; discoveryId: string | null; anchor: string | null } {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/");
  const view = VIEWS.find((v) => v === parts[0]) ?? "catalog";
  const runId = parts[1] && /^run_[0-9a-f]{16}$/.test(parts[1]) ? parts[1] : null; // lab, lineage
  const comparisonId = parts[1] && /^cmp_[0-9a-f]{16}$/.test(parts[1]) ? parts[1] : null;
  const sweepId = parts[1] && /^swp_[0-9a-f]{16}$/.test(parts[1]) ? parts[1] : null;
  const anchor = view === "guide" && parts[1] && /^[a-z_]{1,40}$/.test(parts[1]) ? parts[1] : null;
  const discoveryId = parts[1] && /^dsc_[0-9a-f]{16}$/.test(parts[1]) ? parts[1] : null;
  return { view, runId, comparisonId, sweepId, discoveryId, anchor };
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
  const openComparison = (id: string) => {
    window.location.hash = `/compare/${id}`;
  };
  const openSweep = (id: string) => {
    window.location.hash = `/sweeps/${id}`;
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <strong>{t("app.title")}</strong>
          <span className="muted">{t("app.subtitle")}</span>
        </div>
        <nav>
          {VIEWS.filter((v) => !NAV_HIDDEN.has(v)).map((v) => (
            <button key={v} data-testid={`nav-${v}`} className={route.view === v ? "active" : ""} onClick={() => go(v)}>
              {t(`nav.${v}`)}
            </button>
          ))}
        </nav>
        <ComputeToggle />
        <button className="ghost" data-testid="lang-toggle" onClick={() => setLang(lang === "ja" ? "en" : "ja")}>
          {t("lang.toggle")}
        </button>
      </header>
      <main>
        {route.view === "guide" && <GuidePage anchor={route.anchor} />}
        {route.view === "catalog" && <Catalog onOpen={open} />}
        {route.view === "library" && <Library onOpen={open} />}
        {route.view === "lab" &&
          (route.runId ? (
            <LiveLab key={route.runId} runId={route.runId} onOpenComparison={openComparison} />
          ) : (
            <p className="muted">{t("lab.none")}</p>
          ))}
        {route.view === "compare" &&
          (route.comparisonId ? (
            <CompareView key={route.comparisonId} comparisonId={route.comparisonId} />
          ) : (
            <ComparisonList onOpen={openComparison} />
          ))}
        {route.view === "sweeps" &&
          (route.sweepId ? <SweepDetail key={route.sweepId} sweepId={route.sweepId} /> : <SweepList onOpen={openSweep} />)}
        {route.view === "discoveries" &&
          (route.discoveryId ? (
            <DiscoveryDetailView key={route.discoveryId} id={route.discoveryId} />
          ) : (
            <DiscoveryList
              onOpen={(id) => {
                window.location.hash = `/discoveries/${id}`;
              }}
            />
          ))}
        {route.view === "gallery" && <GalleryPage />}
        {route.view === "lineage" && (route.runId ? <LineagePage key={route.runId} runId={route.runId} /> : <p className="muted">{t("lab.none")}</p>)}
        {route.view === "debugger" && <BffDebuggerPage />}
        {route.view === "diagnostics" && <Diagnostics />}
      </main>
    </div>
  );
}
