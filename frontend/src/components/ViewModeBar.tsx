// View switch above the world view: the 2D measurement view or a presentation view (3D
// terrain, space-time volume, glowing particles), plus fullscreen. The choice is remembered
// per substrate in this browser only (a convenience; never part of the run).
import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { useT } from "../i18n";
import { viewModesFor, type ViewMode } from "./WorldView";

const KEY = "alife.viewMode.";

export function useViewMode(substrate: string): [ViewMode, (m: ViewMode) => void] {
  const modes = viewModesFor(substrate);
  const read = (): ViewMode => {
    try {
      const v = window.localStorage.getItem(KEY + substrate) as ViewMode | null;
      return v && modes.includes(v) ? v : "2d";
    } catch {
      return "2d";
    }
  };
  const [mode, setMode] = useState<ViewMode>(read);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    setMode(read());
  }, [substrate]);
  const set = useCallback(
    (m: ViewMode) => {
      setMode(m);
      try {
        window.localStorage.setItem(KEY + substrate, m);
      } catch {
        /* storage unavailable: keep the choice for this page only */
      }
    },
    [substrate],
  );
  return [mode, set];
}

/** Window height (follows resizes): square 2D views should fit on the screen. */
export function useWindowHeight(): number {
  const [h, setH] = useState(() => window.innerHeight);
  useEffect(() => {
    const on = () => setH(window.innerHeight);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return h;
}

/** Width available to the world view (follows window resizes and fullscreen). Returns a
 * callback ref, so measuring starts whenever the element appears (e.g. after loading). */
export function useElementWidth(fallback = 720): [number, (el: HTMLElement | null) => void] {
  const [w, setW] = useState(fallback);
  const observer = useRef<ResizeObserver | null>(null);
  const attach = useCallback((el: HTMLElement | null) => {
    observer.current?.disconnect();
    observer.current = null;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(320, Math.floor(el.clientWidth))));
    ro.observe(el);
    observer.current = ro;
  }, []);
  useEffect(() => () => observer.current?.disconnect(), []);
  return [w, attach];
}

export function ViewModeBar({ substrate, mode, setMode, fullscreenTarget }: { substrate: string; mode: ViewMode; setMode: (m: ViewMode) => void; fullscreenTarget: RefObject<HTMLElement | null> }) {
  const t = useT();
  const modes = viewModesFor(substrate);
  const fullscreen = () => {
    const el = fullscreenTarget.current;
    if (!el) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void el.requestFullscreen?.().catch(() => undefined);
  };
  return (
    <div className="toolbar view-modes" role="group" aria-label={t("view.title")} data-testid="view-modes">
      <strong>{t("view.title")}:</strong>
      {modes.length > 1 &&
        modes.map((m) => (
          <button key={m} type="button" className={m === mode ? "primary" : ""} aria-pressed={m === mode} data-testid={`view-${m}`} onClick={() => setMode(m)}>
            {t(`view.${m}`)}
          </button>
        ))}
      <button type="button" onClick={fullscreen} data-testid="view-fullscreen">
        ⛶ {t("view.fullscreen")}
      </button>
      {mode !== "2d" && <span className="small muted">{t("view.presentation_note")}</span>}
    </div>
  );
}
