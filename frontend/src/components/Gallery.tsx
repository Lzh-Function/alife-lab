// Gallery: pictures the user liked, kept apart from the discovery ledger (spec §3.10).
// Each card is a visual record (E0) with the run and tick it came from.
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { SUBSTRATE_GUIDE } from "../guide/content";
import { useT } from "../i18n";

export function GalleryPage() {
  const t = useT();
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["gallery"], queryFn: api.gallery });
  const hide = async (id: string) => {
    if (!window.confirm(t("gallery.hide_confirm"))) return;
    await api.hideGallery(id);
    await qc.invalidateQueries({ queryKey: ["gallery"] });
  };
  return (
    <div className="gallery-page" data-testid="gallery-page">
      <h2>{t("gallery.title")}</h2>
      <div className="substrate-intro">
        <p>{t("gallery.intro")}</p>
        <p className="small muted">{t("gallery.e0")}</p>
      </div>
      {!list.data?.length ? (
        <p className="muted" data-testid="gallery-empty">
          {t("gallery.empty")}
        </p>
      ) : (
        <div className="gallery-grid">
          {list.data.map((g) => {
            const guide = SUBSTRATE_GUIDE[g.substrate];
            return (
              <figure key={g.id} className="gallery-card panel" data-testid={`gallery-${g.id}`}>
                <img src={`/api/v1/gallery/${g.id}/image`} alt={g.title} className="gallery-img" />
                <figcaption>
                  <strong>{g.title}</strong>
                  {g.note && <p className="small">{g.note}</p>}
                  <p className="small muted">
                    {guide?.title ?? g.substrate} · tick {g.tick} · {new Date(g.created_at * 1000).toLocaleDateString()}
                  </p>
                  {guide && <p className="small muted">{guide.oneLine}</p>}
                  <span className="toolbar">
                    <a className="button-link" href={`#/lab/${g.run_id}`}>
                      {t("gallery.open_run")}
                    </a>
                    {g.substrate === "search" && (
                      <a className="button-link" href={`#/lineage/${g.run_id}`}>
                        🌳 {t("lineage.open")}
                      </a>
                    )}
                    <a className="button-link" href={`#/guide/${g.substrate}`}>
                      {t("gallery.read_guide")}
                    </a>
                    <button type="button" className="ghost small" onClick={() => hide(g.id)}>
                      {t("gallery.hide")}
                    </button>
                  </span>
                </figcaption>
              </figure>
            );
          })}
        </div>
      )}
    </div>
  );
}
