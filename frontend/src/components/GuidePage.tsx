// Beginner guide (Japanese): basics, how to use the lab, one section per substrate,
// evidence levels and a glossary. Content lives in ../guide/content.ts.
import { useEffect } from "react";
import { BASICS, CATEGORIES, EVIDENCE_LEVELS, GLOSSARY, SUBSTRATE_GUIDE, type SubstrateGuide } from "../guide/content";
import glider from "../guide/figures/01_glider.png";
import reactionDiffusion from "../guide/figures/03_reaction_diffusion.png";

const FIGURES: Record<string, string> = { "01_glider.png": glider, "03_reaction_diffusion.png": reactionDiffusion };

export function Stars({ level }: { level: number }) {
  return (
    <span className="stars" title={`むずかしさ ${level}/3`}>
      {"★".repeat(level)}
      <span className="muted">{"☆".repeat(3 - level)}</span>
    </span>
  );
}

function SubstrateSection({ g }: { g: SubstrateGuide }) {
  return (
    <section className="panel guide-substrate" id={`guide-${g.slug}`} data-testid={`guide-${g.slug}`}>
      <h3>
        {g.title} <Stars level={g.level} />
      </h3>
      <p className="guide-lead">{g.oneLine}</p>
      <div className="guide-grid">
        <div>
          <h4>どんな世界？</h4>
          <p>{g.what}</p>
          {g.figure && FIGURES[g.figure] && <img className="guide-figure" src={FIGURES[g.figure]} alt={g.title} />}
        </div>
        <div>
          <h4>ここを見よう</h4>
          <ul>
            {g.watch.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
          <h4>やってみよう</h4>
          <ol>
            {g.tryThis.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ol>
          <h4>比較用の実験（何とくらべる？）</h4>
          <p>{g.controls}</p>
          <div className="claim">
            <div>
              <strong>言えること：</strong>
              {g.canSay}
            </div>
            <div>
              <strong>まだ言えないこと：</strong>
              {g.cannotSay}
            </div>
          </div>
        </div>
      </div>
      <a href="#/catalog" onClick={() => sessionStorage.setItem("alife.catalog.substrate", g.slug)}>
        → 世界カタログでこの世界を開く
      </a>
    </section>
  );
}

export function GuidePage({ anchor }: { anchor: string | null }) {
  useEffect(() => {
    if (anchor) document.getElementById(`guide-${anchor}`)?.scrollIntoView({ block: "start" });
    else window.scrollTo(0, 0);
  }, [anchor]);

  return (
    <div className="guide" data-testid="guide-page">
      <nav className="guide-toc panel">
        <strong>もくじ</strong>
        <a href="#/guide/basics">はじめに</a>
        {CATEGORIES.map((c) => (
          <div key={c.id}>
            <div className="muted small">{c.title}</div>
            {Object.values(SUBSTRATE_GUIDE)
              .filter((g) => g.category === c.id)
              .map((g) => (
                <a key={g.slug} href={`#/guide/${g.slug}`}>
                  {g.title}
                </a>
              ))}
          </div>
        ))}
        <a href="#/guide/evidence">証拠レベル</a>
        <a href="#/guide/glossary">用語集</a>
      </nav>
      <div className="guide-body">
        <section id="guide-basics">
          <h2>ALife Lab ガイド</h2>
          <p className="muted">
            人工生命に初めて触れる人向けのガイドです。世界カタログや実験画面にある「?」マークにカーソルを合わせるかクリックすると、それぞれの設定や計測値の説明も読めます。
          </p>
          {BASICS.map((b) => (
            <div className="panel" key={b.title}>
              <h3>{b.title}</h3>
              {b.body.map((p) => (
                <p key={p}>{p}</p>
              ))}
            </div>
          ))}
          <div className="panel">
            <h3>どの世界から始める？</h3>
            <p>
              ★ の少ないものから試すのがおすすめです。まずは <a href="#/guide/conway">ライフゲーム</a> のグライダーで「1 tick 進める」や「ブラシ」に慣れ、次に{" "}
              <a href="#/guide/boids">Boids</a> や <a href="#/guide/gray_scott">Gray–Scott</a> で「比較用の実験と並べて比べる」体験を試してみてください。
            </p>
          </div>
        </section>
        <h2>世界ごとのガイド</h2>
        {CATEGORIES.map((c) => (
          <div key={c.id}>
            <h3 className="guide-category">
              {c.title} <span className="muted small">— {c.note}</span>
            </h3>
            {Object.values(SUBSTRATE_GUIDE)
              .filter((g) => g.category === c.id)
              .map((g) => (
                <SubstrateSection key={g.slug} g={g} />
              ))}
          </div>
        ))}
        <section className="panel" id="guide-evidence">
          <h2>証拠レベル（どこまで確かめたか）</h2>
          <p>見つけたことを、どこまで確かめたかで5段階に分けます。ALife Lab の実験記録にもこの考え方を使います。</p>
          <table className="table">
            <tbody>
              {EVIDENCE_LEVELS.map((e) => (
                <tr key={e.level}>
                  <td className="mono">{e.level}</td>
                  <td>
                    <strong>{e.title}</strong>
                  </td>
                  <td>{e.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section className="panel" id="guide-glossary">
          <h2>用語集</h2>
          <dl className="glossary">
            {GLOSSARY.map((g) => (
              <div key={g.term}>
                <dt>{g.term}</dt>
                <dd>{g.text}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>
    </div>
  );
}
