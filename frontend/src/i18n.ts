import { create } from "zustand";

export type Lang = "ja" | "en";

const ja = {
  "app.title": "ALife Lab",
  "app.subtitle": "人工生命実験室",
  "nav.catalog": "世界カタログ",
  "nav.library": "実験ライブラリ",
  "nav.lab": "ライブラボ",
  "nav.diagnostics": "診断",
  "conn.connected": "接続中",
  "conn.connecting": "接続しています…",
  "conn.disconnected": "切断（再接続待ち）",
  "lang.toggle": "English",
  "substrate.conway": "Conway's Game of Life",
  "preset.conway_glider.title": "グライダー",
  "preset.conway_glider.description": "周期4で斜めに移動する最小の宇宙船。基準テスト用。",
  "preset.conway_blinker.title": "ブリンカー",
  "preset.conway_blinker.description": "周期2の振動子。",
  "preset.conway_block.title": "ブロック（否定的対照）",
  "preset.conway_block.description": "静物。重心変位が0のままの対照。",
  "preset.conway_random_soup.title": "ランダムスープ",
  "preset.conway_random_soup.description": "seed依存のランダム初期状態。視覚的観察（E0）のみ。",
  "catalog.presets": "プリセット",
  "catalog.create": "実験を作成",
  "catalog.expected": "期待される観察",
  "catalog.negative_control": "否定的対照",
  "form.preset": "プリセット",
  "form.name": "実験名",
  "form.seed": "seed",
  "form.width": "幅",
  "form.height": "高さ",
  "form.boundary": "境界条件",
  "form.boundary.periodic": "周期（トーラス）",
  "form.boundary.dead": "固定（外側は死）",
  "form.max_steps": "最大tick",
  "form.checkpoint_interval": "チェックポイント間隔",
  "form.profile": "実行プロファイル",
  "form.profile.unimplemented": "未実装",
  "form.tps": "速度 (tick/秒, 0=最大)",
  "form.start_paused": "一時停止状態で開始",
  "form.submit": "作成して起動",
  "form.science_note": "科学条件（ExperimentSpec）と実行条件（ExecutionProfile）は別々にハッシュ化されます。",
  "form.validation": "検証結果",
  "library.empty": "まだ実験はありません。",
  "library.open": "開く",
  "col.run": "Run",
  "col.name": "名前",
  "col.state": "状態",
  "col.tick": "tick",
  "col.mode": "モード",
  "col.created": "作成",
  "lab.none": "ライブラリまたはカタログから実験を選んでください。",
  "lab.play": "再生",
  "lab.pause": "一時停止",
  "lab.step": "1 tick進める",
  "lab.checkpoint": "チェックポイント",
  "lab.restore": "最新チェックポイントから復元",
  "lab.cancel": "中止",
  "lab.speed": "速度 (tick/秒)",
  "lab.speed.apply": "適用",
  "lab.brush": "ブラシ",
  "lab.brush.draw": "描く（生）",
  "lab.brush.erase": "消す（死）",
  "lab.brush.off": "オフ",
  "lab.brush.radius": "半径",
  "lab.brush.note": "ブラシは次のtick境界で適用され、介入として記録されます。",
  "lab.tick": "tick",
  "lab.sim_rate": "シミュレーション速度",
  "lab.ui_fps": "表示FPS",
  "lab.preview_note": "表示はサンプリングされたプレビューです（研究用状態ではありません）。",
  "lab.dropped": "破棄したプレビュー",
  "lab.worker": "ワーカー",
  "lab.metrics": "計測値",
  "lab.checkpoints": "チェックポイント",
  "lab.events": "イベント",
  "lab.interventions": "介入",
  "lab.provenance": "来歴",
  "lab.mode.reference": "reference（忠実再現）",
  "lab.mode.exploratory": "exploratory（独自拡張）",
  "lab.mode.changed": "手動介入により exploratory として記録されています。",
  "lab.pending_command": "コマンド送信中…",
  "lab.not_live": "ワーカーは動作していません。",
  "lab.restorable": "このrunは中断されています。最新チェックポイントから明示的に復元できます。",
  "lab.evidence": "証拠レベル: E0（視覚的観察）。主張は付随しません。",
  "metric.alive_cell_count": "生存セル数",
  "metric.glider_displacement": "重心変位（グライダー）",
  "metric.centroid_displacement": "重心変位",
  "state.pending": "待機前",
  "state.queued": "キュー中",
  "state.starting": "起動中",
  "state.running": "実行中",
  "state.pausing": "一時停止要求中",
  "state.paused": "一時停止",
  "state.resuming": "再開中",
  "state.completed": "完了",
  "state.failed": "失敗",
  "state.cancelled": "中止",
  "state.interrupted": "中断",
  "diag.title": "システム診断",
  "diag.refresh": "再計測",
  "diag.measured_note": "値はすべて実測（nvidia-smi / psutil）。不明な項目は不明として表示します。",
  "common.error": "エラー",
  "common.loading": "読み込み中…",
  "common.none": "なし",
} as const;

export type MessageKey = keyof typeof ja;

const en: Record<MessageKey, string> = {
  "app.title": "ALife Lab",
  "app.subtitle": "Artificial life laboratory",
  "nav.catalog": "World catalog",
  "nav.library": "Experiment library",
  "nav.lab": "Live lab",
  "nav.diagnostics": "Diagnostics",
  "conn.connected": "Connected",
  "conn.connecting": "Connecting…",
  "conn.disconnected": "Disconnected (retrying)",
  "lang.toggle": "日本語",
  "substrate.conway": "Conway's Game of Life",
  "preset.conway_glider.title": "Glider",
  "preset.conway_glider.description": "Smallest spaceship, period 4, moving diagonally. Reference test.",
  "preset.conway_blinker.title": "Blinker",
  "preset.conway_blinker.description": "Period-2 oscillator.",
  "preset.conway_block.title": "Block (negative control)",
  "preset.conway_block.description": "Still life; centroid displacement stays 0.",
  "preset.conway_random_soup.title": "Random soup",
  "preset.conway_random_soup.description": "Seeded random initial state. Visual observation only (E0).",
  "catalog.presets": "Presets",
  "catalog.create": "Create experiment",
  "catalog.expected": "Expected observation",
  "catalog.negative_control": "Negative control",
  "form.preset": "Preset",
  "form.name": "Name",
  "form.seed": "Seed",
  "form.width": "Width",
  "form.height": "Height",
  "form.boundary": "Boundary",
  "form.boundary.periodic": "Periodic (torus)",
  "form.boundary.dead": "Dead (absorbing)",
  "form.max_steps": "Max ticks",
  "form.checkpoint_interval": "Checkpoint interval",
  "form.profile": "Execution profile",
  "form.profile.unimplemented": "not implemented",
  "form.tps": "Speed (ticks/s, 0 = max)",
  "form.start_paused": "Start paused",
  "form.submit": "Create and start",
  "form.science_note": "Scientific spec and execution profile are hashed independently.",
  "form.validation": "Validation",
  "library.empty": "No experiments yet.",
  "library.open": "Open",
  "col.run": "Run",
  "col.name": "Name",
  "col.state": "State",
  "col.tick": "Tick",
  "col.mode": "Mode",
  "col.created": "Created",
  "lab.none": "Choose an experiment from the library or catalog.",
  "lab.play": "Play",
  "lab.pause": "Pause",
  "lab.step": "Step 1 tick",
  "lab.checkpoint": "Checkpoint",
  "lab.restore": "Restore latest checkpoint",
  "lab.cancel": "Cancel",
  "lab.speed": "Speed (ticks/s)",
  "lab.speed.apply": "Apply",
  "lab.brush": "Brush",
  "lab.brush.draw": "Draw (alive)",
  "lab.brush.erase": "Erase (dead)",
  "lab.brush.off": "Off",
  "lab.brush.radius": "Radius",
  "lab.brush.note": "Brush edits apply at the next tick boundary and are logged as interventions.",
  "lab.tick": "Tick",
  "lab.sim_rate": "Simulation rate",
  "lab.ui_fps": "Display FPS",
  "lab.preview_note": "The view is a sampled preview, not the research state.",
  "lab.dropped": "Dropped previews",
  "lab.worker": "Worker",
  "lab.metrics": "Metrics",
  "lab.checkpoints": "Checkpoints",
  "lab.events": "Events",
  "lab.interventions": "Interventions",
  "lab.provenance": "Provenance",
  "lab.mode.reference": "reference (faithful)",
  "lab.mode.exploratory": "exploratory (extended)",
  "lab.mode.changed": "Recorded as exploratory because of manual interventions.",
  "lab.pending_command": "Sending command…",
  "lab.not_live": "No worker is running.",
  "lab.restorable": "This run was interrupted. Restore explicitly from the latest checkpoint.",
  "lab.evidence": "Evidence level: E0 (visual observation). No claim attached.",
  "metric.alive_cell_count": "Alive cells",
  "metric.glider_displacement": "Centroid displacement (glider)",
  "metric.centroid_displacement": "Centroid displacement",
  "state.pending": "Pending",
  "state.queued": "Queued",
  "state.starting": "Starting",
  "state.running": "Running",
  "state.pausing": "Pause requested",
  "state.paused": "Paused",
  "state.resuming": "Resuming",
  "state.completed": "Completed",
  "state.failed": "Failed",
  "state.cancelled": "Cancelled",
  "state.interrupted": "Interrupted",
  "diag.title": "System diagnostics",
  "diag.refresh": "Probe again",
  "diag.measured_note": "All values are measured (nvidia-smi / psutil). Unknown items are shown as unknown.",
  "common.error": "Error",
  "common.loading": "Loading…",
  "common.none": "None",
};

const dictionaries: Record<Lang, Record<MessageKey, string>> = { ja, en };

function initialLang(): Lang {
  try {
    const saved = window.localStorage.getItem("alife.lang");
    if (saved === "ja" || saved === "en") return saved;
  } catch {
    /* storage unavailable */
  }
  return "ja";
}

interface LangState {
  lang: Lang;
  setLang: (lang: Lang) => void;
}

export const useLang = create<LangState>((set) => ({
  lang: initialLang(),
  setLang: (lang) => {
    try {
      window.localStorage.setItem("alife.lang", lang);
    } catch {
      /* ignore */
    }
    document.documentElement.lang = lang;
    set({ lang });
  },
}));

export function translate(lang: Lang, key: string): string {
  const dict = dictionaries[lang] as Record<string, string>;
  return dict[key] ?? key;
}

export function useT(): (key: string) => string {
  const lang = useLang((s) => s.lang);
  return (key: string) => translate(lang, key);
}

export const MESSAGE_KEYS = Object.keys(ja) as MessageKey[];
export const DICTIONARIES = dictionaries;
