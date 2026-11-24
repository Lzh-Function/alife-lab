# 累積文化・原始社会ワールド（`cultural_world`）＆ 統合発展ロードマップ計画書

- **状態:** **Stage 3（`cultural_world` P4〜P5）実装・検証完了（2026-11-24）**（Stage 1 `cultural_world` P0〜P3、Stage 2 `connectome` S1 ＋実活動データ照合に続き、次は **Stage 4: S2〜S4 ＆ F13a–c**）。既存の 23+ 基質・全プリセットの再現性は一切壊さず、独立トラックとして維持されている。
- **提案する基質名:** `cultural_world`（`mode: exploratory` 専用）
- **科学的位置付け:** 探索的研究（exploratory）。現代文明のゲーム的再現や人間級知能・言語の自然発生を保証するものではない。「限られた初期能力の個体群から、知識の保持・改良・伝達がどのような環境・コスト条件で成立するか」を因果介入付きで検証する計算実験基盤である。
- **統合ロードマップ:** 本計画（トラック B：`cultural_world` P0〜P7）と [`docs/plans/connectome_plan.md`](connectome_plan.md) の他生物展開（トラック A：S1〜S7・F13）を、下記の 6 段階で縦割り進行する。

---

## 0. 全体統合ロードマップ（トラック A ＋ トラック B）

各ステージは「世界モデル／計算エンジン → API/CLI → Web GUI → 保存/リプレイ → CPU テスト＆対照実験」の 1 本の縦スライスとして完結させてから次へ進む（`AGENTS.md`）。

| ステージ | トラック | 内容 | 主な完了条件・承認ゲート |
|---|---|---|---|
| **Stage 1（完了）** | B: `cultural_world` **P0〜P3** | ADR 0050 確定 → 複数資源生態系（収支厳密保存）・3 層状態分離（遺伝／生涯／文化）・観察/模倣/教示・`erase_culture` 因果介入・A〜E 群比較スイープ | CPU テスト完結、A〜E 群＋介入結果の実測記録完了（[fidelity report](../reference_fidelity/cultural_world.md)） |
| **Stage 2（完了）** | A: コネクトーム **S1** | 線虫 *C. elegans*（302 細胞）：非発火電位 `MODEL_GRADED` とギャップ結合 `SYNAPSE_GAP_JUNCTION` ＋ 這う体 ＋ 実 GCaMP 活動照合 | FlyWire 無影響確認・5匹/154s・79〜111一致細胞で `spatial_r2`・`rollout_r2` 有意優位を検証済 |
| **Stage 3（完了）** | B: `cultural_world` **P4〜P5** | ADR 0052・許可リスト 10 演算子安全 DSL 構成的技術・保存則 `PlacedArtifact` 外在化・累積文化 5 基準アッセイ・Discovery Ledger E2/E3 連携・2 次元相図スイープ | `cultural_world_compositional` E3 合格 vs `no_composition` E1 棄却、H6 回復 δ=+1.00、保存誤差 `0.0` を検証済 |
| **Stage 4（次）** | A: **S2〜S4 ＆ F13a–c** | ハエ幼虫（3k）・ゴカイ幼生（9k）・ホヤ幼生（180）の種パック ＋ 異種比較統計 ＋ 異種間キメラ（交配・脳移植・キノコ体移植） | **実データ取得前に承認取得**。キメラは由来割合・シャッフル対照を明示 |
| **Stage 5** | B: `cultural_world` **P6〜P7** | 相互依存資源の原子的交換・分業/専門化・`forbid_exchange` 介入・文化系統 DAG ＆ 取引ネットワーク UI・CPU/GPU スケーリング実測 | 取引の資源保存・原子性、独立 run 単位での相補分業検証 |
| **Stage 6** | A: 大型皮質 **S5〜S7** | マウス視覚野 MICrONS（約 20 万細胞・5.2 億シナプス・7.5 万活動記録）＆ ヒト皮質 H01 比較（S7 ゼブラフィッシュは公式公開待ち） | **大容量取得・VRAM 配置を事前見積もりして承認取得** |

---

## 1. 研究の背景と中核の問い

### 1.1 中核の 4 つの問い

1. **遺伝と学習:** 遺伝的進化・生涯学習・他個体からの学習は、環境の変化速度に応じてどのように役割を分担するか。
2. **累積文化:** 個体の死亡を超えて有用な行動手順が残り、さらに改良されるための必要十分条件は何か。
3. **社会と技術:** 局所的なコスト・利益と限られた伝達チャネルだけから、構成的な手順・役割分化・交換が成立するか。
4. **開放性（Open-Endedness）:** 事前定義された技術ツリーをアンロックするのではなく、小さな操作手順の合成により機能的レパートリーが持続的に増えるか。

**最初の成功基準（Stage 1 目標）:** 「知識伝達を許した集団が、伝達を禁止した集団（同じ遺伝・学習条件・同じ環境乱数列）より高い成績を示し、かつその優位が遺伝子・人口・在庫を変えない `erase_culture` 介入で失われる」ことを、複数独立 seed と未観測環境で定量的に検証する。優位が出なかった場合も再現性のある否定的結果として記録する。

### 1.2 「文明」の観察階層（ゲーム的達成目標として扱わない）

| 階層 | 検証対象 | 観測する現象の例 | 主な実装段階 |
|---|---|---|---|
| L0 生態・個体群 | 生存、資源競争、個体群動態 | 絶滅、密度依存、棲み分け | P1 |
| L1 個体適応 | 遺伝的変異、寿命内学習 | 探索・採集方策の改良 | P1〜P2 |
| L2 社会的学習 | 他個体からの情報獲得 | 模倣、教示、知識の拡散 | P3 |
| L3 累積文化 | 世代を超える手順の改良 | 発明者死亡後の存続と段階改良 | P3〜P5 |
| L4 原始社会 | 協力、分業、交換 | 相補的役割、局所取引、共有設備 | P6 |
| L5 制度的構造 | 継続する集団慣行 | 反復的取引慣行、共有資源管理 | P6〜P7 |

### 1.3 科学的に避けるべき混同（非目標と解釈境界）

- 「人口が増えた」≠「知性が進化した」（資源流入増加だけでも増える）。
- 「行動が異なる」≠「分業が成立した」（ランダム個体差と相互依存による集団利得を区別する）。
- 「手順数が増えた」≠「累積文化」（意味のある構造差・収支成立・効用改善・再現伝達・先知識依存の 5 点で判定する）。
- 「巨大 LLM が文明を語った」≠「個体群が文明を創発した」（事前学習済み人間知識を混入させる LLM は自然発生実験には使わない）。
- 人類史の再現、AGI/自然言語の保証、既存 `agent_world` や ALife Lab の再現性破壊、CPU-only テストの喪失は行わない。

---

## 2. 既存 ALife Lab 基盤との関係とモデル設計

### 2.1 再利用する仕組みと新規基質 `cultural_world`

既存の [`agent_world`](../../src/alife_lab/substrates/agent_world/model.py) は数値検証・既存プリセットのハッシュ互換が確立しているため変更しない。代わりに以下の既存資産を活かして独立基質 `cultural_world` を追加する：

- **状態・介入・保存契約:** `ExperimentSpec` / `ExecutionProfile` 分離、`SubstrateAdapter`、決定論的チェックポイント＆リプレイ、`paired_fork`、`sweep`、`Discovery Ledger`（E0〜E4）。
- **再利用する計算手法:** `protocell` 型の厳密収支元帳（`residual <= 1e-9`）、`agent_world` 型の SoA ベクトル配列＋遺伝/非遺伝（`w` vs `dw2`）分離。

### 2.2 世界と経済・物理的制約

- 空間は周期境界の 2D 格子＋連続座標個体。環境は `food`, `material_a`, `material_b` の 3 資源場と環境レジーム（`stationary` / `seasonal` / `shifting_patches`）を持つ。
- すべての資源は非負かつ厳密収支に従う：`resource_initial + inflow == field + agent_inventories + consumed + decayed`。
- 乱数ストリームを **`env` / `birth` / `policy` / `learn` / `transmit` / `tech`** の 6 系統（`SeedSequence` 派生）に分離し、全系統をチェックポイント保存する。これにより、知識伝達や学習の有無を比較するペア実験で**環境変化・資源分布の乱数列を厳密に共有**できる。

### 2.3 個体の 3 層状態分離

1. **Genotype（遺伝・変異あり）:** 基本 MLP 重み（`w1, b1, w2, b2`）、学習率遺伝子・社交性遺伝子。
2. **Lifetime state（生涯状態・非遺伝）:** エネルギー、未加工素材在庫（`inv_a`, `inv_b`）、生涯学習差分 `dw2`（世界内エネルギー変化のみを報酬とする REINFORCE）、私的経験記憶（最良採集地など）。繁殖時に子の `dw2` と私的記憶は必ずゼロ初期化される。
3. **Cultural state（獲得した文化・遺伝子と独立）:** 観察可能な手順スキル `SkillProcedure` の所持リスト（`SkillSlot`：`skill_id`, `teacher_id`, `mode`, `acquired_tick`, `uses`）。伝達チャネルが有効な場合のみ親子間（`vertical`）・同世代間（`horizontal`）・世代間非血縁（`oblique`）・外部物体（`externalized`, P4）で伝わる。

### 2.4 安全な技術手順 DSL（P3〜P4）

- ブラウザや実験設定からの `eval/exec` を禁止し、許可リスト演算（`locate`, `gather`, `process_a`, `process_b`, `extract`, `store`；P4 で `combine`, `transform`, `place_object`, `use_object` 拡張）から成る有限長・非巡回手順 `SkillProcedure` として表現する。
- 事前定義された技術ツリーは用意せず、自己試行（`invention_prob` / `allow_improvement`）と模倣・誤伝達による変異から、手順ID・親手順ID・深さ・発明者生存情報を含む **文化系統（Knowledge DAG）** を追跡する。

---

## 3. 研究仮説・比較設計・必須検証

### 3.1 事前登録対象仮説（H1〜H6）

| ID | 仮説 | 主な指標 | 主な対照 |
|---|---|---|---|
| H1 | 生涯学習は変動環境で遺伝のみより適応的 | 環境切替後の生存・平均エネルギー | 遺伝のみ（`lifetime_learning=false`） |
| H2 | 知識伝達は有用手順の世代間保持を改善する | `skill_retention_after_death`、再発見時間 | 垂直伝達なし・社会伝達なし |
| H3 | 社会的学習と個体内発明の併用が累積効果を生む | `improvement_depth`、`innovation_utility` | 発明のみ（伝達オフ）、模倣のみ |
| H4 | 局所交換は相互依存資源下で専門化を促す（P6） | 専門化指数、相互依存度、生存率 | `forbid_exchange`、役割シャッフル |
| H5 | 文化の有用性は環境変動の頻度と予測可能性に依存する | 長期・未知環境への転移性能、知識喪失率 | 定常環境 vs 急変・予測不能環境 |
| H6 | 外部化された物体/記録は個体死後の知識保持を高める（P4） | 世代越え知識存続率 | 外部記録なし、`destroy_objects` 介入 |

### 3.2 第一比較実験（A〜E 群）と `erase_culture` 因果分岐

| 群 | 遺伝進化 | 生涯学習 | 垂直伝達 | 水平伝達 | 伝達コスト |
|---|---|---|---|---|---|
| A (`genetic_only`) | ○ | × | × | × | — |
| B (`lifetime_only`) | ○ | ○ | × | × | — |
| C (`vertical_only`) | ○ | ○ | ○ | × | あり |
| D (`full_free`) | ○ | ○ | ○ | ○ | 0 |
| E (`baseline_costly`) | ○ | ○ | ○ | ○ | あり |

さらに同一チェックポイントから `paired_fork` ＋ `erase_culture`（遺伝子・`dw2`・在庫・エネルギー・位置・環境場をビット不変のまま `transmissible_skills` のみ消去）および `set_transmission`（伝達遮断）を行い、観察上の優位が「獲得・伝達された文化」に起因するかを因果的に切り分ける。統計上の単位は常に **独立に初期化された 1 run（世界 1 本）** とする。

### 3.3 詳細フェーズ実装計画（P0〜P7）

| 段階 | 対象モジュール | 主な機能と受け入れテスト |
|---|---|---|
| **P0 契約固定** | ADR 0050、モデルカード骨格、`spec.py` | 3 層境界・DSL 定義・6 乱数系統・収支不変条件・操作定義を文書化 |
| **P1 生態・遺伝** | `ecology.py`, `agents.py`, `model.py`, `codecs.py`, `adapter.py` | 3 資源場・厳密収支（`residual <= 1e-9`）・チェックポイントビット一致・2D 可視化・個体インスペクタ |
| **P2 学習・記憶** | `controllers.py`, `memory.py` | `scripted / mlp`、REINFORCE 生涯学習、`private_memory` 分離（子の `dw2 == 0` 検証） |
| **P3 社会伝達・介入** | `transmission.py`, `interventions.py`, `metrics.py`, GUI/CLI/Presets/Sweeps | 垂直/水平/斜め伝達、コスト・誤差、イベントログ、`erase_culture`/`set_transmission`/`shock_environment`、A〜E 群＆因果分岐検証 |
| **P4 構成的技術** | `technology.py`（DSL 拡張＋外部物体） | `combine/transform/place_object`、5 分離指標（Novelty/Validity/Utility/Transfer/Improvement）、`no_composition` 対照 |
| **P5 累積文化アッセイ** | `src/alife_lab/evolution/cultural_assays.py`, `discoveries.py` 連携 | 発明者死後存続・段階改良・伝達遮断・`erase_culture` 比較・未知環境転移の 5 基準自動アッセイ、相図スイープ |
| **P6 社会構造・GUI 強化** | `social.py`、DAG/取引ネットワーク UI | 原子的交換、相互依存資源、専門化 vs 分業、`forbid_exchange` 対照、Cultural Lineage 切替表示 |
| **P7 性能実測＆拡張** | ベンチマークスクリプト・必要時 GPU バッチ | 個体数 × 知識件数グリッド実測、ボトルネック箇所のみ GPU 化、統計的同値性検証 |

---

## 4. 参考文献・境界

- R. Boyd & P. J. Richerson (1985), *Culture and the Evolutionary Process*
- J. M. Epstein & R. Axtell (1996), *Growing Artificial Societies* (Sugarscape)
- M. Tomasello (1999), ラチェット効果（累積文化進化）
- ALife Lab 既存基準実装・対照実験設計: [`src/alife_lab/substrates/agent_world/`](../../src/alife_lab/substrates/agent_world/), [`docs/reference_fidelity/agent_world.md`](../reference_fidelity/agent_world.md), [`docs/decisions/0025-discovery-ledger.md`](../decisions/0025-discovery-ledger.md)
