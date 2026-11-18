# ALife Lab — 統合人工生命研究プラットフォーム 要件・実装設計書

> **文書の役割:** プロジェクトのマスター仕様。`AGENTS.md` は作業規約、`docs/GIT_AND_ARTIFACT_POLICY.md` はデータ管理詳細とする。
> **成果物:** 実験用サーバー、ブラウザGUI、人工生命の複数基質、探索・進化・科学的評価、計算資源に依存しない実行基盤、再現可能な実験ストレージ。
> **実行環境:** CPU開発環境／ローカル対話用GPU／大容量・高性能の実験用GPU／任意の複数GPU・ジョブ管理クラスタ。固有のGPU型番やVRAM容量には依存させない。
> **導入原則:** 開発はローカルで完結可能とし、計算先や保存先のアダプタは必要に応じて追加する。

## 仕様の解釈と優先順位

- **MUST** はプロジェクトの基本契約。**SHOULD** は対象モジュールを採用したときの標準要件。**MAY** は拡張候補。
- **P0〜P3** は実装の依存関係と導入順序であり、研究領域の価値判断や重点分野の宣言ではない。
- シミュレーション基質は独立した研究トラックとして扱う。いずれか一分野に特化することなく、発見・実験・検証に共通の基盤を提供する。
- 掲載するアルゴリズム、API、YAML、スケール、性能値は**要件／設計例**であり、プログラムの実装済み状態や実測性能を意味しない。
- 論文の忠実再現と独自拡張を別モードとし、差分と根拠を記録する。
- 本文と `AGENTS.md` に不整合がある場合は、独断の置換を避けてADRに記録し、科学的要件を優先して解決する。

---

## 0. エグゼクティブサマリー

単なる人工生命シミュレーションの寄せ集めではなく、**人工世界を作り、進化させ、観察し、操作し、保存し、比較し、定量的に研究する「ブラウザ上のALife実験室」**を構築する。

**優先する最終体験:** ローカルサーバーを起動してブラウザの `http://localhost:8000` にアクセスすると、世界のカタログが現れ、プリセットを選ぶだけで世界が動き始める。利用者は個体をクリックして状態・系統・ゲノムを観察し、資源を追加したり世界を一時停止したり、途中の状態から分岐実験を開始できる。複数世界・複数seed・複数アルゴリズムを比較し、実験記録を科学的に再利用できる。

### 0.1 必須の成果物

1. **ALife Engine:** 基質（世界）の共通管理、独立した実験ランナー、プロセス分離、シード管理。
2. **Web Observatory:** 世界一覧、ライブビュー、実験作成、パラメータ操作、個体/セル/領域インスペクタ、イベントタイムライン、比較画面。
3. **Evolution Lab:** 進化戦略、品質多様性（QD）、新規性探索、共進化、アーカイブ、系統追跡。
4. **Scientific Analysis:** 複製・修復・恒常性・多様性・生態的相互作用・オープンエンド性の検証可能な測定。
5. **Experiment Store:** SQLiteメタデータ、Zarr/NPZ状態、Parquet計測値、再実行可能なマニフェスト、MP4/GIF/画像。
6. **Automation / API:** CLIとREST/WebSocketを統一したジョブ実行API。将来のエージェント操作にも対応。
7. **Reference Validation:** 標準モデルと論文/リファレンス実装に対する差分テスト、保存則、統計的対照実験。

### 0.2 対象となるALife系

| 系統 | 基質/環境 | 研究テーマ | 実装優先度 |
|---|---|---|---|
| 古典的セルオートマトン | Conway's Game of Life、Elementary/Outer-totalistic CA | 創発、パターン形成、基準系 | P0 |
| 粒子集合系 | Boids、Particle Life | 群れ、凝集、相互作用則 | P0〜P1 |
| 連続セルオートマトン | Lenia、Flow-Lenia | 自己組織化、質量保存、形態、生態 | P1 |
| デジタルプログラム生命 | BFF primordial soup / CuBFF準拠 | 自己複製コードの自然発生 | P1 |
| 神経セルオートマトン | Growing NCA | 形態形成、再生、損傷回復 | P1 |
| 多主体NCA | Petri Dish NCA / Digital Ecosystems | 競争、共存、寿命内学習 | P2 |
| 反応拡散・資源生態系 | Gray–Scott、栄養/捕食者-被食者 | パターン、生態系の安定性 | P2 |
| 仮想生物身体性 | MuJoCo MJX-JAX / MJX-Warp | 歩行・形態と制御の共進化 | P2 |
| 連続的な身体＋流体 | Microcosmosアダプタ | フィラメント生物、遊泳、物理的身体性 | P3 |
| デジタル進化 | Avida等の外部系へのアダプタ | 突然変異、選択、系統 | P3 |
| 人工化学 | 文字列/グラフ反応ネットワーク（Stringmol型を参考） | 触媒・複製・反応ネットワーク進化 | P2 |
| プロトセル・代謝系 | 2D区画/膜/資源フラックス | 区画形成、代謝・成長の結合 | P2〜P3 |
| 群知能・化学走性 | Ant trails / Physarum-inspired / Swarm | 間接的通信、ネットワーク形成 | P1〜P2 |
| 軽量神経エージェント | Gridworld / particle-agent hybrid | 行動・学習・通信・共進化 | P2 |
| ルール空間探索 | CA rule search / parameterized worlds | 法則ごとの創発可能性 | P2 |
| 複数スケールの世界 | 部分領域拡大・粗視化/異種場結合 | スケール依存の秩序・相互作用 | P3 |

**重要:** 「P0」を最初に実装するのは、容易な系でWeb GUI・実験保存・介入・計測のすべてを垂直統合するため。ALifeの高度な生態系モデルから始めて巨大な未完成モノリスを作らないこと。

---

### 0.3 研究プログラムの全体マップ

| 主題 | 主な世界・系 | 観測したい現象 | 初期検証・反証の軸 |
|---|---|---|---|
| 自己組織化 | CA / Lenia / Particle Life / 反応拡散 | 持続構造・伝播・集団秩序 | ランダム化、境界・時間刻み変更 |
| 自己複製・進化 | BFF / Avida / 人工化学 | 複製子、継承、選択、適応 | 複製アッセイ、非実行・非変異対照 |
| 形態形成・再生 | NCA / プロトセル / Microcosmos | 成長、形状保持、修復 | 損傷、未学習対照、ノイズ摂動 |
| 行動・身体性 | Boids / swarm / MJX / 軽量神経エージェント | 協調・探索・ロコモーション | センサー/通信遮断、静的制御器 |
| 多主体相互作用 | PD-NCA / 資源場 / 共進化 | 競争、協力、安定化、淘汰 | 介入、個体数/資源/学習の対照 |
| 発見アルゴリズム | ASAL / QD / curiosity / rule search | 未知のルール・形態・動態 | ランダム探索との予算一致比較 |
| 計測科学 | イベント検出 / 情報理論 / 因果fork | 創発の操作的定義と再現性 | 複数seed、同時刻対照、独立再評価 |

**「動いて面白い」ことと「新規な科学的発見」を分ける。** デモの魅力はユーザー体験として追求するが、科学的発見の判定には独立したアッセイと対照実験を必要とする。

### 0.4 機能の階層的な完成目標

1. **Playable:** ブラウザで起動・操作・可視化できる。
2. **Observable:** 状態、メトリクス、イベント、性能が対応付けられる。
3. **Reproducible:** checkpoint / manifest / seed / backend を保存・復元できる。
4. **Comparable:** 条件・seed・介入による比較が可能。
5. **Discoverable:** 探索器・イベント検出・アーカイブにより候補を見つけられる。
6. **Testable:** 候補に独立アッセイ・反事実的介入・統計的検証を適用できる。

高度な基質の実装は、この6段階のどこまで達成したかを機能表に明示する。

---

## 1. 設計原則とスコープ

### 1.1 科学的忠実度を第一にする

- **Reference mode（忠実再現）**と**Exploratory mode（独自拡張）**を分ける。設定・結果・UIに区別を明示する。
- 探索の評価関数と、観測のためのメトリクスと、世界の力学を混同しない。
- BFFでは**適応度に基づく選択を勝手に導入しない**。複製子を検出する分析コードもスープの更新には影響させない。
- 「新しい行動」「自己複製」「生命の創発」は単一のヒューリスティック指標だけで宣言しない。検出候補、独立検証、対照群を段階的に扱う。
- GPU非決定性・精度・ソフトウェア更新によりビット完全再現できない場合は、その限界を記録する。

### 1.2 分離可能な4層

1. **World / Substrate:** 基本法則、ステップ更新、状態、必要な乱数。
2. **Experiment / Intervention:** 初期化、刺激、環境変更、ロールアウト、分岐。
3. **Search / Measurement:** 探索器、適応度、観測指標、発見候補検出。観測は原則として非介入。
4. **Service / Presentation:** ジョブ管理、API、可視化、保存、アクセス制御。

UI、動画化、実験DBは基質の時間発展を変更してはいけない。

### 1.3 拡張性のために「無理な統一」をしない

- CA、粒子、剛体、BFFは物理的意味も時刻単位も違う。`State` や `Genome` の単一具体クラスを全基質に強制しない。
- 最小の `SubstrateAdapter` 契約と、`Capabilities`（何が可能か）と、基質専用Schemaによる設定検証を用意する。
- BFFの `epoch`、CAの `tick`、MJXの `physics_step`、進化の `generation` は明示的に区別する。
- モジュールごとにJAX、PyTorch、NumPy、CUDA/C++等を使い分けてよい。**全基質JAXへの強制移植は禁止**。

### 1.4 開発言語・記述言語

- ユーザーとの説明、マスター仕様は日本語。
- ソースコード、型、docstring、コメント、ログキー、APIフィールド、コミットメッセージは英語。
- Web GUIの表示文言は日本語を基本とし、i18nにより英語にも切り替えられるようにする。コード内にUIテキストを直書きしない。
- 各論文に対する独自の再現範囲と実装差分は英語の `docs/reference_fidelity/` に記録する。

---

### 1.5 科学的境界と命名ルール

- `phenomenon_candidate` は特徴量・画像・時系列からの**探索的検出**。`validated_phenomenon` は事前に定めた独立アッセイを通過した事例。
- 個体識別（tracking）、系統推定（lineage inference）、種判定（species assignment）を別の推論パイプラインにし、信頼度を保持する。
- `replication` は直接のコピー機構と再帰的増殖を区別。プログラムの類似度だけでは複製子認定しない。
- `learning`（寿命内パラメータ更新）、`evolution`（世代間の継承と変異）、`search`（外部最適化）、`self-organization`（状態力学）をデータモデルでも別々に記録する。
- 手動・自動の世界ルール変更は `rule_revision` を上げ、同一の固定ダイナミクスの長期実験と混ぜない。
- 世界の内部状態を変更する計測器は `invasive_observer` として登録し、非介入観測と別categoryにする。

### 1.6 設計対象から除外すること（Non-goals）

- あらゆる基質の状態を単一テンソル形式に変換すること。
- 動的・高度な生物学的現象を汎用適応度関数1個で測定すること。
- すべての探索を微分可能にすること。
- 単一の巨大なAIエージェントが全世界の個体行動を毎step推論すること。
- どのローカルGPUでも大規模な3D流体・分子動力学を実時間で忠実計算できると約束すること。
- 大きなVRAMを持つGPUなら、すべての基質が一定倍率で高速化すると約束すること。
- 初期開発段階でマルチノード分散計算や複雑な分散スケジューラを完備すること。ただし将来の実行先変更を妨げる密結合は非目標ではなく**禁止**。
- 実験生成データをすべてGitやGit LFSに入れること。
- 実行プラットフォームを移しても、異なるGPU演算で必ずbitwise identicalに再現されると主張すること。

---

## 2. 基本アーキテクチャ：ローカルサーバー + Web GUI + Worker

### 2.1 推奨スタック

| コンポーネント | 推奨技術 | 役割・判断理由 |
|---|---|---|
| Python環境 | `uv` + `pyproject.toml` + `uv.lock` | 依存解決と再現性 |
| APIサーバー | FastAPI + Pydantic v2 + Uvicorn | 型付きREST、OpenAPI、WebSocket |
| UI | React + TypeScript + Vite | モジュール化された研究UI |
| UI状態/取得 | TanStack Query + Zustand | サーバー状態と操作中状態の分離 |
| 2D Canvas | PixiJS 8 / WebGL2（WebGPUは任意） | CA、粒子、フィールドの描画 |
| 3D Viewer | Three.js | MJXの生物・関節・軌跡表示 |
| グラフ | Plotly.js または uPlot | 経時曲線、分布、散布図 |
| 系統/相互作用グラフ | Cytoscape.js | 系統・系譜・生態ネットワーク |
| メタデータ | SQLite（WAL） + SQLAlchemy/SQLModel | シングルマシン用run/index管理 |
| 数値計測 | Parquet（PyArrow） | 集計・大規模解析 |
| 状態スナップショット | Zarr v3（可変長は別形式）/NPZ | チャンク化・高速ロード・復元 |
| イベント | SQLiteテーブル + JSONL export | 監査・因果的イベント閲覧 |
| 進化 | evosax / QDax | ESとQDの再利用 |
| GPUシミュレーション | JAX、CuPy/独自CUDA、必要に応じPyTorch | 基質固有の最適バックエンド |
| 動画出力 | FFmpeg | 観測フレームからMP4/GIF作成 |
| 開発・テスト | pytest, Ruff, mypy/pyright, Vitest, Playwright | 継続的品質保証 |

**初期実装ではRedis、Celery、PostgreSQL、Kubernetesは導入しない。** シングルワークステーション上では障害点と設定コストが増えるため。プロセス分離した単一ワーカーとSQLiteベースの永続ジョブキューで開始する。

### 2.2 システム構成

```text
Browser (React / PixiJS / Three.js)
  |  REST: configuration, runs, controls, reports
  |  WebSocket: metrics, events, low-rate previews, job status
  v
FastAPI service (CPU process; GPU objects never loaded here)
  +-- experiment registry / configuration validation
  +-- SQLite metadata & queue / artifact index
  +-- static artifact and playback delivery
  +-- client session / access policy / command acknowledgements
  |
  v
Job supervisor (separate process)
  +-- GPU worker (single active GPU-heavy task by default)
  |     +-- JAX / CUDA / PyTorch simulation and search
  |     +-- bounded preview snapshots / event sampling
  |     +-- checkpoint / metrics / artifacts
  +-- CPU workers (lightweight tests, report generation, encoding)
  |
  v
Experiment store
  +-- SQLite (runs, events, lineage metadata, artifacts)
  +-- Parquet (time-series metrics)
  +-- Zarr / NPZ (states, genomes, checkpoint)
  +-- PNG / MP4 / WebP (visualization)
```

### 2.3 GPU Workerの扱い

- **APIサーバーはGPUを確保しない。** JAX / PyTorch / CUDAのimport・初期化はGPUワーカー側。
- 単一GPUでは原則**重量級GPUジョブを1件だけ実行**。それ以外は `queued`。優先度・キャンセル機構を持つ。
- ライブ操作はジョブの既定の安全地点（tick境界等）で適用し、シミュレーション状態に直接非同期書き込みしない。
- サーバー再起動時は中断ジョブを `interrupted` に遷移。最新の正しいcheckpointから**明示的に**再開できるようにする。
- `pending -> queued -> starting -> running -> pausing -> paused -> resuming -> completed/failed/cancelled/interrupted` を状態機械として定義する。
- GPUワーカー異常終了時もAPIは存続。ログ・終了コード・最終heartbeat・OOM診断を保存する。
- `GPU_JOB_CONCURRENCY=1` と `GPU_MEMORY_BUDGET_MB` を設定で変更可。ただし競合実行はベンチマーク完了後のみ許可。

### 2.4 サーバー起動と接続

- `uv run alife serve --host 127.0.0.1 --port 8000` でAPIとビルド済みWeb GUIを配信。
- 開発時は `uv run alife api` と `npm run dev`（Vite proxy）を使用可。
- Windows 11 + WSL2では**まず同一PCの `localhost` アクセスを検証**する。WSLネットワーク構成によって到達性が異なるので想定だけで設定しない。
- 外出先アクセスは任意機能。Tailscale Serve等の認証付き経路を優先し、**GPU計算APIを公開インターネットに直接露出させない**。
- `0.0.0.0` は明示的な認証・アクセス制限・Origin検証が実装されない限り禁止。
- 個人用のローカル構成でも、CSRF/Origin検証、リクエストサイズ上限、ジョブ数・CPU/GPU時間上限、ファイルパストラバーサル防止を実装する。

---

### 2.5 実行スケジューラ・障害復旧の実装契約

- API / Supervisor / Workerの各プロセスに独立した `process_id`、`run_id`、`lease_token` を持たせる。古いworkerからのイベントで新しいジョブの状態を上書きしない。
- ジョブの受付と永続化はトランザクション。実行中の心拍喪失では即時に`failed`とせず`recovering`候補として整合性確認を行う。
- **`pause`は安全地点まで遅延する**。長大なGPU/JAX scan中は反映まで時間差が生じるため、画面に `pause_requested` / `pause_applied_at` を表示。
- `cancel`はまず協調的キャンセル、次に期限付きでプロセス終了。生存確認とcheckpointのアトミック確定を優先する。
- GPUタスクと短時間のCPUデータ解析は異なるキューに置き、動画エンコードで実験ステップが阻害されないようにする。
- 最初はFIFO + 明示優先度だけで十分。将来の複数GPU・Slurm連携は `Executor` 層で後付けする。
- `sampled preview` と `durable observation` を別channelにし、ブラウザが遅い場合はpreviewだけdropできる。
- Workerは可能ならspawn形式で起動し、親APIで先にGPU runtimeを初期化してforkすることを避ける。
- 実行が外部パッケージの未サンドボックス化された任意Pythonプラグインを伴う場合、信頼済みプラグインに限定し権限を明示する。

### 2.6 リソース・費用の見積もり

- 事前計画：推定計算時間、VRAMピークの予測範囲、GPU時間、ディスク容量、書込帯域、プレビュー帯域。
- 予測と実測の差分を蓄積し、同基質・同backendの過去の実測から見積もりを改善する。実測データがなければ `unknown` を返す。
- `fast_preview`, `balanced`, `long_run`, `reproduction` の4実行profileを用意するが、profileによって**シミュレーションの物理法則を黙って変えない**。
- 長期バッチは通知や画面表示を止めても計算継続。`max_walltime_seconds`, `max_gpu_seconds`, `max_disk_mb`を守る。

---

### 2.7 Executor境界とControl Plane / Data Plane分離

`ExecutionPlan`（科学的に何をするか）と`ResourceRequest`（どこでどう動かすか）を分ける。GUI/API/CLIは共通のrun管理操作を用い、workerの具体的な起動経路には依存しない。

```text
Browser / CLI
   | REST + WebSocket (authenticated controller)
   v
ALife Controller (FastAPI / SQLite / scheduler / Artifact index)
   | submit JobSpec + input digests
   |<----- heartbeat/status/metrics/preview descriptors
   |
   +--- LocalExecutor --------> local worker -> local ArtifactStore
   |
   +--- RemoteAgentExecutor ---> authenticated agent -> remote GPU worker
   |                           -> remote scratch -> staged artifact upload
   |
   +--- SSHBatchExecutor ------> SSH-initiated remote process (optional)
   |                           -> return status/artifacts by pull or sync
   |
   +--- SlurmExecutor ---------> sbatch job on allocated compute node (optional)
                               -> scratch and durable artifact sync

           Git revision + immutable resolved config + artifact hashes
```

- Controllerは実験の権威的なrun ID、状態遷移、親子関係、アーティファクトインデックスの**単一ライター**となる。
- Workerはrunごとに分離されたプロセス/コンテナで動く。ControllerからGPU runtimeをimportしない。
- リモートノードからローカルSQLiteのDB/WALファイルをNFS/SMBマウントで直接更新させない。状態の同期にはAPIまたは耐障害なメッセージ/結果同期を用いる。
- ローカルGUIを維持したままリモートGPUを指定可能。ライブ表示は低解像度のプレビューとメトリクスのみを転送し、rawフレーム/全状態を毎step送らない。
- `LocalExecutor`はMUST。`RemoteAgentExecutor`、`SSHBatchExecutor`、`SlurmExecutor`は独立プラグインで導入する。初期段階でSlurmや共有GPU必須にしない。

**抽象契約（概念API）:**

```python
class Executor(Protocol):
    def probe(self) -> ExecutionTarget: ...
    def validate(self, spec: JobSpec) -> ValidationReport: ...
    def stage(self, spec: JobSpec) -> StagingReceipt: ...
    def submit(self, spec: JobSpec) -> JobHandle: ...
    def status(self, handle: JobHandle) -> JobStatus: ...
    def request_control(self, handle: JobHandle, action: ControlAction) -> ControlReceipt: ...
    def collect(self, handle: JobHandle) -> ArtifactReceipt: ...
```

実装は必ず非同期・永続化を考慮し、上記の型名や同期シグネチャをそのまま固定APIとしない。共通操作を持っても対応可能な制御は実行先で異なるため、`ExecutionTarget.capabilities`に`pause_live`, `checkpoint_on_signal`, `interactive_preview`, `remote_cancel`, `multi_gpu`等を明記する。**Slurm batchでの`pause`を通常のリアルタイム一時停止と偽らない**。

### 2.8 Worker登録・能力探査・実行先選択

能力を静的YAMLの値だけで断定せず、開始時に実測プローブする。取得可能な項目：

- 実行先ID、OS/アーキテクチャ、Python/パッケージlock、API protocol version。
- GPU台数、GPU UUID（必要な環境に限る）、製品名、driver、runtime、compute capability、total/free VRAM、温度/電力（権限がある場合）。
- CPUコア、available RAM、scratch空き、永続ストレージ容量、I/O帯域、ネットワーク上り帯域（未知ならunknown）。
- インストール済みbackend、JAX/PyTorch/CuPy/MuJoCo-Warpの実行可否、必要なモデル/データのcache状態。
- Slurm環境ならpartition/QOS/account/GRES/時間・memory上限を**コマンド/サイト規定から検証**。権限のない項目はunknownとして残す。

スケジューラの選択順：`explicit target` → `capability filter` → `resource constraints` → `queue/runtime/cost estimate` → `final confirmation when required`。意図しない高コスト計算資源へのジョブ投入、課金のあるクラスタ利用、共有計算資源への大規模投入は行わない。

`auto`で選択してもresolved target、計算資源、queue時間、モデル及びカーネル選択、依存環境digestをrun manifestへ記録する。見積もりが不明ならunknown。自動選択で科学条件を変更してはいけない。

### 2.9 状態同期・遅延・障害モード

- `JobSpec`は一意の`run_id`と`request_id/idempotency_key`、immutable config digest、source commit、実行backendと資源制約を持つ。
- 送信前に必要な入力とコードのdigestを確認し、データステージング完了後に実行開始。コード配布は**固定commitのcheckout**または**秘密情報除去済み・hash済みsource archive**で行う。`git pull`で動的なmainブランチを無条件に実行しない。
- 実行中の`heartbeat`には`job_epoch/lease_token`を持たせ、遅延イベントが終了後の新runを上書きしない。
- previewは欠損可能、重要イベント/metrics/checkpointは追跡可能な永続アーティファクトとして転送する。アップロードはmultipart/再試行/チェックサム照合を備える。
- Controllerが再起動してもremote batchは存続し得る。復帰後に`job_id`, `external_scheduler_id`, `lease`, `artifact receipts`で再照合。
- 信頼できない/互換性のないworkerを`quarantined`に置く。データ欠損・バージョン不一致時に成功扱いしない。
- interactive workerは利用可能な操作を実行してackを返す。batch workerは原則stop/checkpoint-and-exit/restartで操作を表現。介入を即時適用できない場合はGUIに明示。

### 2.10 リモート接続と権限の原則

- まずはローカルのみ。遠隔はTailscale等の閉域接続、SSH、または認証済みリバース接続を利用する。計算ノードやジョブAPIをインターネットへ直公開しない。
- UIの認証とremote workerの認証は**別の資格情報・権限**を使う。workerのtokenはスコープ・失効期限を持ち、ログへ表示せず、Gitに保存しない。
- 外部のSlurmや共有HPCでポート待受、コンテナ、ファイル転送が許されるかはサイト規程とACLを確認。禁止ならpolling/SSH方式へ切り替える。
- 任意のWeb GUIパラメータをremote shellへ直接連結しない。許可済み定型コマンドと型検証済み引数だけを使用する。SSHのhost key検証を無効化しない。

---

## 3. Web GUI 詳細仕様 — ALife Observatory

**Web UIは「ログを見る補助ツール」ではなく、本プロジェクトの主操作面とする。** CLIから起動した実験も同じUIから閲覧・再生できるようにする。

### 3.1 画面一覧

| 画面 | 主要機能 | 優先度 |
|---|---|---|
| Home / Dashboard | 稼働中世界、GPU/VRAM、待機ジョブ、最近の発見、最近の実験 | P0 |
| World Catalog | 実行可能な基質、説明、サンプル、必要リソース、論文、プリセット | P0 |
| Create Experiment | 動的設定フォーム、seed、予算見積り、実行モード、プリセット | P0 |
| Live Laboratory | ライブ描画、Play/Pause/Step、時刻・速度、計測値、イベント | P0 |
| Experiment Library | 検索、タグ、フィルタ、結果サムネイル、複製・削除・export | P0 |
| Timeline / Replay | スナップショットシーク、発見イベント、チャプター、速度変更 | P1 |
| Inspector | 個体・セル・粒子・プログラム・領域の詳細 | P1 |
| Evolution Explorer | 集団分布、系統樹、エリートアーカイブ、突然変異履歴 | P1 |
| Experiment Compare | seed間、介入対照、異なる条件の同期比較 | P1 |
| ASAL Discovery Gallery | 発見したパターンのカード、埋め込み距離、動画一覧 | P2 |
| BFF Program Microscope | 64/128バイトビュー、2ヘッド、PC、コピー、実行トレース | P2 |
| Experiment Notebook | 仮説・条件・観測・コメント、図表つきMarkdown export | P2 |
| Admin / Diagnostics | GPU診断、依存版、エラー、Disk/VRAM、ワーカー管理 | P0 |

### 3.2 Live Laboratoryの具体的なレイアウト

```text
+--------------------------------------------------------------------+
| ALife Lab | World / Run Name | RUNNING | GPU telemetry | 01:42:18    |
+---------------+--------------------------------------+-------------+
| World Catalog |              World View              | Inspector   |
| Presets       |        PixiJS / Three.js             | Selected ID |
| Parameters    |                                      | Genome      |
| Interventions |                                      | Energy      |
|               |                                      | Lineage     |
+---------------+--------------------------------------+-------------+
| [Play][Pause][Step][Reset][Branch][Snapshot] Tick 15,245   12x    |
+--------------------------------------------------------------------+
| Population / Complexity / Energy / Novelty    | Event timeline    |
| [interactive plots with synchronized cursor] | Emergence alerts |
+--------------------------------------------------------------------+
```

### 3.3 世界のリアルタイム表示

- 共通操作：再生・一時停止・1tick/1epochステップ・速度変更・初期化・画面キャプチャ・MP4出力。
- **2D Grid**：状態チャネル、正規化スケール、RGB合成、擬似カラー、密度場、ベクトル場、境界条件表示。
- **Particle Field**：粒子群、種別/速度/エネルギー別着色、ベクトル矢印、接触・近傍リンク、局所密度。
- **3D**：MuJoCo生物の骨格・幾何、関節・接触点、速度ベクトル、カメラプリセット、軌跡。
- **BFF**：個体（テープ）頻度の時系列・クラスタ、Hex/ASCII/命令表示、ペア間の上書き履歴。
- **描画と計算を分離**。シミュレーションが毎秒1万step進んでも描画を毎step送らない。
- UIのFPSと、シミュレーションのstep/secを**別々に表示**する。

### 3.4 Interaction / Intervention（観察者による介入）

- マウスで個体/セル/領域を選択し、属性・最近の履歴・系統・表現型を表示する。
- グリッドに質量、栄養、毒、障害物、熱源、流れ場などを注入・除去する（基質のcapabilityで制約）。
- Brushツール：点、円、矩形、筆、スタンプ。NCAには損傷・切除・再生テスト。
- 生態系に突然変異率変更、資源パルス、季節変化、捕食者導入、外来種導入などを実施する。
- **Pause -> Checkpoint -> Fork -> Intervention -> Compare** を1操作で行えるようにする。
- 介入は `{tick, kind, payload, author, parent_checkpoint, config_hash}` としてイベントログに永続化する。
- **Run中に変更できるパラメータ**と**再コンパイル/再起動が必要なパラメータ**をSchemaで区別。後者は新規branchを作る。
- BFFのreference modeではUIからの操作が再現実験に混ざらないようにする。介入があった場合は明示的に exploratory として保存。

### 3.5 タイムトラベルと分岐実験

- 保存済みスナップショットから任意時刻へのシーク・再生を可能にする。
- `Fork from here`: 親runを保存したまま別設定で分岐し、親子関係を記録。
- 条件A/Bの動画を同一時刻・同一シミュレーション時刻で同期表示する（異なるtick単位では意味のあるアライン方法を選択）。
- **任意時点の完全復元はチェックポイントが存在する場合のみ保証する**。フレーム動画しかない区間では「visual replay only」と明示。
- 介入replayの完全一致にはRNG状態、浮動小数点非決定性、バックエンド版を含めて保証範囲を表示する。

### 3.6 イベント検出と通知（ローカル）

- イベント例：`possible_replication`, `mass_drift`, `population_collapse`, `novel_lineage`, `new_species_candidate`, `repair_success`, `novelty_spike`, `phase_transition_candidate`。
- 指標は **candidate / validated / rejected** の状態を持てる。
- 急激な変化に対するchange-point検出（例：CUSUM、rolling baseline）を搭載。ただし発見の統計的有意性とは区別。
- イベント時に低頻度の追加チェックポイント、前後時点のクリップ、関連メトリクスを保存。
- 将来の通知連携（メール/Slack等）はプラグインに分け、初期実装には組み込まない。

### 3.7 Web UIのアクセシビリティと性能

- 2D描画は必要なチャンネルのみ送信。凡例と単位、ログ/線形軸を指定できる。
- Reactの再レンダリングを描画ループから分離し、Canvasは独立した描画コンポーネントで更新。
- モバイル/タブレットでは閲覧優先、ブラシ編集はデスクトップ優先。
- グラフはダウンサンプリングして描画し、元系列は保持する。
- 切断・再接続で現在状態と最後に受信したseqを復元し、二重操作を避ける。

---

### 3.8 Scenario Studio — 汎用実験デザイナ（P1〜P2）

生態系専用ではなく、すべての対応基質へ共通に適用できる**世界設定・イベント列・対照群の視覚的編集画面**。

- **World tab:** ルール・境界条件・初期状態・レンダラ層・数値精度。
- **Agent tab:** 初期集団、個体/種の性質、遺伝/学習/制御ポリシー（該当する基質のみ）。
- **Timeline tab:** `at_clock`、`every_n_steps`、`on_event`で刺激・環境変化・観測を構成。
- **Experiment tab:** fork条件、seed集合、parameter sweep、実行予算、対照設定。
- **Analysis tab:** 検出するイベント、観測メトリクス、独立アッセイ、統計比較。
- テキストYAML↔フォームは**losslessな往復**を理想とする。未知項目・コメント保持の扱いを明記し、勝手に消さない。
- 調整対象はSchemaからUIを生成。無効な組合せ、seed未固定、単位の不整合は保存前に警告。
- パラメータ探索範囲を可視化する2D heatmap、散布図、parallel coordinatesを提供。
- クリップ/再生は研究用状態のコピーではないことを常時明示する。

### 3.9 World Comparison / Multiverse View（P1〜P2）

- `side-by-side`（2〜4 runの同期再生）、差分ビュー、並べた指標、発見イベントの重ね合わせ。
- `paired fork`では親checkpointと差分configを1画面に表示。
- 時間単位の異なる系の比較は`normalized progress`等の専用表示へ切替え、tickとgenerationを同じx軸に重ねない。
- 系統表示はツリー/DAG/時間軸切替、系統数に応じてLOD/サンプリングを行う。
- メトリクス・イベントをクリックするとその時点のsnapshot / clipに移動。
- 散布図の点から元runとstateを開き、**代表個体だけでなく失敗ケース**も参照できる。

### 3.10 Discovery Workbench（P2）

- 探索で自動検出された候補のキューを、`new / triaged / assayed / validated / rejected / inconclusive`で管理。
- 画像/動画・時系列・基質パラメータを横断検索。embedding類似検索は補助であり独立した証拠ではない。
- 検出器ごとに誤検出例・閾値・モデル版を閲覧。人手でラベルした結果を評価に使えるようにする。
- `Generate assay`は候補種別に合う対照実験のYAMLテンプレートを生成するが、結果を予断しない。
- 興味深い候補をピン留めし、run, event, checkpoint, metrics, validationを揃えた**Discovery Record**をexport。
- 気に入った動画だけを選ぶギャラリーと、科学的な発見候補の台帳を分ける。

### 3.11 UIパフォーマンスと操作性の受け入れ要件

- ライブ観測は計算の進行をブロックしない。30Hz表示を要求してGPU計算スループットを崩さない。
- 連続グリッドの多チャネル表示ではGPU→CPU readbackをサンプリングし、画像圧縮をworker/CPU側で計測。
- Browser切断時もrunは継続し、再接続後は`run_state`と最新サマリを同期。
- カメラ・レンダラ・色・表示周波数の変更は**科学的な状態変更として記録しない**。状態への介入は必ず別コマンド。
- 誤操作の防止：`Pause` / `Fork` / `Intervene`の状態遷移と適用時刻をユーザーが判別できる。

---

### 3.12 Compute Targets / Run Placement / Artifact UI

- **Compute Targets:** 登録先の状態 (`ready`, `busy`, `unreachable`, `quarantined`)、GPU型番/数/VRAM、backend、driver、scratch空き、最終probe日時、セキュリティ/アクセス可否を表示。
- **Run Placement:** 実験起動時に`local_interactive`, `remote_compute`, `cluster_slurm`等を選択。推定VRAM/計算時間/容量/費用を表示し、未知項目は`unknown`。
- **Queue Inspector:** 予約→投入→スケジューラ待機→実行→upload→整合性検証→completed の状態と所要時間を別表示。
- **Reproducibility Comparison:** 同じ科学設定を異なるbackendで再実行した際の比較メトリクス、数値差/統計差、kernel/precisionの情報を並べる。
- **Artifact Explorer:** Gitで管理される定義、外部ストレージのデータ、モデル、派生動画を区別し、`download`, `verify`, `restore`, `pin`, `export`を提供。
- **Security:** ユーザーにremote権限の無いtargetは非表示/使用不可。生のtokenやprivate storage URLは画面に出さず、署名済み短命URL等を利用。

---

## 4. API・ストリーミング・操作プロトコル

### 4.1 REST API（基本）

```text
GET    /api/v1/health
GET    /api/v1/system/capabilities
GET    /api/v1/system/resources
GET    /api/v1/substrates
GET    /api/v1/substrates/{slug}/schema
GET    /api/v1/presets
POST   /api/v1/experiments/validate
POST   /api/v1/experiments
GET    /api/v1/experiments?status=&substrate=&tag=
GET    /api/v1/experiments/{id}
POST   /api/v1/experiments/{id}/start
POST   /api/v1/experiments/{id}/pause
POST   /api/v1/experiments/{id}/resume
POST   /api/v1/experiments/{id}/step
POST   /api/v1/experiments/{id}/cancel
POST   /api/v1/experiments/{id}/interventions
POST   /api/v1/experiments/{id}/fork
GET    /api/v1/experiments/{id}/metrics
GET    /api/v1/experiments/{id}/events
GET    /api/v1/experiments/{id}/snapshots
GET    /api/v1/experiments/{id}/artifacts
GET    /api/v1/experiments/{id}/lineages
POST   /api/v1/experiments/{id}/export
GET    /api/v1/jobs
GET    /api/v1/jobs/{id}
GET    /api/v1/artifacts/{artifact_id}
WS     /api/v1/stream/experiments/{id}
WS     /api/v1/stream/system
```

- 書き込みAPIは明示的な認可と、重複リクエスト対策（`Idempotency-Key`、`command_id`）を実装する。
- 設定はPydantic discriminated union / JSON Schemaで型検証し、危険な値は起動前に拒否。
- 大容量データはJSONの巨大な数値配列にしない。バイナリframe/ファイルAPIを用いる。
- 全APIに`api_version`とサーバー実装versionを含める。

### 4.2 WebSocketイベントの例

```json
{
  "type": "metrics",
  "run_id": "run_00001234",
  "seq": 381,
  "timestamp": "<ISO-8601 timestamp>",
  "simulation_clock": { "unit": "tick", "value": 15245 },
  "payload": {
    "population": 324,
    "mass": 193.18,
    "novelty_score": 0.17,
    "steps_per_second": 1240.0
  }
}
```

- 通知種別：`run_state`, `metrics`, `event`, `preview_descriptor`, `checkpoint_ready`, `error`, `command_ack`, `heartbeat`。
- 大容量プレビューは別バイナリメッセージにheader（run/seq/shape/codec/clock）を付ける、または一時的な署名付きアーティファクトURLを用いる。
- サーバーはクライアントの遅延を見て**古いpreviewを捨てる**。メトリクス・重要イベントは別の永続ストアから再取得できる。
- `preview_fps=2〜10` を初期値候補とし、帯域・CPU/GPU readback・エンコード負荷に応じて自動調整。
- 画像はまずWebP/JPEG/PNGの選択可能な圧縮プレビュー。研究状態データそのものと混同しない。
- 高フレームレート配信は将来のオプション（WebRTC/H.264等）とし、初期必須要件ではない。

### 4.3 Browserからの介入コマンド

```json
{
  "command_id": "f7d7f4e5-9f7c-4b36-8857-13be5510e949",
  "kind": "inject_resource",
  "requested_at_clock": 15000,
  "payload": { "x": 0.50, "y": 0.35, "radius": 0.06, "amount": 1.2 }
}
```

- APIは受付時刻ではなく、**workerが適用したsimulation_clock**を返す。
- 手動操作は「状態を書き換える」だけでなく研究イベントとしてログに残す。
- 誤操作をUndoで隠さず、取消しも新たな介入として記録。状態を完全に戻したい場合はforkを作る。

---

### 4.4 研究ワークフローAPI

```text
GET    /api/v1/scenarios
POST   /api/v1/scenarios/validate
POST   /api/v1/scenarios
GET    /api/v1/experiments/{id}/observations
GET    /api/v1/experiments/{id}/resource-usage
GET    /api/v1/experiments/{id}/provenance
GET    /api/v1/experiments/{id}/identities
GET    /api/v1/experiments/{id}/lineage-graph
GET    /api/v1/experiments/{id}/state/{checkpoint_id}/inspect
POST   /api/v1/experiments/{id}/assays
GET    /api/v1/assays/{assay_id}
POST   /api/v1/sweeps
GET    /api/v1/sweeps/{sweep_id}
POST   /api/v1/comparisons
GET    /api/v1/comparisons/{comparison_id}
GET    /api/v1/discoveries
GET    /api/v1/discoveries/{discovery_id}
POST   /api/v1/discoveries/{discovery_id}/review
GET    /api/v1/artifacts/{artifact_id}/metadata
```

- UI/CLIが同じRequest/Responseスキーマを参照する。スキーマの破壊的変更は `/api/v2` 等で版を分ける。
- 状態変更は `expected_run_revision` と `command_id` を使用し、競合するクライアントによるlost updateを回避。
- ブラウザWS切断後は `seq` に基づきイベント差分を取得できる。**プレビューをイベントログで完全復元しない**。
- どのAPIもユーザー設定の任意ファイルパスを直接openしない。artifact ID→検証済みroot→実パスの方式を採用。
- 計測系列はカーソル/期間指定/集計粒度をサポートし、巨大Parquetを無制限に一括返却しない。
- `POST /experiments` と `POST /start` をまとめる場合は `auto_start` を明示し、再送に対する重複ジョブを防ぐ。

### 4.5 イベントの整合性・時刻単位

`EventEnvelope`は `run_id`, `event_id`, `seq`, `clock_unit`, `clock_value`, `observed_at_utc`, `source`, `schema_version`, `payload` を必須とする。`detected_at`と`occurred_at`は、サンプリング間隔のため異なり得る。イベント検出器の遅延/時間解像度を保持する。

---

### 4.6 Compute / Artifact API

| method | endpoint | 意味 |
|---|---|---|
| `GET` | `/api/v1/compute-targets` | 権限のある計算先と能力概要 |
| `POST` | `/api/v1/compute-targets/{id}/probe` | 再計測（負荷/認可付き） |
| `POST` | `/api/v1/runs/{id}/placement/validate` | プランと計算先の適合性、費用/容量見積もり |
| `POST` | `/api/v1/runs/{id}/dispatch` | 冪等key付きの投入 |
| `GET` | `/api/v1/runs/{id}/execution` | scheduler/job ID、状態、使用量 |
| `GET` | `/api/v1/runs/{id}/artifacts` | 状態/メトリクス/動画/重要checkpoint一覧 |
| `POST` | `/api/v1/runs/{id}/artifacts/verify` | checksumと参照の照合 |
| `POST` | `/api/v1/runs/{id}/reproduce` | 元runの科学設定から別backendで新run |

remote worker APIは別のcredential・URI namespaceを使用し、UI用tokenで登録・heartbeat・uploadできないようにする。スキーマはOpenAPIでバージョン管理し、Controller/Worker protocolが互換か確認する。

---

## 5. Core Pythonインターフェース

### 5.1 モジュール契約（概念例、API詳細は実装時に調整）

```python
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class Capabilities:
    state_kind: Literal["grid", "particles", "rigid_body", "program_soup"]
    supports_pause: bool
    supports_interventions: bool
    supports_inspection: bool
    supports_lineage: bool
    supports_differentiation: bool
    clock_unit: Literal["tick", "epoch", "physics_step"]


class SubstrateAdapter(Protocol):
    @property
    def capabilities(self) -> Capabilities: ...
    def initialize(self, config: Any, rng: Any) -> Any: ...
    def step(self, state: Any, config: Any, rng: Any) -> Any: ...
    def metrics(self, state: Any, context: Any) -> dict[str, float]: ...
    def render_payload(self, state: Any, request: Any) -> Any: ...
    def serialize_state(self, state: Any) -> Any: ...
    def deserialize_state(self, payload: Any, metadata: Any) -> Any: ...
```

実装上の注意：

- 高速JAX経路ではPythonの `Protocol` 自体をJIT内部で毎step呼び出さない。基質ごとに `jax.jit` / `jax.lax.scan` 可能な純粋な計算関数に落とす。
- `rng` の意味が不要なモデルでは省略可能なオーバーロードを設け、JAXに動的Pythonオブジェクトを渡さない。
- `metrics` / `render_payload` は**毎stepではなくサンプリング間隔**で実施できる契約とする。
- 差分checkpointと完全checkpoint、バックエンドの保存形式の違いはadapterで吸収する。
- `supports_inspection=False` の基質にUIで個体の祖先情報を勝手に生成しない。

### 5.2 共通型

- `ExperimentConfig`: substrate, backend, seed, steps, preview, checkpoint, evaluator, search, resources, interventions, tags。
- `RunManifest`: run ID, parent run ID, config hash, Git commit, lock hash, GPU/driver info, package versions, model checkpoints, fidelity mode。
- `ExperimentState`: run state machine, clock, last checkpoint, last event, error summary。
- `Observation`: metrics, events, sampled state descriptors, optional preview。
- `Genome`: 基質ごとの型付きペイロード。強制的に1本のfloatベクトルへ変換しない。
- `Intervention`: validation, applicability, requested/applied clock, reversible flag。

---

### 5.3 プラグインManifestとCapabilities

すべての基質・探索器・評価器・描画器は登録時に次のメタデータを提供する。プラグインの存在だけで「使用可能」にしない。

```yaml
plugin_id: "alife_lab.substrates.example"
plugin_version: "0.1.0"
api_contract_version: 1
kind: "substrate"
scientific_mode: "exploratory"
clock: {unit: "tick", monotonic: true}
capabilities:
  state_kinds: ["grid"]
  intervention_types: ["damage", "inject_field"]
  supports_checkpoint: true
  supports_deterministic_cpu: true
  supports_lineage: false
  supports_gradient: false
requirements:
  backend: ["numpy", "jax"]
  gpu_required: false
reference:
  paper_urls: []
  implementation_commit: null
```

**Plugin contractの追加項目:** `parameter_schema`, `metric_schema`, `render_layers`, `inspection_schema`, `serialize_version`, `migrations`, `reference_tests`, `resource_estimator`, `health_check`, `licence`。UIはcapabilityを読み、対象基質で不可能なコマンドを非表示または無効化する。

### 5.4 連続時間と多時間尺度

- tick固定幅以外のODE/PDE系では `simulation_time`（物理時間、float）と `integration_step`（整数）を併記。
- 生涯学習のgradient step、個体のbirth/death、世代のturnover、外部探索のevaluationを別時計で管理。
- 適応的時間刻みの場合は `dt_history`・受理/棄却ステップ・数値解法を保存。
- 定期介入は「物理時間指定」か「step指定」かをSchemaで区別。浮動小数点時刻の厳密一致をトリガー条件にしない。

---

## 6. 物理・計算基質（Substrates）

### 6.1 P0: 古典的CAとBoids — UIの基盤テスト

- Conway Life、HighLife等のOuter-totalistic CAをNumPy/JAXで実装。ルールは `B3/S23` のような可読記法。
- 連続/周期/反射等の境界条件は基質が対応する範囲で選択。
- 標準パターン（glider、blinker等）の既知周期をテスト。
- Boidsは分離・整列・結合の3規則と障害物・捕食者を追加可能。**行動のスクリプトと創発を混同しない**。
- Particle Lifeは種間引力・斥力行列による粒子系。近傍探索はcell list / spatial hashingを導入し、全ペアO(N²)を避ける。
- UI基準：ブラウザからプリセット起動、描画、停止、ブラシ介入、run保存、branch比較まで縦貫で完成。

### 6.2 P1: Lenia / Flow-Lenia

- **Lenia reference**：連続状態、リングカーネル、成長関数、FFT周期畳み込み。
- **Flow-Lenia reference**：質量保存型の移流（reintegration tracking）、局所パラメータ埋め込み・輸送。公式JAX実装を検証基準とする。
- `FlowLeniaParams` を外して通常Flow-Leniaを検証後、param advectionを有効にする。
- 境界条件、数値丸め、移流実装、負値のクリップ処理をバージョン別に記録。
- 初期単一個体のロールアウトに加え、複数形態の同居、質量の移動、接触、分裂候補を観察可能にする。
- **測定:** 総質量変化、局所構造の持続、移動量、形状変動、分裂/融合候補、種・表現型多様性。
- **テスト:** 質量の非負性（モデル仕様に依存）、保存則、周期境界、CPU比較、既知パラメータの参考画像との一致。
- 誤差はfp32/fp64・解像度・step数・ライブラリに応じてプロファイリングした許容値を採用。`1e-5`を全条件で機械的に強制しない。

### 6.3 P1: BFF Primordial Soup / Computational Life

**必須:** 論文のreference implementation `paradigms-of-intelligence/cubff` を照合し、独自の`bff_vm`を作る前にC/CUDAでの既存実装の再利用・ラップとライセンスを評価する。

- 原型：`(N_tapes, 64)`, `uint8`。ランダムにペア化して128バイト結合、実行後分割・書戻し。
- 10命令 `< > { } + - . , [ ]`、ASCIIコードを使用。その他は実行時NOP（ただし値0は条件判定上特別）。
- `noheads` variantと`heads` variantを**別名の明示的な設定**にする。
- 結合テープ上でのポインタ移動・ヘッド初期化・命令PC・括弧の探索は**reference一致を要求**。
- 対応しない括弧を「無視する」ことは禁止。`cubff`準拠では、必要な方向に対応括弧が見つからない場合は実行停止。括弧ジャンプはテープ端でwrapしない。
- 背景突然変異は**有効/無効を選択可能**。無突然変異baselineを必ず用意し、突然変異で自然発生したかのように誤報しない。
- `mutation.enabled=true` の場合、referenceに合わせたバイト単位の一様置換を検証する。
- 標準規模候補：`N = 2^17 = 131,072`、最大8192命令/ペア。**100 epochs/secを受入基準に設定しない**。この規模の命令数は大きく、実機ベンチマーク後に妥当な性能目標を設定する。
- 実装は正しさ優先でスカラーoracle -> CPUベクトル版 -> CUDA版。ランダムペア・初期状態を固定した**差分テスト**を実施。
- 保存：dominant tapes、命令頻度、unique tape数、シャノンエントロピー、圧縮率、コピー頻度、相転移候補。
- ヒューリスティックな自己複製検出と、独立の`replication_assay`（異なるpartnerでの複製頑健性）は分離。
- ブラウザではVMを1命令ずつ実行できるデバッガを追加。スープ全体の実行器をデバッガの低速ループで置き換えない。

### 6.4 P1: Neural Cellular Automata（NCA）

- Growing NCA: 格子の各セルに複数状態チャネル、局所perception（Sobel等）、共有MLP更新則、非同期更新。
- Grow/Repair/Persistの3モードで、標的形状を成長させる基礎タスクを実装。
- ブラウザCanvasから標的画像を描画/アップロードして学習を開始できる。
- 生育過程、損傷後再生、複数seed、学習Loss、細胞状態チャネルの可視化。
- `model_weights`、`cell_states`、`training_optimizer_state`、`rng`を別々に保存。
- 従来NCAでの形態再現と、「無制約に生命が創発した」ことは同一視しない。

### 6.5 P2: PD-NCA / Digital Ecosystems

- 既存のPD-NCA研究と2026年のDigital Ecosystemsを**参照実装の対象**とする。
- 多種類のNCAが1つのグリッドで領域競争。エージェントごとに重み更新が継続する設定を支持。
- 生存・学習・進化の時間尺度を区別し、各エージェントの重み・領域・相互作用を保存。
- UIで各NCAの占有率、学習量、競争ネットワーク、共存状態と崩壊を可視化。
- 同一条件で `frozen weights`、`online learning`、`no competition` の対照を用意する。
- 基準のlossと本システム独自のnovelty探索は別run設定とする。

### 6.6 P2: 反応拡散・資源循環型生態系

- Gray–Scott等の反応拡散を基準系として実装。時間刻み、数値安定条件、境界条件を検証。
- 設計可能なリソース場：栄養、エネルギー、廃棄物、毒、温度、化学勾配、季節変動。
- 個体が資源を取り込み、代謝・維持費を消費し、増殖・死亡・分解で資源を戻す閉鎖系/開放系モード。
- 捕食・競争・共生・寄生は行動の帰結として可能になるよう相互作用則で表現し、最初から特定の個体の行動を固定しない。
- 資源・エネルギーの収支を計測し、**内部生成してよい量と外部から供給する量を別に定義**。
- 追加機能：空間分割、ニッチ占有、侵略種、環境変動、個体の寿命と繁殖。

### 6.7 P2: MuJoCo MJXによる形態と制御の共進化

- **MJX-JAX** と **MJX-Warp/MJWarp** を別バックエンドにする。MJX-JAXはJAXでの微分可能性を使用可能。一方、**MJX-Warpは自動微分をサポートしない**（確認時点）。
- 進化戦略/ブラックボックス最適化ではWarpの高スループットが利点になり得るため、ベンチマークして選択。
- Ant/Swimmer等の固定トポロジーから開始し、後に肢長・節数等の形態進化へ拡張。
- GPUバッチ処理中の動的XML/形状トポロジー変更による再コンパイルを避ける。**同一構造のバッチ・static形状グループ**を基本とする。
- 制御遺伝子（MLP/RNNなど）、身体遺伝子、環境パラメータの区別。
- 目標：歩行速度、移動効率、エネルギー費、方向転換、地形適応、形態多様性。
- `episode_steps=500`、`batch_size=512`等は**検証対象の候補値**であり実機保証ではない。
- 3D viewerには骨格、接触位置、センサ、アクション値とfitness・系統情報。

### 6.8 P3: Microcosmos / フィラメント生命

- ALIFE 2026のMicrocosmos（JAX、流体＋弾性フィラメント、生物の動作や最適化）を新たな研究候補として登録。
- まず外部コード・ライセンス・API・性能を評価し、独自の再実装ではなく**外部アダプタ**方式を優先。
- 2D流体と柔軟な体の連成、流れ場、遊泳姿勢、エネルギー勾配への探索、QDを観察。
- 形態最適化・勾配ベース学習・進化戦略を比較できるようにする。

---

### 6.9 P2: Artificial Chemistry / 反応ネットワーク生命

**目的:** バイトコード生命と連続セル・オートマトンの間を埋める、物質/情報/反応のレベルでの創発を研究する独立した基質。

- `abstract_reaction_network`: 化学種をID、反応を反応物・生成物・速度/確率で定義。反応の化学的妥当性を自動保証しない**抽象人工化学**と明示。
- `string_chemistry`: Stringmol型の文字列分子、結合・マッチング・切断/コピー作用等を**原論文・ライセンス照合後に別reference adapter**で導入。
- `graph_chemistry`（後期）: グラフ書換え規則、触媒/抑制/消費を扱い、部分グラフマッチングの計算量制御を設ける。
- 反応は「規則に明記したもの」と「規則の相互作用から新たに成立した経路」を区別。物理化学的実在性は主張しない。
- 開放系の流入/流出、閉鎖系の分子数/保存量、空間区画の有無をconfigで固定。
- **観測:** 化学種の生成・消失、反応ネットワークの連結性、触媒サイクル候補、再帰的な合成、情報の保存/増幅。
- **検証:** 原料だけの対照、反応規則シャッフル、触媒無効化、質量/原子数保存が定義されたモデルでの収支、反応順序依存性。
- **実装:** 最初はCPU上の小規模離散イベント法。GPU化は十分なバッチ並列性とプロファイル結果が得られた後。Gillespie型シミュレーションは標準的な確率過程として実装できるが、元モデル固有の速度法則と混同しない。
- **DoD:** 同一seedで決定論的CPU再生が可能。反応履歴をgraphとして閲覧でき、同じ観測器で未知の反応ループ**候補**を抽出できる。

### 6.10 P2〜P3: Protocell / 区画化・代謝・分裂

- 空間セル/粒子が**境界（膜・区画）**、内外の物質、透過性、維持コストを持つ抽象プロトセルを導入。
- まずは2Dグリッドまたは非重複円/ポリゴンの軽量モデル。流体・膜の分子動力学レベルを初期スコープに含めない。
- 内部反応が体積/膜面積変化へつながるフィードバック、資源消費と成長・破裂・分裂の条件を明示。
- 「分裂らしく見える」描画と、両娘個体へ内部構成/遺伝情報が継承されることを別々に計測。
- 発達課題：膜の維持、外乱下の内外濃度勾配、区画寿命、自己修復候補。
- **対照:** 無膜、非代謝、ランダム分配、透過性固定、体積増加だけのモデル。
- **DoD:** 物質の出入りとエネルギー支出の元帳が一致し、分裂イベントの親子識別が保存され、介入によって寿命・継承率を比較できる。

### 6.11 P1〜P2: Stigmergy / Swarm / Physarum-inspired Systems

- Ant-like trail / chemical field: 個体が局所場へ痕跡を残し、他個体がその勾配を感知して移動。痕跡は拡散・減衰する。
- Physarum-inspired network formation: 栄養点を結ぶネットワーク状の動的構造を作る**現象モデル**を追加。実際の粘菌の生物学的忠実再現とは別ラベル。
- Boidsとの違いを明示：直接相互作用、間接環境媒介、通信チャネルをアブレーションできる。
- **観測:** 集団の秩序、経路効率、障害物迂回、ネットワーク安定性、trail依存度、場/個体間情報流の候補。
- **実験:** trailを消す、擾乱を入れる、センサーを無効化、ランダム化した場を注入、障害物を変更する。
- **計算:** 2DフィールドFFT/局所畳み込み、spatial hashing、粒子位置更新。VRAMを増やすだけでなくO(N²)を避ける。
- **DoD:** いずれかの協調的行動を再現し、環境場の除去によって行動が変わることを同初期状態の対照から示せる。

### 6.12 P2: Lightweight Learning Agents / 行動と意思決定の進化

- 低次元センサー（視界、近傍、方向、エネルギー、化学勾配）を持つ2Dエージェントを `AgentWorld` として定義。
- Controller backend: hand-coded baseline、MLP、GRU、小型NCA policy、有限状態機械。高価なLLMを個体ごとに呼ぶことを既定としない。
- 固定コントローラ、寿命内強化学習、集団進化、学習則自体の進化を**別モード**で比較。
- Action: move/turn/consume/emit signal/attack/defend/reproduce等。ただし世界で定義される行為のみ提供。
- 資源競争は実験シナリオの一つ。別シナリオとして迷路、探索、協調搬送、通信、地形適応も登録。
- 適応度を最適化する場合は明示的な外部目標を持つ`optimization_run`、外部rewardなしのダイナミクスは`autonomous_run`として混同しない。
- **観測:** 行動レパートリー、環境横断の汎化、社会的ネットワーク、通信の必要性、寿命内学習と遺伝変化の寄与。
- **対照:** rewardなし、通信遮断、policy shuffle、記憶オフ、固定世界、環境外挿。
- **DoD:** ライブ個体インスペクタでセンサー・内部状態・アクションを時系列に表示し、同集団の対照比較ができる。

### 6.13 P2: Evolvable World Rules / 法則そのものの探索

- 基質パラメータだけでなく、CAのルールテーブルやローカル相互作用関数を探索対象として登録する。
- `RuleGenome`と`OrganismGenome`を**区別**し、世界法則の進化と生物個体の進化を混同しない。
- 初期はOuter-totalistic CAのルール探索、Particle Lifeの相互作用行列、反応拡散パラメータを扱う。
- 指標：持続するパターンの出現、時空間複雑性、構造の移動、スケール依存性、計算コスト。
- **重要:** あるルールの探索中にパターンが出現しても、そのルール内で生物のダーウィン進化が起きたとは言えない。
- `rule_search`はASALやMAP-Elitesの探索器を再利用し、同じ予算のRandom/Sobolを比較対照にする。
- 生成ルールは**宣言的なDSL/安全な構文木**で表現し、外部からPythonコードを自動実行しない。
- **DoD:** ルール探索から発見した個体をgalleryに保存し、ルール・初期条件・採点器版・シードが再生可能。

### 6.14 P3: Hybrid / Multi-scale Worlds

- 異なる基質を1つに統合する機能は、**結合の科学的意味を明示できる場合だけ**提供する。
- 例：particle agents ↔ 拡散場、NCA ↔ 可観測な資源場、化学ネットワーク ↔ 区画、剛体 ↔ 場への局所deposit。
- `CouplingPort`: 単位、座標系、更新頻度、遅延、保存量、補間法を持つ型付き入出力。
- `CouplingScheduler`: operator splitting / multi-rate steppingを明示し、数値安定性と循環依存を検証。
- `WorldComposition`の実験manifestに構成基質のcommit/versions、接続グラフ、変換関数hashを保存。
- CPU/GPUの頻繁な往復が計算量を圧倒する場合は構成を拒否・警告する。
- **検証:** 結合OFF、片方向結合、時間刻み縮小、総量の収支、GPU/CPUの差分。
- **DoD:** まず粒子群+拡散場の最小例で、2方向結合と観測・保存・forkが破綻しないことを示す。

### 6.15 P3: Digital Evolution / Tierra・Avida・Stringmol互換アダプタ

- 既存の研究ソフトは全面再実装より、再現用の**外部アダプタ**を優先する。
- 実行モデルの違い（個体単位、共有メモリ、時間尺度、評価軸）をreportに明示する。
- Avida/他VM系は複製、計算タスク、突然変異、競争、祖先解析などモデル固有の仕組みを保持。
- 入出力：設定生成、ジョブ実行、系統/メトリクス取り込み、元のreference成果物へのリンク、可視化snapshot。
- 同じ「Fitness」「Species」の語が別基質で意味を持つ場合は、**名前空間付き指標**として管理する。
- **DoD:** 1件のreference runをアダプタで呼び出し、正本ログから系統図と時系列を構築できる。

### 6.16 P2〜P3: Evolutionary Development / 形態と制御の階層化

- NCAの局所成長則、軽量神経エージェントの制御器、MJXの身体パラメータを、別々の**遺伝子→発達→表現型**変換として扱う。
- `DevelopmentProcess` は必要なら世代外の段階（embryogenesis / maturation）を記録する。
- 静的遺伝子→表現型と、環境依存の発達可塑性を比較。
- 発達の頑健性、異形態への転換、ノイズ耐性、学習しやすさを探索。
- **対照:** 発達固定、形態固定、制御器固定、ノイズなし、損傷なし。
- **DoD:** 同じゲノムを異環境で発達させた表現型を比較し、遺伝/発達/制御の寄与を個別に分析可能。

### 6.17 基質導入用の共通評価表

新基質は、次を記した `docs/substrates/<id>.md` を伴う。

| 項目 | 記載内容 |
|---|---|
| Scientific question | 何を観測/反証したいか |
| Reference fidelity | 原論文・公式repo・commit、実装差分 |
| Mathematical rules | 更新式、境界、時間、保存則、乱数 |
| State and identity | データ形状、個体同定・親子関係の可否 |
| Compute profile | CPU/GPU、batching、static shape、想定ピーク |
| Interventions | 状態変更できる項目と安全時点 |
| Metrics | 数式・単位・サンプリング・検証法 |
| Controls | 最低2つの独立対照 |
| Acceptance tests | 再現、正しさ、性能、UI、checkpoint |
| Failure modes | 数値発散、崩壊、reward hacking、誤検出 |

---

## 7. Evolution & Search — 遺伝・進化・発見

### 7.1 共通の探索インターフェース

- `SearchAlgorithm`: `initialize`, `propose`, `observe`, `update`, `checkpoint`, `restore`。
- `PhenotypeDescriptor`: 基質別の行動特徴。例：移動距離、空間占有率、形態円形度、周波数、複製成功率。
- `Archive`: 個体、遺伝子、表現型、評価条件、世代、系統情報を保存。
- すべての探索で**乱数seed、評価予算、再評価回数、環境条件**を記録。

### 7.2 探索アルゴリズム

| アルゴリズム | 主な用途 | 実装 |
|---|---|---|
| Random Search / Grid / Sobol | 初期baseline、アブレーション | P0 |
| CMA-ES / Sep-CMA-ES / OpenES / PGPE | 連続パラメータ探索、行動最適化 | P1 (`evosax`) |
| MAP-Elites / CVT-MAP-Elites | 行動特徴空間の多様な解の探索 | P1 (`QDax`) |
| Novelty Search / kNN archive | 特徴の新規性探索 | P1 |
| Pareto / NSGA-II | 生命特性のトレードオフ探索 | P2 |
| NEAT / CPPN | 神経構造・身体パターン進化 | P2〜P3 |
| Population Based Training | NCA等の学習中パラメータ探索 | P3 |
| Competitive / cooperative coevolution | 捕食・共生・適応レース | P2 |
| ASAL supervised / open-ended / illumination | 基盤モデル誘導探索 | P2 |

- **evosaxは2026-10公開のv0.3.2を含む導入時点の現行版を確認してから組み込む。** v0.1/v0.2の`Strategy`契約を決め打ちしない。
- QDの品質・多様性・coverage・QD-scoreを別指標として保存する。
- QDのdescriptor境界は事前定義・固定し、途中変更する場合はarchive versionを上げる。

### 7.3 オープンエンド性を単一スコアとみなさない

- 累積新規性（cumulative novelty）、時間的な行動変化、複雑性、探索空間の広がり、ニッチ形成、持続性をそれぞれ記録。
- MODES toolboxの着想（change / novelty / complexity / ecological potential）を参考に、計測可能な項目を基質ごとに段階導入する。
- 埋め込みが動くと新規性スコアも変わるため、**encoder version・正規化方式・archive version**を保持。
- CLIPスコアは「視覚モデルが興味深いと判定する」指標であり、「生物学的意味で生命である」証拠ではない。
- オープンエンド性を主張する場合、少なくとも**時間に伴う継続的な新規性**と、静的/ランダム/既知baselineとの比較を用意する。

### 7.4 系統と進化の観察

- Genotype/Phenotype/Identityを分離。
- 個体が分裂し、両親がある場合は祖先を木でなく**DAG**として記録できるようにする。
- BFFのように個体の「親」を一意に定義できない系は、過剰な親子推定を避け**推定可能性の制約**を明示。
- 種分化は遺伝距離・生殖隔離・表現型差等の明示的基準により候補として検出。クラスタラベルを自動的に「種」と断定しない。
- Lineage viewerは個体の祖先・派生・絶滅・表現型変化を探索できる。

---

### 7.5 進化実験の「条件」を明確にする

- `evolution_mode`を `none / implicit / explicit_selection / coevolution / lifetime_learning / hybrid` に分ける。
- 集団を明示する系では `birth`, `death`, `mutation`, `recombination`, `selection`, `migration` を一貫したイベントモデルで保存する。
- 任意の形態クラスタに自動的に繁殖系統があると仮定しない。Lineage DAGは観測可能な親子イベントまたは推定根拠付きで生成。
- 演算予算を `num_evaluations`、`environment_steps`、`gpu_seconds`、`walltime_seconds` に分解し、探索手法間で何を揃えたか明記。
- 同一個体の乱数環境を変えた再評価を保管し、偶然の高スコアをeliteとして固定しない。
- `ArchiveRecord`に `genome_ref`, `descriptor`, `score`, `evaluation_seed`, `model_version`, `run_id`, `validation_status` を含める。

### 7.6 探索戦略の追加候補

| 手法 | 使いどころ | 誤解防止・対照 |
|---|---|---|
| Intrinsic curiosity / prediction error | 未踏の挙動の探索、世界の動態発見 | ノイズを最大化する落とし穴 |
| Goal-conditioned exploration / IMGEP | 目標を連続生成して多様な結果を探索 | 目標分布・達成距離を保存 |
| Unsupervised Environment Design | 多様な環境の自動作成 | 環境の難化≠生命の進化 |
| Power spectrum / complexity search | 空間的・時間的複雑性探索 | ランダムノイズ、定常パターン対照 |
| Evolutionary game dynamics | 利他・裏切り・共存等の相互作用 | 固定報酬・固定利得行列の事前設定 |
| Coevolutionary arms race | 相手の戦略が変わる継続的適応 | cycling、forgetting、相対評価のみの改善 |
| Quality-Diversity with learned descriptors | 手設計特徴で捉えにくい行動探索 | 埋め込みversionの固定と再評価 |

### 7.7 Cross-run discovery archive

- 世界をまたぐ共通表現は `summary_descriptor` に限定し、物理量・時計・単位を壊してまで同一空間に埋め込まない。
- `within-substrate archive` を正本とし、`cross-substrate gallery` は参考閲覧・探索のレイヤー。
- 発見候補を別の初期条件で `retest` し、条件に依存する一過性挙動と持続的挙動を分ける。
- candidate generationに使ったデータと評価用holdout条件を分け、ASAL/進化探索の過学習を検出。

---

## 8. ASAL — Foundation Model-guided Discovery

### 8.1 位置付け

- ASALを**特定の世界ではなく探索レイヤー**として実装する。Lenia/Boids/CA/Particle Life/NCAなどへ適用できるようにする。
- Sakana AIの元論文および公式`SakanaAI/asal`の目的関数・フレームサンプリング・基質例を参照する。
- 使用する視覚基盤モデルは固定し、モデルID・重みハッシュ・前処理・入力解像度・テキストpromptを記録。

### 8.2 3つの探索モード

1. **Supervised target discovery**: `text -> image/video embedding`の類似度を利用して所望のパターンを探索。時間イベント列を指定する拡張は**オリジナルASALとの違い**として別モードにする。
2. **Open-ended discovery**: 時間差分・歴史アーカイブからの新規性を指標化。自己相関・映像ノイズ・表示色の変化を新規性として誤認しない対照が必要。
3. **Illumination**: descriptor / embedding空間を多様なシミュレーション結果で満たす。MAP-Elites型のアーカイブとの連携を可能にする。

### 8.3 VLM evaluator

- `open_clip`を基本に、CLIP / SigLIPをアダプタで入れ替え可能にする。初期は軽量モデル1種類で検証。
- 適切な`eval()`・`requires_grad_(False)`・mixed precisionで推論。
- テキストembeddingを先に計算しcache。動画フレームは指定間隔・均一/重要イベントサンプリングで取得。
- プロンプト類似度を最大化する際、**色調や静止画だけを最適化してしまうreward hacking**を検出する。複数view、持続性指標、時間的変動指標との比較を設ける。
- 時間平均埋め込み以外に、frame embeddingの変化量、時間順序を保つ集約法も実験的に追加可能。
- CLIPで見つけた候補は独立したルールベース/物理ベースの評価で再試験。

### 8.4 JAX ⇄ PyTorch DLPack連携

- 従来のcapsule経由に加え、双方の`__dlpack__`が使用可能な場合は `torch.from_dlpack(jax_array)` / `jax.dlpack.from_dlpack(torch_tensor)` という**モダンなArray API**を優先検証。
- 同一GPU、対応dtype、連続/受入可能なstrides、所有権・参照寿命、stream同期、read-only/mutabilityの条件を満たす場合に限りゼロコピーを主張する。
- 不適切な暗黙コピー/デバイス転送を検出するtelemetryを実装。ゼロコピー不可時は明示的なfallbackと警告を出す。
- **GPU間テンソル共有とブラウザへのストリーミングは別問題。** ブラウザへ見せるプレビューには圧縮・CPU転送が通常必要であり、シミュレーション→VLMがゼロコピーでもブラウザへの完全ゼロコピーは要求しない。
- JAX/PyTorchのGPUメモリ共存が重い場合は、`evaluate chunks -> free references -> next chunk`等でピーク削減。別プロセス化による実用的な通信コストとも比較。

### 8.5 発見ギャラリー

- 探索で見つけた候補を「サムネイル、短いループ動画、世界ルール、seed、埋め込み類似度、定量指標、generation」で一覧。
- 実行時の評価器バージョンが異なるcandidateは値を直接同列比較しない。
- 類似候補のクラスタリングと重複除去を導入し、見た目の色が違うだけのcandidateを独立発見と扱わない。
- クリックで基質パラメータを復元し、**再生・同条件再評価・条件変更・branch**へ遷移。

---

### 8.6 発見器の複線化

- CLIP/SigLIPは「見た目の意味的類似度」を得る手段。探索用の特徴空間にはルールベース記述子、手作り動的指標、時系列表現、専用軽量動画encoderも追加可能とする。
- Flow-Leniaの好奇心駆動型探索の先行研究（arXiv:2505.15998）も参照し、**ASAL由来の実装ではない探索器は独立プラグイン**で表現する。
- 複数評価器の結果を無検証で単一の「生命らしさスコア」に統合しない。モデル間一致/不一致をそのまま分析できるようにする。
- VLMはバッチ候補の**事後的な順位付け**に用いるモードも設け、常時GPUに常駐させずピークVRAMを抑えられるようにする。
- 文字列プロンプトで意図した現象と、学習したモデルが画像中に検出した概念を区別。時間イベントを扱う際は明示的な動画・軌跡指標も確認する。

---

## 9. ALife Scientific Metrics — 何を「生命らしさ」と測るか

**原則:** 生命性の単一スコアは定義しない。個別の仮説に対応する複数メトリクスと統計的対照を管理する。

### 9.1 指標分類

| 分類 | 候補指標 | 想定基質 | 主な注意点 |
|---|---|---|---|
| 持続 | alive duration、形態保持率、局所凝集寿命 | Lenia/NCA/粒子 | 「個体」認識の定義を明示 |
| 自己修復 | 損傷後の機能/形状回復率、回復時間 | NCA/Lenia | 単なる形状平滑化と区別 |
| 複製 | 複製成功率、世代間情報継承、partner頑健性 | BFF/CA/NCA | 類似出現 ≠ 自己複製の証明 |
| 恒常性 | 外乱前後の内部状態変動、収束時間 | 資源系/NCA | 静止・死滅との区別 |
| 移動・agency | 方向性、目標追従、障害物応答、再現性 | Boids/MJX/Lenia | 環境に依存した因果的介入 |
| 多様性 | 遺伝子多様性、表現型分布、archive coverage | 進化系 | 正規化・descriptor定義に依存 |
| 複雑性 | 高次エントロピー、圧縮比、統計的構造 | BFF/全系 | 乱数/ノイズも複雑に見える |
| 生態 | 共存時間、種数、資源フラックス、食物網 | 資源系/PD-NCA | 種・相互作用の観測可能性 |
| 進化 | 新規系統出現、表現型変化、適応度改善 | 進化系 | 探索アルゴリズムの導入に注意 |
| Open-ended | 累積新規性、記録更新、ニッチ追加、複雑性の長期増加 | 全系 | 有限実験から「無限」は証明できない |

### 9.2 BFFの複雑性と複製の検証

- 時間ごとに全バイトのShannon entropy `H1`、全体/ブロック/条件付きentropy、正規化圧縮サイズを測定。
- `H1 - compressed_bits_per_symbol`等の差分を採用する場合は、単位（bit/byteとbit/symbol）と圧縮器・サンプルサイズ・header overheadを固定。
- 圧縮率が改善/悪化しただけで「相転移」と確定しない。頻度分布、tracer origin、複製アッセイで裏取りする。
- 候補テープをランダムな複数partnerと反応させ、自己/類似子孫の生成、再帰的な持続を測る。
- 対照：初期ランダム集団、シャッフル対照、非実行対照、命令の一部をNOP化したアブレーション。
- 重要な複製候補はテープ・反応partner・rng・初期状態・全step traceを小規模で保存。

### 9.3 因果実験（自動化可能なテンプレート）

- `baseline`: 介入なし。
- `damage_repair`: 定めた面積を破壊し、機能回復を測定。
- `resource_pulse`: 栄養供給の有無と量を変化。
- `environment_shift`: 温度/粘性/摩擦等を段階的に変更。
- `predator_introduction`: 捕食圧を変更。
- `genetic_ablation`: 遺伝子/制御器の一部を固定または削除。
- `no_evolution`: 突然変異・交叉を止めた対照。
- `no_learning`: PD-NCA等で重み学習を止めた対照。
- `reward_shuffling`: 視覚評価器/最適化目標の不正なshortcut確認。

**paired design:** 同じ初期seed/初期状態のcheckpointから介入条件とcontrol条件をforkする。確率的実行ではrun間の乱数をどう対応させたか記録する。

### 9.4 統計

- ユーザーが指定しなければ、探索的な初回比較で**最低5 seed**を推奨（証明力を保証する数ではない）。論文化を想定する場合はばらつき・事前効果量に基づいて必要サンプルを再見積りする。
- seed別系列を保持し、中央値・平均・ばらつき・95%区間を表示。
- イベント発生までの時間には右打ち切りを考慮できるようにする。
- 同じrun内の多数のフレームを独立サンプルとして扱わない（疑似反復の禁止）。
- 複数探索・多数仮説を試した場合は選択バイアスを明示。
- Run比較画面で明示的に「差の効果量」「未検証の解釈」「対照条件」を表示。

### 9.5 エージェントによる自動研究（将来）

- `Hypothesis -> Configs -> Runs -> Metrics -> Statistical test -> Summary` のワークフローを宣言的に定義。
- 研究エージェントが実験候補・対照条件を**提案**できるようにするが、各runには費用上限（GPU秒数、VRAM、disk）が必要。
- 多数の探索のうち都合がよい結果だけ選んで結論を書くことを防ぎ、試した全条件を記録。
- 要約文は計測値・run ID・figure IDへの参照を必須とする。
- 自動論文風レポートは「観測結果」「仮説」「限界」「次の実験」を明確に分ける。

---

### 9.6 Metric Registry — 全指標に数式とテストを持たせる

各metricのmetadataに `id`, `display_name`, `unit`, `clock`, `definition`, `aggregation`, `sampling_interval`, `required_observations`, `applicable_substrates`, `implementation_version`, `null_policy`, `uncertainty_method`を持たせる。

- エントロピー: `H(X)`, conditional entropy、時系列自己相関、圧縮率を別指標にする。バイト列を単に連結した圧縮比はデータ順序・header・window長で変わる。
- 時空間秩序: パワースペクトル、相関長、動的構造因子、連結成分、寿命、軌跡。
- 形態: 分裂候補、面積、周長、対称性、フラクタル的指標、空間的持続性。grid解像度・閾値を記録。
- 情報・制御: 予測可能性、情報フロー候補、センサー→アクション相互情報量。**Transfer EntropyやMIから因果を直接主張しない**。
- 身体/行動: 移動の効率・消費エネルギー・反応速度・外乱回復・環境外汎化。
- 進化: gene/phenotype diversity、系統の存続率、novelty discovery rate、絶滅、適応度分布。
- 相互作用: 競争係数の推定、接触ネットワーク、相互支援/抑制効果の介入差、資源収支。
- オープンエンド: 新規性の発見曲線、可到達多様性、長期の適応・複雑度変化。**有限長runから数学的無限性は立証できない**。
- すべての指標に `test_metrics_known_cases.py` と、ノイズ・無変化・周期変動・既知signalの対照ケースを用意。

### 9.7 Discovery Evidence Ladder — 証拠レベル

| レベル | 判定 | 要求する証拠 |
|---|---|---|
| E0 | visual curiosity | 見た目が面白い、またはヒューリスティック検出 |
| E1 | measurable event | 定義済み指標で差分が観測された |
| E2 | reproducible phenomenon | 独立seed/初期条件で再現されるか、再現条件が明確 |
| E3 | intervention-supported | 適切な対照・アブレーションで因果機構の一部を支持 |
| E4 | reference/external validation | 別実装、外部既知事例、独立評価などで確認 |

- E0/E1をUIで「生命を発見」「自己複製が証明された」と表示しない。
- E3/E4でも「自然界の生命である」ことが証明されたわけではない。
- `DiscoveryRecord`は主張（claim）、利用指標、run/seed集合、対照、反証された代替仮説、限界を保持する。
- 検証失敗、結論保留、指標の依存関係を削除しない。

### 9.8 よくある「驚き」を機構へ落とす実験テンプレート

| 観測候補 | 単純な代替説明 | 最低限の検証 |
|---|---|---|
| 自己複製のような模様 | 両側に同じ形が偶然発生 | トラッキング、情報継承、継続的複製、partner/環境変更 |
| 群れや協調 | 外力/同一勾配を追っているだけ | センサー・個体間相互作用・場を段階的に無効化 |
| 自己修復 | 拡散平滑化・目標画像への学習 | 損傷位置/規模変更、untrained/no-update対照 |
| 多種共存 | 個体同定ミスや反応停止 | 資源・再生産・個体入替・侵入実験 |
| 複雑性の増大 | ノイズ増加・圧縮器依存 | shuffled/noise/periodic対照と複数指標 |
| 相転移 | 閾値・描画スケール・サンプル選択 | 指標の頑健性、changepoint再評価、独立seed |
| 継続的進化 | archiveだけ増え、実系は停滞 | 新規性/機能/系統を分離、holdoutで再評価 |
| 予想外の知的行動 | reward仕様の抜け道 | 報酬OFF、未知環境、行動アブレーション |

### 9.9 科学的比較のルール

- 比較の実験単位は原則 `seed-run` または独立して初期化された世界。時系列の全フレームを独立標本にしない。
- 同じcheckpointから作る分岐実験はpaired design。ただし乱数が介入で消費系列を変える可能性があるので、共通乱数の厳密対応は保証対象として明示する。
- `exploratory` と `confirmatory` をRunにタグ付け。探索から選び出した候補の性能だけでp値を出さない。
- 多数の探索条件・数十の特徴量を同時に検定した際は、多重比較とwinner's curseへの対応をreportに明記。
- CPU決定論的baseline、GPU浮動小数点結果、モデル近似版を同一の精度であるかのように表示しない。
- negative resultsの保存を必須とする。E2まで達しなかった探索も成功/失敗を分けて記録。

### 9.10 Automated Assay Planner（P2〜P3）

- 検出器のcategoryに従い、E1候補から `assay_plan.yaml` を生成。例：損傷修復→位置/面積/環境のスイープ、複製→partner/突然変異/濃度のスイープ。
- 人間が確認可能な `expected_observations`, `null_hypothesis`, `confounders`, `budget`, `stopping_rule` を提示。
- アッセイの実行は通常のジョブキューへ投入。専用の隠れたrunnerや非永続runを作らない。
- 探索エージェントがレポートを書く場合、claimごとに少なくとも1つの `run_id` / `metric_id` / `assay_id` を添付する。

---

## 10. Data Model・実験保存・再現性

### 10.1 Runディレクトリ（例）

```text
runs/<run_id>/
  manifest.json                 # immutable run provenance
  resolved_config.yaml          # fully expanded configuration
  environment.json             # git SHA, uv.lock hash, GPU, driver, backend
  logs/
    worker.jsonl
    events.jsonl
  metrics/
    summary.json
    history.parquet
    lineage_stats.parquet
  checkpoints/
    step_000000000.npz
    step_001000000.npz
    index.json
  states/
    sampled_states.zarr/
  archive/
    genomes.parquet
    descriptors.parquet
  media/
    cover.webp
    preview.mp4
    event_000001.mp4
  reports/
    experiment_report.md
```

- 画像と動画は**可視化の派生物**。正確な数値解析には元状態/メトリクスを使用。
- JAXの大規模pytreeではOrbax等の使用を検討し、`npz`だけで全型を無理に表現しない。
- `manifest.json` に基質実装バージョンとcheckpoint形式バージョンを記録。
- 重要実験の状態保存と、ディスク節約のサンプル保存を別ポリシーにする。
- 生成動画は必要なら再レンダリング可能なだけの状態サンプル・renderer configを保存する。
- データ容量をrun開始前に概算し、ユーザーに提示。上限超過時は自動停止か保存頻度低下（ユーザーの設定通り）を行う。

### 10.2 SQLiteスキーマ（概略）

```text
experiments(id, name, substrate, mode, created_at, config_hash, config_json)
runs(id, experiment_id, parent_run_id, parent_checkpoint_id,
     state, seed, backend, started_at, finished_at, manifest_path)
checkpoints(id, run_id, simulation_clock, clock_unit, path, checksum)
events(id, run_id, simulation_clock, type, severity, data_json)
metrics_index(run_id, metric_name, units, file_path, clock_unit)
artifacts(id, run_id, kind, mime_type, path, size_bytes, checksum)
organisms(id, run_id, birth_clock, death_clock, genome_ref)
lineage_edges(run_id, parent_id, child_id, relation, confidence)
interventions(id, run_id, requested_clock, applied_clock, kind, payload_json)
jobs(id, run_id, status, worker_pid, heartbeat_at, error_json)
```

- SQLiteを大容量のフレーム列ストアとして使わない。データはParquet/Zarrへ。
- migrationをAlembic等で管理。
- ユーザーがrun名を変更してもID・hash・原設定は変わらない。
- checkpointは一時ファイル作成→検証→atomic rename→DB反映の順で公開。

### 10.3 チェックポイント復旧要件

保存すべきもの：

1. 完全な世界状態と個体集団。
2. シミュレーション時刻（単位付き）。
3. PRNG状態・キーの系列管理方法。
4. 進化戦略の内部状態、archive、optimizer状態。
5. 介入の適用済みoffset。
6. バージョン、dtype、backend、格子サイズ、固定形状情報。
7. データ整合性checksum。

**復旧テスト:** run Aを500step進めてcheckpointし、再開後さらに500step進める。連続1000step実行と比較し、決定論的CPUなら完全一致、GPUなら仕様に応じた許容差/統計一致を報告する。

### 10.4 RunごとのProvenance

- `git_commit`, `git_dirty`, `uv_lock_hash`, `python_version`, `packages`, `cuda_driver`, `jax/jaxlib`, `torch`, `mujoco`。
- GPU型番、VRAM、precision、XLA設定、CUDA kernelバージョン。
- 参照論文/参照実装commit、設定ファイルhash、promptとVLM ID。
- run開始時のハードウェア負荷・GPU共有状況（収集できる範囲）。
- 実験の観測バージョン・レンダラーバージョンをそれぞれ記録。

---

### 10.5 研究ワークフローの保存モデル

```text
scenarios(id, name, config_hash, schema_version, created_at)
sweeps(id, scenario_id, plan_json, state, budget_json, created_at)
sweep_members(sweep_id, run_id, sample_index, factor_json)
assays(id, discovery_id, config_hash, state, created_at)
assay_runs(assay_id, run_id, arm, replicate_index)
discoveries(id, category, claim, evidence_level, status, created_at)
discovery_evidence(discovery_id, run_id, event_id, metric_id, assay_id, role)
rule_revisions(id, run_id, revision_hash, parent_revision, applied_clock)
observation_versions(id, module_id, module_version, schema_hash)
resource_usage(run_id, sampled_at, cpu_pct, ram_mb, vram_mb, disk_bytes, device_json)
```

- DB migrationは既存runの読取互換性を優先し、マイグレーション前のDBバックアップを作る。
- `evidence`は単なるイベントと区別。どの主張をどのrunで支持/反証したか双方向に追えるようにする。
- Artifactをcontent hashによって重複排除する場合、**削除時の参照カウントと改ざん検査**を合わせて実装。
- DBには生の数百万個体の時系列を入れない。重い配列はParquet/Zarr、元の科学データが不必要に丸められない形式を採用。
- Snapshotとcheckpointを別オブジェクトにする。前者は観測用部分状態、後者は完全な再開に必要な状態。
- External adapterの著作権・ライセンス上再配布不可なモデル/データはartifactに埋め込まず、hashと参照元を保存。

### 10.6 ストレージ予算と保持ポリシー

- `minimal`: manifest + metric summary + 最終checkpoint + 少数preview。
- `standard`: 上記+間引きseries、周期checkpoint、主要イベントclip。
- `research`: 状態の高頻度サンプル、全lineage、検証用assay、解析中間生成物。
- 各runの `retention_policy` を開始前に固定。ストレージ不足時に重要checkpointを自動削除せず、設定に沿って安全停止する。
- 古いrunの動画のみ削除可能にしても、元状態・アッセイ証拠を巻き込まないようにする。
- `alife runs verify`でmanifest、checkpoint、artifact checksum、DB参照を検査できる。

### 10.7 可搬な研究パッケージ

`alife export <run>` の `research_bundle`形式は、少なくとも config, manifest, metric definitions, checkpointsまたは復元不能の明示、主要イベント、図表、出典、環境再現手順を含める。**全GPU/全環境でのビット一致を約束しない**。

---

### 10.8 科学的条件と実行条件の分離

`ExperimentSpec`には世界ルール、初期条件、seed/RNG、時間刻み、精度・許容誤差、探索目的関数、観測指標、停止条件を保持する。`ExecutionProfile`にはtarget、空きVRAMに対するメモリ予算、microbatch/chunk、並列job、preview転送、scratch、時間・ディスク制約を保持する。

- 科学的シナリオの設計例は `configs/scenarios/conway_reference.yaml`、計算先に関する設計例は `configs/execution_profiles/` を参照する。両者は別々のYAMLとして管理し、開始時に検証済みの`ResolvedRunPlan`へ解決する。
- previewの品質・転送頻度、検証済みmicrobatch、chunk、計算先のみの変更であれば科学条件は維持される。ただし浮動小数点の演算順序が変わった可能性は計算環境の差分として記録する。
- world size、個体数、seed、精度、solver、探索母集団サイズ、実験終了条件などを変更する場合は**新しい科学的設定**として扱う。
- `experiment_spec_hash`、`execution_profile_hash`、`source_digest`、`input_digest`、`reference_digest`は独立して記録し、同一条件の移植実験と大規模化した別条件実験を区別する。
- GPUの違いによるbitwise同一性は保証せず、適切な再現性ラベルと数値検査を付ける。

### 10.9 Artifact Store / Git / DVC の3レイヤー

1. **Git = 実装・研究プロトコルの正本。** コード、テスト、テキスト設計、スキーマ、small fixtures、実験テンプレート、lockfile、CI、マイグレーションを保管。
2. **ArtifactStore = 実験出力の正本。** checkpoints、Zarr状態、Parquet大規模指標、MP4、系統、イベント、モデル重み、ログ、run DBのバックアップ等。容量とretentionを制御。
3. **DVC/同等のdata registry = 共有・再利用したい厳選された固定大型資産に限る（オプション）。** curated dataset, benchmark corpus, reference checkpointのポインタ/manifestはGitで追い、実体は外部保存先。全runをDVCで登録する必要はない。

大きなアーティファクトには `sha256`, `logical_name`, `kind`, `mime`, `size`, `created_at`, `run_id`, `storage_uri`, `retention_class`, `producer_commit`, `input_digests`, `access_scope`を持たせる。`storage_uri`には署名付き秘密URL/認証文字列を保存せず、論理パスを保存する。

`ArtifactStore`は初期`local_fs`、後から`ssh/sftp`, `S3-compatible`, `NAS-mounted`等をAdapter追加可能にする。リモートの`upload -> verify -> finalize -> manifest update`は原子的に見える状態遷移とし、途中で切れたアップロードを有効checkpointとして扱わない。

### 10.10 Gitと実験出力の関連付け

**runから次を必ず逆引きできること:**

- `git_commit`, `git_dirty`, `source_digest`, `uv_lock_digest`、フロント/worker依存版。
- `experiment_spec_hash`, `execution_profile_hash`, `resolved_config`, `seed / RNG state`, `solver precision / numerical_policy`。
- `execution_target_id`, `GPU UUID/model/VRAM`, `driver`, `CUDA/XLA/runtime`, `backend`, `container_image_digest`（あれば）。
- `dataset_hash`, `checkpoint hashes`, `model weights hashes`, `reference implementation commit/license`。
- `artifact_index_uri`, `run DB schema revision`, `storage retention policy`, `restore compatibility`, `external scheduler job ID`。

`git_dirty=true`の状態からリモート実行する場合、実際に実行した内容を固定できるよう、認証情報を除外した差分またはソーススナップショットのdigestを保存し、実験を`non_release_reproducibility`等の状態にする。原則として**重要な再現・共有実験はclean commitから開始**する。GitHubへのpushを強制する必要はなく、ローカルのcommitと検証済みsource archiveでも追跡可能。

### 10.11 保存媒体・移行・バックアップ

- 開発機の`runs/`と、研究室の大容量共有ストレージは別々の実装が可能。`ALIFE_RUNS_DIR`等の環境変数で設定し、source repo直下へ巨大データを吐かない。
- 初期local SQLiteはControllerのみが更新。バックアップはSQLite公式のバックアップAPIまたは整合性のある方法で取得し、DBファイルだけを稼働中に雑にコピーしない。
- 複数ワーカー・複数ユーザー運用に拡張する際はメタデータDBをPostgreSQL等へ移行可能なrepository層に隔離。初期段階で必須にしない。
- 永続化したいrunは`pinned`、短期プレビューは`ephemeral`。GC/アーカイブはdry-run→参照確認→承認→削除の段階を踏む。
- バックアップの成功だけでなく**別の空ディレクトリへのrestore**を定期テストする。
- 削除・破損・共有ストレージのアクセス不能はGit履歴で回復できない。研究上重要なartifactは別媒体/別障害ドメインへバックアップする。

---

## 11. ハードウェア可搬性・GPUメモリ・性能設計

### 11.1 実行環境を分類する目的

| 実行Tier | 想定される役割 | 最低限の契約 | 実装優先度 |
|---|---|---|---|
| `cpu_dev` | GPUのない開発機・CI・参照計算 | スカラーoracle、API/UI、軽量実験、テスト | MUST |
| `local_interactive` | 日常作業用のローカル単一GPU | 低遅延プレビュー、対話操作、単一実験、メモリ節約 | MUST |
| `remote_compute` | 大容量メモリ・高演算性能の実験用GPU | 大規模世界、長期探索、大型評価器、遠隔成果物回収 | SHOULD |
| `multi_gpu` | 複数GPUを持つ計算ノード | 独立run並列／能力検証済み分散評価 | MAY |
| `cluster_batch` | Slurm等のジョブスケジューラ環境 | ジョブ投入、状態取得、チェックポイント、長期保存 | MAY |

Tierは**性能・容量の序列を固定値として定義するものではない**。`local_interactive`は実行環境の役割であり、CLIの`--target local`はprobeで発見した具体的なターゲットIDを指す。各実行先のGPU種類、空きVRAM、RAM、scratch、compute capability、利用可能backend、権限・料金を動的probeして選択する。同じ`ExperimentSpec`を実行できないtargetは明示的に拒否し、暗黙のダウンサイジングを行わない。

### 11.2 科学的条件と実行条件の分離

- `ExperimentSpec`（科学条件）には世界法則、初期状態、seed、時間刻み、個体数、精度要件、探索母集団サイズ、停止条件、観測指標を記述する。
- `ExecutionProfile`（資源条件）にはtarget、メモリ利用上限、microbatch、chunk、preview転送、worker数、scratch、保存先とwalltimeを記述する。
- microbatchと**進化集団サイズ**は異なる。粒子数やworld grid、dtype・数値解法を変える場合は科学条件の新revisionとして扱う。
- 大容量GPUではより大きい実験を**新しいExperimentSpec**として作成できるが、計算環境だけ変えて同一条件を再現するrunと混同しない。
- `auto` はprobe情報と実測プロファイルに基づくresource配置だけに使用する。推定値と実測値を必ず分離する。

### 11.3 GPUメモリと実行予算

- 空きVRAMを起動時とrun直前に取得し、他プロセスの使用量・描画環境・allocator reserveを考慮する。ハードコードした容量や製品名に依存しない。
- 使用予定メモリを `world_state`, `optimizer`, `activations`, `model_weights`, `temporary_buffers`, `preview`, `runtime_cache` に分解し、unknown項目はunknownと報告する。
- `memory.max_fraction_of_available` と `memory.reserve_fraction` はsoft-budget。予算の一例は `min(available * max_fraction_of_available, available - total_vram * reserve_fraction)`（下限0）であり、実際のallocatable量はruntime probeで補正する。厳格なハード上限を保証しない。
- メモリ不足時の再試行は、科学的意味が変わらないとテスト済みの `microbatch`, `chunk_size`, `preview_quality`, `device staging` に限定する。
- 低メモリ環境では`chunking`, `streaming`, `host offload`, `state compression`, `checkpoint/remat`を利用可能にする。ただし転送/I/O/計算時間の増大は計測・表示する。
- モデルとsimulationが同一GPUを使う場合、JAXの事前確保とPyTorchのallocator競合を分離・検証する。APIサーバープロセスはGPU runtimeをロードしない。
- `max_walltime_seconds`, `max_gpu_seconds`, `max_disk_bytes`, `max_artifact_bytes`, `max_queue_jobs` をrunごとに持ち、無制限な計算・保存を禁止する。

### 11.4 依存・バックエンド・CUDA環境

- Pythonは`uv`で管理する。最小のCPU/base環境と、JAX、PyTorch/VLM、MuJoCo、CUDA VM等のoptional worker環境を分離する。
- 相互に整合しないCUDA/JAX/Torch/Warp/CuPy依存を1つのvirtualenvへ強制しない。別`uv` project、worker process、コンテナを利用する。
- JAX CUDA plugin、GPU driver、PyTorch、CuPy、Warp、MuJoCo、evosax/QDaxの互換性は**実装時の公式情報と実機smoke test**で確認する。CUDAの特定リリースをプロジェクト要件に固定しない。
- `jax.random.key()`等のAPIはインストール済み版の公式仕様で照合する。外部ライブラリのPRNG変換は別テストで扱う。
- CUDAアーキテクチャに対応するwheelとnative kernelを実機で検証する。driver表示だけで動作を保証しない。
- JAXのメモリ関連環境変数は当該版の公式ドキュメントで確認し、importより前に設定する。用途ごとに事前確保の有効/無効を評価する。
- DLPackによるJAX↔PyTorch連携はdevice・dtype・ownership・stream同期・copyの有無を実機で検証する。ゼロコピーを無条件に約束しない。
- MJX-JAXとMJX-Warpは微分可能性と計算特性が異なる。選択したbackendと参照との相違をrunに記録する。

### 11.5 性能評価ハーネス

- compile/warmup時間、steady-state throughput、end-to-end throughput（描画・計測・保存を含む）を別々に記録する。
- `steps/sec`, `epochs/sec`, `agent_steps/sec`, `p50/p95 latency`, peak VRAM, RSS, I/O, checkpoint time, joules/step（取得可能な場合）を記録する。
- `jax.block_until_ready()`などを用い、非同期実行を正しく同期して測る。
- 粒子数・格子・チャネル・複製テープ数・並列環境数などを段階的にスイープし、OOM条件も欠測理由と共に保存する。
- ランダム試行のbest caseだけでなく中央値・ばらつき・system loadを記録する。

### 11.6 基質ごとのスケーリング戦略

| 基質 | 主な計算制約 | 小規模環境での工夫 | 大規模環境での拡張 |
|---|---|---|---|
| CA/Lenia | 格子・近傍計算・FFT | 細分化preview、kernel再利用 | 高解像度・多チャネル・複数世界 |
| Flow-Lenia | 移流・中間バッファ | fused操作、分割、質量誤差検証 | 多様な局所パラメータ・大きな格子 |
| Particle Life/Boids | 近傍探索 | spatial hashing、局所接触 | 大集団・大空間・複雑な相互作用 |
| BFF | 可変長命令実行と分岐 | uint8圧縮、CPU oracle、batching | 大規模soup、多seed並列 |
| NCA / PD-NCA | 学習時activationと状態 | truncated rollout、remat | 多主体・長rollout・豊富な状態 |
| Artificial chemistry / Protocell | sparse反応・結合 | event-driven update | 複雑な反応ネットワーク・多空間系 |
| MJX / agent worlds | 接触計算・policy評価 | 小型policy、段階的並列化 | 多個体/形態の同時評価 |
| ASAL / VLM | 埋め込み推論・モデル常駐 | frame sampling、逐次評価 | 大型評価器・多候補評価 |
| Hybrid worlds | 複数solverの同期 | 疎なcoupling、時間刻み契約 | 高解像度・異種基質の同時実行 |

数値スケールの目標は環境ごとにベンチマーク実行で決める。VRAMが増えてもGPU計算速度が同じ割合で増えるとは限らない。

### 11.7 複数GPUとクラスタ実行

1. まず**独立runのGPU単位分散**を行い、異なるseed/条件を同時実行する。
2. つぎに独立して評価できる探索候補・学習バッチをshardingする。
3. その後、特定の基質に限り単一worldの跨GPU分割（halo交換、グローバルreduce等）を検討する。通信コストと科学的同等性を実測する。
4. Slurm等では `submit/status/cancel/collect` を独立adapterとし、共有クラスタのsite policy、partition/QOS、job array、walltime、権限・予算を事前確認する。
5. batch jobの`pause`をライブ停止と偽らない。可能ならcheckpoint-and-exit後に再投入する。
6. scratch内のファイルは永続ではない。出力を検証済みArtifactStoreへ確定してからrun成功とする。
7. remote workerからControllerのSQLite/WALを直接書き換えさせない。Controllerを唯一のメタデータライターとする。

### 11.8 可搬性検証と再現性ラベル

- `bitwise`: 整数CA等で状態・出力が完全一致。
- `numerically_close`: 浮動小数点solverの誤差、保存則、許容差を明示して比較。
- `statistically_consistent`: カオス的な長期進化・確率系は複数独立seedで分布や効果量を比較。
- 同一条件の別target実行には新しいrun IDを与え、親run・コードdigest・科学条件hashを保持する。
- 速度や容量の差から世界サイズ、人口、世代数、計算精度を変更した場合は**スケーリング実験**として新しい科学条件を登録する。
- CPU基準実装とGPU高速化実装の差分テストはGPU移行で勝手に緩和しない。

---

## 12. CLI・シナリオ・実験プリセット

### 12.1 CLIの基本契約

```bash
uv run alife doctor
uv run alife serve --host 127.0.0.1 --port 8000
uv run alife substrates list
uv run alife presets list
uv run alife targets list
uv run alife targets probe local
uv run alife run configs/scenarios/conway_reference.yaml --target local
uv run alife runs list --status completed
uv run alife runs inspect <run_id>
uv run alife runs checkpoint <run_id>
uv run alife runs resume <run_id> --checkpoint latest
uv run alife runs fork <run_id> --checkpoint latest --intervention <intervention.yaml>
uv run alife runs reproduce <run_id> --target local
uv run alife runs verify <run_id>
uv run alife compare <run_id_a> <run_id_b>
uv run alife report <run_id> --format markdown
uv run alife export <run_id> --format research_bundle
uv run alife sweep <sweep.yaml> --dry-run
uv run alife discoveries list
uv run alife assays plan --discovery <discovery_id>
uv run alife benchmark --substrate <id> --target local
```

コマンドは**実装すべきCLI契約案**。CLIとWeb APIは同じサービス層、同じvalidation、同じrun manifestを使う。未実装コマンドを実行可能と偽らない。リモートの`dispatch`、Slurm等はcapabilityが実装されるまで表示を無効化する。

### 12.2 科学的シナリオと実行プロファイルの分離

科学的条件ファイル：`configs/scenarios/conway_reference.yaml`。実行先・保存先の条件は`configs/execution_profiles/local_interactive.yaml`などで別に管理する。ユーザーがGUIで実験を作成した場合も、科学条件をcanonical JSONに解決してdigestを付ける。

- 科学条件には`substrate`, `initialization`, `solver`, `steps`, `seed`, `experiment_design`, `observations`などを含める。
- 実行profileには`target_selector`, `memory`, `scheduling`, `preview`, `storage`, `limits`を含める。
- 観測メトリクスの計算間隔が統計的解釈に影響する場合は科学条件。描画用フレーム転送間隔は実行profileに置く。
- `seed`は単一run、`seeds`はsweep。混在させない。`schema_version`は保存データの互換性を扱う技術フィールド。
- 実行開始前に科学条件/数値ポリシーの互換性、入力data hash、コスト・ストレージ制約をチェックする。

### 12.3 推奨プリセット群

| プリセットID | 世界・手法 | 検証の中心 |
|---|---|---|
| `conway_glider` / `conway_guns` | 古典CA | 既知の周期と伝播 |
| `boids_flocking` | Boids | 対話的な群れの形成 |
| `particle_life_clusters` | Particle Life | 引力・斥力の相互作用 |
| `lenia_orbium` | Lenia | 形態持続と移動 |
| `flow_lenia_reference` | Flow-Lenia | 質量保存と局所輸送 |
| `bff_reference` | BFF | VM正しさと複製子候補 |
| `nca_grow` / `nca_regenerate` | Growing NCA | 成長・損傷回復 |
| `pd_nca_competition` | PD-NCA | 学習と集団の相互作用 |
| `gray_scott_pattern_sweep` | 反応拡散 | パターン相図 |
| `ecosystem_resource_cycle` | 資源循環 | 収支・相互作用 |
| `artificial_chemistry_cycles` | 人工化学 | 触媒・反応グラフ |
| `protocell_growth_split` | プロトセル | 区画・分裂・継承 |
| `swarm_stigmergy` / `physarum_path_network` | 群知能 | 信号場と経路 |
| `agent_memory_ablation` | 軽量神経エージェント | 学習・記憶 |
| `agent_communication_game` | 多主体エージェント | 介入による協力の検証 |
| `ca_rule_illumination` | ルール空間探索 | QDとrandomの比較 |
| `flow_lenia_curiosity` | 好奇心駆動探索 | 発見の多様性 |
| `asal_lenia_target` | VLM誘導探索 | テキストと形態 |
| `mjx_ant_evolution` / `morphology_development` | 身体性進化 | 形態と制御器 |
| `replicator_partner_assay` | 複製子アッセイ | partner・非実行対照 |
| `multiverse_paired_fork` | 同一checkpoint分岐 | 介入の対照比較 |

各presetは、目的、参照文献、許容パラメータ、期待観察、否定的対照、必要なbackend、実測済みならメモリ/速度指標、忠実再現と独自拡張の範囲をmanifestに記述する。表はカタログの**開発目標**であり、全presetの実装を同時に要求しない。

### 12.4 スイープ・介入・アッセイ

- `Scenario Studio`は基質schemaからフォームを作る。任意の式のevalは行わない。
- factor sweep、複数seed、paired fork、探索budget、介入の時刻・適用順序を再現可能な宣言形式で保存する。
- 一つのsweepには親IDと複数run IDを発行し、停止/再開/部分失敗・課金を記録する。
- `assay_plan`には null hypothesis、confounders、armとseed、expected observation、停止規則、budget、どのcandidateを検証するかを保存する。
- 参照文献との比較は`reference mode`だけで行い、同じ設定名で独自改変ルールを混ぜない。

### 12.5 計算先操作（任意の拡張）

```bash
uv run alife targets list
uv run alife targets probe <target_id>
uv run alife runs dispatch <run_id> --target <target_id>
uv run alife runs status <run_id>
uv run alife artifacts list <run_id>
uv run alife artifacts verify <run_id>
uv run alife artifacts export <run_id> --format research_bundle
```

- `dispatch`前に実行環境の互換性・容量・権限・見積もり・入力digestを検証する。
- 同じジョブを二重submitしないようにidempotency keyを要求する。
- 実験条件は変更せず、配置やchunkを変更する場合はrun manifestに明示する。

---

## 13. リポジトリ構造・Git管理・大規模データの扱い

### 13.1 目標構造

```text
alife-lab/
├── AGENTS.md                         # Agent execution contract
├── README.md                         # Setup, run and implementation status
├── pyproject.toml / uv.lock          # Python environment
├── frontend/                         # React/TypeScript application and lockfile
├── docs/
│   ├── ALIFE_LAB_SPEC.md             # Master specification
│   ├── GIT_AND_ARTIFACT_POLICY.md    # Source / artifact boundaries
│   ├── architecture.md / api.md / data_schema.md
│   ├── experiment_methods.md / roadmap.md / hardware_benchmarks.md
│   ├── remote_execution.md / reproducibility_matrix.md
│   ├── decisions/                     # ADR
│   ├── substrates/                    # Scientific contracts
│   ├── reference_fidelity/            # Difference from primary implementations
│   ├── assays/                         # Validation protocols
│   └── discoveries/                   # Reports, links to archived evidence
├── configs/
│   ├── scenarios/ / presets/ / sweeps/ / interventions/ / assays/
│   ├── execution_profiles/            # Local / remote / cluster templates
│   └── compute_targets/               # No credentials
├── src/alife_lab/
│   ├── api/ / service/ / core/
│   ├── execution/                     # local, remote, batch, scheduler adapters
│   ├── artifacts/ / provenance/ / storage/
│   ├── substrates/
│   │   ├── ca/ / boids/ / particle_life/ / lenia/ / flow_lenia/
│   │   ├── bff/ / nca/ / pd_nca/ / reaction_diffusion/ / ecosystem/
│   │   ├── artificial_chemistry/ / protocell/ / swarm_stigmergy/
│   │   ├── agent_world/ / rule_search/ / hybrid_world/
│   │   └── mjx/ / microcosmos/ / external_evolution/
│   ├── evolution/                    # ES, QD, novelty, curiosity, lineages
│   ├── evaluators/                   # Metrics, ASAL, evidence, assays
│   ├── rendering/ / benchmarks/ / utils/
│   └── cli.py
├── tests/
│   ├── unit/ / api/ / integration/ / end_to_end/
│   ├── reference/ / property/ / scientific_assays/
│   ├── execution/ / artifact_store/ / portability/ / gpu/
│   └── fixtures/
├── scripts/ / migrations/ / benchmarks/
├── .github/workflows/                # CPU CI; restricted optional GPU CI
├── .gitignore / .gitattributes / .env.example
└── runs/                             # Runtime output; never Git-tracked
```

上記は**目標ディレクトリ構造**。記述されたすべてのディレクトリや実行ファイルが既に存在するという意味ではない。依存関係を持たない空モジュールを大量生成せず、vertical sliceの完成に合わせて追加する。

### 13.2 Git / ArtifactStore / DVCの境界

| 管理対象 | Git | 外部ストレージ | 取扱い |
|---|---|---|---|
| Python/TypeScript/CUDAコード、UI、API schema、CI、テスト、ドキュメント | **Yes** | 不要 | 通常のsource history |
| 設定、プリセット、科学的protocol、migration、lockfile | **Yes** | 不要 | 再現性の核心 |
| 小さなfixture、ライセンス確認済みの画像/サンプル | 任意 | 不要 | サイズ審査 |
| 大型で再利用する固定dataset・benchmark・モデル | ポインタのみ | DVC remoteまたは同等 | hashと取得手順 |
| run毎のcheckpoint、Zarr、Parquet、動画、lineage、探索出力 | **No** | ArtifactStore | retentionとbackup |
| live DB、WAL/SHM、コンパイルcache、logs | **No** | ローカルと必要時backup | 変更頻度が高い |
| password、SSH鍵、token、資格情報、signed URL | **Never** | OS credential store | Private repoでも厳禁 |

- **Gitはソースと宣言的研究プロトコルの正本**。実験生成物の正本はArtifactStore。
- DVCは厳選した共有固定大型資産だけ。LFSは少数の安定的なバイナリ配布物への例外であり、runの一括保管手段ではない。
- Artifactは`run_id`, `sha256`, `producer_commit`, `scientific_config_hash`, `execution_hash`, `storage_uri`に関連付ける。
- `git clone`とCPU依存関係だけで基礎UI/APIと小型CAを試せる状態を維持する。
- `.gitignore`は既に追跡済みの秘密/巨大ファイルには効かない。secret scanとstaged diffの容量検査を導入する。
- 大量の静的データをGit履歴へpushしない。GitHubなどのホスティング制限・LFS課金は利用前に公式ガイドで確認する。

### 13.3 コミット・CI・安全規約

- `git status --short`を変更前後に確認する。`git add -A`を無条件に実行しない。
- コードのcommitは英語で小粒に分け、科学的更新則の変更には回帰テストとfidelity記録を付ける。
- Python/TS lockfileとDB migrationはGit追跡する。GPU依存を互換検証した環境のlockとimage digestはrun manifestに保存。
- CPU CIは通常のPRで実行可能とし、共有GPUのself-hosted runnerは信頼されたブランチ・許可済みアカウントだけで起動する。
- ライセンス未確認の外部コード・学習済みモデル・論文図を勝手にvendorしない。
- run出力の削除はGCのdry-run、pinned参照確認、明示承認を要する。

---

## 14. 実装ロードマップと受け入れ基準

**実装方針:** まず最小の基質で`simulation → service → API → GUI → checkpoint → test`を端から端まで完成させる。その後、他のALife系を独立プラグインとして導入する。多くの基質の空実装を先に作らない。

### Phase 0 — 契約とローカル開発基盤

- 実行環境を`doctor`でprobe（OS/CPU/GPU/VRAM/driver/uv/Node）。CPU-onlyモードを維持する。
- 公式資料を調査して依存関係・ライセンス・reference実装の状態を確認する。実機で動く組合せだけlockする。
- `ExperimentSpec`, `ExecutionProfile`, `SubstrateAdapter`, `LocalExecutor`, `ArtifactStore`の最小契約を定義する。
- `.gitignore`, `.env.example`, secret/large-file guard、ADR、CPU CIを準備する。
- **DoD:** `uv run alife doctor`が正確に診断し、GPU無しでも最低限のテストが動く。

### Phase 1 — Webサーバー・GUI・Conwayの垂直スライス

- FastAPI、SQLite、job supervisor、CPU worker、REST/WebSocket、React、PixiJSを接続する。
- プリセットからConway世界を起動。Play/Pause/Step/brush、検査、メトリクス、checkpoint、終了後再開を実装する。
- CLIから同じサービス契約で実行・保存し、ブラウザからrunを開ける。
- **DoD:** `uv run alife serve --host 127.0.0.1 --port 8000`でGUIが開き、gliderの基準テストと保存・復元・再接続が成功する。

### Phase 2 — 観測・介入・分岐比較と実行基盤

- Live Laboratory、Experiment Library、Timeline/Replay、Inspector、Fork/Intervention/Compareを実装。
- SQLiteのController単一ライター、`LocalExecutor`、fake executor、artifact checksum/atomic publish、run manifestを完成。
- GUI/CLIのtarget一覧とcapability validation、previewのdrop/backpressure、worker異常終了からの復旧を実装。
- Boids / Particle Lifeを追加し、汎用2D rendererとinteractionを検証。
- **DoD:** 同checkpointからA/B分岐し介入比較できる。GUI再起動でrunが失われず、Git状態が実験生成物で汚れない。

### Phase 3 — BFF・Lenia・Flow-Lenia

- BFFのscalar oracle、batch版、GPU最適化（参照比較後）、VM debugger、複製子アッセイを実装。
- Lenia / Flow-LeniaでFFT、移流、質量保存、パラメータ輸送を再現し、参照との差分を定量化。
- **DoD:** VM命令と保存則のテスト、基準runの再生、複数seedの統計比較が成功する。

### Phase 4 — NCAと追加の人工世界（独立トラック）

- Growing NCA、損傷・再生対照、PD-NCAの参照モードを段階導入。
- 反応拡散・資源循環・人工化学・プロトセル・Swarm・軽量学習エージェントを**独立した研究モジュール**として選択実装。
- 各モジュールに `source → oracle → invariant → plugin → GUI → assay` を要求。
- **DoD:** 性質の異なる基質を少なくとも2系統追加し、保存・介入・対照・metricsが共通基盤で動作する。

### Phase 5 — Evolution Lab・ASAL・Discovery Workbench

- evosax / QDax / Novelty / MAP-Elites / curiosity、系統追跡、ASALのVLM評価を実装。
- Discovery候補とE0〜E4の証拠段階、assay planner、探索アルゴリズムの公平比較を実装。
- **DoD:** 1件の候補を自動探索で発見し、独立アッセイと否定的対照により`validated/rejected/inconclusive`と判定できる。

### Phase 6 — 身体性・長期実験・大容量実行先

- MJX-JAX/MJX-Warp、3D viewer、形態・制御共進化、必要ならMicrocosmos adapterを追加。
- RemoteAgentまたはSSHBatchのいずれかと安全な成果物同期を実装し、必要ならSlurm adapterを追加する。
- `multi_gpu`は独立run分散から開始し、単一world分散は実測後に判断。
- **DoD:** ローカルと別計算先で同じ科学条件を実行し、正しい再現性ラベル・artifactの照合・通信断復旧を提示できる。

### 継続的な拡張トラック

Rule search、Hybrid worlds、外部digital evolution、研究エージェント、人工化学、プロトセル、エージェント社会、複数時間尺度、未知ルールの探索等は、基質ごとに独立した技術選定・実装・検証を行う。科学的関心に応じて優先順位を変更してよいが、未検証の能力を実装済みと主張しない。

---

## 15. テスト戦略と完成判定

### 15.1 単体テスト

- CA: glider周期/移動、still lifeの不変性、boundary。
- Flow-Lenia: 質量保存とreference snapshot、負質量の禁止条件。
- BFF: 10命令、256値の解釈、head wrap、bracket jump、停止条件、pair execution、mutation。
- NCA: 状態shape、微分の有限値、checkpointからの学習継続。
- MJX: sensor/state shape、NaN検出、一定環境でのstep妥当性。
- DLPack: 値一致、デバイス一致、所有権、想定外のコピー、同期。
- Metrics: 圧縮・entropy・replication assayの既知入力での値/振る舞い。

### 15.2 Property-based / Differential Tests

- JAX版とCPU版の同一初期状態ステップ比較。
- BFF scalar oracle / CPU batch / CUDA をランダムテープで同条件比較。
- save -> restoreで時間・状態・RNG・archiveの整合性。
- checkpointを10回繰り返してもrun ID、親子run、artifact indexが壊れない。
- フレームdrop・接続断時もworkerのsimulation結果は同一（観測による非介入性）。

### 15.3 API / GUI / E2E

```text
1. Open localhost GUI.
2. Create Conway experiment from preset.
3. Start and inspect live state.
4. Pause, single-step, apply a brush intervention.
5. Create checkpoint and fork run.
6. Resume both and compare metrics / state.
7. Refresh browser and reconnect to active run.
8. Stop server; restart and inspect durable artifacts.
9. Export report and validate artifact references.
10. Run a CPU-only smoke test without NVIDIA GPU.
```

- 無効パラメータに400系と説明可能なerror message、許可されない操作に409/422等を返す。
- 未認証アクセス/CSRFを検証し、ファイルダウンロード経由のパストラバーサルを防ぐ。
- WebSocket 2接続以上から監視してもworker計算が二重実行されない。
- WebSocket負荷でメモリが増大し続けない（bounded queue）。

### 15.4 ベンチマーク/長期安定性

- CPU/P0 runで30分〜1時間の連続実行 + UI再接続。
- GPU/P1 runでVRAM測定、OOMフェイルセーフ、繰り返し起動のメモリリーク確認。
- 停電/kill -9を模した異常終了からのステータス整合性と部分ファイル復旧。
- 設定を変えても古いrunが破壊されないmigrationテスト。

### 15.5 対応するDefinition of Done (DoD)

- **Correct:** リファレンスや明確な不変条件で正しさを説明可能。
- **Observable:** UIで挙動・計測・イベントを見られる。
- **Recoverable:** checkpointがある時点へ復旧可能。
- **Reproducible:** seed、config、versions、backendを保存。
- **Measured:** GPU/CPU性能とVRAMが測定済み。
- **Tested:** 単体・統合テストが通過。
- **Documented:** 制約・未実装機能・参照との差分を記録。

---

### 15.6 追加研究モジュールの科学的テスト

| 対象 | テスト | 合格条件 |
|---|---|---|
| 反応ネットワーク | 解析解が既知の最小反応、収支 | 設定誤差内で一致 |
| 文字列人工化学 | 既知の反応・相補作用 | oracleの出力と一致 |
| プロトセル | 膜透過・代謝・分裂時の収支 | 保存則と親子イベント整合 |
| Swarm | 非相互作用/相互作用 | 介入が期待通り作用、経路が再現 |
| 軽量エージェント | sensor/action shape、policy無効化 | actionの整合と対照適用 |
| ルール空間探索 | Genotype→Rulesの可逆性と安定性 | 同一rule hashで再現 |
| Hybrid coupling | 単方向/双方向の量の受け渡し | unit検証・データ損失の検知 |
| Evidence Ladder | 偽検出・不十分な結果 | E0/E1をE3/E4へ昇格させない |
| Scenario Studio | YAML↔Schema往復、未知設定値 | データ破壊なし・わかるエラー |
| Sweep/assay queue | 重複POST、クラッシュ、resume | 同じ`command_id`で二重runなし |
| GPU/WS streaming | 重いGPU run中の切断・再接続 | 数値結果不変、previewのみdrop可能 |

### 15.7 科学的正しさのCI層別

1. **Tier A / < 2分:** pure Python/NumPy単体テスト、APIスキーマ、フロントLint。
2. **Tier B / CPU integration:** checkpoint/fork、Web API、SQLite migration、科学指標。
3. **Tier C / GPU nightly or manual:** JAX/Torch/MJX/DLPack、VRAM budget、参考実装比較。
4. **Tier D / long-running research:** 多seed、耐障害、アーカイブ成長、相転移検証。通常のPR CIに要求しない。

- テストの実行基準は設計案。実測が得られたらCI実行時間・GPU資源に応じて再定義。
- 失敗結果、数値精度の不一致はskipで隠さず、skip理由・対応version・問題番号を残す。

---

### 15.8 Git・ストレージ・リモート移行テスト

| テスト | 期待する結果 |
|---|---|
| `git status --short` after normal run | `runs/`, DB, caches, models and videos do not appear |
| Fake Executor disconnect / reconnect | Run ID/state/order restored; no duplicate submission |
| `source digest / config hash / lock hash` | All run manifests contain validated immutable provenance |
| Failed artifact upload / corrupted chunk | No corrupted checkpoint is published; retry works |
| SQLite while multiple remote workers report | Controller remains sole writer; migration/backups consistent |
| `local_interactive` target capability vs fake `remote_compute` | Placement respects memory constraint and explicit selection |
| CPU integer CA local vs remote | Exact or explained deterministic equivalence |
| Float Lenia on two GPU backends | Error envelope, invariant metrics, baseline difference recorded |
| Population-based stochastic ALife over seeds | Statistical comparison, not single-trajectory overclaim |
| Slurm adapter with fake sbatch/sacct | Job ID validation, completed/failed, idempotent cancel, arrays |
| Browser refresh during remote run | No duplicate simulation; view reconnects |
| Missing model/data artifact | Explicit error and retrieval guidance, not silent substitution |
| Secrets scan and file size gate | Accidental token and giant binary blocked before push |
| Restore on clean clone + clean data directory | CPU/P0 runs reproducible; external data dependencies surfaced |

常時GPUを占有するフルE2Eは標準PR CIの必須条件にしない。95%のCPU/fake testsに加え、GPU実機acceptanceをハードウェアごとに独立した検証レポートとして蓄積する。

---

## 16. 初期のハードウェア受け入れプロトコル

**この表の性能数値は目標値ではなく、報告すべき項目と成功条件を定義したもの。未知の速度は測定後に追記する。**

| チェック | コマンド/手順 | 成功条件 |
|---|---|---|
| WSL GPU可視性 | `nvidia-smi`, Python device query | 現在利用可能なGPUを正しく認識（GPUなしなら明示） |
| uv environment | `uv sync` またはprofile別sync | lockから再現可能 |
| JAX GPU | `jax.devices()`, small jit kernel | NVIDIA GPUで演算 |
| PyTorch GPU | tensor CUDA演算 | 適切なcompute kernelが実行できる |
| JAX/PyTorch共存 | DLPack -> CLIP inference | 値正常、OOMなし、ピークを報告 |
| MuJoCo backend | tiny batchでMJX-JAX/Warpそれぞれ試験 | 利用可能backendを明示 |
| BFF | reference semantics test | scalar/reference差分なし |
| Live streaming | Canvasで10分描画、タブ再読込 | workerは継続、GUI再接続 |
| Run recovery | checkpoint -> worker stop -> resume | 一貫した状態に復帰 |
| Storage | metricsとstateの書込/読込 | checksum正常 |

確認後に `docs/hardware_benchmarks.md`へ実測値・環境・失敗条件・修正理由を記載する。

---

## 17. ソフトウェアセキュリティ・信頼性

- BFFのプログラムは**独自VM内のデータとして解釈**する。ホストのshell、Pythonコード、ネイティブ実行ファイルへ変換して実行しない。
- UIから受け付ける実験設定に任意Python式、`eval`、外部URL取得、コマンド実行を許可しない。
- 作業ディレクトリ外への書込禁止。モデル/実験名を生のファイルパスに連結しない。
- 任意のpickleをブラウザ経由で読み込まない。必要な内部checkpoint形式は信頼範囲を区別。
- Workerには時間・VRAM・RAM・Disk・ジョブ数のquotaを適用。制限到達時は説明可能な失敗として記録。
- 既定でローカル限定。ユーザーが明示的に有効化するまでWAN公開、外部アカウント連携、クラウドコスト発生機能を付けない。
- APIとworkerで詳細ログを記録するが、秘密鍵・個人情報をログしない。
- `--dry-run`で実験設定の検証、必要資源概算、互換性チェックのみ実行可能。

---

### 17.1 API・プラグインの信頼境界

- `localhost`限定でも悪意のあるWebページからアクセスされる可能性を想定し、Origin検証・認証/トークン・CSRFへの対策を設ける。
- Tailscale等のprivate network経由のアクセスでも、アプリ側の権限と認証方式を検討する。
- 内部の実験YAML、DSL、テープはデータとして処理。エージェント生成の任意コードは別reviewと隔離した開発環境の対象。
- インターネットから取得した論文コード/依存パッケージをいきなり実行せず、ライセンス/maintainer/commit/依存をレビュー。
- モデル重み・checkpointにpickle形式が含まれる場合、取得元が信頼できない限りアンシリアライズしない。

---

### 17.2 Git / remote execution security

- GitHub secret scanning/push protectionとローカルpre-commit secret scanは多層防御として併用。push protectionの有無で機密情報のGit保管可否を変えない。
- `.env`, SSH private key, access token, Slurm認証トークン、社内private URL、署名付きURL、個人のホームpath等をGit追跡しない。
- 共有GPU・HPCにGitHub Actions self-hosted runnerを配置する場合、public PRの任意コード実行によるホスト侵害リスクがある。原則として保護されたprivate repoと明示承認済みbranch・権限制限でのみ運用。
- リモート実験は利用者が認可されたノード/accountで実行。ジョブ投入、費用発生、データ移転、外部ネットワーク公開の際には個別の権限境界を守る。
- 研究データのライセンスと機密区分をartifact manifestへ記録。`public`/`private`/`restricted`を意図せず混在させない。

---

## 18. 開発エージェントへの実行契約

### 18.1 作業開始時

1. `AGENTS.md`と本仕様を読む。既存ファイルを破壊せず実装済み/未実装を調査する。
2. `git status --short`を確認し、branch・既存差分・秘密情報・未追跡の実験データを保護する。
3. Phase 0とPhase 1の具体的な実装計画を作る。重要な決定を`docs/decisions/`にADRとして記録する。
4. 依存関係とALife論文・reference実装は導入直前に公式資料とリリースノートを確認する。
5. 最初から全基質を実装せず、GUIで操作・保存・復元できる単一基質を完成させる。

### 18.2 科学的・実装上の禁止事項

- 実測せずにGPU性能、ゼロコピー、スループット、GPU間での厳密再現を断言しない。
- reference modeを無断で近似版へ変更しない。外部探索器のrewardをBFF基質の内部ルールに混ぜない。
- 相転移・複製・協調らしい映像だけで、新たな生命現象が確認されたと報告しない。
- シミュレーションを変更する観測器・レンダラ・監視イベントを隠して導入しない。
- 資源不足を隠すためにworld規模、人口、seed、数値精度、探索予算を自動変更しない。
- GUI操作で任意Python/shellコードを評価しない。リモート資源への接続や課金を無断で行わない。
- APIプロセスで重量級GPU runtimeをロードしない。GUIの遅さで数値計算を停止しない。
- 実装していないAPIや拡張モードを実装済みとして報告しない。
- 既存run、保存データ、checkpoint、Git履歴、環境設定を無断で破壊しない。

### 18.3 完了の判定と報告形式

機能ごとに `backend → API → GUI/CLI → tests → docs → measured resources` を完遂する。各作業の終了時に以下を記載する。

```text
Implemented: <features and source paths>
Verified: <exact commands, OS/backend and results>
Measured: <peak VRAM, throughput or 'not measured'>
Scientific fidelity: <reference commit, deviations, uncertainty>
Artifacts: <run IDs, storage paths, checksum status>
Security: <secret scan, permission checks, outstanding risks>
Known limitations: <partial and unimplemented items>
Next slice: <smallest demonstrable end-to-end milestone>
```

### 18.4 最初の実装依頼

`AGENTS.md`と本仕様をマスター契約として、環境調査・ADR・開発基盤・Conway CAのWeb垂直スライスを実装する。起動後、ブラウザから実験作成、ライブ観察、停止、一歩進める、brushによる介入、checkpoint、再起動後の復旧まで動作させる。CLI、REST/WebSocket、run manifest、SQLite、CPU oracle、参照テスト、ローカルGPU診断を含める。成功したコマンドと実測値だけを報告し、実装していない機能は明確に列挙する。

---

## 19. 参考文献・コード・一次ソース

以下は研究・実装の参照先。エージェントは使用前にリリース状況・ライセンス・API・参照コードの変更を再確認すること。

### ALife基礎・主要基質

- Sakana AI, **ASAL: Automating the Search for Artificial Life with Foundation Models** (2024): <https://github.com/SakanaAI/asal> / <https://asal.sakana.ai/>
- Plantec et al., **Flow-Lenia**: <https://github.com/erwanplantec/FlowLenia> / <https://arxiv.org/abs/2212.07906> / <https://deepmind.google/research/publications/106327/>
- Agüera y Arcas et al., **Computational Life**: <https://arxiv.org/abs/2406.19108> / **CuBFF author reference** <https://github.com/paradigms-of-intelligence/cubff>
- Mordvintsev et al., **Growing Neural Cellular Automata**: <https://distill.pub/2020/growing-ca/>
- Sakana AI, **Petri Dish Neural Cellular Automata**: <https://pub.sakana.ai/pdnca/> / <https://github.com/SakanaAI/petri-dish-nca>
- Sakana AI, **Digital Ecosystems: Interactive Multi-Agent Neural Cellular Automata**: <https://pub.sakana.ai/digital-ecosystem/> / <https://github.com/SakanaAI/digital-ecosystem>
- Artificial Life Institute, **Microcosmos: Reimagining Artificial Life for the GPU Era**: <https://alife.institute/en/blog/microcosmos-release/> / <https://github.com/alife-institute/microcosmos>
- **Avida Digital Evolution Platform**: <https://alife.org/encyclopedia/digital-evolution/avida/> / <https://github.com/devosoft/avida>
- **MODES Toolbox — Measurements of Open-ended Dynamics in Evolving Systems**: <https://pubmed.ncbi.nlm.nih.gov/30933626/> / <https://github.com/emilydolson/MODES-toolbox-paper>

### 進化・物理・GPU

- evosax: <https://github.com/RobertTLange/evosax> / <https://github.com/RobertTLange/evosax/releases>
- QDax: <https://qdax.readthedocs.io/en/latest/> / <https://github.com/adaptive-intelligent-robotics/QDax>
- MuJoCo MJX (JAX / Warpの区別): <https://mujoco.readthedocs.io/en/latest/mjx.html>
- MuJoCo Warp: <https://mujoco.readthedocs.io/en/latest/mjwarp/index.html>
- JAX installation (CUDA 12 / CUDA 13): <https://docs.jax.dev/en/latest/installation.html>
- JAX GPU memory allocation: <https://docs.jax.dev/en/latest/gpu_memory_allocation.html>
- PyTorch DLPack interop: <https://docs.pytorch.org/docs/stable/generated/torch.from_dlpack.html>

### Web / Data / GUI

- FastAPI WebSocket: <https://fastapi.tiangolo.com/advanced/websockets/>
- PixiJS rendering: <https://pixijs.com/guides/components/renderers/>
- Zarr-Python array specification: <https://zarr.readthedocs.io/en/latest/user-guide/arrays/>
- React: <https://react.dev/>
- Vite: <https://vite.dev/>
- Three.js: <https://threejs.org/>

### 研究・実装出典（発展領域）

- **Artificial Life encyclopedia（分野全体・Artificial Chemistry / Digital Evolution）:** <https://alife.org/encyclopedia/introduction/artificial-life/> / <https://alife.org/encyclopedia/digital-evolution/introduction-to-digital-evolution/>
- **Stringmol系の分子相互依存の背景:** 上記ALife Encyclopedia Digital Evolution章。特定の原論文・reference実装の特定は導入時に別途行うこと。
- **Flow-Lenia curiosity-driven scientist** (2025): <https://arxiv.org/abs/2505.15998>
- **BFF: Simple explanations for complex phenomena** (2026, 反証・比較上の検討候補): <https://arxiv.org/abs/2607.01483>
- **Open-ended evolution** (ALife encyclopedia): <https://alife.org/encyclopedia/introduction/open-ended-evolution/>
- **MODES Toolbox**: <https://pubmed.ncbi.nlm.nih.gov/30933626/>
- **uv workspaces / dependency conflict**: <https://docs.astral.sh/uv/concepts/projects/workspaces/> / <https://docs.astral.sh/uv/concepts/projects/dependencies/>
- **evosax package versions (PyPI)**: <https://pypi.org/project/evosax/>
- **Digital Ecosystems (Sakana AI)**: <https://pub.sakana.ai/digital-ecosystem/>
- **Microcosmos**: <https://alife.institute/en/blog/microcosmos-release/>
- **QDax MAP-Elites API**: <https://qdax.readthedocs.io/en/latest/api_documentation/core/map_elites/>

**新規研究を取り込むときの注意:** これらの文献は動機と参考実装の出発点であり、本設計中の独自追加機能の存在・性能・正しさを保証するものではない。

### 参考UI/アプローチ（実装を無批判に流用しない）

- **LIFE — Digital Evolution Observatory**: <https://github.com/dominikvytisk/life-simulation> （多機能なブラウザ上の人工生命観測UIの設計例）
- **Computational Life Lab**: <https://github.com/ioannisantoniadis/computational-life-lab> （BFFの観測・保存・系統の設計例。ただし科学的な正本はCuBFF）
- **Growing NCA WebUI**: <https://github.com/semvdn/Growing-NCA-WebUI> （標的描画・学習・損傷介入の体験例）

---

### 実行基盤・Git管理の一次資料

- GitHub: Large files on GitHub — <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>
- GitHub: Git LFS and billing — <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage> / <https://docs.github.com/en/billing/concepts/product-billing/git-lfs>
- GitHub: Secret scanning and history-cleaning guidance — <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository>
- GitHub: Secure use of Actions / self-hosted runner risks — <https://docs.github.com/en/actions/reference/security/secure-use>
- DVC: Remote storage and data management — <https://doc.dvc.org/user-guide/data-management/remote-storage> / <https://doc.dvc.org/start>
- Slurm: sbatch / sacct / job arrays — <https://slurm.schedmd.com/sbatch.html> / <https://slurm.schedmd.com/sacct.html> / <https://slurm.schedmd.com/job_array.html>
- JAX: Installation / GPU memory allocation — <https://docs.jax.dev/en/latest/installation.html> / <https://docs.jax.dev/en/latest/gpu_memory_allocation.html>

---

## 20. プラットフォームの受け入れ条件

### 20.1 基本機能の合格条件

- Web GUIから基準世界を起動・操作・停止・介入・保存・復旧できる。
- 互いに異なる複数の基質を共通Run Manifest、CLI、ArtifactStoreで管理できる。
- CPU oracleとGPU高速化実装を同じ科学条件で検証する仕組みがある。
- checkpointから分岐し、介入群・対照群を比較して結果を保存できる。
- 複数seedのsweep、条件比較、統計レポートが実行できる。
- 発見候補の根拠と否定的結果をEvidence Ladderに従い記録できる。
- 実行先の能力をprobeでき、資源不足や互換性不足を明示的に拒否する。
- Git cloneによってソースと小さなテストが復元でき、大きな実験成果物は別のArtifactStoreで整合性検証できる。
- 少なくとも1つのGPU計算基質について、実際の環境でpeak VRAMとend-to-end throughputを記録する。

### 20.2 科学的トラックの選定

研究テーマは独創性、測定可能性、reference実装の確かさ、計算効率、介入・比較の価値、保守負荷で評価する。どの分野にも固定的な特権を与えず、独立モジュールとして追加・停止・入替できるようにする。

### 20.3 必ず維持する体験

1. ブラウザから人工世界を観察し、操作できる。
2. 実験条件・結果・チェックポイントが失われず、再生できる。
3. 分岐・介入・対照実験によって因果的な仮説を検証できる。
4. 自動探索で候補を発見し、その主張の根拠と限界を追跡できる。
5. 同じ研究基盤をローカル対話用GPU、より高性能な研究GPU、任意の計算クラスタへ拡張できる。ただし実測と能力検査が完了するまで性能・機能を保証しない。

**最終原則:** 多くのALife基質を実装すること自体を完成条件としない。観察、進化、介入、分岐比較、科学的検証、データ再現性を、どの基質にも再利用できる形で成立させる。
