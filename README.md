# ALife Lab

**人工生命の研究・実験・探索をブラウザから操作するためのソフトウェア開発プロジェクト。**

仕様・開発指示・設定テンプレートに加え、Phase 0–1（CPU Conwayの垂直スライス）の実装を含みます。GPUエンジンや他の基質は未実装です。

## 開発に必要な文書

- [`AGENTS.md`](AGENTS.md) — Claude Code / Codex等に与える実行契約。
- [`docs/ALIFE_LAB_SPEC.md`](docs/ALIFE_LAB_SPEC.md) — 科学・計算基質・Web UI・API・実装ロードマップのマスター仕様。
- [`docs/GIT_AND_ARTIFACT_POLICY.md`](docs/GIT_AND_ARTIFACT_POLICY.md) — Git、チェックポイント、巨大データ、バックアップの境界。
- [`configs/scenarios/`](configs/scenarios/) — `ExperimentSpec`の参考設定。
- [`configs/execution_profiles/`](configs/execution_profiles/) — 実行先のロール別profile例。**容量固定のpresetではない**。

## 現在の実装状況（Phase 0–1 ＋ Phase 2 第1〜第3スライス＋A・B、2026-11-18）

| 区分 | 内容 |
|---|---|
| 実装済み | `alife doctor`／`alife serve`、スイープ（条件×seed、seed単位の統計：効果量・ブートストラップCI・順列検定・Holm補正、GUI・API・CLI）、サーバー強制終了からの復旧（staging隔離・未索引checkpointの報告）、バックアップ／検証／復元（`alife backup`）、Fork／ペア分岐（対照・処置）／状態比較（Hamming距離）、タイムライン（決定論的リプレイ・リプレイ検証・任意tickからの分岐）、FastAPI + SQLite（WAL、Controller単一ライター）、LocalExecutor（spawnワーカー）＋FakeExecutor、ArtifactStore（local_fs、sha256＋atomic publish）、CPU Conway（NumPy＋スカラーoracle）、Boids（cell-list近傍探索＋O(N²) oracle、exploratoryのみ）、React Web GUI（作成・観察・再生/一時停止・1 tick・ブラシ・チェックポイント・再起動後の復元） |
| 部分実装 | CLI（`run`, `runs list/inspect/verify/fork/replay-verify`, `compare`, `sweep`, `backup`, `targets`, `presets`, `substrates`）、メトリクスはJSONL（Parquetは未導入） |
| 計画のみ | Particle Life、Inspector、remote/multi-GPU/cluster executor、Lenia/BFF等のPhase 3以降の基質、GPU backend |

## 起動方法

```bash
uv sync
npm --prefix frontend ci && npm --prefix frontend run build   # Web GUIのビルド（初回・更新時）
uv run alife doctor
uv run alife serve --host 127.0.0.1 --port 8000              # http://127.0.0.1:8000/
```

テスト：

```bash
uv run pytest -q                       # unit / reference / property / integration / API
npm --prefix frontend test             # frontend unit
ALIFE_E2E_CHROMIUM=<chromium path> npm --prefix frontend run e2e   # ブラウザE2E（サーバー再起動を含む）
uv run python scripts/check_staged.py  # commit前のsecret/大容量ファイル検査
```

設計判断は [`docs/decisions/`](docs/decisions/)、構成は [`docs/architecture.md`](docs/architecture.md)、API は [`docs/api.md`](docs/api.md)、実測値は [`docs/hardware_benchmarks.md`](docs/hardware_benchmarks.md) を参照。

## 設計上の基礎ルール

- 科学的条件と実行環境の指定を分離する。
- CPU開発、ローカル対話用GPU、研究用の大規模GPU、必要なら計算クラスタを、同じランナー・保存契約で扱う。
- 人工生命基質の優先度は実装依存関係によって決め、特定の研究領域を固定的に優先しない。
- 科学的な主張には参照・定量指標・再現試験・否定的対照を対応付ける。
- コードと設定はGit、実験によって生じる大量データはArtifactStoreに保管する。

## 開発エージェントへの開始指示

まず`AGENTS.md`とマスター仕様を読み、技術的互換性を公式一次情報で調査し、Phase 0から実装すること。初期段階で全ての研究モジュールを構築しようとせず、Web GUI・API・実験保存・CPU referenceが完成した基準系を作ること。
