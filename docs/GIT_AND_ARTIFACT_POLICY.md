# Git・データ・実験成果物の管理規約

## 目的と原則

Gitはコード・実験プロトコル・設定の履歴を追う。実験で増殖する状態・動画・モデル重み・イベント時系列の保管庫ではない。研究上の再現性は、Git commitと実験manifestおよび独立したArtifactStoreの対応関係で担保する。

| 対象 | 管理先 | 代表例 |
|---|---|---|
| ソース・仕様・Web GUI・スキーマ・テスト | Git | `.py`, `.ts`, `.tsx`, `.md`, migrations, CI |
| 科学条件・実験プリセット・実行profile・依存lock | Git | YAML, `uv.lock`, `package-lock.json` |
| 小さな固定reference fixture・権利を確認した静的素材 | Git（サイズ審査） | baseline JSON/PNG |
| 再利用する大型の固定データ・重み | DVC / data registry + 外部保存先 | curated dataset, benchmark, reference checkpoint |
| checkpoint、Zarr/Parquet、大規模ログ・動画・世代系統 | ArtifactStore | run-specific numerical state, MP4 |
| 実行中DB・WAL、コンパイルキャッシュ、一時ファイル | ローカルruntime | `.alife/`, caches, tmp |
| トークン、秘密鍵、資格情報、署名URL | OS秘密情報管理 | `.env`, SSH key, remote token |

## 必須ルール

1. 1 runに永続`run_id`を割り当て、`git_commit`, `git_dirty`, `source_digest`, `spec_digest`, `execution_digest`, `seed`, `backend`, `input_digests`, `artifact_checksums`, `target_capabilities`を残す。
2. `ExperimentSpec`を科学的正本、`ExecutionProfile`を資源・配置の正本として独立hashで保存。大規模環境向けにworld sizeやpopulationを変更したら、別科学条件・別runとして扱う。
3. `runs/`, `.alife/`, `data/raw/`, `models/weights/`, `scratch/` 等を`.gitignore`。ログ、動画、checkpoint、稼働SQLiteをcommitしない。
4. DVCは共有される不変の大容量研究資産にだけ導入。DVCポインタやlockをGitに置き、実体は別ストレージ。個々のrunをすべてDVC管理する必要はない。
5. Git LFSは安定した少数の大容量配布素材のみに例外使用。ホスティング側の保存・転送課金とファイル上限を都度確認。
6. ArtifactStoreは `stage → sha256 verify → atomic finalize → manifest/index publish`。部分アップロードや欠損を成功状態にしない。
7. 重要checkpointは定期的に独立ストレージへバックアップし、空環境へのrestoreを試験する。Gitだけでは実験生成物を復元できない。
8. `git status --short`と`git diff --cached --stat`を確認し、staged secret scan/大容量ファイルgateを通してからcommitする。既に追跡された機密情報は`.gitignore`では保護されない。
9. 外部コード・実装・研究素材・学習済み重みのlicenseと出典を記録し、再配布可能なものだけvendorする。
10. CIの通常ジョブはCPUで完結。共有GPUのself-hosted runnerは未信頼PRや無権限ユーザーに開放しない。
11. Artifactの削除は`pinned`保護・参照検査・dry-run・明示承認を経る。大規模なGCを黙って実行しない。
12. dirty working treeから研究runを動かした場合は、安全なsource snapshot digestを保存し、重要な再現実験ではclean commitを推奨する。

## 実験manifestの最低限の論理形

```json
{
  "schema_version": 1,
  "run_id": "run_example",
  "source": {"git_commit": "<commit-sha>", "git_dirty": false, "source_sha256": "<sha256>"},
  "science": {"experiment_spec_sha256": "<sha256>", "seed": 42, "numeric_policy": "exact_integer"},
  "execution": {"target_id": "local", "profile_sha256": "<sha256>", "backend": "numpy"},
  "artifacts": [{"kind": "checkpoint", "sha256": "<sha256>", "storage_uri": "artifact://run_example/checkpoints/step_1000.npz"}]
}
```

このJSONは論理スキーマ例であり、`<...>`は実値で置き換える。secretや一時的な署名URLを保存しない。

## ローカル・研究用GPU・クラスタ間の移行

- 実行先は`local_interactive`, `remote_compute`, `multi_gpu`, `cluster_batch`などのロールで表現する。容量値やGPU型番をロール名に埋め込まない。
- リモート実行前にprobeと権限・必要backend・入力hash・source digestを照合する。リモートで無条件の`git pull main`をしない。
- ControllerのみがメタDBを書き込み、workerはheartbeat/成果物と署名済みメッセージで通信する。Controllerから認証済み経路で回収する。
- batch環境はlive pauseができる前提を置かず、対応する基質はcheckpoint-and-exit/requeueを使う。
- 同一科学条件のハードウェア移行は`bitwise`/`numerically_close`/`statistically_consistent`を適切に選択して比較する。
- 未確認のGPU speed-up/VRAM余裕/スケジューラ待ち時間/課金予測を事実として表示しない。

## 公式ドキュメント

- GitHub large files: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- GitHub LFS: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage
- DVC remote storage: https://doc.dvc.org/user-guide/data-management/remote-storage
- Slurm: https://slurm.schedmd.com/sbatch.html
- JAX GPU memory: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
