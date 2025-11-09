# Tasks Document

- [x] 1. 建立 HuggingFace 分片加载与 LazyFrame 基础
  - File: papersys/data_sources.py
  - 实现 `_repo_cache_dir`、`_snapshot_shards`、`_load_sharded_dataset`、`load_metadata`、`load_embeddings` 的分片 + LazyFrame 逻辑，替换旧的单文件下载流程。
  - Purpose: 让 metadata/embeddings 支持分片缓存与惰性读取，满足 Requirement 1。
  - _Leverage: huggingface_hub.snapshot_download, polars.scan_parquet_
  - _Requirements: 1_
  - _Prompt: Role: 数据工程老司机，Task: 把 HF 分片加载改成 snapshot + LazyFrame，保持列裁剪与 streaming collect 特性，文件 papersys/data_sources.py，遵循 Requirement 1。_

- [x] 2. 升级 GitStore 仓库与偏好 CSV 管理
  - File: papersys/storage/git_store.py
  - 清理非 git 目录再 clone、加载/保存偏好 CSV（补列、Polars DataFrame），`summary_store` 延迟实例化，删除推荐快照相关字段。
  - Purpose: 保障 Git 仓库一致性并实现 CSV 偏好读写，满足 Requirement 2。
  - _Leverage: SummaryStore, pl.read_csv_
  - _Requirements: 2_
  - _Prompt: Role: DevOps + 数据管道开发者，Task: 强化 GitStore，确保 CSV 偏好与 summary ids 逻辑健壮，文件 papersys/storage/git_store.py，对标 Requirement 2。_

- [x] 3. 让 Recommender 支持 LazyFrame + streaming collect
  - File: papersys/recommend/recommender.py
  - 在构造函数中应用 `_to_lazy`，训练/预测阶段保持 DataFrame/LazyFrame 兼容；`_prepare_dataset` join 后 streaming collect，确保过滤 `_preference_ids`/`_excluded_ids`；保留 NaN 过滤。
  - Purpose: 使训练/预测在大数据集下稳定运行并避免重复推荐，满足 Requirement 1/2。
  - _Leverage: adaptive_sample, train_model_
  - _Requirements: 1,2_
  - _Prompt: Role: Recommender 系统工程师，Task: 改造 Recommender 以支持 LazyFrame、一次 collect、排除偏好/摘要 ID，文件 papersys/recommend/recommender.py。_

- [x] 4. 修订 recommend CLI 流程与输出
  - File: papersys/cli/recommend_cmd.py
  - 记录 LazyFrame 日志信息，引入 `effective_last_n`，禁止 `--last-n-days` 与 `--start/--end` 混用，输出只允许 `.jsonl/.ndjson/.csv`，移除自动快照推送。
  - Purpose: 提供明确参数语义与可控输出，满足 Requirement 3。
  - _Leverage: typer.BadParameter, GitStore_
  - _Requirements: 3_
  - _Prompt: Role: CLI/后端工程师，Task: 强化 recommend 命令参数与输出逻辑，文件 papersys/cli/recommend_cmd.py，满足 Requirement 3。_

- [x] 5. 更新配置结构
  - File: config.toml, papersys/config.py
  - 将 metadata/embedding 字段改为 `shard_prefix`，移除旧的 `parquet` 字段；保持其他配置项不变。
  - Purpose: 让配置与分片逻辑匹配，满足 Requirement 1。
  - _Leverage: Pydantic BaseModel_
  - _Requirements: 1_
  - _Prompt: Role: 配置管理工程师，Task: 调整配置结构支持分片前缀，文件 config.toml & papersys/config.py。_

- [x] 6. 更新 README/文档（若需要）
  - File: README.md or docs/
  - 如 CLI 用法/配置示例需同步，则补充文档（根据需求确定）。
  - Purpose: 保证使用者了解新配置与 CLI 行为。
  - _Requirements: 1,2,3_
  - _Prompt: Role: 文档维护者，Task: 检查文档是否需要更新，若有则补充。_
