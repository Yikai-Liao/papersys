# Design Document

## Overview

本设计将 2025-11-09 recommend 管线修复正式化：
1. HuggingFace 数据按分片拉取并以 LazyFrame 使用，封装在 `papersys/data_sources.py` 的辅助函数中，避免 CLI 直接处理下载逻辑。
2. GitStore 负责外部仓库生命周期与 CSV 偏好文件管理，延迟初始化 SummaryStore 并去除 push 推荐快照的副作用。
3. `Recommender` 全面接受 LazyFrame，在 join 后一次 collect，配合 CLI 的严格参数校验和输出管控，形成无副作用的推理流程。

## Steering Document Alignment

- `.spec-workflow/steering` 下暂无 product/tech/structure 文档，本设计沿用代码库现有约定：中文日志、Typer CLI、Pydantic 配置等。

## Code Reuse Analysis

### Existing Components to Leverage
- **`papersys/data_sources.py`**：复用 `load_metadata/load_embeddings` 入口，在其上抽象 `_snapshot_shards`、`_load_sharded_dataset`、`_to_lazy`。
- **`papersys/storage/git_store.py`**：继续使用 Git clone/pull 模式，只增强校验与 CSV 逻辑。
- **`papersys/recommend/recommender.py`**：延续训练/预测实现，新增 LazyFrame 支持与 streaming collect。
- **`papersys/cli/recommend_cmd.py`**： Typer CLI 结构保持不变，仅强化参数和输出。

### Integration Points
- **HuggingFace dataset**：通过 snapshot API 下载 `metadata_*`/`embedding_*` parquet，缓存到 `data/hf_cache/<repo>`，Polars scan 读取。
- **外部 Git 仓库**：GitStore 负责 clone/fetch/pull、CSV 偏好读写以及 `summary_store` 的 `existing_ids`。
- **Recommender ← CLI**：CLI 把 LazyFrame + 偏好 DF + 已总结 ID 注入到 Recommender，后者完成训练/预测并返回 `RecommendationResult`。

## Architecture

```
Typer CLI (recommend_cmd)
  ├─ load_config(AppConfig)
  ├─ GitStore.ensure_local_copy()
  ├─ load_metadata()/load_embeddings()  ← LazyFrame
  ├─ GitStore.load_preferences() + summary_store.existing_ids()
  ├─ Recommender.fit(categories)
  └─ Recommender.predict(...)
        └─ adaptive_sample + logistic regression
```

- **data_sources** 只负责下载缓存与 LazyFrame 创建；
- **GitStore** 专注 Git + CSV；
- **Recommender** 聚合元数据/向量 + 偏好，再筛除已处理 ID、过滤非法向量；
- **CLI** 负责参数解析、日志、输出（JSONL/CSV）。

## Components and Interfaces

### `_repo_cache_dir(config: HuggingFaceDatasetConfig) -> Path`
- 缓存目录以 `hf_repo` 派生，保证 metadata/embedding 共用逻辑。

### `_snapshot_shards(config) -> str`
- 调用 `snapshot_download` 下载 `shard_prefix_*.parquet`，若无匹配文件则报错。
- 返回 glob pattern，供 Polars scan。

### `_load_sharded_dataset(config, columns=None, lazy=False)`
- 统一 metadata/embedding 读取逻辑，支持列裁剪和 LazyFrame。

### `GitStore`
- `ensure_local_copy`：若目录存在但非 git，清理后重新 clone。
- `load_preferences/save_preferences`：改用 CSV，以 Polars DataFrame 输出，自动补列。
- `summary_store`：惰性属性，首次访问才初始化 `SummaryStore`。

### `Recommender`
- 构造函数使用 `_to_lazy` 将 DataFrame 转 LazyFrame。
- `_prepare_dataset` 只在 join 后 streaming collect。
- 训练前校验正样本、背景样本、偏好数据；预测阶段过滤 `_preference_ids` 与 `_excluded_ids`。

### CLI `recommend`
- `_parse_date` 继续校验日期格式。
- `effective_last_n`：优先参数，否则用配置。
- 输出：仅允许 `.jsonl/.ndjson/.csv`，无输出则不写文件。

## Data Models

```
HuggingFace metadata shard:
  id: str
  title: str
  authors: list[str]
  categories: list[str]
  update_date: date
  ...

Embedding shard:
  id: str
  embedding: list[float] (长度 dim)

Preferences CSV:
  id,preference
  0000.12345,like
```

`RecommendationResult.frame` 包含 join 后字段 + `score` + `show`。CLI 将 `score` round(6)、`show`==1 的记录输出。

## Error Handling

1. **分片缺失**：`_snapshot_shards` 直接 `FileNotFoundError` + logger.error，包含仓库与模式。
2. **Git 仓库损坏**：`ensure_local_copy` 清理后重新 clone，并记录 warning。
3. **偏好/正样本缺失**：`Recommender.fit` 抛 `ValueError`，CLI 捕捉并 `typer.Exit(1)`。
4. **CLI 参数非法**：日期解析或 `--last-n-days` 与 `--start/--end` 冲突时抛 `BadParameter`；输出后缀非法同理。
5. **预测无结果**：多个阶段记录 warning（无候选、全部排除、NaN embedding）。

## Testing Strategy

### Unit Testing
- `data_sources`
  - Mock `snapshot_download`，断言允许模式、空分片抛错。
  - 验证 `columns` 参数只返回指定列。
- `GitStore`
  - 使用临时目录模拟非 git 路径，确保清理 + clone。
  - CSV 读写：无文件返回空 DF，保存时补列。
- `Recommender`
  - 构造小型 LazyFrame，验证 `_to_lazy` 行为与 streaming collect。
  - 测试 `_filter_nan_embeddings` 清理 NaN。

### Integration Testing
- CLI 在本地小数据集下运行：
  - 成功路径：检查日志含 lazy 信息、输出文件格式正确。
  - 参数冲突/非法输出：捕获 `BadParameter`。
  - 无推荐：确保 warning 并无异常退出。

### End-to-End Testing
- 结合真实 GitStore 仓库 + 伪造 HF 分片：模拟 `uv run papersys recommend` 全流程，包含 clone、lazy join、训练、输出。
