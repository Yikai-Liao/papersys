# Requirements Document

## Introduction

为了解决 recommend 管线在 2025-11-09 的紧急修复中暴露出的数据体量、偏好存储以及导出方式混乱的问题，本规范要求为该管线补齐正式需求描述。目标是让大规模 HuggingFace 分片数据能够以惰性方式被加载，偏好与摘要状态统一由 Git 仓库托管，并让 CLI 推理流程在导出与时间窗控制上更加可控、可观测。

## Alignment with Product Vision

papersys 的核心是把高质量论文筛选、总结后送往 Notion/下游消费者。稳定、低资源占用且可重复的推荐结果是整个内容生产流水线的入口。本需求把推荐环节从一次性脚本修修补补拉回到受控、可审计的流程，确保与产品“稳定获取值得总结的论文”这一愿景对齐。

## Requirements

### Requirement 1

**User Story:** 作为维护推荐任务的运维（ops engineer），我希望一次命令即可下载 HuggingFace 仓库内按年份切分的 metadata/embedding 分片并以惰性方式参与训练，这样服务器不会因为一次性加载巨大 parquet 而 OOM。

#### Acceptance Criteria

1. WHEN 仓库包含 `metadata_*.parquet` 或 `embedding_*.parquet` 时 THEN 系统 SHALL 通过 `snapshot_download` 把全部匹配分片缓存到 `data/hf_cache/<repo>`，若无匹配分片 MUST 抛出带具体仓库名的错误日志。
2. IF recommend CLI 需要 metadata/embedding THEN 系统 SHALL 通过 Polars lazy scan 返回 `LazyFrame`，只在真正训练前进行 streaming collect，且 `load_embeddings` MUST 支持列裁剪。
3. WHEN 数据源调用成功 THEN CLI SHALL 记录“是否 lazy、行数信息”等诊断日志，便于在未 collect 前判断数据规模。

### Requirement 2

**User Story:** 作为人工标注者，我希望偏好与已出摘要记录全部托管在外部 Git 仓库并用 CSV 管理，以便换机器或批量编辑时不会受到 JSONL 顺序/格式问题影响，同时推荐阶段能够自动跳过已经总结过的论文。

#### Acceptance Criteria

1. WHEN GitStore 初始化时 THEN 系统 SHALL 校验本地路径是否真的是 Git 仓库，若不是则清理后重新 clone 指定 `repo_url` + `branch`。
2. IF 偏好文件不存在 THEN GitStore SHALL 返回带 `id`、`preference` 两列的空 Polars DataFrame 并写 warning；否则 SHALL 以 CSV 读写并保持列顺序，保存时自动补全缺失列。
3. WHEN recommend CLI 收集候选 ID THEN 系统 SHALL 通过 `git_store.summary_store.existing_ids()` 与偏好 ID 组成排除集合，防止推荐重复内容。

### Requirement 3

**User Story:** 作为日常执行 recommend 命令的运营，我希望推理阶段有确定的时间窗口策略、导出格式受控且不再强行写入数据仓库，这样可以按需生成 CSV/JSONL 文件并在无输出需求时保持数据仓库干净。

#### Acceptance Criteria

1. IF CLI 调用未传 `--last-n-days` THEN 系统 SHALL 使用配置 `recommend.predict.last_n_days`；若传入则优先使用显式值，且禁止与 `--start/--end` 混用。
2. WHEN `--output` 被指定 THEN 系统 SHALL 仅接受 `.jsonl/.ndjson/.csv` 三种后缀，其他格式直接抛出 `BadParameter` 并指明支持范围。
3. WHEN CLI 运行成功 THEN 系统 SHALL 仅在用户提供 `--output` 时写文件，不得再自动在 GitStore 下生成 `recommendations-*.parquet` 并推送。
4. WHEN 模型挑选推荐集合 THEN CLI SHALL 把 show==1 的记录按分数降序输出、默认展示前 20 行 `ID/TITLE/SCORE`，空结果需要清晰 warning。

## Non-Functional Requirements

### Code Architecture and Modularity
- Loader、GitStore、Recommender、CLI 四层职责清晰：数据下载与缓存逻辑不能侵入 CLI；GitStore 只关注仓库同步与 CSV；推理类只关心数据帧。
- 新增的惰性处理工具 `_to_lazy`、`_load_sharded_dataset` 必须独立函数，方便复用与测试。
- CLI 必须保持 I/O 纯度，不得隐式产生副作用文件。

### Performance
- 读取 HuggingFace 数据时需使用 streaming collect 避免把全部分片一次性载入内存。
- 仅在 join metadata 与 embedding 后进行一次 collect，保证推荐阶段可在单机内完成。

### Security
- Git 仓库 clone 需支持 `PAPERSYS_DATA_TOKEN` 注入，但不得在日志中打印 token。
- 未授权时使用只读 https 链接，避免将凭证写入配置。

### Reliability
- 缺失分片、空偏好、或无正样本等场景必须通过 `logger.warning/error` 并抛出合理异常（typer.Exit 或 ValueError），避免 silent failure。
- Git 操作失败时要保留命令 stderr 供排查。

### Usability
- CLI 错误信息保持中文提示、明确问题（格式、无分类、无推荐等），并在成功路径输出行数与导出路径。
- 日志要指出 lazy/collect 状态，方便日常巡检。
