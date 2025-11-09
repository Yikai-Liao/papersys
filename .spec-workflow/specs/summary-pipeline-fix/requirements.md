# Requirements Document

## Introduction

摘要管线需要重新梳理存储策略：当前 JSONL 仍按出版年份分片且没有一次性快照，导致后续推荐/分析难以定位最近一次摘要结果，也无法按月滚动归档。该特性旨在确保摘要结果以“摘要发生月份”进行 jsonl 分片，并额外生成 `last.jsonl` 快照，方便下游增量处理与人工审计。

## Alignment with Product Vision

papersys 作为科研情报工作流，需要保证数据资产有序沉淀。按月份分片 + 最新快照可以让推荐、统计、外部同步脚本快速消费最新摘要，同时保持 Git 仓库体积可控，与产品“自动化、可审计的数据流水线”愿景保持一致。

## Requirements

### Requirement 1

**User Story:** 作为摘要管线的维护者，我希望摘要结果可以按摘要发生的月份（例如 2024-11）归档到对应的 JSONL 文件，这样仓库可以自然按时间滚动，定位问题也更高效。

#### Acceptance Criteria

1. WHEN 摘要命令写入 N 条结果 THEN 系统 SHALL 将它们写入 `summaries/<YYYY-MM>.jsonl`，其中 `<YYYY-MM>` 来源于 UTC 语义下的 `SUMMARY_DATE`（若无则使用 fallback）。
2. IF 记录缺少 `SUMMARY_DATE` THEN 系统 SHALL 回退到 `PUBLISH_DATE` 或 `UPDATE_DATE` 以推导月份，否则写入 `summaries/unknown_month.jsonl`。
3. WHEN 不同批次写入同一个月份文件 THEN 系统 SHALL 直接附加/覆盖记录，允许同一论文存在多个摘要版本，并按批次写入顺序排序。

### Requirement 2

**User Story:** 作为下游消费者，我想要一个始终包含“最后一次摘要批次”的 JSONL 文件，方便快速加载最新结果或进行增量校验。

#### Acceptance Criteria

1. WHEN 摘要命令完成批量 upsert THEN 系统 SHALL 生成/覆盖 `summaries/last.jsonl`，其内容仅包含本次批次的记录，顺序与传入列表一致。
2. IF `last.jsonl` 存在历史损坏或字段缺失 THEN 摘要命令 SHALL 覆盖写入最新合法 JSONL，以保证后续处理不会因解析错误而终止。

## Non-Functional Requirements

### Code Architecture and Modularity
- Summary 存储逻辑需封装在 `SummaryStore`，CLI 只调用清晰接口。
- `last.jsonl` 写入逻辑与月度分片写入逻辑分离，便于单元测试。

### Performance
- 每次 upsert 仅需触达本次涉及的月份文件及 `last.jsonl`，写入顺序与批次顺序一致以便按时间回溯。

### Security
- 不在非必要位置记录模型 API key，继续沿用现有 env 读取方式。

### Reliability
- 对无法解析的 JSON 行继续记录 warning 并跳过，保证迭代式写入不崩溃。

### Usability
- CLI 成功后需明确日志提示月度分片和 `last.jsonl` 路径，方便排障。
