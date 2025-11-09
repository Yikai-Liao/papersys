# GitHub Actions 工作流说明

## 📌 概述

已创建 GitHub Actions 工作流 `daily_paper_workflow.yml`，自动执行论文推荐、摘要生成和 Notion 同步流程。

## 🔧 需要配置的 Secrets

在 GitHub 仓库的 **Settings → Secrets and variables → Actions** 中添加以下 secrets：

| Secret 名称 | 用途 | 是否必需 |
|------------|------|---------|
| `GEMINI_API_KEY` | Gemini API 密钥（用于生成摘要） | ✅ 必需 |
| `NOTION_TOKEN` | Notion Integration Token（用于同步） | ✅ 必需 |
| `HF_TOKEN` | Hugging Face Token（用于访问数据集） | ✅ 必需 |
| `PAPERSYS_DATA_TOKEN` | Git 仓库访问 Token（用于推送数据） | ⚠️ 可选* |

\* 如果 `config.toml` 中的 `git_store.repo_url` 使用 SSH 方式，则不需要此 token。

## 🚀 运行方式

### 1️⃣ 自动运行
- **时间**：每天 UTC 3:00（北京时间 11:00）
- **执行内容**：完整流程，处理所有论文
  - `uv run papersys recommend`
  - `uv run papersys summary`
  - `uv run papersys notion-sync`

### 2️⃣ 手动运行 - 完整模式
1. 进入仓库的 **Actions** 标签页
2. 选择 **Daily Paper Workflow**
3. 点击 **Run workflow**
4. 将 `limit` 留空
5. 点击 **Run workflow** 按钮

### 3️⃣ 手动运行 - 测试模式
1. 进入仓库的 **Actions** 标签页
2. 选择 **Daily Paper Workflow**
3. 点击 **Run workflow**
4. 在 `limit` 中输入数字（如 `5`）
5. 点击 **Run workflow** 按钮

**测试模式行为**：
- `recommend`：只推荐指定数量的论文
- `summary`：处理 recommend 输出的所有论文（已被限制）
- `notion-sync`：同步 summary 输出的所有论文（已被限制）

## 📊 工作流特性

- ✅ 使用 `uv` 管理 Python 环境（Python 3.12）
- ✅ 启用依赖缓存，加速后续运行
- ✅ Secret 验证机制，确保必需的环境变量已配置
- ✅ 并发控制，同一时间只运行一个实例
- ✅ 自动生成执行摘要报告

## 🔍 查看运行结果

1. 进入仓库的 **Actions** 标签页
2. 点击最近的工作流运行记录
3. 查看各步骤的详细日志
4. 在 **Summary** 中可以看到执行摘要

## ⚠️ 注意事项

- 首次运行前必须配置好所有必需的 Secrets
- 如果 Secret 缺失，工作流会在验证步骤失败并提示具体缺失的 Secret
- 建议先使用测试模式（`limit=3`）验证配置是否正确
- 定期轮换 API keys 和 tokens 以确保安全

## 📁 相关文件

- 工作流文件：`.github/workflows/daily_paper_workflow.yml`
- 详细配置指南：`.github/SECRETS_SETUP.md`（英文）
