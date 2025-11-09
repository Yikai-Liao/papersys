# GitHub Secrets 配置指南

## 概述

`daily_paper_workflow.yml` 工作流需要以下环境变量作为 GitHub Secrets 配置。请按照以下步骤在你的 GitHub 仓库中添加这些 secrets。

## 必需的 Secrets

### 1. GEMINI_API_KEY
- **用途**: 用于论文摘要生成（summary 命令）
- **获取方式**: 从 Google AI Studio 获取 Gemini API Key
- **设置路径**: Repository Settings → Secrets and variables → Actions → New repository secret

### 2. NOTION_TOKEN
- **用途**: 用于同步论文摘要到 Notion 数据库（notion-sync 命令）
- **获取方式**: 从 Notion Integration 设置页面获取 Internal Integration Token
- **设置路径**: Repository Settings → Secrets and variables → Actions → New repository secret

### 3. HF_TOKEN
- **用途**: 用于访问 Hugging Face 数据集（recommend 命令需要）
- **获取方式**: 从 Hugging Face Settings → Access Tokens 创建
- **设置路径**: Repository Settings → Secrets and variables → Actions → New repository secret

### 4. PAPERSYS_DATA_TOKEN (可选)
- **用途**: 用于推送数据到 Git 仓库（如果 `config.toml` 中 `git_store.repo_url` 使用 HTTPS）
- **获取方式**: 
  - 如果是 GitHub 仓库，使用 Personal Access Token (需要 repo 权限)
  - 如果使用 SSH，则不需要此 token
- **设置路径**: Repository Settings → Secrets and variables → Actions → New repository secret
- **注意**: 如果不设置此 secret，工作流会显示警告但不会失败（除非实际需要推送数据）

## 设置步骤

1. 进入你的 GitHub 仓库页面
2. 点击 **Settings** 标签
3. 在左侧菜单中找到 **Secrets and variables** → **Actions**
4. 点击 **New repository secret** 按钮
5. 输入 Secret 名称（如 `GEMINI_API_KEY`）和对应的值
6. 点击 **Add secret** 保存
7. 重复步骤 4-6 添加其他所需的 secrets

## 工作流触发方式

### 1. 定时触发
- 工作流会在每天 UTC 时间 3:00（北京时间 11:00）自动运行
- 默认会执行所有三个步骤：recommend、summary、notion-sync
- 处理所有推荐的论文（无数量限制）

### 2. 手动触发（完整模式）
- 进入 **Actions** 标签页
- 选择 **Daily Paper Workflow**
- 点击 **Run workflow** 按钮
- 将 **limit** 字段留空
- 点击 **Run workflow** 开始执行
- 将处理所有论文

### 3. 手动触发（测试模式）
- 进入 **Actions** 标签页
- 选择 **Daily Paper Workflow**
- 点击 **Run workflow** 按钮
- 在 **limit** 字段输入一个数字（如 `5`）
- 点击 **Run workflow** 开始执行
- 将只处理指定数量的论文（用于测试）
  - `recommend` 命令会限制推荐数量为指定值
  - `summary` 命令会处理 recommend 输出的所有论文（已被限制）
  - `notion-sync` 命令会同步 summary 输出的所有论文（已被限制）

## 验证配置

添加所有 secrets 后，可以通过以下方式验证：

1. 进入 **Actions** 标签页
2. 手动触发一次工作流
3. 查看 "Validate required secrets" 步骤是否通过
4. 如果所有 secrets 配置正确，会看到 "✅ All required secrets are set" 消息

## 故障排除

### Secret 未设置
如果某个必需的 secret 缺失，工作流会在 "Validate required secrets" 步骤失败，并显示具体缺失的 secret 名称。

### PAPERSYS_DATA_TOKEN 警告
如果看到 "⚠️ Warning: PAPERSYS_DATA_TOKEN is not set"，这是正常的警告信息（如果你的配置不需要通过 HTTPS 推送数据）。工作流会继续执行。

### 权限问题
确保你拥有仓库的管理员权限，才能添加和管理 secrets。

## 安全提示

- ⚠️ **永远不要**将 API keys 或 tokens 直接写在代码中或提交到仓库
- 所有敏感信息必须通过 GitHub Secrets 配置
- 定期轮换你的 API keys 和 tokens
- 如果怀疑某个 token 泄露，立即在对应平台撤销并重新生成
